#!/usr/bin/env python
"""Run the real agent scenarios sequentially and compare runtime modes.

Example:
    uv run python scripts/run_live_agent_benchmark.py \
        --provider ollama --model qwen3.5:9b --modes multiagent single_agent
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_FILE = PROJECT_ROOT / "tests" / "test_live_agent_scenarios.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".test_runs"
MODEL_ENV_BY_PROVIDER = {
    "gigachat": "GIGACHAT_MODEL",
    "ollama": "OLLAMA_MODEL",
    "openrouter": "OPENROUTER_MODEL",
}


@dataclass
class ModeResult:
    mode: str
    return_code: int
    transcript_path: Path
    junit_path: Path
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    pytest_seconds: float = 0.0
    scenario_statuses: dict[str, str] = field(default_factory=dict)
    measured_runs: int = 0
    agent_seconds: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    http_500: int = 0


def _slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean.strip("._-") or "default"


def _scenario_targets(names: Sequence[str]) -> list[str]:
    if not names:
        return [str(SCENARIO_FILE)]
    targets: list[str] = []
    for name in names:
        clean = name.strip()
        if not clean:
            continue
        if "::" in clean or clean.endswith(".py"):
            targets.append(clean)
        else:
            targets.append(f"{SCENARIO_FILE}::{clean}")
    return targets or [str(SCENARIO_FILE)]


def _parse_junit(result: ModeResult) -> None:
    if not result.junit_path.is_file():
        return
    root = ET.parse(result.junit_path).getroot()
    testcases = list(root.iter("testcase"))
    result.pytest_seconds = sum(
        float(case.attrib.get("time") or 0.0) for case in testcases
    )
    for case in testcases:
        name = str(case.attrib.get("name") or "unknown")
        if case.find("skipped") is not None:
            status = "skipped"
            result.skipped += 1
        elif case.find("failure") is not None:
            status = "failed"
            result.failed += 1
        elif case.find("error") is not None:
            status = "error"
            result.errors += 1
        else:
            status = "passed"
            result.passed += 1
        result.scenario_statuses[name] = status


def _metric_values(text: str, pattern: str) -> list[int]:
    return [
        int(value)
        for value in re.findall(pattern, text, flags=re.MULTILINE)
    ]


def _parse_transcript(result: ModeResult) -> None:
    if not result.transcript_path.is_file():
        return
    text = result.transcript_path.read_text(encoding="utf-8")
    seconds = [
        float(value)
        for value in re.findall(r"^agent_seconds: ([0-9.]+)$", text, re.MULTILINE)
    ]
    result.measured_runs = len(seconds)
    result.agent_seconds = sum(seconds)
    result.llm_calls = sum(
        _metric_values(text, r"^llm_calls: (\d+)$")
    )
    token_rows = re.findall(
        r"^tokens: input=(\d+), output=(\d+), total=(\d+), cache_read=(\d+)$",
        text,
        re.MULTILINE,
    )
    result.input_tokens = sum(int(row[0]) for row in token_rows)
    result.output_tokens = sum(int(row[1]) for row in token_rows)
    result.total_tokens = sum(int(row[2]) for row in token_rows)
    result.cache_read_tokens = sum(int(row[3]) for row in token_rows)
    result.http_500 = len(re.findall(r"^### Ответ — HTTP 500$", text, re.MULTILINE))
    for tools_line in re.findall(r"^tools: (.+)$", text, re.MULTILINE):
        clean = tools_line.strip()
        if clean and clean != "Нет":
            result.tool_calls += len(
                [item for item in clean.split(",") if item.strip()]
            )


def _status_mark(status: str | None) -> str:
    return {
        "passed": "✅",
        "failed": "❌",
        "error": "💥",
        "skipped": "⏭",
    }.get(status or "", "—")


def _comparison_report(
    *,
    provider: str,
    model: str,
    results: Sequence[ModeResult],
    report_path: Path,
) -> None:
    lines = [
        f"# Live agent benchmark: {provider} / {model}",
        "",
        "Запросы выполнялись последовательно через реальный HTTP `/chat`, "
        "без mock и параллельных LLM-вызовов.",
        "",
        "| Режим | Passed | Failed | Skipped | HTTP 500 | Agent, с | "
        "LLM calls | Tool calls | Total tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.mode} | {result.passed} | "
            f"{result.failed + result.errors} | {result.skipped} | "
            f"{result.http_500} | {result.agent_seconds:.3f} | "
            f"{result.llm_calls} | {result.tool_calls} | "
            f"{result.total_tokens} |"
        )

    scenario_names = sorted(
        {
            name
            for result in results
            for name in result.scenario_statuses
        }
    )
    if scenario_names:
        lines.extend(
            [
                "",
                "## Сценарии",
                "",
                "| Сценарий | "
                + " | ".join(result.mode for result in results)
                + " |",
                "|---|" + "---:|" * len(results),
            ]
        )
        for name in scenario_names:
            label = name.removeprefix("test_live_agent_")
            lines.append(
                f"| {label} | "
                + " | ".join(
                    _status_mark(result.scenario_statuses.get(name))
                    for result in results
                )
                + " |"
            )

    lines.extend(["", "## Артефакты", ""])
    for result in results:
        lines.append(
            f"- `{result.mode}`: `{result.transcript_path}`; "
            f"JUnit: `{result.junit_path}`"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_mode(
    *,
    mode: str,
    provider: str,
    model: str,
    targets: Sequence[str],
    pytest_args: Sequence[str],
    output_dir: Path,
    run_label: str,
) -> ModeResult:
    transcript_path = output_dir / f"{run_label}_{mode}.md"
    junit_path = output_dir / f"{run_label}_{mode}.xml"
    env = os.environ.copy()
    env.update(
        {
            "RUN_LIVE_AGENT_SCENARIOS": "1",
            "LIVE_AGENT_MODE": mode,
            "LIVE_AGENT_TRANSCRIPT_PATH": str(transcript_path),
            "LLM_PROVIDER": provider,
            "PYTHONUTF8": "1",
        }
    )
    if model:
        env[MODEL_ENV_BY_PROVIDER[provider]] = model

    command = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-q",
        f"--junitxml={junit_path}",
        *pytest_args,
    ]
    print(f"\n=== {mode}: sequential live run ===", flush=True)
    print(" ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    result = ModeResult(
        mode=mode,
        return_code=completed.returncode,
        transcript_path=transcript_path,
        junit_path=junit_path,
    )
    _parse_junit(result)
    _parse_transcript(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Последовательно прогнать реальные agent-сценарии и сравнить режимы."
        )
    )
    parser.add_argument(
        "--provider",
        choices=sorted(MODEL_ENV_BY_PROVIDER),
        default=os.getenv("LLM_PROVIDER", "ollama"),
    )
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("multiagent", "single_agent"),
        default=("multiagent", "single_agent"),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help=(
            "Имя test-функции или полный pytest node id. Можно повторять; "
            "без параметра запускаются все 10 сценариев."
        ),
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Дополнительный аргумент pytest; можно повторять.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Вернуть код 0 после benchmark, даже если acceptance-тесты упали.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = str(args.model or os.getenv(MODEL_ENV_BY_PROVIDER[args.provider], ""))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = "_".join(
        (
            "LIVE_AGENT_BENCHMARK",
            _slug(args.provider),
            _slug(model or "default"),
            timestamp,
        )
    )
    targets = _scenario_targets(args.scenario)
    results = [
        _run_mode(
            mode=mode,
            provider=args.provider,
            model=model,
            targets=targets,
            pytest_args=args.pytest_arg,
            output_dir=output_dir,
            run_label=run_label,
        )
        for mode in args.modes
    ]
    report_path = output_dir / f"{run_label}_comparison.md"
    _comparison_report(
        provider=args.provider,
        model=model or "default",
        results=results,
        report_path=report_path,
    )
    print(f"\nComparison report: {report_path}", flush=True)
    for result in results:
        print(
            f"{result.mode}: passed={result.passed}, "
            f"failed={result.failed + result.errors}, "
            f"tokens={result.total_tokens}, seconds={result.agent_seconds:.3f}",
            flush=True,
        )
    if args.allow_failures:
        return 0
    return 1 if any(result.return_code for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
