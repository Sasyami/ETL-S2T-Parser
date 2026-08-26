#!/usr/bin/env python
"""Run the real agent scenarios sequentially and compare runtime modes.

Example:
    uv run python scripts/run_live_agent_benchmark.py \
        --provider ollama --model qwen3.5:9b --modes multiagent single_agent
"""

from __future__ import annotations

import argparse
import json
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
    stage_usage: dict[str, dict[str, int | float]] = field(default_factory=dict)
    http_500: int = 0
    presentation_warnings: int = 0
    efficiency_warnings: int = 0
    warning_details: list[dict[str, str]] = field(default_factory=list)
    scenario_warnings: dict[str, list[str]] = field(default_factory=dict)
    semantic_statuses: dict[str, str] = field(default_factory=dict)


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
    stage_rows = re.findall(
        r"^stage_tokens\[([^\]]+)\]: calls=(\d+), errors=(\d+), "
        r"input=(\d+), output=(\d+), total=(\d+), cache_read=(\d+), "
        r"seconds=([0-9.]+)$",
        text,
        re.MULTILINE,
    )
    for row in stage_rows:
        stage = row[0]
        usage = result.stage_usage.setdefault(
            stage,
            {
                "calls": 0,
                "errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "elapsed_seconds": 0.0,
            },
        )
        usage["calls"] += int(row[1])
        usage["errors"] += int(row[2])
        usage["input_tokens"] += int(row[3])
        usage["output_tokens"] += int(row[4])
        usage["total_tokens"] += int(row[5])
        usage["cache_read_tokens"] += int(row[6])
        usage["elapsed_seconds"] += float(row[7])
    result.http_500 = len(re.findall(r"^### Ответ — HTTP 500$", text, re.MULTILINE))
    warning_rows = re.findall(
        r"^<!-- LIVE_WARNING (\{.+\}) -->$",
        text,
        re.MULTILINE,
    )
    for raw_warning in warning_rows:
        try:
            warning = json.loads(raw_warning)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(warning, dict):
            continue
        category = str(warning.get("category") or "warning").strip().lower()
        scenario = str(warning.get("scenario") or "unknown").strip()
        message = str(warning.get("message") or "").strip()
        detail = {
            "category": category,
            "scenario": scenario,
            "message": message,
        }
        result.warning_details.append(detail)
        result.scenario_warnings.setdefault(scenario, []).append(category)
        if category == "presentation":
            result.presentation_warnings += 1
        elif category == "efficiency":
            result.efficiency_warnings += 1
    semantic_rows = re.findall(
        r"^<!-- LIVE_SEMANTIC (\{.+\}) -->$",
        text,
        re.MULTILINE,
    )
    for raw_status in semantic_rows:
        try:
            semantic_status = json.loads(raw_status)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(semantic_status, dict):
            continue
        scenario = str(semantic_status.get("scenario") or "unknown").strip()
        status = str(
            semantic_status.get("status") or "not_evaluated"
        ).strip()
        result.semantic_statuses[scenario] = status
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


def _scenario_mark(result: ModeResult, scenario: str) -> str:
    technical_status = result.scenario_statuses.get(scenario)
    semantic_status = result.semantic_statuses.get(scenario)
    if technical_status == "passed":
        mark = {
            "passed": "✅",
            "failed": "❌",
            "judge_error": "💥",
            "not_evaluated": "📝",
        }.get(semantic_status or "not_evaluated", "📝")
    else:
        mark = _status_mark(technical_status)
    categories = result.scenario_warnings.get(scenario, [])
    suffixes = []
    presentation_count = categories.count("presentation")
    efficiency_count = categories.count("efficiency")
    if presentation_count:
        suffixes.append(f"⚠P×{presentation_count}")
    if efficiency_count:
        suffixes.append(f"⚠E×{efficiency_count}")
    return " ".join((mark, *suffixes))


def _comparison_report(
    *,
    provider: str,
    model: str,
    results: Sequence[ModeResult],
    report_path: Path,
) -> None:
    evaluated = any(
        status not in {"", "not_evaluated"}
        for result in results
        for status in result.semantic_statuses.values()
    )
    semantic_note = (
        "Содержательная корректность оценена LLM-as-judge: `✅` означает "
        "semantic pass, `❌` — semantic либо technical failure, `💥` — ошибку "
        "judge."
        if evaluated
        else (
            "Содержательная корректность пока не оценивается автоматически: "
            "`📝` означает, что ответ получен и сохранён для ручного разбора. "
            "Позже этот статус сможет заменить LLM-as-judge."
        )
    )
    lines = [
        f"# Live agent benchmark: {provider} / {model}",
        "",
        "Запросы выполнялись последовательно через реальный HTTP `/chat`, "
        "без mock и параллельных LLM-вызовов.",
        "",
        semantic_note,
        "",
        "| Режим | Pytest passed | Pytest failures | Semantic failures | "
        "Skipped | HTTP 500 | Presentation warnings | Efficiency warnings | "
        "Agent, с | LLM calls | Tool calls | Total tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        semantic_failures = sum(
            status in {"failed", "judge_error"}
            for status in result.semantic_statuses.values()
        )
        lines.append(
            f"| {result.mode} | {result.passed} | "
            f"{result.failed + result.errors} | {semantic_failures} | "
            f"{result.skipped} | "
            f"{result.http_500} | {result.presentation_warnings} | "
            f"{result.efficiency_warnings} | {result.agent_seconds:.3f} | "
            f"{result.llm_calls} | {result.tool_calls} | "
            f"{result.total_tokens} |"
        )

    if any(result.stage_usage for result in results):
        lines.extend(
            [
                "",
                "## Расход LLM по этапам",
                "",
                "| Режим | Этап | Calls | Errors | Input | Output | "
                "Total | Cache read | LLM, с |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for result in results:
            for stage, usage in result.stage_usage.items():
                lines.append(
                    f"| {result.mode} | {stage} | {usage['calls']} | "
                    f"{usage['errors']} | {usage['input_tokens']} | "
                    f"{usage['output_tokens']} | {usage['total_tokens']} | "
                    f"{usage['cache_read_tokens']} | "
                    f"{usage['elapsed_seconds']:.3f} |"
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
                    _scenario_mark(result, name)
                    for result in results
                )
                + " |"
            )

    warning_details = [
        (result.mode, detail)
        for result in results
        for detail in result.warning_details
    ]
    if warning_details:
        lines.extend(
            [
                "",
                "## Некритичные предупреждения",
                "",
                "`P` — presentation, `E` — efficiency. Они не переводят "
                "сценарий в failed.",
                "",
                "| Режим | Сценарий | Категория | Детали |",
                "|---|---|---|---|",
            ]
        )
        for mode, detail in warning_details:
            clean_message = detail["message"].replace("|", "\\|")
            lines.append(
                f"| {mode} | {detail['scenario']} | "
                f"{detail['category']} | {clean_message} |"
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
    llm_judge: bool,
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
            "LIVE_AGENT_LLM_JUDGE": "1" if llm_judge else "0",
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
            "без параметра запускаются все сценарии."
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
        "--llm-judge",
        action="store_true",
        help=(
            "Оценить answer и display-results каждого завершённого сценария "
            "настроенной LLM."
        ),
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
            llm_judge=args.llm_judge,
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
