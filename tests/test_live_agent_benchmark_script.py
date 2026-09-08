import ast
from pathlib import Path

import pytest

from scripts import run_live_agent_benchmark as benchmark
from scripts.run_live_agent_benchmark import (
    LIVE_SCENARIO_GROUP_MARKERS,
    SCENARIO_FILE,
    ModeResult,
    _comparison_report,
    _group_pytest_args,
    _parse_transcript,
    _scenario_mark,
    build_parser,
)


def test_benchmark_parser_sums_multiline_execution_metrics(tmp_path):
    transcript = tmp_path / "run.md"
    transcript.write_text(
        """
### Ответ — HTTP 200
agent_seconds: 1.250
llm_calls: 3
tokens: input=100, output=20, total=120, cache_read=10
stage_tokens[supervisor]: calls=1, errors=0, input=20, output=5, total=25, cache_read=2, seconds=0.250
stage_tokens[router]: calls=2, errors=0, input=80, output=15, total=95, cache_read=8, seconds=1.000
tools: run_sql

### Ответ — HTTP 500
agent_seconds: 2.750
llm_calls: 5
tokens: input=200, output=30, total=230, cache_read=15
stage_tokens[supervisor]: calls=1, errors=0, input=30, output=5, total=35, cache_read=3, seconds=0.500
stage_tokens[upstream]: calls=4, errors=1, input=170, output=25, total=195, cache_read=12, seconds=2.000
tools: run_sql, run_cypher
<!-- LIVE_WARNING {"category":"presentation","scenario":"test_live_agent_path","message":"missing display"} -->
<!-- LIVE_WARNING {"category":"efficiency","scenario":"test_live_agent_path","message":"llm_calls=14 exceeds budget=12"} -->
<!-- LIVE_SEMANTIC {"scenario":"test_live_agent_path","status":"not_evaluated"} -->
""".strip(),
        encoding="utf-8",
    )
    result = ModeResult(
        mode="multiagent",
        return_code=1,
        transcript_path=transcript,
        junit_path=Path("missing.xml"),
    )

    _parse_transcript(result)

    assert result.measured_runs == 2
    assert result.agent_seconds == 4.0
    assert result.llm_calls == 8
    assert result.tool_calls == 3
    assert result.input_tokens == 300
    assert result.output_tokens == 50
    assert result.total_tokens == 350
    assert result.cache_read_tokens == 25
    assert result.stage_usage == {
        "supervisor": {
            "calls": 2,
            "errors": 0,
            "input_tokens": 50,
            "output_tokens": 10,
            "total_tokens": 60,
            "cache_read_tokens": 5,
            "elapsed_seconds": 0.75,
        },
        "router": {
            "calls": 2,
            "errors": 0,
            "input_tokens": 80,
            "output_tokens": 15,
            "total_tokens": 95,
            "cache_read_tokens": 8,
            "elapsed_seconds": 1.0,
        },
        "upstream": {
            "calls": 4,
            "errors": 1,
            "input_tokens": 170,
            "output_tokens": 25,
            "total_tokens": 195,
            "cache_read_tokens": 12,
            "elapsed_seconds": 2.0,
        },
    }
    assert result.http_500 == 1
    assert result.presentation_warnings == 1
    assert result.efficiency_warnings == 1
    assert result.scenario_warnings == {
        "test_live_agent_path": ["presentation", "efficiency"]
    }
    assert result.warning_details == [
        {
            "category": "presentation",
            "scenario": "test_live_agent_path",
            "message": "missing display",
        },
        {
            "category": "efficiency",
            "scenario": "test_live_agent_path",
            "message": "llm_calls=14 exceeds budget=12",
        },
    ]
    assert result.semantic_statuses == {
        "test_live_agent_path": "not_evaluated"
    }


def test_benchmark_report_marks_semantics_as_not_evaluated(tmp_path):
    report = tmp_path / "comparison.md"
    result = ModeResult(
        mode="multiagent",
        return_code=0,
        transcript_path=tmp_path / "run.md",
        junit_path=tmp_path / "run.xml",
        passed=1,
        scenario_statuses={"test_live_agent_path": "passed"},
        semantic_statuses={"test_live_agent_path": "not_evaluated"},
        presentation_warnings=1,
        efficiency_warnings=1,
        warning_details=[
            {
                "category": "presentation",
                "scenario": "test_live_agent_path",
                "message": "missing display",
            },
            {
                "category": "efficiency",
                "scenario": "test_live_agent_path",
                "message": "extra call",
            },
        ],
        scenario_warnings={
            "test_live_agent_path": ["presentation", "efficiency"]
        },
        stage_usage={
            "upstream": {
                "calls": 2,
                "errors": 0,
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "cache_read_tokens": 10,
                "elapsed_seconds": 1.25,
            }
        },
    )

    _comparison_report(
        provider="gigachat",
        model="GigaChat-3-Ultra",
        results=[result],
        report_path=report,
    )

    text = report.read_text(encoding="utf-8")
    assert "Pytest passed" in text
    assert "Pytest failures" in text
    assert "Semantic failures" in text
    assert "Presentation warnings" in text
    assert "Efficiency warnings" in text
    assert "📝 ⚠P×1 ⚠E×1" in text
    assert "LLM-as-judge" in text
    assert "сценарий в failed" in text
    assert "## Расход LLM по этапам" in text
    assert "| multiagent | upstream | 2 | 0 | 100 | 20 | 120 | 10 | 1.250 |" in text


def test_benchmark_mark_uses_llm_judge_verdict():
    result = ModeResult(
        mode="multiagent",
        return_code=0,
        transcript_path=Path("run.md"),
        junit_path=Path("run.xml"),
        scenario_statuses={
            "semantic-pass": "passed",
            "semantic-fail": "passed",
            "judge-error": "passed",
        },
        semantic_statuses={
            "semantic-pass": "passed",
            "semantic-fail": "failed",
            "judge-error": "judge_error",
        },
    )

    assert _scenario_mark(result, "semantic-pass") == "✅"
    assert _scenario_mark(result, "semantic-fail") == "❌"
    assert _scenario_mark(result, "judge-error") == "💥"


def test_live_group_filter_builds_stable_or_expression():
    assert _group_pytest_args([]) == []
    assert _group_pytest_args(["history", "catalog", "history"]) == [
        "-m",
        "live_history or live_catalog",
    ]
    with pytest.raises(ValueError, match="unknown live scenario group: missing"):
        _group_pytest_args(["missing"])


def test_benchmark_parser_accepts_only_named_live_groups():
    parser = build_parser()

    args = parser.parse_args(["--group", "history", "--group", "handoff"])

    assert args.group == ["history", "handoff"]
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--group", "missing"])
    assert exc_info.value.code == 2


def test_benchmark_main_combines_exact_scenario_with_group(
    monkeypatch,
    tmp_path,
):
    calls = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs)
        return ModeResult(
            mode=kwargs["mode"],
            return_code=0,
            transcript_path=tmp_path / "run.md",
            junit_path=tmp_path / "run.xml",
        )

    monkeypatch.setattr(benchmark, "_run_mode", fake_run_mode)
    monkeypatch.setattr(benchmark, "_comparison_report", lambda **kwargs: None)

    return_code = benchmark.main(
        [
            "--modes",
            "multiagent",
            "--scenario",
            "test_live_agent_resolves_history_reference_into_task",
            "--group",
            "history",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert return_code == 0
    assert len(calls) == 1
    assert calls[0]["targets"] == [
        f"{SCENARIO_FILE}::test_live_agent_resolves_history_reference_into_task"
    ]
    assert calls[0]["pytest_args"] == ["-m", "live_history"]


def test_benchmark_rejects_competing_marker_expressions(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        benchmark.main(
            [
                "--group",
                "history",
                "--pytest-arg=-m",
                "--pytest-arg=live_graph",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert exc_info.value.code == 2


def _pytest_marker_name(decorator: ast.expr) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if not isinstance(target, ast.Attribute):
        return None
    mark = target.value
    if not isinstance(mark, ast.Attribute) or mark.attr != "mark":
        return None
    if not isinstance(mark.value, ast.Name) or mark.value.id != "pytest":
        return None
    return target.attr


def test_every_live_scenario_belongs_to_exactly_one_semantic_group():
    tree = ast.parse(SCENARIO_FILE.read_text(encoding="utf-8"))
    group_markers = set(LIVE_SCENARIO_GROUP_MARKERS.values())
    assignments = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_live_agent_"):
            continue
        markers = [
            marker
            for decorator in node.decorator_list
            if (marker := _pytest_marker_name(decorator)) in group_markers
        ]
        assignments[node.name] = markers

    assert assignments
    assert all(len(markers) == 1 for markers in assignments.values()), assignments
    assert {markers[0] for markers in assignments.values()} == group_markers
