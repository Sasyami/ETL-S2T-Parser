from pathlib import Path

from scripts.run_live_agent_benchmark import (
    ModeResult,
    _comparison_report,
    _parse_transcript,
    _scenario_mark,
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
