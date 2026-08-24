from pathlib import Path

from scripts.run_live_agent_benchmark import (
    ModeResult,
    _comparison_report,
    _parse_transcript,
)


def test_benchmark_parser_sums_multiline_execution_metrics(tmp_path):
    transcript = tmp_path / "run.md"
    transcript.write_text(
        """
### Ответ — HTTP 200
agent_seconds: 1.250
llm_calls: 3
tokens: input=100, output=20, total=120, cache_read=10
tools: run_sql

### Ответ — HTTP 500
agent_seconds: 2.750
llm_calls: 5
tokens: input=200, output=30, total=230, cache_read=15
tools: run_sql, run_cypher
<!-- LIVE_WARNING {"category":"presentation","scenario":"test_live_agent_path","message":"missing display"} -->
<!-- LIVE_WARNING {"category":"efficiency","scenario":"test_live_agent_path","message":"llm_calls=14 exceeds budget=12"} -->
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


def test_benchmark_report_separates_critical_failures_from_warnings(tmp_path):
    report = tmp_path / "comparison.md"
    result = ModeResult(
        mode="multiagent",
        return_code=0,
        transcript_path=tmp_path / "run.md",
        junit_path=tmp_path / "run.xml",
        passed=1,
        scenario_statuses={"test_live_agent_path": "passed"},
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
    )

    _comparison_report(
        provider="gigachat",
        model="GigaChat-3-Ultra",
        results=[result],
        report_path=report,
    )

    text = report.read_text(encoding="utf-8")
    assert "Critical failures" in text
    assert "Presentation warnings" in text
    assert "Efficiency warnings" in text
    assert "✅ ⚠P×1 ⚠E×1" in text
    assert "сценарий в failed" in text
