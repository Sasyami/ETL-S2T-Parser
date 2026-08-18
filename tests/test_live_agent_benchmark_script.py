from pathlib import Path

from scripts.run_live_agent_benchmark import ModeResult, _parse_transcript


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
