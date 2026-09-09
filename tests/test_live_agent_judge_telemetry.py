import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

import test_live_agent_scenarios as live_scenarios
from test_live_agent_scenarios import _JudgeTelemetryCallback


def _response(input_tokens: int, output_tokens: int) -> LLMResult:
    return LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="{}",
                        usage_metadata={
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                            "input_token_details": {"cache_read": 2},
                        },
                    )
                )
            ]
        ]
    )


def test_judge_telemetry_counts_recovered_retry_and_provider_usage():
    callback = _JudgeTelemetryCallback()
    failed_attempt = uuid4()
    audit_attempt = uuid4()
    verdict_attempt = uuid4()

    callback.on_chat_model_start({}, [], run_id=failed_attempt)
    callback.on_llm_error(RuntimeError("retry"), run_id=failed_attempt)
    callback.on_chat_model_start({}, [], run_id=audit_attempt)
    callback.on_llm_end(_response(100, 10), run_id=audit_attempt)
    callback.on_chat_model_start({}, [], run_id=verdict_attempt)
    callback.on_llm_end(_response(200, 20), run_id=verdict_attempt)

    assert callback.snapshot(model="GigaChat-2-Max") == {
        "model": "GigaChat-2-Max",
        "attempts": 3,
        "completed": 2,
        "errors": 1,
        "input_tokens": 300,
        "output_tokens": 30,
        "total_tokens": 330,
        "cache_read_tokens": 4,
    }


@pytest.mark.parametrize("semantic_status", ["failed", "judge_error"])
def test_chat_reports_semantic_rejection_as_call_phase_failure_after_recording(
    monkeypatch,
    tmp_path,
    semantic_status,
):
    transcript_path = tmp_path / "semantic.md"

    class Response:
        status_code = 200

        @staticmethod
        def get_json():
            return {"answer": "answer", "display_items": []}

    class Client:
        @staticmethod
        def post(_path, *, json):
            del json
            return Response()

    def record_exchange(*_args, **_kwargs):
        transcript_path.write_text(
            "semantic result recorded\n",
            encoding="utf-8",
        )
        return {
            "status": semantic_status,
            "reason": "semantic reason",
        }

    monkeypatch.setattr(live_scenarios, "_record_live_exchange", record_exchange)
    monkeypatch.setattr(
        live_scenarios,
        "consume_agent_run_metrics",
        lambda _session_id: object(),
    )

    with pytest.raises(pytest.fail.Exception) as failure:
        live_scenarios._chat(Client(), "query")

    assert transcript_path.read_text(encoding="utf-8") == (
        "semantic result recorded\n"
    )
    assert semantic_status in str(failure.value)
    assert "semantic reason" in str(failure.value)


@pytest.mark.parametrize("semantic_status", ["failed", "judge_error"])
def test_exchange_recorder_persists_semantic_result_before_returning_it(
    monkeypatch,
    tmp_path,
    semantic_status,
):
    from agents import llm_factory, semantic_judge

    transcript_path = tmp_path / "semantic.md"
    monkeypatch.setattr(live_scenarios, "LIVE_AGENT_LLM_JUDGE", True)
    monkeypatch.setattr(
        live_scenarios,
        "LIVE_TRANSCRIPT_PATH",
        str(transcript_path),
    )
    monkeypatch.setenv(
        "PYTEST_CURRENT_TEST",
        "tests/test_live.py::test_semantic_result (call)",
    )
    monkeypatch.setattr(
        llm_factory,
        "get_judge_model_name",
        lambda: "GigaChat-2-Max",
    )
    monkeypatch.setattr(
        llm_factory,
        "create_judge_chat_model",
        lambda **_kwargs: SimpleNamespace(callbacks=[]),
    )
    if semantic_status == "failed":
        monkeypatch.setattr(
            semantic_judge,
            "judge_agent_response",
            lambda **_kwargs: SimpleNamespace(
                status="failed",
                reason="semantic reason",
            ),
        )
    else:
        def fail_judge(**_kwargs):
            raise RuntimeError("judge unavailable")

        monkeypatch.setattr(
            semantic_judge,
            "judge_agent_response",
            fail_judge,
        )

    evaluation = live_scenarios._record_live_exchange(
        "query",
        200,
        {"answer": "answer", "display_items": []},
        metrics=None,
        http_elapsed_seconds=0.1,
    )

    assert evaluation is not None
    assert evaluation["status"] == semantic_status
    transcript = transcript_path.read_text(encoding="utf-8")
    marker = transcript.split("<!-- LIVE_SEMANTIC ", 1)[1].split(" -->", 1)[0]
    assert json.loads(marker) == {
        "scenario": "test_semantic_result",
        "status": semantic_status,
        "reason": evaluation["reason"],
    }
