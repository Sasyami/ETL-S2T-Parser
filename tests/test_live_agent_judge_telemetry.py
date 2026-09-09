from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

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
