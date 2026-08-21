from types import SimpleNamespace
from uuid import uuid4

from agents.run_metrics import (
    capture_agent_run,
    consume_agent_run_metrics,
    get_run_metrics_callback,
    record_aggregate_result,
    record_coordinate_result,
    record_coordinator_plan,
    record_display_tools,
    record_worker_observation,
    record_worker_route,
    record_worker_task,
)


def test_run_metrics_capture_real_callback_events(monkeypatch):
    monkeypatch.setenv("AGENT_RUN_METRICS_ENABLED", "1")
    session_id = f"metrics-{uuid4()}"
    llm_run_id = uuid4()
    tool_run_id = uuid4()

    with capture_agent_run(session_id):
        callback = get_run_metrics_callback()
        assert callback is not None
        callback.on_chat_model_start(
            {"id": ["langchain", "GigaChat"]},
            [[]],
            run_id=llm_run_id,
        )
        callback.on_llm_end(
            SimpleNamespace(
                llm_output={
                    "token_usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 30,
                        "total_tokens": 150,
                        "precached_prompt_tokens": 20,
                    }
                },
                generations=[],
            ),
            run_id=llm_run_id,
        )
        callback.on_tool_start(
            {"name": "run_sql"},
            '{"query":"SELECT 1"}',
            run_id=tool_run_id,
        )
        callback.on_tool_end("result", run_id=tool_run_id)
        record_worker_task("Выполни SELECT 1")
        record_coordinator_plan(
            [{"step": 1, "goal": "Получить единицу", "presentation": "answer_only"}]
        )
        record_worker_route(
            worker_task="Выполни SELECT 1",
            routing_attempt=1,
            tools=["run_sql"],
            skills=["SQLite SQL"],
            schemas=["SQLite ETL"],
        )
        record_worker_route(
            worker_task="Выполни SELECT 1",
            routing_attempt=2,
            tools=["list_s2t_transformations"],
            skills=["S2T-строки"],
            schemas=["S2T-маппинг"],
            reroute_reason="Нужен точный S2T-фильтр.",
        )
        record_worker_observation(
            worker_task="Выполни SELECT 1",
            cycle=1,
            routing_attempt=1,
            observation={
                "summary": "Получено значение 1.",
                "goal_satisfied": True,
                "mismatches": [],
                "has_error": False,
                "important_facts": ["Значение равно 1."],
                "limitations": [],
                "reroute_required": False,
                "reroute_reason": None,
            },
        )
        record_coordinate_result(
            {"answer": "Единица получена.", "display_result_keys": []}
        )
        record_aggregate_result("Единица получена.")
        record_display_tools(["run_sql"])

    metrics = consume_agent_run_metrics(session_id)
    assert metrics is not None
    assert metrics.input_tokens == 120
    assert metrics.output_tokens == 30
    assert metrics.total_tokens == 150
    assert metrics.cache_read_tokens == 20
    assert len(metrics.llm_calls) == 1
    assert [item.name for item in metrics.tool_calls] == ["run_sql"]
    assert metrics.worker_tasks == ["Выполни SELECT 1"]
    assert metrics.coordinator_plan[0]["goal"] == "Получить единицу"
    assert len(metrics.worker_routes) == 2
    route = metrics.worker_routes[0]
    assert route.worker_task == "Выполни SELECT 1"
    assert route.routing_attempt == 1
    assert route.tools == ["run_sql"]
    assert route.skills == ["SQLite SQL"]
    assert route.schemas == ["SQLite ETL"]
    assert route.reroute_reason is None
    reroute = metrics.worker_routes[1]
    assert reroute.routing_attempt == 2
    assert reroute.tools == ["list_s2t_transformations"]
    assert reroute.skills == ["S2T-строки"]
    assert reroute.schemas == ["S2T-маппинг"]
    assert reroute.reroute_reason == "Нужен точный S2T-фильтр."
    assert len(metrics.observations) == 1
    observation = metrics.observations[0]
    assert observation.worker_task == "Выполни SELECT 1"
    assert observation.cycle == 1
    assert observation.routing_attempt == 1
    assert observation.summary == "Получено значение 1."
    assert observation.goal_satisfied is True
    assert observation.important_facts == ["Значение равно 1."]
    assert metrics.coordinate_result == {
        "answer": "Единица получена.",
        "display_result_keys": [],
    }
    assert metrics.aggregate_result == "Единица получена."
    assert metrics.display_tools == ["run_sql"]
    assert metrics.elapsed_seconds >= 0
    assert consume_agent_run_metrics(session_id) is None


def test_run_metrics_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_RUN_METRICS_ENABLED", raising=False)
    monkeypatch.delenv("RUN_LIVE_AGENT_SCENARIOS", raising=False)

    with capture_agent_run("disabled-session"):
        assert get_run_metrics_callback() is None

    assert consume_agent_run_metrics("disabled-session") is None
