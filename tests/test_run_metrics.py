from types import SimpleNamespace
from uuid import uuid4

from agents.run_metrics import (
    capture_agent_run,
    consume_agent_run_metrics,
    get_run_metrics_callback,
    llm_stage,
    record_coordinator_plan,
    record_display_tools,
    record_worker_observation,
    record_worker_route,
    record_worker_task,
    record_upstream_output,
)


def test_run_metrics_capture_real_callback_events(monkeypatch):
    monkeypatch.setenv("AGENT_RUN_METRICS_ENABLED", "1")
    session_id = f"metrics-{uuid4()}"
    llm_run_id = uuid4()
    tool_run_id = uuid4()

    with capture_agent_run(session_id):
        callback = get_run_metrics_callback()
        assert callback is not None
        with llm_stage("supervisor"):
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
            [
                {
                    "cycle": 1,
                    "step": 1,
                    "task": "Получить единицу",
                }
            ]
        )
        record_coordinator_plan(
            [
                {
                    "cycle": 2,
                    "step": 1,
                    "task": "Добрать проверку единицы",
                }
            ]
        )
        record_worker_route(
            worker_task="Выполни SELECT 1",
            routing_attempt=1,
            tools=["run_sql"],
            skills=[],
            schemas=["SQLite ETL"],
        )
        record_worker_route(
            worker_task="Выполни SELECT 1",
            routing_attempt=2,
            tools=["list_s2t_transformations"],
            skills=["S2T-строки"],
            schemas=["S2T-маппинг"],
            gap="Нужен точный S2T-фильтр.",
        )
        record_worker_observation(
            worker_task="Выполни SELECT 1",
            cycle=1,
            routing_attempt=1,
            observation={
                "status": "complete",
                "gap": None,
                "accepted_tool_call_ids": ["call-sql"],
                "facts": [
                    {
                        "text": "Значение равно 1.",
                        "evidence_ids": ["evidence-sql"],
                    }
                ],
                "limitations": [],
            },
        )
        record_upstream_output(
            {
                "answer": "Единица получена.",
                "used_evidence_ids": ["evidence-sql"],
                "display_evidence_ids": ["evidence-sql"],
            }
        )
        record_display_tools(["run_sql"])

    metrics = consume_agent_run_metrics(session_id)
    assert metrics is not None
    assert metrics.input_tokens == 120
    assert metrics.output_tokens == 30
    assert metrics.total_tokens == 150
    assert metrics.cache_read_tokens == 20
    assert len(metrics.llm_calls) == 1
    assert metrics.llm_calls[0].stage == "supervisor"
    assert metrics.llm_stages[0].model_dump() == {
        "stage": "supervisor",
        "calls": 1,
        "error_calls": 0,
        "elapsed_seconds": metrics.llm_calls[0].elapsed_seconds,
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
        "cache_read_tokens": 20,
    }
    assert [item.name for item in metrics.tool_calls] == ["run_sql"]
    assert metrics.tool_calls[0].arguments == {"query": "SELECT 1"}
    assert metrics.tool_calls[0].input_preview == '{"query":"SELECT 1"}'
    assert metrics.worker_tasks == ["Выполни SELECT 1"]
    assert metrics.coordinator_plan[0]["task"] == "Получить единицу"
    assert [item["cycle"] for item in metrics.coordinator_plan] == [1, 2]
    assert "depends_on" not in metrics.coordinator_plan[0]
    assert "needs_from_previous" not in metrics.coordinator_plan[0]
    assert "required_evidence" not in metrics.coordinator_plan[0]
    assert len(metrics.worker_routes) == 2
    route = metrics.worker_routes[0]
    assert route.worker_task == "Выполни SELECT 1"
    assert route.routing_attempt == 1
    assert route.tools == ["run_sql"]
    assert route.skills == []
    assert route.schemas == ["SQLite ETL"]
    assert route.gap is None
    reroute = metrics.worker_routes[1]
    assert reroute.routing_attempt == 2
    assert reroute.tools == ["list_s2t_transformations"]
    assert reroute.skills == ["S2T-строки"]
    assert reroute.schemas == ["S2T-маппинг"]
    assert reroute.gap == "Нужен точный S2T-фильтр."
    assert len(metrics.observations) == 1
    observation = metrics.observations[0]
    assert observation.worker_task == "Выполни SELECT 1"
    assert observation.cycle == 1
    assert observation.routing_attempt == 1
    assert observation.status == "complete"
    assert observation.gap is None
    assert observation.accepted_tool_call_ids == ["call-sql"]
    assert observation.facts == [
        {
            "text": "Значение равно 1.",
            "evidence_ids": ["evidence-sql"],
        }
    ]
    assert metrics.upstream_output == {
        "answer": "Единица получена.",
        "used_evidence_ids": ["evidence-sql"],
        "display_evidence_ids": ["evidence-sql"],
    }
    assert metrics.display_tools == ["run_sql"]
    assert metrics.elapsed_seconds >= 0
    assert consume_agent_run_metrics(session_id) is None


def test_run_metrics_parse_python_repr_tool_arguments(monkeypatch):
    monkeypatch.setenv("AGENT_RUN_METRICS_ENABLED", "1")
    session_id = f"metrics-{uuid4()}"

    with capture_agent_run(session_id):
        callback = get_run_metrics_callback()
        assert callback is not None
        callback.on_tool_start(
            {"name": "read_s2t_source_to_target"},
            "{'source_table': 'source_name', 'target_table': 'target_name'}",
            run_id=uuid4(),
        )

    metrics = consume_agent_run_metrics(session_id)
    assert metrics is not None
    assert metrics.tool_calls[0].arguments == {
        "source_table": "source_name",
        "target_table": "target_name",
    }


def test_run_metrics_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_RUN_METRICS_ENABLED", raising=False)
    monkeypatch.delenv("RUN_LIVE_AGENT_SCENARIOS", raising=False)

    with capture_agent_run("disabled-session"):
        assert get_run_metrics_callback() is None

    assert consume_agent_run_metrics("disabled-session") is None
