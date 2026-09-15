import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from agents.run_metrics import (
    _metrics_enabled,
    capture_agent_run,
    count_agent_reroutes,
    consume_agent_run_metrics,
    get_run_metrics_callback,
    llm_stage,
    record_coordinator_plan,
    record_display_tools,
    record_entity_resolution,
    record_sql_risk_facts,
    record_sql_risk_operation,
    record_supervisor_decision,
    record_validation_protocol,
    record_worker_observation,
    record_worker_outcome,
    record_worker_route,
    record_worker_task,
    record_upstream_output,
)
def test_reroute_count_does_not_treat_worker_observation_cycle_as_data_cycle():
    metrics = SimpleNamespace(
        worker_routes=[
            SimpleNamespace(routing_attempt=1),
            SimpleNamespace(routing_attempt=2),
        ],
        coordinator_plan=[
            {"cycle": 1, "step": 1},
            {"cycle": 1, "step": 2},
            {"cycle": 2, "step": 1},
        ],
        observations=[SimpleNamespace(cycle=5)],
    )

    assert count_agent_reroutes(metrics) == 2


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
        record_supervisor_decision(
            route="delegate",
            resolved_references="«в ней» = таблица example",
            context="Отвечай кратко.",
        )
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
        record_worker_observation(
            worker_task="Прочитай SQL и граф",
            cycle=2,
            routing_attempt=2,
            observation={
                "status": "reroute",
                "gap": "Не хватает двух источников.",
                "accepted_tool_call_ids": [],
                "facts": [],
                "limitations": [],
                "reroute_reason": "missing_capability",
                "required_capabilities": ["sql_read", "graph_read"],
            },
        )
        record_worker_outcome(
            cycle=1,
            step=1,
            status="partial",
            stop_reason="missing_capability",
            unmet_requirements=["Нужен graph_read."],
            evidence_count=1,
            dataset_count=1,
        )
        record_entity_resolution(
            [
                {
                    "role": "target",
                    "mention": "t_targte",
                    "status": "resolved",
                    "method": "fuzzy",
                    "canonical_name": "t_target",
                    "candidates": ["t_target"],
                }
            ]
        )
        record_sql_risk_facts(
            [
                {
                    "source_table": "src",
                    "source_field": "id",
                    "target_table": "tgt",
                    "target_field": "id",
                    "conclusion": "not_detected",
                    "mechanism": "direct_column",
                    "matching_rows": 1,
                    "target_expressions": ["src.id"],
                    "evidence_ids": ["evidence-sql"],
                }
            ],
            cycle=2,
        )
        record_validation_protocol(
            {
                "mode": "explicit",
                "status": "partial_protocol",
                "readers": ["read_s2t_source_to_target"],
                "checks": [
                    {"kind": "row_count", "status": "ready", "phase": 1},
                    {
                        "kind": "required_null_rate",
                        "status": "unavailable",
                        "phase": 1,
                    },
                ],
            }
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
    assert metrics.supervisor_decision is not None
    assert metrics.supervisor_decision.model_dump() == {
        "route": "delegate",
        "resolved_references": "«в ней» = таблица example",
        "context": "Отвечай кратко.",
    }
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
    assert len(metrics.observations) == 2
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
    assert observation.reroute_reason is None
    assert observation.required_capabilities == []
    reroute_observation = metrics.observations[1]
    assert reroute_observation.status == "reroute"
    assert reroute_observation.reroute_reason == "missing_capability"
    assert reroute_observation.required_capabilities == [
        "sql_read",
        "graph_read",
    ]
    assert metrics.worker_outcomes == [
        {
            "cycle": 1,
            "step": 1,
            "status": "partial",
            "stop_reason": "missing_capability",
            "unmet_requirements": ["Нужен graph_read."],
            "evidence_count": 1,
            "dataset_count": 1,
        }
    ]
    assert metrics.entity_resolution == [
        {
            "role": "target",
            "mention": "t_targte",
            "status": "resolved",
            "method": "fuzzy",
            "canonical_name": "t_target",
            "candidates": ["t_target"],
        }
    ]
    assert metrics.sql_risk_facts == [
        {
            "source_table": "src",
            "source_field": "id",
            "target_table": "tgt",
            "target_field": "id",
            "conclusion": "not_detected",
            "mechanism": "direct_column",
            "matching_rows": 1,
            "target_expressions": ["src.id"],
            "evidence_ids": ["evidence-sql"],
            "cycle": 2,
        }
    ]
    assert metrics.validation_protocol == {
        "mode": "explicit",
        "status": "partial_protocol",
        "readers": ["read_s2t_source_to_target"],
        "checks": [
            {"kind": "row_count", "status": "ready", "phase": 1},
            {
                "kind": "required_null_rate",
                "status": "unavailable",
                "phase": 1,
            },
        ],
    }
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


def test_entity_resolution_metrics_are_bounded_without_provenance(monkeypatch):
    monkeypatch.setenv("AGENT_RUN_METRICS_ENABLED", "1")
    session_id = f"metrics-{uuid4()}"
    candidates = [
        {
            "canonical_name": f"t_candidate_{index}",
            "entity_type": "table",
            "role": "target",
            "file_id": index + 1,
            "score": 0.9,
            "method": "semantic",
            "provenance": [
                {"record_id": index, "full_row": "SECRET" * 1000},
                {"record_id": index + 100, "full_row": "SECRET" * 1000},
            ],
        }
        for index in range(25)
    ]
    event = {
        "mention": "таблица клиентов",
        "entity_type": "table",
        "role": "target",
        "status": "ambiguous",
        "method": "semantic",
        "error_code": "ambiguous_entity",
        "reason": "неоднозначно " * 200,
        "resolver_invoked": True,
        "candidate_set": {
            "coverage": "truncated",
            "source": "semantic_search_descriptions",
            "total_candidates": 90,
            "threshold": 0.6,
            "minimum_gap": 0.05,
            "candidates": candidates,
        },
    }

    with capture_agent_run(session_id):
        record_entity_resolution([event] * 30)
        record_entity_resolution([event] * 30)

    metrics = consume_agent_run_metrics(session_id)
    assert metrics is not None
    assert len(metrics.entity_resolution) == 50
    recorded = metrics.entity_resolution[0]
    assert recorded["status"] == "ambiguous"
    assert recorded["method"] == "semantic"
    assert len(recorded["reason"]) <= 600
    candidate_set = recorded["candidate_set"]
    assert candidate_set["coverage"] == "truncated"
    assert candidate_set["total_candidates"] == 90
    assert candidate_set["candidate_count"] == 25
    assert candidate_set["candidates_truncated"] is True
    assert len(candidate_set["candidates"]) == 20
    assert candidate_set["candidates"][0] == {
        "canonical_name": "t_candidate_0",
        "entity_type": "table",
        "role": "target",
        "file_id": 1,
        "score": 0.9,
        "method": "semantic",
        "provenance_count": 2,
    }
    serialized = json.dumps(metrics.entity_resolution, ensure_ascii=False)
    assert '"provenance"' not in serialized
    assert "SECRET" not in serialized


def test_sql_risk_metrics_accept_payload_and_are_bounded_without_rows(
    monkeypatch,
):
    monkeypatch.setenv("AGENT_RUN_METRICS_ENABLED", "1")
    session_id = f"metrics-{uuid4()}"
    facts = [
        {
            "source_table": f"src_{index}" + "s" * 300,
            "source_field": "id",
            "target_table": f"tgt_{index}",
            "target_field": "id",
            "conclusion": "may_change",
            "mechanism": "value_expression",
            "matching_rows": index,
            "target_expressions": ["x" * 500] * 7,
            "evidence_ids": ["e" * 200] * 12,
            "raw_rows": [{"transformation_rule": "SECRET" * 1000}],
            "unexpected": {"full_result": "SECRET" * 1000},
            "cycle": 777,
        }
        for index in range(12)
    ]

    with capture_agent_run(session_id):
        record_sql_risk_facts(
            {
                "authority": "deterministic_sqlglot_full_saved_result",
                "facts": facts,
                "rows": [{"secret": "SECRET" * 1000}],
            },
            cycle=999,
        )
        record_sql_risk_facts(facts)

    metrics = consume_agent_run_metrics(session_id)
    assert metrics is not None
    assert len(metrics.sql_risk_facts) == 8
    first = metrics.sql_risk_facts[0]
    assert set(first) == {
        "source_table",
        "source_field",
        "target_table",
        "target_field",
        "conclusion",
        "mechanism",
        "matching_rows",
        "target_expressions",
        "evidence_ids",
        "cycle",
    }
    assert len(first["source_table"]) <= 200
    assert len(first["target_expressions"]) == 4
    assert len(first["target_expressions"][0]) <= 300
    assert len(first["evidence_ids"]) == 8
    assert len(first["evidence_ids"][0]) <= 120
    assert first["cycle"] == 100
    serialized = json.dumps(metrics.sql_risk_facts, ensure_ascii=False)
    assert "raw_rows" not in serialized
    assert "full_result" not in serialized
    assert "SECRET" not in serialized


def test_upstream_answer_source_is_optional_and_bounded(monkeypatch):
    monkeypatch.setenv("AGENT_RUN_METRICS_ENABLED", "1")
    session_id = f"metrics-{uuid4()}"

    with capture_agent_run(session_id):
        record_upstream_output(
            {
                "answer": "Детерминированный ответ.",
                "used_evidence_ids": ["evidence-sql"],
                "display_evidence_ids": [],
                "answer_source": "deterministic_value_changes" + "x" * 200,
            }
        )

    metrics = consume_agent_run_metrics(session_id)
    assert metrics is not None
    assert metrics.upstream_output is not None
    assert metrics.upstream_output["answer_source"].startswith(
        "deterministic_value_changes"
    )
    assert len(metrics.upstream_output["answer_source"]) <= 120


def test_sql_risk_metrics_preserve_bounded_constraint_fact(monkeypatch):
    monkeypatch.setenv("AGENT_RUN_METRICS_ENABLED", "1")
    session_id = f"metrics-{uuid4()}"

    with capture_agent_run(session_id):
        record_sql_risk_facts(
            {
                "facts": [
                    {
                        "file_id": 17,
                        "source_table": "src_alpha",
                        "source_field": "code",
                        "target_table": "tgt_beta",
                        "target_field": "code",
                        "source_not_null": 0,
                        "target_not_null": 1,
                        "conclusion": "conditional_rejection_risk",
                        "mechanism": (
                            "nullable_source_to_not_null_target"
                        ),
                        "mapping_rows": 2,
                        "exact_field_rows": 1,
                        "source_metadata_rows": 1,
                        "target_metadata_rows": 1,
                        "evidence_ids": ["evidence-m", "evidence-c"],
                        "rows": [{"secret": "must-not-leak"}],
                    }
                ]
            },
            cycle=1,
        )

    metrics = consume_agent_run_metrics(session_id)
    assert metrics is not None
    assert metrics.sql_risk_facts == [
        {
            "source_table": "src_alpha",
            "source_field": "code",
            "target_table": "tgt_beta",
            "target_field": "code",
            "conclusion": "conditional_rejection_risk",
            "mechanism": "nullable_source_to_not_null_target",
            "file_id": 17,
            "source_not_null": 0,
            "target_not_null": 1,
            "mapping_rows": 2,
            "exact_field_rows": 1,
            "source_metadata_rows": 1,
            "target_metadata_rows": 1,
            "evidence_ids": ["evidence-m", "evidence-c"],
            "cycle": 1,
        }
    ]


def test_sql_risk_metrics_preserve_bounded_cardinality_join_fact(monkeypatch):
    monkeypatch.setenv("AGENT_RUN_METRICS_ENABLED", "1")
    session_id = f"metrics-{uuid4()}"

    with capture_agent_run(session_id):
        record_sql_risk_facts(
            {
                "facts": [
                    {
                        "source_table": "src_alpha",
                        "target_table": "tgt_beta",
                        "conclusion": "conditional_duplicate_risk",
                        "mechanism": "join_fanout",
                        "condition": "full_join_key_uniqueness_unknown",
                        "matching_rows": 2,
                        "joins": [
                            {
                                "join_type": "JOIN",
                                "relation": "aux_table AS d",
                                "predicate": "d.id = s.id",
                                "join_key_equalities": ["d.id = s.id"],
                                "uniqueness_condition": (
                                    "full_join_key_uniqueness_unknown"
                                ),
                                "raw_rows": ["must-not-leak"],
                            }
                        ],
                        "evidence_ids": ["evidence-mapping"],
                        "raw_rows": [{"secret": "must-not-leak"}],
                    }
                ]
            },
            cycle=2,
        )

    metrics = consume_agent_run_metrics(session_id)
    assert metrics is not None
    assert metrics.sql_risk_facts == [
        {
            "source_table": "src_alpha",
            "target_table": "tgt_beta",
            "conclusion": "conditional_duplicate_risk",
            "mechanism": "join_fanout",
            "condition": "full_join_key_uniqueness_unknown",
            "matching_rows": 2,
            "joins": [
                {
                    "join_type": "JOIN",
                    "relation": "aux_table AS d",
                    "predicate": "d.id = s.id",
                    "uniqueness_condition": (
                        "full_join_key_uniqueness_unknown"
                    ),
                    "join_key_equalities": ["d.id = s.id"],
                }
            ],
            "evidence_ids": ["evidence-mapping"],
            "cycle": 2,
        }
    ]
    serialized = json.dumps(metrics.sql_risk_facts, ensure_ascii=False)
    assert "raw_rows" not in serialized
    assert "must-not-leak" not in serialized


def test_sql_risk_operation_trace_round_trips_without_runtime_or_rows(monkeypatch):
    monkeypatch.setenv("AGENT_RUN_METRICS_ENABLED", "1")
    session_id = f"metrics-{uuid4()}"

    with capture_agent_run(session_id):
        record_sql_risk_operation(
            {
                "pipeline": "sql_risk_scope",
                "status": "complete",
                "execution_mode": "conditional_cardinality",
                "scope": "src_alpha → tgt_beta",
                "answer_source": "sql_risk_scope_llm",
                "silent_fallback": False,
                "facts": [
                    {
                        "assessment_status": "complete",
                        "outcome": "risk_present",
                        "structure_status": "ready",
                        "limitations": ["Uniqueness was not supplied."],
                        "reviewed_rule_ids": ["sql_rule_1"],
                        "raw_rows": [{"secret": "must-not-leak"}],
                    }
                ],
                "reads": [
                    {
                        "tool_name": "read_s2t_source_to_target",
                        "arguments": {
                            "source_table": "src_alpha",
                            "target_table": "tgt_beta",
                        },
                        "dataset_ref": "runtime-secret",
                        "raw_rows": [{"secret": "must-not-leak"}],
                    }
                ],
            }
        )

    metrics = consume_agent_run_metrics(session_id)
    assert metrics is not None
    assert metrics.sql_risk_operation is not None
    assert metrics.sql_risk_operation["pipeline"] == "sql_risk_scope"
    assert metrics.sql_risk_operation["silent_fallback"] is False
    assert metrics.sql_risk_operation["assessment"] == {
        "status": "complete",
        "outcome": "risk_present",
        "structure_status": "ready",
        "limitations": ["Uniqueness was not supplied."],
        "reviewed_rule_ids": ["sql_rule_1"],
    }
    serialized = json.dumps(metrics.sql_risk_operation, ensure_ascii=False)
    assert "runtime-secret" not in serialized
    assert "must-not-leak" not in serialized


def test_run_metrics_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_RUN_METRICS_ENABLED", raising=False)
    monkeypatch.delenv("RUN_LIVE_AGENT_SCENARIOS", raising=False)

    with capture_agent_run("disabled-session"):
        assert get_run_metrics_callback() is None

    assert consume_agent_run_metrics("disabled-session") is None


def test_run_metrics_can_be_enabled_by_either_strict_binary_flag(monkeypatch):
    monkeypatch.setenv("AGENT_RUN_METRICS_ENABLED", "1")
    monkeypatch.setenv("RUN_LIVE_AGENT_SCENARIOS", "0")
    assert _metrics_enabled() is True

    monkeypatch.setenv("AGENT_RUN_METRICS_ENABLED", "0")
    monkeypatch.setenv("RUN_LIVE_AGENT_SCENARIOS", "1")
    assert _metrics_enabled() is True


@pytest.mark.parametrize(
    ("invalid_name", "other_name"),
    [
        ("AGENT_RUN_METRICS_ENABLED", "RUN_LIVE_AGENT_SCENARIOS"),
        ("RUN_LIVE_AGENT_SCENARIOS", "AGENT_RUN_METRICS_ENABLED"),
    ],
)
def test_run_metrics_reject_invalid_values_even_when_other_flag_is_enabled(
    monkeypatch,
    invalid_name,
    other_name,
):
    monkeypatch.setenv(invalid_name, "true")
    monkeypatch.setenv(other_name, "1")

    with pytest.raises(
        ValueError,
        match=rf"{invalid_name} must be 0 or 1",
    ):
        _metrics_enabled()
