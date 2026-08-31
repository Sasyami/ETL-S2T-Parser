import json
from contextlib import nullcontext
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from agents.chat_graph import (
    Observation as ObservationContract,
    WorkerCycleTrace,
    WorkerDisplayItem,
    WorkerResponseError,
    WorkerRunResult as WorkerRunResultContract,
    run_worker_graph,
)
from agents.tools import get_tools, load_schemas, load_skills
from agents.tools.routing import ToolRoute


def Observation(
    *,
    summary="",
    goal_satisfied=True,
    problem=None,
    accepted_tool_call_ids=None,
    important_facts=None,
    limitations=None,
    reroute_required=False,
    status=None,
    gap=None,
    facts=None,
):
    """Build the new Observation contract from concise test fixtures."""
    selected_status = status or (
        "complete"
        if goal_satisfied
        else "reroute" if reroute_required else "continue"
    )
    selected_gap = gap if gap is not None else problem
    selected_facts = facts
    if selected_facts is None:
        selected_facts = [
            {"text": text, "evidence_ids": []}
            for text in (important_facts or [])
        ]
        if not selected_facts and selected_status == "complete" and summary:
            selected_facts = [{"text": summary, "evidence_ids": []}]
    return ObservationContract(
        status=selected_status,
        gap=selected_gap,
        accepted_tool_call_ids=list(accepted_tool_call_ids or []),
        facts=selected_facts,
        limitations=list(limitations or []),
    )


def WorkerRunResult(
    *,
    answer,
    display_items=None,
    cycle_history=None,
    goal_satisfied=True,
    problem=None,
    reroute_required=False,
    status=None,
    gap=None,
    facts=None,
    accepted_tool_call_ids=None,
):
    selected_status = status or (
        "complete"
        if goal_satisfied or not reroute_required
        else "reroute"
    )
    selected_gap = gap if gap is not None else problem
    return WorkerRunResultContract(
        answer=answer,
        display_items=list(display_items or []),
        cycle_history=list(cycle_history or []),
        status=selected_status,
        gap=selected_gap,
        facts=list(facts or []),
        accepted_tool_call_ids=list(accepted_tool_call_ids or []),
    )


def _as_tool(function, name=None):
    tool_name = name or function.__name__
    return StructuredTool.from_function(
        func=function,
        name=tool_name,
        description=f"Test tool {tool_name}",
    )


def _finish_message(summary, *, extra_args=None):
    args = {"summary": summary}
    args.update(extra_args or {})
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "finish_worker",
                "args": args,
                "id": "finish-1",
                "type": "tool_call",
            }
        ],
    )


def test_worker_prompt_does_not_require_full_table_in_text():
    from agents.chat_graph import _WORKER_PLANNER_PROMPT

    normalized_prompt = " ".join(_WORKER_PLANNER_PROMPT.split())
    assert "не копируй результаты" in normalized_prompt
    assert "краткую внутреннюю отметку" in normalized_prompt
    assert "не формулируй финальный ответ" in normalized_prompt
    assert "для upstream coordinator" not in normalized_prompt
    assert "не переименовывай заданную операцию" in _WORKER_PLANNER_PROMPT
    assert "Не конструируй отсутствующий объект" in _WORKER_PLANNER_PROMPT
    assert "аргументы бери из" in _WORKER_PLANNER_PROMPT
    assert "Если обязательного входа нет, этот tool не подходит" in (
        normalized_prompt
    )
    assert "Палитра worker никогда не пуста" in _WORKER_PLANNER_PROMPT
    assert "внутренний `analyze_known_facts`" in _WORKER_PLANNER_PROMPT
    assert "действие `analyze`" not in _WORKER_PLANNER_PROMPT
    assert "Обычный текст без tool_calls\nзапрещён" in _WORKER_PLANNER_PROMPT
    assert "`finish_worker` разрешён на любом шаге" in _WORKER_PLANNER_PROMPT
    assert "`result_schema` рядом с description" in _WORKER_PLANNER_PROMPT
    assert "одним batch-вызовом" in _WORKER_PLANNER_PROMPT
    assert "не вызывай `read_previous_result` повторно" in (
        _WORKER_PLANNER_PROMPT
    )
    assert "scrollable" not in _WORKER_PLANNER_PROMPT


class _ObserverModel:
    def __init__(self, responses=None):
        self.messages = []
        self.responses = list(responses or [])
        self.last_observation = None

    def invoke(self, messages):
        self.messages.append(messages)
        payload = next(
            json.loads(message.content)
            for message in messages
            if str(getattr(message, "content", "")).lstrip().startswith("{")
        )
        prior_ids = [
            tool_call_id
            for observation in payload.get("prior_state", [])
            for tool_call_id in observation.get("accepted_tool_call_ids", [])
        ]
        current_ids = [
            str(result.get("tool_call_id") or "")
            for result in payload.get("tool_results", [])
            if not result.get("is_error")
            and result.get("name") != "analyze_known_facts"
            and str(result.get("tool_call_id") or "")
        ]
        prior_facts = [
            fact
            for observation in payload.get("prior_state", [])
            for fact in observation.get("facts", [])
        ]
        current_evidence_ids = [
            str(result.get("evidence_id") or "")
            for result in payload.get("tool_results", [])
            if not result.get("is_error")
            and result.get("name") != "analyze_known_facts"
            and str(result.get("evidence_id") or "")
        ]
        response = (
            self.responses.pop(0)
            if self.responses
            else Observation(
                status="complete",
                accepted_tool_call_ids=list(
                    dict.fromkeys([*prior_ids, *current_ids])
                ),
                facts=[
                    *prior_facts,
                    *(
                        [
                            {
                                "text": "Превью результата получено.",
                                "evidence_ids": current_evidence_ids,
                            }
                        ]
                        if current_evidence_ids
                        else []
                    ),
                ],
            )
        )
        observation = (
            response
            if isinstance(response, ObservationContract)
            else ObservationContract.model_validate(response)
        )
        if observation.status == "complete" and not observation.accepted_tool_call_ids:
            observation = observation.model_copy(
                update={
                    "accepted_tool_call_ids": list(
                        dict.fromkeys([*prior_ids, *current_ids])
                    )
                }
            )
        accepted_evidence_ids = list(
            dict.fromkeys(
                [
                    str(result.get("evidence_id") or "")
                    for result in payload.get("tool_results", [])
                    if str(result.get("tool_call_id") or "")
                    in observation.accepted_tool_call_ids
                    and str(result.get("evidence_id") or "")
                ]
                + [
                    evidence_id
                    for fact in prior_facts
                    for evidence_id in fact.get("evidence_ids", [])
                ]
            )
        )
        if accepted_evidence_ids and any(
            not fact.evidence_ids for fact in observation.facts
        ):
            observation = observation.model_copy(
                update={
                    "facts": [
                        fact.model_copy(
                            update={"evidence_ids": accepted_evidence_ids}
                        )
                        if not fact.evidence_ids
                        else fact
                        for fact in observation.facts
                    ]
                }
            )
        self.last_observation = observation
        return observation


class _WorkerModel:
    def __init__(
        self,
        responses,
        *,
        observer_responses=None,
    ):
        self.responses = list(responses)
        self.bound_tools = []
        self.messages = []
        self.observer = _ObserverModel(observer_responses)
        self.structured_methods = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def with_structured_output(self, schema, method=None):
        self.structured_methods.append((schema, method))
        if schema is ObservationContract:
            return self.observer
        raise AssertionError(f"Unexpected structured schema: {schema}")

    def invoke(self, messages, **kwargs):
        del kwargs
        if messages and "Ты observer многошагового агента" in str(
            messages[0].content
        ):
            return self.observer.invoke(messages)
        self.messages.append(messages)
        return self.responses.pop(0)


class _BoundToolChoiceModel:
    def __init__(self, parent, tool_choice=None):
        self.parent = parent
        self.tool_choice = tool_choice

    def invoke(self, messages, **kwargs):
        del kwargs
        return self.parent.invoke_bound(self.tool_choice, messages)


class _ToolChoiceFallbackModel:
    def __init__(self):
        self.forced_lookup_calls = 0
        self.regular_calls = 0
        self.observer = _ObserverModel()

    def bind_tools(self, tools, tool_choice=None):
        del tools
        return _BoundToolChoiceModel(self, tool_choice)

    def with_structured_output(self, schema, method=None):
        assert method == "function_calling"
        if schema is ObservationContract:
            return self.observer
        raise AssertionError(f"Unexpected structured schema: {schema}")

    def invoke(self, messages, **kwargs):
        del kwargs
        if messages and "Ты observer многошагового агента" in str(
            messages[0].content
        ):
            return AIMessage(content="Значение подтверждено.")
        raise AssertionError("Unexpected unbound planner call")

    def invoke_bound(self, tool_choice, messages):
        del messages
        if tool_choice == "lookup":
            self.forced_lookup_calls += 1
            raise RuntimeError("malformed forced tool call")

        self.regular_calls += 1
        if self.regular_calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-fallback",
                        "type": "tool_call",
                    }
                ],
            )
        return _finish_message("Подтверждено через fallback.")


def test_worker_repairs_observer_without_repeating_data_tool():
    tool_calls = []
    stages = []

    def stage_scope(stage):
        stages.append(stage)
        return nullcontext()

    def lookup():
        tool_calls.append("lookup")
        return {"value": 42}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-lookup",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Значение: 42."),
        ],
        observer_responses=[
            {
                "summary": "Невалидная отрицательная observation.",
                "goal_satisfied": False,
                "problem": None,
            },
            Observation(
                summary="Значение 42 подтверждено.",
                goal_satisfied=True,
                important_facts=["Значение: 42."],
            ),
        ],
    )

    with patch("agents.chat_graph.llm_stage", side_effect=stage_scope):
        result = run_worker_graph(
            task="Получи значение.",
            system_prompt="Системный контекст",
            model=model,
            tools=(_as_tool(lookup),),
            max_steps=2,
        )

    assert result.answer == "Значение: 42."
    assert tool_calls == ["lookup"]
    assert len(model.observer.messages) == 2
    repair_messages = model.observer.messages[1]
    assert "Data tool уже выполнен" in repair_messages[-1].content
    assert "не требуй его повторного" in repair_messages[-1].content
    assert stages == [
        "worker_planner",
        "observer",
        "observer",
        "finish_worker",
    ]


def test_worker_repairs_provider_markup_before_executing_tool():
    tool_calls = []

    def lookup(data_type=None):
        tool_calls.append(data_type)
        return {"data_type": "uuid"}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {
                            "data_type": (
                                "}}!#native#!#tool_call_id-00001"
                                "#!#/native#!#native_result!#{"
                            )
                        },
                        "id": "call-invalid",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-valid",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Тип uuid подтверждён."),
        ]
    )

    result = run_worker_graph(
        task="Получи тип данных.",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=2,
    )

    assert result.answer == "Тип uuid подтверждён."
    assert tool_calls == [None]
    assert len(model.observer.messages) == 1
    repair_prompt = str(model.messages[1][-1].content)
    assert "служебную разметку LLM-провайдера" in repair_prompt
    assert "Неизвестные необязательные аргументы полностью опусти" in (
        repair_prompt
    )
    assert "call-invalid" not in str(result)


def test_worker_rejects_unknown_accepted_tool_call_id_without_repeating_tool():
    tool_calls = []

    def lookup():
        tool_calls.append("lookup")
        return {"value": 42}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-lookup",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Значение: 42."),
        ],
        observer_responses=[
            Observation(
                summary="Значение получено.",
                goal_satisfied=True,
                accepted_tool_call_ids=["call-unknown"],
            ),
            Observation(
                summary="Значение 42 подтверждено.",
                goal_satisfied=True,
                accepted_tool_call_ids=["call-lookup"],
            ),
        ],
    )

    result = run_worker_graph(
        task="Получи значение.",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=2,
    )

    assert result.accepted_tool_call_ids == ["call-lookup"]
    assert tool_calls == ["lookup"]
    assert len(model.observer.messages) == 2
    assert "accepted_tool_call_ids" in model.observer.messages[1][-1].content


def test_worker_raises_after_five_observer_retries_without_repeating_tool():
    tool_calls = []

    def lookup():
        tool_calls.append("lookup")
        return {"value": 42}

    invalid_observation = {
        "summary": "Невалидная отрицательная observation.",
        "goal_satisfied": False,
        "problem": None,
    }
    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-lookup",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Значение 42 получено."),
        ],
        observer_responses=[invalid_observation] * 6,
    )

    with pytest.raises(WorkerResponseError, match="6 попыток"):
        run_worker_graph(
            task="Получи значение.",
            system_prompt="Системный контекст",
            model=model,
            tools=(_as_tool(lookup),),
            max_steps=2,
        )
    assert tool_calls == ["lookup"]
    assert len(model.observer.messages) == 6


def test_worker_rejects_complete_when_exact_lineage_scope_was_shortened():
    calls = []

    def trace_neo4j_lineage(
        column_reference: str,
        direction: str = "both",
        max_depth: int = 1,
    ):
        calls.append(
            {
                "column_reference": column_reference,
                "direction": direction,
                "max_depth": max_depth,
            }
        )
        return {
            "rows": [] if max_depth == 1 else [{"target_table": "branch::1"}],
            "max_depth": max_depth,
        }

    full_table = "s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract"
    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "trace_neo4j_lineage",
                        "args": {
                            "column_reference": (
                                "a_000025_t_loanscontract.c_closedate"
                            ),
                            "direction": "downstream",
                        },
                        "id": "call-shortened",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "trace_neo4j_lineage",
                        "args": {
                            "column_reference": f"{full_table}.c_closedate",
                            "direction": "downstream",
                            "max_depth": 5,
                        },
                        "id": "call-exact",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Полный downstream lineage получен."),
        ],
        observer_responses=[
            Observation(
                status="continue",
                gap=(
                    f"Точный column_reference={full_table}.c_closedate и "
                    "требуемая полнота ещё не подтверждены."
                ),
                accepted_tool_call_ids=[],
            ),
            Observation(
                summary="Точный транзитивный lineage получен.",
                goal_satisfied=True,
                accepted_tool_call_ids=["call-exact"],
            ),
        ],
    )

    result = run_worker_graph(
        task=(
            "Выполни reverse lineage для "
            f"{full_table}.c_closedate и перечисли все зависимые transformations."
        ),
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(trace_neo4j_lineage),),
        max_steps=3,
    )

    assert [item["column_reference"] for item in calls] == [
        "a_000025_t_loanscontract.c_closedate",
        f"{full_table}.c_closedate",
    ]
    assert result.status == "complete"
    assert result.accepted_tool_call_ids == ["call-exact"]
    assert [item.tool_call_id for item in result.display_items] == ["call-exact"]
    first_observation = result.cycle_history[0].observation
    assert first_observation.status == "continue"
    assert full_table in str(first_observation.gap)
    assert "column_reference" in str(first_observation.gap)


def test_worker_keeps_valid_lineage_while_fetching_transformation_rules():
    executed = []
    reference = "schema.source_table.c_closedate"

    def trace_neo4j_lineage(
        column_reference: str,
        direction: str = "both",
        max_depth: int = 1,
    ):
        executed.append(("trace_neo4j_lineage", column_reference, max_depth))
        return {
            "rows": [
                {
                    "transformation_id": 118,
                    "source_table": "branch::1",
                    "target_table": "target_table",
                }
            ],
            "column_reference": column_reference,
            "direction": direction,
            "max_depth": max_depth,
        }

    def run_sql(query: str):
        executed.append(("run_sql", query))
        if "WHERE transformation_id" in query:
            return {
                "error": "SQL query failed",
                "error_message": "no such column: transformation_id",
                "query": query,
            }
        return {
            "rows": [
                {
                    "id": 118,
                    "transformation_rule": "UNION ALL",
                }
            ]
        }

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "trace_neo4j_lineage",
                        "args": {
                            "column_reference": reference,
                            "direction": "downstream",
                            "max_depth": 5,
                        },
                        "id": "call-lineage",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_sql",
                        "args": {
                            "query": (
                                "SELECT transformation_rule "
                                "FROM s2t_transformations "
                                "WHERE transformation_id IN (118)"
                            )
                        },
                        "id": "call-rules-wrong-column",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_sql",
                        "args": {
                            "query": (
                                "SELECT id, transformation_rule "
                                "FROM s2t_transformations WHERE id IN (118)"
                            )
                        },
                        "id": "call-rules",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Lineage и зависимые правила получены."),
        ],
        observer_responses=[
            Observation(
                status="continue",
                gap=(
                    "Lineage получен, но фактические transformation rules "
                    "для найденных записей ещё не прочитаны."
                ),
                accepted_tool_call_ids=["call-lineage"],
                facts=[{"text": "Найдена transformation 118."}],
            ),
            Observation(
                status="continue",
                gap=(
                    "Текущий SQL завершился ошибкой; повтори чтение правила "
                    "по фактической схеме источника."
                ),
                accepted_tool_call_ids=["call-lineage"],
                facts=[{"text": "Найдена transformation 118."}],
            ),
            Observation(
                status="complete",
                accepted_tool_call_ids=["call-lineage", "call-rules"],
                facts=[
                    {"text": "Transformation 118 использует UNION ALL."}
                ],
            ),
        ],
    )

    result = run_worker_graph(
        task=(
            f"Выполни reverse lineage для {reference} и перечисли "
            "downstream transformations."
        ),
        system_prompt="Системный контекст",
        model=model,
        tools=(
            _as_tool(trace_neo4j_lineage),
            _as_tool(run_sql),
        ),
        max_steps=3,
    )

    assert result.status == "complete"
    assert result.accepted_tool_call_ids == ["call-lineage", "call-rules"]
    assert result.cycle_history[0].observation.status == "continue"
    assert result.cycle_history[0].observation.accepted_tool_call_ids == [
        "call-lineage"
    ]
    assert "transformation rules" in str(
        result.cycle_history[0].observation.gap
    )
    assert result.cycle_history[1].observation.status == "continue"
    assert result.cycle_history[1].observation.accepted_tool_call_ids == [
        "call-lineage"
    ]
    assert "фактической схеме" in str(
        result.cycle_history[1].observation.gap
    )
    assert [item[0] for item in executed] == [
        "trace_neo4j_lineage",
        "run_sql",
        "run_sql",
    ]


def test_worker_fetches_lineage_rules_by_ids_without_free_sql():
    executed = []
    reference = "schema.source_table.c_closedate"

    def trace_neo4j_lineage(
        column_reference: str,
        direction: str = "both",
        max_depth: int = 1,
    ):
        executed.append(("trace_neo4j_lineage", column_reference, max_depth))
        return {
            "rows": [
                {
                    "transformation_id": 118,
                    "source_table": "branch::1",
                    "target_table": "target_table",
                }
            ],
            "column_reference": column_reference,
            "direction": direction,
            "max_depth": max_depth,
        }

    def get_s2t_rules_by_ids(transformation_ids: list[int]):
        executed.append(("get_s2t_rules_by_ids", transformation_ids))
        return {
            "rows": [
                {
                    "id": 118,
                    "transformation_rule": "UNION ALL",
                }
            ],
            "requested_ids": transformation_ids,
            "missing_ids": [],
        }

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "trace_neo4j_lineage",
                        "args": {
                            "column_reference": reference,
                            "direction": "downstream",
                            "max_depth": 5,
                        },
                        "id": "call-lineage",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_s2t_rules_by_ids",
                        "args": {"transformation_ids": [118]},
                        "id": "call-rules",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Lineage и зависимое правило получены."),
        ],
        observer_responses=[
            Observation(
                status="continue",
                gap=(
                    "Lineage получен, но правила найденных transformations "
                    "ещё не прочитаны доступным tool."
                ),
                accepted_tool_call_ids=["call-lineage"],
                facts=[{"text": "Найдена transformation 118."}],
            ),
            Observation(
                summary="Lineage и правило подтверждены.",
                goal_satisfied=True,
                accepted_tool_call_ids=["call-lineage", "call-rules"],
                facts=[
                    {"text": "Transformation 118 использует UNION ALL."}
                ],
            ),
        ],
    )

    result = run_worker_graph(
        task=(
            f"Выполни reverse lineage для {reference} и перечисли "
            "downstream transformations."
        ),
        system_prompt="Системный контекст",
        model=model,
        tools=(
            _as_tool(trace_neo4j_lineage),
            _as_tool(get_s2t_rules_by_ids),
        ),
        max_steps=2,
    )

    assert result.status == "complete"
    assert result.accepted_tool_call_ids == ["call-lineage", "call-rules"]
    assert result.cycle_history[0].observation.status == "continue"
    assert "доступным tool" in str(
        result.cycle_history[0].observation.gap
    )
    assert result.cycle_history[1].observation.status == "complete"
    assert executed == [
        ("trace_neo4j_lineage", reference, 5),
        ("get_s2t_rules_by_ids", [118]),
    ]


def test_worker_rejects_empty_s2t_result_with_swapped_role_filter():
    executed = []

    def list_s2t_transformations(
        source_table: str | None = None,
        target_table: str | None = None,
        source_field: str | None = None,
        target_field: str | None = None,
    ):
        executed.append(
            {
                "source_table": source_table,
                "target_table": target_table,
                "source_field": source_field,
                "target_field": target_field,
            }
        )
        return {
            "rows": []
        }

    def list_s2t_table_mapping(source_table: str, target_table: str):
        executed.append(
            {
                "source_table": source_table,
                "target_table": target_table,
            }
        )
        return {
            "rows": [
                {
                    "source_table": source_table,
                    "source_field": "src_id",
                    "target_table": target_table,
                    "target_field": "optn_id",
                }
            ]
        }

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_s2t_transformations",
                        "args": {
                            "source_table": "b3050000420005_paymentdetails",
                            "target_field": "t_optn",
                        },
                        "id": "call-wrong-role",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_s2t_table_mapping",
                        "args": {
                            "source_table": "b3050000420005_paymentdetails",
                            "target_table": "t_optn",
                        },
                        "id": "call-correct-role",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Полный mapping получен."),
        ],
        observer_responses=[
            Observation(
                status="continue",
                gap=(
                    "Пустой результат получен с перепутанной ролью; "
                    "нужен точный target_table=t_optn."
                ),
                accepted_tool_call_ids=[],
            ),
            Observation(
                goal_satisfied=True,
                accepted_tool_call_ids=["call-correct-role"],
            ),
        ],
    )

    result = run_worker_graph(
        task=(
            "Покажи полный маппинг b3050000420005_paymentdetails -> "
            "t_optn: source column -> target column."
        ),
        system_prompt="Системный контекст",
        model=model,
        tools=(
            _as_tool(list_s2t_transformations),
            _as_tool(list_s2t_table_mapping),
        ),
        max_steps=2,
    )

    assert result.status == "complete"
    assert result.accepted_tool_call_ids == ["call-correct-role"]
    assert [item.tool_call_id for item in result.display_items] == [
        "call-correct-role"
    ]
    assert result.cycle_history[0].observation.status == "continue"
    assert "target_table=t_optn" in str(
        result.cycle_history[0].observation.gap
    )
    assert executed == [
        {
            "source_table": "b3050000420005_paymentdetails",
            "target_table": None,
            "source_field": None,
            "target_field": "t_optn",
        },
        {
            "source_table": "b3050000420005_paymentdetails",
            "target_table": "t_optn",
        },
    ]


def test_worker_requires_target_roles_for_exact_loaded_field():
    executed = []
    target_table = "b700000025_agr_cred::subquery::v_agr_cred1"

    def search_s2t_transformations(needle: str):
        executed.append(("search_s2t_transformations", needle))
        return {"rows": []}

    def list_s2t_transformations(
        target_table: str | None = None,
        target_field: str | None = None,
    ):
        executed.append(
            ("list_s2t_transformations", target_table, target_field)
        )
        return {
            "rows": [
                {
                    "target_table": target_table,
                    "target_field": target_field,
                    "source_field": "ctl_action",
                    "transformation_rule": "CASE ... END",
                }
            ]
        }

    full_reference = f"{target_table}.del_dt"
    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_s2t_transformations",
                        "args": {"needle": full_reference},
                        "id": "call-search",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_s2t_transformations",
                        "args": {
                            "target_table": target_table,
                            "target_field": "del_dt",
                        },
                        "id": "call-list",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Mappings получены."),
        ],
        observer_responses=[
            Observation(
                status="continue",
                gap=(
                    "Подстрочный поиск не подтвердил точные target_table и "
                    "target_field исходной task."
                ),
                accepted_tool_call_ids=[],
            ),
            Observation(
                goal_satisfied=True,
                accepted_tool_call_ids=["call-list"],
            ),
        ],
    )

    result = run_worker_graph(
        task=f"Найди все S2T mappings, которые загружают {full_reference}.",
        system_prompt="Системный контекст",
        model=model,
        tools=(
            _as_tool(search_s2t_transformations),
            _as_tool(list_s2t_transformations),
        ),
        max_steps=2,
    )

    assert result.status == "complete"
    assert result.accepted_tool_call_ids == ["call-list"]
    assert [item.tool_call_id for item in result.display_items] == ["call-list"]
    assert executed == [
        ("search_s2t_transformations", full_reference),
        ("list_s2t_transformations", target_table, "del_dt"),
    ]


def test_worker_requires_dependent_value_in_current_tool_filter():
    from agents.contracts import (
        WORKER_PREVIOUS_RESULTS_MARKER,
        parse_worker_request,
    )

    queries = []
    selected_target = "t_rate_rule_param"

    def run_sql(query: str):
        queries.append(query)
        return {
            "rows": [
                {
                    "distinct_source_tables": (
                        5 if selected_target in query else 236
                    )
                }
            ]
        }

    wrong_query = (
        "SELECT COUNT(DISTINCT source_table) AS distinct_source_tables "
        "FROM s2t_transformations WHERE source_table IS NOT NULL"
    )
    correct_query = wrong_query + f" AND target_table = '{selected_target}'"
    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_sql",
                        "args": {"query": wrong_query},
                        "id": "call-global-count",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_sql",
                        "args": {"query": correct_query},
                        "id": "call-filtered-count",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Зависимый count получен."),
        ],
        observer_responses=[
            Observation(
                status="continue",
                gap=(
                    f"Глобальный count не применяет target_table="
                    f"{selected_target} из прошлого outcome."
                ),
                accepted_tool_call_ids=[],
            ),
            Observation(
                goal_satisfied=True,
                accepted_tool_call_ids=["call-filtered-count"],
            ),
        ],
    )
    previous_payload = {
        "previous_results": [
            {
                "result_id": "result-first",
                "description": (
                    "run_sql: "
                    "target_table с максимальным числом строк: "
                    f"{selected_target}: 110"
                ),
            }
        ]
    }
    task = (
        "Используя target_table, полученную на предыдущем шаге, посчитай "
        "число различных непустых source_table в s2t_transformations."
        + WORKER_PREVIOUS_RESULTS_MARKER
        + " Используй краткие описания прошлых результатов:\n"
        + json.dumps(previous_payload, ensure_ascii=False)
    )
    request_parts = parse_worker_request(task)
    assert request_parts.current_task.startswith("Используя target_table")
    assert [
        item.model_dump(mode="json", exclude_none=True)
        for item in (request_parts.previous_results or [])
    ] == previous_payload["previous_results"]

    result = run_worker_graph(
        task=task,
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(run_sql),),
        max_steps=2,
    )

    assert result.status == "complete"
    assert result.accepted_tool_call_ids == ["call-filtered-count"]
    assert result.cycle_history[0].observation.status == "continue"
    assert selected_target in str(result.cycle_history[0].observation.gap)
    assert "schema.previous.table" not in str(
        result.cycle_history[0].observation.gap
    )
    assert queries == [wrong_query, correct_query]


def test_worker_reroutes_when_description_is_claimed_as_s2t_rule():
    calls = []

    def semantic_search_descriptions(query: str):
        calls.append(query)
        return {
            "rows": [
                {
                    "column_name": "DEL_DT",
                    "description": "Дата удаления",
                }
            ]
        }

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search_descriptions",
                        "args": {"query": "дата удаления записи"},
                        "id": "call-description",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        observer_responses=[
            Observation(
                status="reroute",
                gap=(
                    "Описание поля не подтверждает transformation_rule; "
                    "нужен другой источник данных."
                ),
                accepted_tool_call_ids=[],
            )
        ],
    )

    result = run_worker_graph(
        task=(
            "Найди техническое поле для даты удаления записи и "
            "соответствующее S2T-правило."
        ),
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(semantic_search_descriptions),),
        max_steps=2,
    )

    assert calls == ["дата удаления записи"]
    assert result.status == "reroute"
    assert "transformation_rule" in str(result.gap)
    assert result.accepted_tool_call_ids == []
    assert result.display_items == []


def test_worker_uses_function_calling_for_observer_contracts():
    def lookup():
        return {"value": 42}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-lookup",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Значение: 42."),
        ]
    )

    run_worker_graph(
        task="Получи значение.",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=2,
    )

    assert model.structured_methods == [
        (ObservationContract, "function_calling"),
    ]


def test_structured_observer_trims_only_surrounding_call_name_spaces():
    from agents.chat_graph import _with_structured_output

    class _RawRunnable:
        def invoke(self, messages, **kwargs):
            del messages, kwargs
            return {
                "raw": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": " Observation ",
                            "args": {
                                "status": "complete",
                                "gap": None,
                                "accepted_tool_call_ids": [],
                                "facts": [],
                                "limitations": [],
                            },
                            "id": "observation-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                "parsed": None,
                "parsing_error": ValueError("unknown tool type"),
            }

    class _RawModel:
        def with_structured_output(
            self,
            schema,
            *,
            method=None,
            include_raw=False,
        ):
            assert schema is ObservationContract
            assert method == "function_calling"
            assert include_raw is True
            return _RawRunnable()

    observer = _with_structured_output(_RawModel(), ObservationContract)

    result = observer.invoke([])

    assert result == ObservationContract(status="complete")


def test_worker_without_tools_is_observed_before_returning_answer():
    candidate_answer = "Первая пара\nВторая пара"
    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_known_facts",
                        "args": {"answer": candidate_answer},
                        "id": "call-analysis",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message(candidate_answer),
        ],
        observer_responses=[
            Observation(
                summary="Обе пары точно перенесены из task.",
                goal_satisfied=True,
                important_facts=["Ответ содержит две пары."],
            )
        ],
    )

    result = run_worker_graph(
        task=(
            "Верни две строки из уже известных фактов: "
            "Первая пара; Вторая пара."
        ),
        system_prompt="Системный контекст",
        model=model,
        tools=(),
        max_steps=2,
    )

    assert result.answer == candidate_answer
    assert result.status == "complete"
    assert result.gap is None
    assert result.display_items == []
    assert len(result.cycle_history) == 1
    cycle = result.cycle_history[0]
    assert cycle.tool_calls == [
        {
            "name": "analyze_known_facts",
            "args": {"answer": candidate_answer},
        }
    ]
    assert len(cycle.tool_results) == 1
    assert cycle.tool_results[0]["name"] == "analyze_known_facts"
    assert cycle.observation.status == "complete"
    observer_payload = json.loads(model.observer.messages[0][-1].content)
    assert observer_payload["tool_calls"][0]["name"] == "analyze_known_facts"
    assert observer_payload["tool_results"][0]["name"] == "analyze_known_facts"
    planner_system = str(model.messages[0][0].content)
    assert "Доступные worker tools:\nanalyze_known_facts" in planner_system
    assert "Палитра worker никогда не пуста" in planner_system
    observer_system = str(model.observer.messages[0][0].content)
    assert "не является новым evidence" in observer_system
    assert observer_payload["candidate_answer"] == ""


def test_worker_keeps_text_preview_and_returns_full_successful_result():
    tail_marker = "FULL_RESULT_TAIL"

    def long_result():
        return {
            "payload": ("x" * 200) + tail_marker,
            "truncated": True,
        }

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "long_result",
                        "args": {},
                        "id": "call-long",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Результат получен."),
        ]
    )

    result = run_worker_graph(
        task="Получи длинный результат",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(long_result),),
        max_steps=2,
        tool_message_preview_chars=50,
    )

    assert result.answer == "Результат получен."
    assert len(model.messages) == 2
    assert len(result.display_items) == 1
    assert result.display_items[0].name == "long_result"
    assert tail_marker in result.display_items[0].content
    assert result.display_items[0].truncated is True
    assert len(result.cycle_history) == 1
    cycle = result.cycle_history[0]
    assert cycle.cycle == 1
    assert cycle.routing_attempt == 1
    assert cycle.tool_calls == [{"name": "long_result", "args": {}}]
    assert cycle.tool_results[0]["name"] == "long_result"
    assert len(cycle.tool_results[0]["content"]) <= 50
    assert tail_marker not in cycle.tool_results[0]["content"]
    assert cycle.observation.facts[0].text == "Превью результата получено."
    assert cycle.observation.facts[0].evidence_ids == [
        cycle.tool_results[0]["evidence_id"]
    ]

    llm_prompts = [*model.messages, *model.observer.messages]
    assert all(
        tail_marker not in str(message.content)
        for prompt in llm_prompts
        for message in prompt
    )
    preview_messages = [
        message
        for prompt in llm_prompts
        for message in prompt
        if isinstance(message, ToolMessage)
    ]
    assert preview_messages
    assert all(isinstance(message.content, str) for message in preview_messages)
    assert all(len(message.content) <= 50 for message in preview_messages)


def test_worker_returns_all_successful_tool_results_for_coordinator():
    def first_result():
        return {"rows": [{"value": "first"}]}

    def second_result():
        return {"rows": [{"value": "second"}]}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "first_result",
                        "args": {},
                        "id": "call-first",
                        "type": "tool_call",
                    },
                    {
                        "name": "second_result",
                        "args": {},
                        "id": "call-second",
                        "type": "tool_call",
                    },
                ],
            ),
            _finish_message("Готово."),
        ]
    )

    result = run_worker_graph(
        task="Получи результаты",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(first_result), _as_tool(second_result)),
        max_steps=3,
    )

    assert [item.name for item in result.display_items] == [
        "first_result",
        "second_result",
    ]
    assert "first" in result.display_items[0].content
    assert "second" in result.display_items[1].content


def test_worker_graph_materializes_sqlite_tool_rows_in_active_store():
    from agents.tools.saved_results import (
        query_saved_result,
        saved_result_store_scope,
    )

    def lookup_rows():
        return {
            "total": 2,
            "rows": [
                {"name": "first", "score": 1},
                {"name": "second", "score": 2},
            ],
        }

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_s2t_transformations",
                        "args": {},
                        "id": "call-sqlite",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Строки получены."),
        ]
    )

    with saved_result_store_scope() as store:
        result = run_worker_graph(
            task="Получи строки",
            system_prompt="Системный контекст",
            model=model,
            tools=(
                _as_tool(
                    lookup_rows,
                    name="list_s2t_transformations",
                ),
            ),
            max_steps=2,
        )

        descriptors = store.descriptors()
        assert len(descriptors) == 1
        descriptor = descriptors[0]
        assert descriptor.source_tool == "list_s2t_transformations"
        assert descriptor.row_count == 2
        assert descriptor.truncated is False
        assert "saved_result" in result.display_items[0].content
        assert "saved_result" in result.cycle_history[0].tool_results[0][
            "content"
        ]

        queried = query_saved_result.invoke(
            {
                "result_ref": descriptor.result_ref,
                "query": "SELECT name FROM result WHERE score = 2",
            }
        )
        assert queried["rows"] == [{"name": "second"}]


def test_worker_binds_referenced_saved_result_schema_into_selected_tool():
    from agents.tools.saved_results import saved_result_store_scope
    from agents.worker import worker_chat

    route = ToolRoute(
        tools=["query_saved_result"],
        skills=[],
        schemas=[],
    )
    graph_result = WorkerRunResult(
        answer="Найдено: 1.",
        display_items=[],
        goal_satisfied=True,
    )

    with saved_result_store_scope() as store:
        descriptor = store.save_payload(
            source_tool="run_sql",
            payload={
                "columns": ["target_table", "row_count"],
                "rows": [{"target_table": "t_example", "row_count": 1}],
            },
        )
        assert descriptor is not None
        task = f"Отфильтруй сохранённый результат {descriptor.result_ref}."

        with (
            patch("agents.worker.select_chat_route", return_value=route) as router,
            patch(
                "agents.worker.run_worker_graph",
                return_value=graph_result,
            ) as run_graph,
        ):
            result = worker_chat(task)

    assert result.summary == "Найдено: 1."
    assert result.datasets == []
    available_tools = router.call_args.kwargs["available_tools"]
    routed_tool = next(
        item for item in available_tools if item.name == "query_saved_result"
    )
    assert descriptor.result_ref in routed_tool.description
    assert '"target_table" TEXT' in routed_tool.description
    selected_tool = run_graph.call_args.kwargs["tools"][0]
    assert selected_tool.name == "query_saved_result"
    assert descriptor.result_ref in selected_tool.description


def test_worker_exposes_only_saved_results_accepted_by_observer():
    from agents.contracts import EvidenceFact
    from agents.tools.saved_results import (
        get_active_saved_result_store,
        read_previous_result,
        saved_result_store_scope,
    )
    from agents.worker import worker_chat

    route = ToolRoute(tools=["run_sql"], skills=[], schemas=[])

    def run_graph(**kwargs):
        del kwargs
        store = get_active_saved_result_store()
        assert store is not None
        store.save_payload(
            source_tool="run_sql",
            source_tool_call_id="call-wrong",
            payload={"rows": [{"value": "wrong"}]},
        )
        store.save_payload(
            source_tool="run_sql",
            source_tool_call_id="call-correct",
            payload={"rows": [{"value": "correct"}]},
        )
        return WorkerRunResult(
            answer="Получен correct.",
            goal_satisfied=True,
            display_items=[
                WorkerDisplayItem(
                    name="run_sql",
                    content=json.dumps({"rows": [{"value": "correct"}]}),
                    evidence_id="evidence-correct",
                    tool_call_id="call-correct",
                    arguments={"query": "SELECT value FROM result"},
                )
            ],
            facts=[
                EvidenceFact(
                    text="Получено значение correct.",
                    evidence_ids=["evidence-correct"],
                )
            ],
            accepted_tool_call_ids=["call-correct"],
        )

    with (
        saved_result_store_scope(),
        patch("agents.worker.select_chat_route", return_value=route),
        patch("agents.worker.run_worker_graph", side_effect=run_graph),
    ):
        result = worker_chat("Получи корректное значение.")
        assert len(result.previous_results) == 1
        reference = result.previous_results[0]
        assert set(reference.model_dump()) == {
            "result_id",
            "description",
            "result_schema",
        }
        assert reference.description == (
            'run_sql: args={"query":"SELECT value FROM result"}'
        )
        assert "correct" not in reference.description
        assert reference.result_schema is not None
        assert reference.result_schema.row_count == 1
        assert [
            (column.name, column.sqlite_type)
            for column in reference.result_schema.columns
        ] == [("value", "TEXT")]
        resolved = read_previous_result.invoke(
            {"result_id": reference.result_id}
        )
        assert resolved["result"]["rows"] == [{"value": "correct"}]

    assert len(result.datasets) == 1
    descriptor = result.datasets[0]
    assert descriptor.source_tool_call_id == "call-correct"
    assert "source_tool_call_id" not in descriptor.model_dump()


def test_worker_repairs_plain_planner_finish_to_native_finish_call():
    def lookup():
        return {"value": "confirmed"}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-lookup",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Подтверждённое значение: confirmed."),
            _finish_message("Подтверждённое значение: confirmed."),
        ]
    )

    result = run_worker_graph(
        task="Получи значение",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=2,
    )

    assert result.answer == "Подтверждённое значение: confirmed."
    assert [item.name for item in result.display_items] == ["lookup"]
    assert len(model.messages) == 3
    assert "Обычный текст planner недопустим" in str(
        model.messages[2][-1].content
    )


def test_worker_raises_when_plain_text_repair_still_has_no_native_call():
    def lookup():
        return {"value": "confirmed"}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-lookup",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Подтверждённое значение: confirmed."),
            AIMessage(content="Подтверждённое значение: confirmed."),
        ]
    )

    with pytest.raises(WorkerResponseError, match="native data-tool call"):
        run_worker_graph(
            task="Получи значение",
            system_prompt="Системный контекст",
            model=model,
            tools=(_as_tool(lookup),),
            max_steps=2,
        )


def test_worker_repairs_plain_text_before_first_data_tool_call():
    def lookup():
        return {"value": "confirmed"}

    model = _WorkerModel(
        [
            AIMessage(content="Сначала я выполню поиск."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-lookup",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Значение получено."),
        ]
    )

    result = run_worker_graph(
        task="Получи значение",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=2,
    )

    assert result.answer == "Значение получено."
    assert [item.name for item in result.display_items] == ["lookup"]
    assert result.status == "complete"
    assert len(model.messages) == 3
    assert "Сначала я выполню поиск." not in str(model.messages[1])


def test_worker_allows_native_finish_before_first_data_tool_call():
    def lookup():
        return {"value": "confirmed"}

    model = _WorkerModel(
        [_finish_message("Данных для проверки недостаточно.")]
    )

    result = run_worker_graph(
        task="Получи значение",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=2,
    )

    assert result.answer == "Данных для проверки недостаточно."
    assert result.display_items == []
    assert result.status == "complete"
    assert len(model.messages) == 1


def test_worker_does_not_force_one_tool_when_finish_is_also_allowed():
    def lookup():
        return {"value": "confirmed"}

    model = _ToolChoiceFallbackModel()

    result = run_worker_graph(
        task="Получи значение",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=2,
    )

    assert result.answer == "Подтверждено через fallback."
    assert [item.name for item in result.display_items] == ["lookup"]
    assert model.forced_lookup_calls == 0
    assert model.regular_calls == 2


def test_worker_planner_keeps_only_latest_tool_exchange():
    def first_result():
        return {"value": "first"}

    def second_result():
        return {"value": "second"}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "first_result",
                        "args": {},
                        "id": "call-first",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "second_result",
                        "args": {},
                        "id": "call-second",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Готово."),
        ],
        observer_responses=[
            Observation(
                summary="Получен только первый результат.",
                goal_satisfied=False,
                problem="Второй результат ещё не получен.",
                accepted_tool_call_ids=["call-first"],
            ),
            Observation(
                summary="Получены оба результата.",
                goal_satisfied=True,
                accepted_tool_call_ids=["call-first", "call-second"],
            ),
        ],
    )

    result = run_worker_graph(
        task="Получи два результата",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(first_result), _as_tool(second_result)),
        max_steps=3,
    )

    assert [item.name for item in result.display_items] == [
        "first_result",
        "second_result",
    ]
    second_prompt_tool_ids = [
        message.tool_call_id
        for message in model.messages[1]
        if isinstance(message, ToolMessage)
    ]
    final_prompt_tool_ids = [
        message.tool_call_id
        for message in model.messages[2]
        if isinstance(message, ToolMessage)
    ]
    assert second_prompt_tool_ids == ["call-first"]
    assert final_prompt_tool_ids == ["call-second"]


def test_worker_planner_keeps_only_latest_cumulative_observation():
    from agents.chat_graph import _runtime_context

    first = Observation(
        summary="Первый результат неполон.",
        goal_satisfied=False,
        problem="Не найден источник.",
    )
    latest = Observation(
        summary="Источник найден, правило ещё отсутствует.",
        goal_satisfied=False,
        problem="Не найдено правило преобразования.",
        important_facts=["Источник: source_contracts."],
    )

    context = _runtime_context({"observations": [first, latest]})

    assert context is not None
    assert "Observation для шага 2" in context
    assert "Источник найден, правило ещё отсутствует." not in context
    assert "Не найдено правило преобразования." in context
    assert "Источник: source_contracts." in context
    assert "Observation для шага 1" not in context
    assert "Первый результат неполон." not in context
    assert "Не найден источник." not in context


def test_worker_observer_evaluates_current_result_with_prior_state():
    calls = []

    def find_source():
        calls.append("source")
        return {"source_table": "source_contracts"}

    def find_rule():
        calls.append("rule")
        return {"transformation_rule": "source.c_closedate"}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_source",
                        "args": {},
                        "id": "call-source",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_rule",
                        "args": {},
                        "id": "call-rule",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Источник и правило подтверждены."),
        ],
        observer_responses=[
            Observation(
                summary="Источник: source_contracts.",
                goal_satisfied=False,
                problem="Правило преобразования ещё не подтверждено.",
                important_facts=["Источник: source_contracts."],
                accepted_tool_call_ids=["call-source"],
            ),
            Observation(
                summary=(
                    "Источник: source_contracts. Правило: "
                    "source.c_closedate."
                ),
                goal_satisfied=True,
                important_facts=[
                    "Источник: source_contracts.",
                    "Правило: source.c_closedate.",
                ],
                accepted_tool_call_ids=["call-source", "call-rule"],
            ),
        ],
    )

    result = run_worker_graph(
        task="Найди источник и правило преобразования c_closedate.",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(find_source), _as_tool(find_rule)),
        max_steps=3,
    )

    assert result.status == "complete"
    assert calls == ["source", "rule"]
    assert len(model.observer.messages) == 2
    second_observer_prompt = model.observer.messages[1]
    second_payload = json.loads(second_observer_prompt[-1].content)
    assert len(second_payload["prior_state"]) == 1
    prior_fact = second_payload["prior_state"][0]["facts"][0]
    assert prior_fact["text"] == "Источник: source_contracts."
    assert len(prior_fact["evidence_ids"]) == 1
    assert prior_fact["evidence_ids"][0].startswith("evidence_")
    observer_system_prompt = " ".join(
        str(second_observer_prompt[0].content).split()
    )
    assert "`prior_state` и `accepted_evidence` накоплены ранее" in (
        observer_system_prompt
    )
    assert "не требуй их повторно" in observer_system_prompt


def test_worker_rejects_legacy_display_selection_field():
    def lookup():
        return {"rows": [{"value": 1}]}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-real",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message(
                "Готово.",
                extra_args={"display_tool_call_ids": ["call-real"]},
            ),
        ]
    )

    with pytest.raises(WorkerResponseError, match="невалидные аргументы"):
        run_worker_graph(
            task="Получи значение",
            system_prompt="Системный контекст",
            model=model,
            tools=(_as_tool(lookup),),
            max_steps=2,
        )


def test_worker_asks_llm_to_finish_after_step_limit():
    def lookup():
        return {"rows": [{"value": 1}]}

    repeated_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "lookup",
                "args": {},
                "id": "call-repeated",
                "type": "tool_call",
            }
        ],
    )
    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-first",
                        "type": "tool_call",
                    }
                ],
            ),
            repeated_call,
            _finish_message("Значение получено."),
        ]
    )

    result = run_worker_graph(
        task="Получи значение",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=1,
    )

    assert result.answer == "Значение получено."
    assert len(model.messages) == 3
    assert "Больше не вызывай data tools" in str(
        model.messages[2][-1].content
    )


def test_worker_returns_unsatisfied_status_after_step_limit():
    def lookup():
        return {"rows": []}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-empty",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Данные не найдены."),
        ],
        observer_responses=[
            Observation(
                summary="Результат пуст.",
                goal_satisfied=False,
                problem="Task ожидала значение, но tool вернул пустой результат.",
            )
        ],
    )

    result = run_worker_graph(
        task="Получи значение.",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=1,
    )

    assert result.status == "complete"
    assert result.gap == (
        "Task ожидала значение, но tool вернул пустой результат."
    )


def test_worker_llm_handles_tool_error_without_backend_branch():
    def lookup(item: str):
        return {
            "error": "Источник недоступен",
            "rows": [],
            "item": item,
        }

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {"item": "значение"},
                        "id": "call-lookup",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message(
                "Источник недоступен; значение проверить не удалось."
            ),
        ]
    )

    result = run_worker_graph(
        task="Проверь значение в источнике.",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=3,
    )

    assert result.answer == (
        "Источник недоступен; значение проверить не удалось."
    )
    assert result.display_items == []
    assert len(model.messages) == 2
    assert "Источник недоступен" in str(model.messages[1])


def test_worker_llm_can_correct_its_tool_call_after_observation():
    executed_queries = []

    def run_sql(query: str):
        executed_queries.append(query)
        return {"rows": [{"target_table": "t_rate_rule_param", "row_count": 55}]}

    wrong_sql = "SELECT source_table, COUNT(*) FROM s2t_transformations GROUP BY source_table"
    correct_sql = (
        "SELECT target_table, COUNT(*) AS row_count FROM s2t_transformations "
        "GROUP BY target_table ORDER BY row_count DESC LIMIT 1"
    )
    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_sql",
                        "args": {"query": wrong_sql},
                        "id": "call-wrong",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_sql",
                        "args": {"query": correct_sql},
                        "id": "call-correct",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Максимум: t_rate_rule_param, 55 строк."),
        ],
        observer_responses=[
            Observation(
                summary="Получена агрегация по source_table вместо target_table.",
                goal_satisfied=False,
                problem="Task требует агрегацию по target_table.",
            ),
            Observation(
                summary="Получена требуемая агрегация по target_table.",
                goal_satisfied=True,
            ),
        ],
    )

    result = run_worker_graph(
        task="Найди target_table с наибольшим числом строк.",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(run_sql),),
        max_steps=2,
    )

    assert result.answer == "Максимум: t_rate_rule_param, 55 строк."
    assert executed_queries == [wrong_sql, correct_sql]
    assert [item.name for item in result.display_items] == ["run_sql"]
    assert result.display_items[0].tool_call_id == "call-correct"
    assert result.display_items[0].arguments == {"query": correct_sql}
    assert "t_rate_rule_param" in result.display_items[0].content


def test_worker_repairs_finish_attempt_after_observer_reports_semantic_mismatch():
    selected_fields = []

    def lookup(field_name: str):
        selected_fields.append(field_name)
        return {field_name: "value"}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {"field_name": "source_table"},
                        "id": "call-wrong-role",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Ошибочно считаю source_table целевой таблицей."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {"field_name": "target_table"},
                        "id": "call-correct-role",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Получено значение target_table."),
        ],
        observer_responses=[
            Observation(
                summary="Tool получил значение поля source_table.",
                goal_satisfied=False,
                problem="Task просит target_table, но tool получил source_table.",
            ),
            Observation(
                summary="Tool получил требуемое значение поля target_table.",
                goal_satisfied=True,
                accepted_tool_call_ids=["call-correct-role"],
            ),
        ],
    )

    result = run_worker_graph(
        task="Получи target_table.",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=2,
    )

    assert result.answer == "Получено значение target_table."
    assert selected_fields == ["source_table", "target_table"]
    assert len(model.observer.messages) == 2
    assert result.status == "complete"
    assert result.gap is None
    assert [item.tool_call_id for item in result.display_items] == [
        "call-correct-role"
    ]
    repair_prompt = str(model.messages[2][-1].content)
    assert "завершать worker сейчас запрещено" in repair_prompt


def test_worker_graph_returns_reroute_after_current_palette_cannot_repair():
    def list_names():
        return {"rows": [{"table_name": "t_example"}]}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_names",
                        "args": {},
                        "id": "call-list-names",
                        "type": "tool_call",
                    }
                ],
            ),
        ],
        observer_responses=[
            Observation(
                summary="Получен только список имён без агрегирования.",
                goal_satisfied=False,
                problem=(
                    "Task требует сравнить количества строк, но текущий tool "
                    "возвращает только имена; нужна возможность произвольной "
                    "агрегации данных."
                ),
                reroute_required=True,
            ),
        ],
    )

    result = run_worker_graph(
        task="Найди таблицу с максимальным числом строк.",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(list_names),),
        max_steps=3,
    )

    assert result.status == "reroute"
    assert result.gap == (
        "Task требует сравнить количества строк, но текущий tool возвращает "
        "только имена; нужна возможность произвольной агрегации данных."
    )
    assert result.display_items == []
    assert len(model.messages) == 1
    observer_payload = str(model.observer.messages[0][-1].content)
    assert '"available_tools"' in observer_payload
    assert '"name": "list_names"' in observer_payload


def test_native_finish_after_semantic_mismatch_does_not_invent_reroute():
    lookup_calls = []

    def lookup():
        lookup_calls.append(True)
        return {"value": "wrong"}

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-lookup",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Текущего результата достаточно."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {},
                        "id": "call-lookup-retry",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Нужное значение не подтверждено."),
        ],
        observer_responses=[
            Observation(
                summary="Получен неподходящий результат.",
                goal_satisfied=False,
                problem="Нужное значение не подтверждено.",
            ),
            Observation(
                summary="Повторно получен неподходящий результат.",
                goal_satisfied=False,
                problem="Нужное значение не подтверждено.",
            ),
        ],
    )

    result = run_worker_graph(
        task="Получи нужное значение.",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=2,
    )

    assert result.answer == "Нужное значение не подтверждено."
    assert result.status == "complete"
    assert result.gap == "Нужное значение не подтверждено."
    assert len(lookup_calls) == 2


def test_worker_loop_does_not_rewrite_llm_tool_arguments_in_python():
    executed_queries = []

    def run_sql(query: str):
        executed_queries.append(query)
        return {"rows": [{"source_count": 5}]}

    wrong_sql = "SELECT COUNT(*) AS source_count FROM target_tables"
    correct_sql = (
        "SELECT COUNT(DISTINCT source_table) AS source_count "
        "FROM s2t_transformations "
        "WHERE target_table = 't_rate_rule_param' "
        "AND source_table IS NOT NULL AND TRIM(source_table) <> ''"
    )
    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_sql",
                        "args": {"query": wrong_sql},
                        "id": "call-wrong-table",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_sql",
                        "args": {"query": correct_sql},
                        "id": "call-correct-table",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Уникальных source_table: 5."),
        ],
        observer_responses=[
            Observation(
                summary="Подсчитаны строки target_tables, а не source_table S2T.",
                goal_satisfied=False,
                problem="Запрос выполнен не по s2t_transformations.",
            ),
            Observation(
                summary="Подсчитаны distinct непустые source_table S2T.",
                goal_satisfied=True,
            ),
        ],
    )

    result = run_worker_graph(
        task=(
            "В s2t_transformations для target_table=t_rate_rule_param "
            "посчитай distinct непустых source_table."
        ),
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(run_sql),),
        max_steps=2,
    )

    assert result.answer == "Уникальных source_table: 5."
    assert executed_queries == [wrong_sql, correct_sql]


def test_public_worker_contract_exposes_evidence_and_opaque_runtime_refs(
    caplog,
):
    from agents.worker import (
        WorkerOutcome,
        resolve_worker_display_refs,
        worker_chat,
    )

    hidden_full_result = "FULL_RESULT_MUST_NOT_LEAVE_WORKER"
    graph_result = WorkerRunResult(
        answer="Готово.",
        display_items=[
            WorkerDisplayItem(
                name="list_files",
                content=hidden_full_result,
                evidence_id="evidence-files",
                tool_call_id="call-files",
                arguments={"limit": 3},
                preview="ограниченное превью",
                truncated=True,
            )
        ],
        cycle_history=[
            WorkerCycleTrace(
                cycle=1,
                tool_calls=[{"name": "list_files", "args": {}}],
                tool_results=[
                    {
                        "name": "list_files",
                        "tool_call_id": "call-files",
                        "content": "ограниченное превью",
                        "status": "success",
                        "is_error": False,
                    }
                ],
                observation=Observation(
                    summary="Файлы получены.",
                    goal_satisfied=False,
                    problem="Нужна дополнительная проверка.",
                    important_facts=["Найдено 3 файла."],
                ),
            )
        ],
        goal_satisfied=False,
        problem="Нужна дополнительная проверка.",
        facts=[
            {
                "text": "Найдено 3 файла.",
                "evidence_ids": ["evidence-files"],
            }
        ],
        accepted_tool_call_ids=["call-files"],
    )
    route = ToolRoute(
        tools=["list_files"],
        skills=["Excel и описания"],
        schemas=["Excel-маппинги"],
    )
    caplog.set_level("INFO", logger="agents.worker")
    with (
        patch("agents.worker.select_chat_route", return_value=route) as router,
        patch(
            "agents.worker.run_worker_graph",
            return_value=graph_result,
        ) as run_graph,
        patch("agents.worker.record_worker_route") as route_recorder,
        patch("agents.worker.record_worker_observation") as observation_recorder,
        patch("agents.worker.load_skills", wraps=load_skills),
        patch("agents.worker.load_schemas", wraps=load_schemas),
    ):
        result = worker_chat("  Покажи файлы  ")

    assert isinstance(result, WorkerOutcome)
    assert result.summary == (
        "Готово.\n"
        "Причина незавершённости: Нужна дополнительная проверка."
    )
    assert not hasattr(result, "gap")
    assert result.facts[0].text == "Найдено 3 файла."
    assert len(result.evidence) == 1
    assert result.evidence[0].evidence_id == "evidence-files"
    assert result.evidence[0].tool_name == "list_files"
    assert result.evidence[0].compact_args == {"limit": 3}
    assert result.evidence[0].preview == "ограниченное превью"
    assert result.evidence[0].truncated is True
    assert hidden_full_result not in result.model_dump_json()
    assert "display_ref" not in result.model_dump_json()
    refs = [
        item.display_ref for item in result.evidence if item.display_ref
    ]
    assert resolve_worker_display_refs(refs) == graph_result.display_items
    assert resolve_worker_display_refs(refs) == []
    assert router.call_args.args == ("Покажи файлы",)
    assert "history" not in router.call_args.kwargs
    assert router.call_args.kwargs["available_tools"] == tuple(
        tool
        for tool in get_tools()
        if tool.name not in {"read_previous_result", "query_saved_result"}
    )
    assert result.previous_results == []
    assert run_graph.call_args.kwargs["task"] == "Покажи файлы"
    assert "## Актуальная схема SQLite" not in run_graph.call_args.kwargs[
        "system_prompt"
    ]
    assert "history" not in run_graph.call_args.kwargs
    assert "file_id" not in run_graph.call_args.kwargs
    route_recorder.assert_called_once_with(
        worker_task="Покажи файлы",
        routing_attempt=1,
        tools=["list_files"],
        skills=["Excel и описания"],
        schemas=["Excel-маппинги"],
        gap=None,
    )
    observation_recorder.assert_called_once()
    observation_call = observation_recorder.call_args.kwargs
    assert observation_call["worker_task"] == "Покажи файлы"
    assert observation_call["cycle"] == 1
    assert observation_call["routing_attempt"] == 1
    assert observation_call["observation"]["status"] == "continue"
    assert observation_call["observation"]["gap"] == (
        "Нужна дополнительная проверка."
    )
    assert "Worker route:" in caplog.text
    assert "Worker observation:" in caplog.text
    assert '"status": "continue"' in caplog.text


def test_public_worker_allows_empty_palette_and_returns_observation():
    from agents.worker import worker_chat

    observation = Observation(
        summary="Форматирование известных фактов выполнено.",
        goal_satisfied=True,
    )
    graph_result = WorkerRunResult(
        answer="Готовый форматированный ответ",
        display_items=[],
        cycle_history=[
            WorkerCycleTrace(
                cycle=1,
                tool_calls=[],
                tool_results=[],
                observation=observation,
            )
        ],
        goal_satisfied=True,
    )
    route = ToolRoute(tools=[], skills=[], schemas=[])

    with (
        patch("agents.worker.select_chat_route", return_value=route),
        patch(
            "agents.worker.run_worker_graph",
            return_value=graph_result,
        ) as run_graph,
    ):
        result = worker_chat("Отформатируй уже известные пары")

    assert result.summary == graph_result.answer
    assert result.evidence == []
    assert result.datasets == []
    assert [
        tool.name for tool in run_graph.call_args.kwargs["tools"]
    ] == ["analyze_known_facts"]


def test_public_worker_adds_previous_result_reader_outside_router():
    from agents.contracts import WORKER_PREVIOUS_RESULTS_MARKER
    from agents.tools.saved_results import saved_result_store_scope
    from agents.worker import worker_chat

    route = ToolRoute(
        tools=["list_s2t_transformations"],
        skills=[],
        schemas=[],
    )
    graph_result = WorkerRunResult(
        answer="Прошлый результат доступен.",
        display_items=[],
        goal_satisfied=True,
    )

    with saved_result_store_scope() as store:
        reference = store.register_previous_result(
            source_tool="semantic_search_descriptions",
            source_tool_call_id="call-semantic",
            content=json.dumps({"rows": [{"column_name": "c_debtlimit"}]}),
            description="semantic_search_descriptions: найден кандидат колонки",
        )
        task = (
            "Получи S2T для найденной колонки."
            + WORKER_PREVIOUS_RESULTS_MARKER
            + "\n"
            + json.dumps(
                {"previous_results": [reference.model_dump(mode="json")]},
                ensure_ascii=False,
            )
        )
        with (
            patch("agents.worker.select_chat_route", return_value=route) as router,
            patch(
                "agents.worker.run_worker_graph",
                return_value=graph_result,
            ) as run_graph,
        ):
            result = worker_chat(task)

    assert result.summary == "Прошлый результат доступен."
    assert "read_previous_result" not in {
        tool.name for tool in router.call_args.kwargs["available_tools"]
    }
    assert [
        tool.name for tool in run_graph.call_args.kwargs["tools"]
    ] == ["list_s2t_transformations", "read_previous_result"]


def test_public_worker_reroutes_original_task_after_observer_request():
    from agents.worker import worker_chat

    routes = [
        ToolRoute(
            tools=["list_s2t_transformations"],
            skills=["S2T-строки"],
            schemas=[],
        ),
        ToolRoute(
            tools=["list_s2t_transformations", "trace_transformation_path"],
            skills=[],
            schemas=["S2T-маппинг"],
        ),
    ]
    graph_results = [
        WorkerRunResult(
            answer="Нужна другая палитра.",
            cycle_history=[
                WorkerCycleTrace(
                    cycle=1,
                    tool_calls=[
                        {"name": "list_s2t_transformations", "args": {}}
                    ],
                    tool_results=[],
                    observation=Observation(
                        summary="Агрегирование не выполнено.",
                        goal_satisfied=False,
                        problem="Текущий tool не строит многошаговый путь с rules.",
                        reroute_required=True,
                    ),
                )
            ],
            goal_satisfied=False,
            problem="Текущий tool не строит многошаговый путь с rules.",
            reroute_required=True,
        ),
        WorkerRunResult(
            answer="Максимум: t_example, 55 строк.",
            cycle_history=[
                WorkerCycleTrace(
                    cycle=1,
                    tool_calls=[{"name": "trace_transformation_path", "args": {}}],
                    tool_results=[],
                    observation=Observation(
                        summary="Максимум найден.",
                        goal_satisfied=True,
                        important_facts=["t_example: 55 строк."],
                    ),
                )
            ],
            goal_satisfied=True,
        ),
    ]

    with (
        patch(
            "agents.worker.select_chat_route",
            side_effect=routes,
        ) as router,
        patch(
            "agents.worker.run_worker_graph",
            side_effect=graph_results,
        ) as run_graph,
    ):
        result = worker_chat("  Найди target_table с максимумом строк  ")

    assert result.summary == "Максимум: t_example, 55 строк."
    assert router.call_count == 2
    assert router.call_args_list[0].args == (
        "Найди target_table с максимумом строк",
    )
    assert "reroute_context" not in router.call_args_list[0].kwargs
    assert router.call_args_list[1].args == (
        "Найди target_table с максимумом строк",
    )
    reroute_context = router.call_args_list[1].kwargs["reroute_context"]
    assert reroute_context == {
        "gap": "Текущий tool не строит многошаговый путь с rules.",
        "previous_tool_palettes": [["list_s2t_transformations"]],
        "attempt": 1,
    }
    selected_tool_names = [
        [tool.name for tool in item.kwargs["tools"]]
        for item in run_graph.call_args_list
    ]
    assert selected_tool_names[0] == ["list_s2t_transformations"]
    assert set(selected_tool_names[1]) == {
        "list_s2t_transformations",
        "trace_transformation_path",
    }


def test_public_worker_can_execute_repeated_reroute_palette():
    from agents.worker import worker_chat

    repeated_route = ToolRoute(
        tools=["list_s2t_transformations"],
        skills=["S2T-строки"],
        schemas=[],
    )
    graph_results = [
        WorkerRunResult(
            answer="Нужно исправить запрос.",
            goal_satisfied=False,
            problem="SQL не учитывает нужный фильтр.",
            reroute_required=True,
        ),
        WorkerRunResult(
            answer="Максимум: t_example, 55 строк.",
            goal_satisfied=True,
        ),
    ]

    with (
        patch(
            "agents.worker.select_chat_route",
            return_value=repeated_route,
        ) as router,
        patch(
            "agents.worker.run_worker_graph",
            side_effect=graph_results,
        ) as run_graph,
    ):
        result = worker_chat("Найди target_table с максимумом строк")

    assert result.summary == "Максимум: t_example, 55 строк."
    assert router.call_count == 2
    assert run_graph.call_count == 2
    assert [
        [tool.name for tool in item.kwargs["tools"]]
        for item in run_graph.call_args_list
    ] == [["list_s2t_transformations"], ["list_s2t_transformations"]]
    second_system_prompt = run_graph.call_args_list[1].kwargs["system_prompt"]
    assert "<reroute_feedback>" in second_system_prompt
    assert "SQL не учитывает нужный фильтр." in second_system_prompt
    assert "с помощью расширенной палитры" in second_system_prompt
    assert router.call_args_list[1].kwargs["reroute_context"] == {
        "gap": "SQL не учитывает нужный фильтр.",
        "previous_tool_palettes": [["list_s2t_transformations"]],
        "attempt": 1,
    }
