import json
from contextlib import nullcontext
from unittest.mock import call, patch

import pytest
from langchain_core.messages import AIMessage

from agents.coordinator import CoordinatorAnswer, CoordinatorResponseError
from agents.worker import WorkerAnswer, WorkerResultRef


def _tool_message(name, args, call_id):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


class _BoundModel:
    def __init__(self, parent, tool_name):
        self.parent = parent
        self.tool_name = tool_name

    def invoke(self, messages, **kwargs):
        del kwargs
        self.parent.messages.append((self.tool_name, list(messages)))
        return self.parent.responses[self.tool_name].pop(0)


class _CoordinatorModel:
    def __init__(self, responses):
        self.responses = {name: list(items) for name, items in responses.items()}
        self.messages = []

    def bind_tools(self, tools):
        return _BoundModel(self, tools[0]["function"]["name"])


def _payload(model, tool_name, occurrence=0):
    matching = [messages for name, messages in model.messages if name == tool_name]
    return json.loads(matching[occurrence][1].content)


def _patches(model):
    return (
        patch("agents.coordinator.chat_model", model),
        patch("agents.coordinator.get_callback_handler", return_value=None),
        patch(
            "agents.coordinator.langfuse_trace_context",
            return_value=nullcontext(),
        ),
    )


def test_coordinator_graph_has_llm_plan_worker_loop_and_aggregate():
    from agents.coordinator import build_coordinator_graph

    graph = build_coordinator_graph(_CoordinatorModel({}))
    graph_view = graph.get_graph()

    assert {"plan", "materialize", "worker", "aggregate"}.issubset(
        graph_view.nodes
    )
    edges = {(edge.source, edge.target) for edge in graph_view.edges}
    assert ("__start__", "plan") in edges
    assert ("plan", "materialize") in edges
    assert ("materialize", "worker") in edges
    assert ("aggregate", "__end__") in edges


def test_coordinator_prompts_are_generic_not_domain_contracts():
    from agents.coordinator import (
        _AGGREGATE_PROMPT,
        _DISPATCH_PROMPT,
        _PLAN_PROMPT,
        _plan_tool_schema,
    )

    combined = "\n".join(
        (_PLAN_PROMPT, _DISPATCH_PROMPT, _AGGREGATE_PROMPT)
    )
    for domain_detail in (
        "target_table",
        "source_table",
        "GROUP BY",
        "COUNT(DISTINCT",
        "Neo4j",
        "SQLite",
    ):
        assert domain_detail not in combined

    assert "явно задал N отдельных операций с данными" in _PLAN_PROMPT
    assert "Главный приоритет — правильный итоговый ответ" in _PLAN_PROMPT
    assert "не достраивай отсутствующую предметную схему" in _PLAN_PROMPT
    normalized_plan = " ".join(_PLAN_PROMPT.split()).lower()
    assert "не оставляй местоимение в goal" in normalized_plan
    assert "не выбирай объект по умолчанию" in normalized_plan
    assert "не являются новыми worker-шагами" in _PLAN_PROMPT
    assert "верни один step" in _PLAN_PROMPT
    assert "никогда не становится отдельным goal" in _PLAN_PROMPT
    assert "Никогда не создавай повторный шаг" in _PLAN_PROMPT
    assert "это один step" in _PLAN_PROMPT
    assert "только число" in _PLAN_PROMPT
    assert "не переноси `full_results`" in _PLAN_PROMPT
    assert "понадобятся зависимым шагам" in _PLAN_PROMPT
    assert "answer_only по умолчанию" in str(_plan_tool_schema())
    assert "не показ уже полученного результата" in str(_plan_tool_schema())
    assert "не придумывай отсутствующие во входе" in _DISPATCH_PROMPT.lower()
    assert "original_task" in _DISPATCH_PROMPT
    assert "Отдельный показ, экспорт" in _DISPATCH_PROMPT
    assert "не отвлекай worker" in _DISPATCH_PROMPT
    assert "только число" in " ".join(_DISPATCH_PROMPT.split())
    assert "только если предыдущий ответ" in _DISPATCH_PROMPT
    assert "неоднозначному значению нужный смысл" in _DISPATCH_PROMPT
    assert "копируй посимвольно" in _DISPATCH_PROMPT
    assert "не создавай их варианты" in _DISPATCH_PROMPT
    normalized_dispatch = " ".join(_DISPATCH_PROMPT.split()).lower()
    assert "не опускай явно названный исходный набор данных" in normalized_dispatch
    assert "значение роли остаётся условием" in _DISPATCH_PROMPT
    assert "Worker не должен восстанавливать источник" in _DISPATCH_PROMPT
    assert "обязательно замени" in _DISPATCH_PROMPT
    assert "worker должен получить «таблица X»" in " ".join(_DISPATCH_PROMPT.split())
    assert "не проси текущий worker повторно" in " ".join(_DISPATCH_PROMPT.split()).lower()
    assert "только новые факты текущей операции" in _DISPATCH_PROMPT
    assert "Для `answer_only` не выбирай result keys" in _AGGREGATE_PROMPT
    assert "Главный приоритет — правильный текстовый ответ" in _AGGREGATE_PROMPT
    assert "Считай шаг выполненным только если" in _AGGREGATE_PROMPT
    assert "при сомнении" in _AGGREGATE_PROMPT
    assert "Не повторяй в answer отрицательные ограничения" in _AGGREGATE_PROMPT
    assert "молча соблюдай такие запреты" in _AGGREGATE_PROMPT
    assert "При требовании «только» верни ровно" in _AGGREGATE_PROMPT
    assert "без\nвступления, заключения" in _AGGREGATE_PROMPT
    assert "`answer`: всегда строка" in _AGGREGATE_PROMPT
    assert "сериализуй" in _AGGREGATE_PROMPT


def test_coordinator_llms_plan_dispatch_dependencies_and_select_results(caplog):
    from agents.coordinator import coordinator_chat

    model = _CoordinatorModel(
        {
            "submit_worker_plan": [
                _tool_message(
                    "submit_worker_plan",
                    {
                        "steps": [
                            {"goal": "Найти имя", "presentation": "answer_only"},
                            {
                                "goal": "Проверить найденное имя",
                                "presentation": "full_results",
                            },
                        ]
                    },
                    "plan-1",
                )
            ],
            "dispatch_worker": [
                _tool_message(
                    "dispatch_worker",
                    {"task": "Найди точное имя."},
                    "dispatch-1",
                ),
                _tool_message(
                    "dispatch_worker",
                    {"task": "Проверь точное имя t_example."},
                    "dispatch-2",
                ),
            ],
            "finish_coordination": [
                _tool_message(
                    "finish_coordination",
                    {
                        "answer": "Имя t_example проверено.",
                        "display_result_keys": ["step-2:result-1"],
                    },
                    "finish-1",
                )
            ],
        }
    )
    worker_results = [
        WorkerAnswer(
            answer="Точное имя: t_example.",
            result_refs=[WorkerResultRef(ref="ref-first", name="lookup")],
        ),
        WorkerAnswer(
            answer="Имя t_example проверено.",
            result_refs=[WorkerResultRef(ref="ref-second", name="inspect")],
        ),
    ]
    model_patch, callback_patch, trace_patch = _patches(model)
    with caplog.at_level("INFO", logger="agents.coordinator"):
        with (
            model_patch,
            callback_patch,
            trace_patch,
            patch(
                "agents.coordinator.worker_chat",
                side_effect=worker_results,
            ) as worker,
            patch("agents.coordinator.discard_worker_result_refs") as discard,
        ):
            result = coordinator_chat(
                "Найди имя и проверь его.",
                context="Общий фон",
            )

    assert result == CoordinatorAnswer(
        answer="Имя t_example проверено.",
        display_refs=["ref-second"],
    )
    assert worker.call_args_list == [
        call("Найди точное имя."),
        call("Проверь точное имя t_example."),
    ]
    assert discard.call_args_list == [call(["ref-first"])]
    second_dispatch = _payload(model, "dispatch_worker", 1)
    assert second_dispatch["original_task"] == "Найди имя и проверь его."
    assert second_dispatch["current_step"] == {
        "step": 2,
        "goal": "Проверить найденное имя",
    }
    assert second_dispatch["completed_workers"][0]["answer"] == (
        "Точное имя: t_example."
    )
    aggregate = _payload(model, "finish_coordination")
    assert aggregate["worker_results"][1]["available_results"] == [
        {"result_key": "step-2:result-1", "tool_name": "inspect"}
    ]
    assert "Coordinator planned worker_steps=2 plan=" in caplog.text
    assert '"step": 1' in caplog.text
    assert '"goal": "Найти имя"' in caplog.text
    assert '"presentation": "answer_only"' in caplog.text
    assert '"step": 2' in caplog.text
    assert '"goal": "Проверить найденное имя"' in caplog.text
    assert '"presentation": "full_results"' in caplog.text


def test_coordinator_rejects_unknown_result_key_and_cleans_refs():
    from agents.coordinator import coordinator_chat

    model = _CoordinatorModel(
        {
            "submit_worker_plan": [
                _tool_message(
                    "submit_worker_plan",
                    {"steps": [{"goal": "Получить факт", "presentation": "full_results"}]},
                    "plan-1",
                )
            ],
            "dispatch_worker": [
                _tool_message(
                    "dispatch_worker",
                    {"task": "Получи факт."},
                    "dispatch-1",
                )
            ],
            "finish_coordination": [
                _tool_message(
                    "finish_coordination",
                    {"answer": "Факт получен.", "display_result_keys": ["unknown"]},
                    "finish-1",
                )
            ],
        }
    )
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch(
            "agents.coordinator.worker_chat",
            return_value=WorkerAnswer(
                answer="Факт получен.",
                result_refs=[WorkerResultRef(ref="ref-result", name="lookup")],
            ),
        ),
        patch("agents.coordinator.discard_worker_result_refs") as discard,
    ):
        with pytest.raises(CoordinatorResponseError):
            coordinator_chat("Получи факт.")

    discard.assert_called_once_with(["ref-result"])


def test_coordinator_aggregates_unsatisfied_worker_without_internal_error():
    from agents.coordinator import coordinator_chat

    model = _CoordinatorModel(
        {
            "submit_worker_plan": [
                _tool_message(
                    "submit_worker_plan",
                    {"steps": [{"goal": "Получить факт", "presentation": "full_results"}]},
                    "plan-1",
                )
            ],
            "dispatch_worker": [
                _tool_message(
                    "dispatch_worker",
                    {"task": "Получи факт."},
                    "dispatch-1",
                )
            ],
            "finish_coordination": [
                _tool_message(
                    "finish_coordination",
                    {
                        "answer": (
                            "Не удалось подтвердить факт: tool вернул данные "
                            "не по той сущности."
                        ),
                        "display_result_keys": [],
                    },
                    "finish-1",
                )
            ],
        }
    )
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch(
            "agents.coordinator.worker_chat",
            return_value=WorkerAnswer(
                answer="Факт не подтверждён.",
                result_refs=[WorkerResultRef(ref="ref-failed", name="lookup")],
                goal_satisfied=False,
                mismatches=["Tool вернул данные не по той сущности."],
            ),
        ),
        patch("agents.coordinator.discard_worker_result_refs") as discard,
    ):
        result = coordinator_chat("Получи факт.")

    assert result == CoordinatorAnswer(
        answer=(
            "Не удалось подтвердить факт: tool вернул данные не по той "
            "сущности."
        ),
        display_refs=[],
    )
    aggregate = _payload(model, "finish_coordination")
    assert aggregate["worker_results"] == [
        {
            "step": 1,
            "goal": "Получить факт",
            "task": "Получи факт.",
            "answer": "Факт не подтверждён.",
            "goal_satisfied": False,
            "mismatches": ["Tool вернул данные не по той сущности."],
            "available_results": [],
        }
    ]
    discard.assert_called_once_with(["ref-failed"])


def test_coordinator_normalizes_structured_aggregate_answer_without_repair():
    from agents.coordinator import coordinator_chat

    model = _CoordinatorModel(
        {
            "submit_worker_plan": [
                _tool_message(
                    "submit_worker_plan",
                    {"steps": [{"goal": "Получить факт", "presentation": "answer_only"}]},
                    "plan-1",
                )
            ],
            "dispatch_worker": [
                _tool_message(
                    "dispatch_worker",
                    {"task": "Получи факт."},
                    "dispatch-1",
                )
            ],
            "finish_coordination": [
                _tool_message(
                    "finish_coordination",
                    {
                        "answer": [{"value": 42}],
                        "display_result_keys": [],
                    },
                    "finish-structured",
                ),
            ],
        }
    )
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch(
            "agents.coordinator.worker_chat",
            return_value=WorkerAnswer(answer="Значение: 42.", result_refs=[]),
        ),
    ):
        result = coordinator_chat("Верни JSON-массив со значением 42.")

    assert result == CoordinatorAnswer(answer='[{"value":42}]', display_refs=[])
    finish_messages = [
        messages
        for name, messages in model.messages
        if name == "finish_coordination"
    ]
    assert len(finish_messages) == 1


def test_coordinator_empty_task_does_not_call_llm():
    from agents.coordinator import coordinator_chat

    result = coordinator_chat("   ")

    assert result.display_refs == []
    assert "пустой" in result.answer.lower() or "пуст" in result.answer.lower()
