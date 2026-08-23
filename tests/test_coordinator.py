import json
from contextlib import nullcontext
from unittest.mock import call, patch

import pytest
from langchain_core.messages import AIMessage

from agents.chat_graph import Observation, WorkerCycleTrace
from agents.coordinator import CoordinatorAnswer, CoordinatorResponseError
from agents.worker import WorkerAnswer, WorkerResultRef
from agents.tools.saved_results import SavedResultColumn, SavedResultDescriptor


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
        self.tool_choices = []

    def bind_tools(self, tools, tool_choice=None):
        tool_name = tools[0]["function"]["name"]
        self.tool_choices.append((tool_name, tool_choice))
        return _BoundModel(self, tool_name)


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


def test_coordinator_graph_routes_tasks_downstream_and_results_upstream():
    from agents.coordinator import build_coordinator_graph

    model = _CoordinatorModel({})
    graph = build_coordinator_graph(model)
    graph_view = graph.get_graph()

    assert model.tool_choices == [
        ("submit_worker_plan", "submit_worker_plan"),
        ("dispatch_worker", "dispatch_worker"),
        ("submit_upstream_evidence", "submit_upstream_evidence"),
        ("finish_upstream_answer", "finish_upstream_answer"),
    ]

    assert {
        "downstream_plan",
        "downstream_materialize",
        "worker",
        "upstream_evidence",
        "upstream_answer",
    }.issubset(graph_view.nodes)
    edges = {(edge.source, edge.target) for edge in graph_view.edges}
    assert ("__start__", "downstream_plan") in edges
    assert ("downstream_plan", "downstream_materialize") in edges
    assert ("downstream_materialize", "worker") in edges
    assert ("upstream_evidence", "upstream_answer") in edges
    assert ("upstream_answer", "__end__") in edges


def test_coordinator_prompts_are_generic_not_domain_contracts():
    from agents.coordinator import (
        _DOWNSTREAM_DISPATCH_PROMPT,
        _DOWNSTREAM_PLAN_PROMPT,
        _UPSTREAM_ANSWER_PROMPT,
        _UPSTREAM_EVIDENCE_PROMPT,
        _dispatch_tool_schema,
        _plan_tool_schema,
        _upstream_answer_tool_schema,
        _upstream_evidence_tool_schema,
    )

    combined = "\n".join(
        (
            _DOWNSTREAM_PLAN_PROMPT,
            _DOWNSTREAM_DISPATCH_PROMPT,
            _UPSTREAM_EVIDENCE_PROMPT,
            _UPSTREAM_ANSWER_PROMPT,
        )
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

    assert len(_DOWNSTREAM_PLAN_PROMPT) < 2600
    assert "пользовательскую проверку или один самостоятельно" in _DOWNSTREAM_PLAN_PROMPT
    assert "атрибутов одного найденного объекта или записи" in _DOWNSTREAM_PLAN_PROMPT
    assert "прямое объяснение" in _DOWNSTREAM_PLAN_PROMPT
    assert "одна проверка, сравнение или объяснение" in _DOWNSTREAM_PLAN_PROMPT
    assert "Не разделяй получение опорных фактов" in _DOWNSTREAM_PLAN_PROMPT
    assert "следующий вызов данных сформировать невозможно" in _DOWNSTREAM_PLAN_PROMPT
    assert "по умолчанию верни ровно один step" in _DOWNSTREAM_PLAN_PROMPT
    assert "только при истинной зависимости" in _DOWNSTREAM_PLAN_PROMPT
    assert "сами по себе не являются зависимостью" in _DOWNSTREAM_PLAN_PROMPT
    assert "не влияет на\nдекомпозицию" in _DOWNSTREAM_PLAN_PROMPT
    assert "не достраивай предметную" in _DOWNSTREAM_PLAN_PROMPT
    assert "сверху вниз" in _DOWNSTREAM_PLAN_PROMPT
    normalized_plan = " ".join(_DOWNSTREAM_PLAN_PROMPT.split()).lower()
    assert "используй `context` только" in normalized_plan
    assert "оформления, пересказа или повторного показа" in normalized_plan
    assert "единственная операция текущего worker" in _DOWNSTREAM_DISPATCH_PROMPT
    assert "технический режим передачи результата" in str(
        _plan_tool_schema()
    ).lower()
    assert "самостоятельно проверяемый результат" in str(_plan_tool_schema())
    assert "original_task" in _DOWNSTREAM_DISPATCH_PROMPT
    assert "показ или экспорт" in _DOWNSTREAM_DISPATCH_PROMPT
    assert "неоднозначному значению роль не назначай" in _DOWNSTREAM_DISPATCH_PROMPT
    assert "копируй\n  посимвольно" in _DOWNSTREAM_DISPATCH_PROMPT
    assert "downstream dispatcher" in _DOWNSTREAM_DISPATCH_PROMPT
    normalized_dispatch = " ".join(_DOWNSTREAM_DISPATCH_PROMPT.split()).lower()
    assert "не заменяй её операцией прошлого или соседнего step" in normalized_dispatch
    assert "не проси повторно получать или проверять" in normalized_dispatch
    assert "цели предыдущего и следующего steps отсутствуют" in normalized_dispatch
    assert "cycle_history" in _DOWNSTREAM_DISPATCH_PROMPT
    assert "не формулируй пользовательский ответ" in _UPSTREAM_EVIDENCE_PROMPT
    assert "cycle_history" in _UPSTREAM_EVIDENCE_PROMPT
    assert "goal_satisfied=true" in _UPSTREAM_EVIDENCE_PROMPT
    assert "Не смешивай evidence разных объектов" in _UPSTREAM_EVIDENCE_PROMPT
    assert "unresolved_requirements" in _UPSTREAM_EVIDENCE_PROMPT
    assert "не занимайся его оформлением" in _UPSTREAM_EVIDENCE_PROMPT
    assert "снизу вверх" in _UPSTREAM_EVIDENCE_PROMPT
    assert "сформируй окончательный ответ" in _UPSTREAM_ANSWER_PROMPT
    assert "только\nпо `original_task`" in _UPSTREAM_ANSWER_PROMPT
    assert "не переоценивай доказательства" in _UPSTREAM_ANSWER_PROMPT
    assert "Если пользователь потребовал «только»" in _UPSTREAM_ANSWER_PROMPT
    assert "имя=<значение>" in _UPSTREAM_ANSWER_PROMPT
    assert "знак `=` дословно" in _UPSTREAM_ANSWER_PROMPT
    assert "worker_results" not in _UPSTREAM_ANSWER_PROMPT
    assert len(_UPSTREAM_EVIDENCE_PROMPT) < 2400
    assert len(_UPSTREAM_ANSWER_PROMPT) < 1800

    upstream_schema = _upstream_evidence_tool_schema()["function"][
        "parameters"
    ]
    assert upstream_schema["required"] == [
        "confirmed_facts",
        "unresolved_requirements",
    ]
    upstream_answer_schema = _upstream_answer_tool_schema()["function"][
        "parameters"
    ]
    assert upstream_answer_schema["required"] == ["answer"]

    plan_schema = _plan_tool_schema()["function"]["parameters"]
    goal_schema = plan_schema["properties"]["steps"]["items"]["properties"][
        "goal"
    ]
    assert "самостоятельно проверяемый результат" in goal_schema["description"]
    assert "Несколько чтений остаются одним шагом" in goal_schema["description"]
    assert "аргумента из предыдущего результата" in goal_schema["description"]
    assert "По умолчанию один шаг" in plan_schema["properties"]["steps"][
        "description"
    ]
    dispatch_schema = _dispatch_tool_schema()["function"]["parameters"]
    task_schema = dispatch_schema["properties"]["task"]
    assert "ровно с одной операцией" in task_schema["description"]


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
            "submit_upstream_evidence": [
                _tool_message(
                    "submit_upstream_evidence",
                    {
                        "confirmed_facts": ["Имя t_example проверено."],
                        "unresolved_requirements": [],
                    },
                    "upstream-1",
                )
            ],
            "finish_upstream_answer": [
                _tool_message(
                    "finish_upstream_answer",
                    {"answer": "Имя t_example проверено."},
                    "downstream-1",
                )
            ],
        }
    )
    worker_results = [
        WorkerAnswer(
            answer="Точное имя: t_example.",
            result_refs=[WorkerResultRef(ref="ref-first", name="lookup")],
            saved_results=[
                SavedResultDescriptor(
                    result_ref="saved-first",
                    source_tool="lookup",
                    row_count=1,
                    columns=[
                        SavedResultColumn(name="name", sqlite_type="TEXT")
                    ],
                )
            ],
            cycle_history=[
                WorkerCycleTrace(
                    cycle=1,
                    tool_calls=[{"name": "lookup", "args": {}}],
                    tool_results=[
                        {
                            "name": "lookup",
                            "tool_call_id": "call-lookup",
                            "content": "{\"name\":\"t_example\"}",
                            "status": "success",
                            "is_error": False,
                        }
                    ],
                    observation=Observation(
                        summary="Точное имя найдено.",
                        goal_satisfied=True,
                        important_facts=["Имя: t_example."],
                    ),
                )
            ],
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
    assert second_dispatch["completed_workers"][0]["status"] == "completed"
    assert second_dispatch["completed_workers"][0]["saved_results"] == [
        {
            "result_ref": "saved-first",
            "source_tool": "lookup",
            "row_count": 1,
            "source_total": None,
            "truncated": False,
            "columns": [{"name": "name", "sqlite_type": "TEXT"}],
        }
    ]
    first_history = second_dispatch["completed_workers"][0]["cycle_history"]
    assert first_history[0]["tool_calls"] == [
        {"name": "lookup", "args": {}}
    ]
    assert first_history[0]["observation"]["important_facts"] == [
        "Имя: t_example."
    ]
    upstream = _payload(model, "submit_upstream_evidence")
    assert upstream["worker_results"][0]["status"] == "completed"
    assert upstream["worker_results"][0]["cycle_history"] == first_history
    assert "available_results" not in upstream["worker_results"][1]
    assert "saved_results" not in upstream["worker_results"][1]
    upstream_answer = _payload(model, "finish_upstream_answer")
    assert upstream_answer == {
        "original_task": "Найди имя и проверь его.",
        "context": "Общий фон",
        "upstream_evidence": {
            "confirmed_facts": ["Имя t_example проверено."],
            "unresolved_requirements": [],
        },
    }
    assert "Coordinator planned worker_steps=2 plan=" in caplog.text
    assert '"step": 1' in caplog.text
    assert '"goal": "Найти имя"' in caplog.text
    assert '"presentation": "answer_only"' in caplog.text
    assert '"step": 2' in caplog.text
    assert '"goal": "Проверить найденное имя"' in caplog.text
    assert '"presentation": "full_results"' in caplog.text
    assert "Upstream coordinator evidence:" in caplog.text
    assert "Upstream coordinator answer:" in caplog.text


def test_coordinator_selects_full_results_without_llm_display_keys():
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
                    {"task": "Получи факт без точного фильтра."},
                    "dispatch-1",
                )
            ],
            "submit_upstream_evidence": [
                _tool_message(
                    "submit_upstream_evidence",
                    {
                        "confirmed_facts": ["Факт получен."],
                        "unresolved_requirements": [],
                    },
                    "upstream-1",
                )
            ],
            "finish_upstream_answer": [
                _tool_message(
                    "finish_upstream_answer",
                    {"answer": "Факт получен."},
                    "downstream-1",
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
        ) as worker,
        patch("agents.coordinator.discard_worker_result_refs") as discard,
    ):
        result = coordinator_chat("Получи факт.")

    assert result == CoordinatorAnswer(
        answer="Факт получен.",
        display_refs=["ref-result"],
    )
    worker.assert_called_once_with("Получи факт.")
    discard.assert_not_called()


def test_upstream_answer_reports_unresolved_evidence_without_internal_error():
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
            "submit_upstream_evidence": [
                _tool_message(
                    "submit_upstream_evidence",
                    {
                        "confirmed_facts": [],
                        "unresolved_requirements": [
                            "Не удалось подтвердить факт: tool вернул данные "
                            "не по той сущности."
                        ],
                    },
                    "upstream-1",
                )
            ],
            "finish_upstream_answer": [
                _tool_message(
                    "finish_upstream_answer",
                    {
                        "answer": (
                            "Не удалось подтвердить факт: tool вернул данные "
                            "не по той сущности."
                        )
                    },
                    "downstream-1",
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
    upstream = _payload(model, "submit_upstream_evidence")
    assert upstream["worker_results"] == [
        {
            "step": 1,
            "status": "failed",
            "goal": "Получить факт",
            "task": "Получи факт.",
            "answer": "Факт не подтверждён.",
            "cycle_history": [],
            "goal_satisfied": False,
            "mismatches": ["Tool вернул данные не по той сущности."],
        }
    ]
    discard.assert_called_once_with(["ref-failed"])


def test_upstream_answer_normalizes_structured_output_without_repair():
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
            "submit_upstream_evidence": [
                _tool_message(
                    "submit_upstream_evidence",
                    {
                        "confirmed_facts": ["Значение равно 42."],
                        "unresolved_requirements": [],
                    },
                    "upstream-structured",
                ),
            ],
            "finish_upstream_answer": [
                _tool_message(
                    "finish_upstream_answer",
                    {"answer": '[{"value":42}]'},
                    "downstream-structured",
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
            return_value=WorkerAnswer(answer="Значение: 42.", result_refs=[]),
        ),
    ):
        result = coordinator_chat("Верни JSON-массив со значением 42.")

    assert result == CoordinatorAnswer(answer='[{"value":42}]', display_refs=[])
    upstream_answer_messages = [
        messages
        for name, messages in model.messages
        if name == "finish_upstream_answer"
    ]
    assert len(upstream_answer_messages) == 1


def test_coordinator_empty_task_does_not_call_llm():
    from agents.coordinator import coordinator_chat

    result = coordinator_chat("   ")

    assert result.display_refs == []
    assert "пустой" in result.answer.lower() or "пуст" in result.answer.lower()


def test_coordinator_repairs_plan_that_exceeds_worker_limit():
    from agents.coordinator import coordinator_chat

    invalid_steps = [
        {"goal": f"Проверка {index}", "presentation": "answer_only"}
        for index in range(1, 10)
    ]
    model = _CoordinatorModel(
        {
            "submit_worker_plan": [
                _tool_message(
                    "submit_worker_plan",
                    {"steps": invalid_steps},
                    "plan-invalid",
                ),
                _tool_message(
                    "submit_worker_plan",
                    {
                        "steps": [
                            {
                                "goal": "Выполнить все девять связанных проверок",
                                "presentation": "answer_only",
                            }
                        ]
                    },
                    "plan-repaired",
                ),
            ],
            "dispatch_worker": [
                _tool_message(
                    "dispatch_worker",
                    {"task": "Выполни все девять связанных проверок."},
                    "dispatch-1",
                )
            ],
            "submit_upstream_evidence": [
                _tool_message(
                    "submit_upstream_evidence",
                    {
                        "confirmed_facts": ["Все проверки выполнены."],
                        "unresolved_requirements": [],
                    },
                    "upstream-1",
                )
            ],
            "finish_upstream_answer": [
                _tool_message(
                    "finish_upstream_answer",
                    {"answer": "Все проверки выполнены."},
                    "downstream-1",
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
            return_value=WorkerAnswer(answer="Все проверки выполнены."),
        ),
    ):
        result = coordinator_chat("Выполни девять связанных проверок.")

    assert result == CoordinatorAnswer(
        answer="Все проверки выполнены.",
        display_refs=[],
    )
    plan_messages = [
        messages
        for name, messages in model.messages
        if name == "submit_worker_plan"
    ]
    assert len(plan_messages) == 2
    assert "от 1 до 8 элементов" in plan_messages[1][-1].content


def test_coordinator_repairs_missing_dispatch_native_call():
    from agents.coordinator import coordinator_chat

    model = _CoordinatorModel(
        {
            "submit_worker_plan": [
                _tool_message(
                    "submit_worker_plan",
                    {
                        "steps": [
                            {"goal": "Получить факт", "presentation": "answer_only"}
                        ]
                    },
                    "plan-1",
                )
            ],
            "dispatch_worker": [
                AIMessage(content="Получи факт без native call."),
                _tool_message(
                    "dispatch_worker",
                    {"task": "Получи факт."},
                    "dispatch-repaired",
                ),
            ],
            "submit_upstream_evidence": [
                _tool_message(
                    "submit_upstream_evidence",
                    {
                        "confirmed_facts": ["Факт получен."],
                        "unresolved_requirements": [],
                    },
                    "upstream-1",
                )
            ],
            "finish_upstream_answer": [
                _tool_message(
                    "finish_upstream_answer",
                    {"answer": "Факт получен."},
                    "downstream-1",
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
            return_value=WorkerAnswer(answer="Факт получен."),
        ),
    ):
        result = coordinator_chat("Получи факт.")

    assert result == CoordinatorAnswer(answer="Факт получен.", display_refs=[])
    dispatch_messages = [
        messages
        for name, messages in model.messages
        if name == "dispatch_worker"
    ]
    assert len(dispatch_messages) == 2
    assert "единственным непустым" in dispatch_messages[1][-1].content
