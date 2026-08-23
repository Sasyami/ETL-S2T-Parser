import json
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from agents.chat_graph import (
    Observation,
    WorkerCycleTrace,
    WorkerDisplayItem,
    WorkerResponseError,
    WorkerRunResult,
    run_worker_graph,
)
from agents.tools import get_tools, load_schemas, load_skills
from agents.tools.routing import ToolRoute


def _as_tool(function, name=None):
    tool_name = name or function.__name__
    return StructuredTool.from_function(
        func=function,
        name=tool_name,
        description=f"Test tool {tool_name}",
    )


def _finish_message(answer, *, extra_args=None):
    args = {"answer": answer}
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
    assert "Полные результаты не копируй" in normalized_prompt
    assert "не переименовывай заданную операцию" in _WORKER_PLANNER_PROMPT
    assert "Не конструируй отсутствующий объект" in _WORKER_PLANNER_PROMPT
    assert "аргументы бери из" in _WORKER_PLANNER_PROMPT
    assert "Если обязательного входа\nнет, этот tool не подходит" in (
        _WORKER_PLANNER_PROMPT
    )
    assert "answer должен содержать ровно их" in _WORKER_PLANNER_PROMPT
    assert "без вступления, заключения" in _WORKER_PLANNER_PROMPT
    assert "Палитра worker никогда не пуста" in _WORKER_PLANNER_PROMPT
    assert "внутренний `analyze_known_facts`" in _WORKER_PLANNER_PROMPT
    assert "scrollable" not in _WORKER_PLANNER_PROMPT


class _ObserverModel:
    def __init__(self, responses=None):
        self.messages = []
        self.responses = list(responses or [])

    def invoke(self, messages):
        self.messages.append(messages)
        response = (
            self.responses.pop(0)
            if self.responses
            else Observation(
                summary="Превью результата получено.",
                goal_satisfied=True,
            )
        )
        return (
            response
            if isinstance(response, Observation)
            else Observation.model_validate(response)
        )


class _WorkerModel:
    def __init__(self, responses, *, observer_responses=None):
        self.responses = list(responses)
        self.bound_tools = []
        self.messages = []
        self.observer = _ObserverModel(observer_responses)

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def with_structured_output(self, schema):
        assert schema is Observation
        return self.observer

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

    def with_structured_output(self, schema):
        assert schema is Observation
        return self.observer

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
            if self.forced_lookup_calls == 1:
                return AIMessage(content="Сначала выполню lookup.")
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
    assert result.goal_satisfied is True
    assert result.mismatches == []
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
    assert cycle.observation.goal_satisfied is True
    observer_payload = json.loads(model.observer.messages[0][-1].content)
    assert observer_payload["tool_calls"][0]["name"] == "analyze_known_facts"
    assert observer_payload["tool_results"][0]["name"] == "analyze_known_facts"
    planner_system = str(model.messages[0][0].content)
    assert "Доступные worker tools:\nanalyze_known_facts" in planner_system
    assert "Палитра worker никогда не пуста" in planner_system
    observer_system = str(model.observer.messages[0][0].content)
    assert "не является новым внешним фактом" in observer_system
    assert observer_payload["candidate_answer"] == ""


def test_worker_keeps_text_preview_and_returns_full_successful_result():
    tail_marker = "FULL_RESULT_TAIL"

    def long_result():
        return {"payload": ("x" * 200) + tail_marker}

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
    assert len(result.cycle_history) == 1
    cycle = result.cycle_history[0]
    assert cycle.cycle == 1
    assert cycle.routing_attempt == 1
    assert cycle.tool_calls == [{"name": "long_result", "args": {}}]
    assert cycle.tool_results[0]["name"] == "long_result"
    assert len(cycle.tool_results[0]["content"]) <= 50
    assert tail_marker not in cycle.tool_results[0]["content"]
    assert cycle.observation.summary == "Превью результата получено."

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
        skills=["SQLite SQL"],
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

    assert result.answer == "Найдено: 1."
    assert result.saved_results == []
    available_tools = router.call_args.kwargs["available_tools"]
    routed_tool = next(
        item for item in available_tools if item.name == "query_saved_result"
    )
    assert descriptor.result_ref in routed_tool.description
    assert '"target_table" TEXT' in routed_tool.description
    selected_tool = run_graph.call_args.kwargs["tools"][0]
    assert selected_tool.name == "query_saved_result"
    assert descriptor.result_ref in selected_tool.description


def test_worker_accepts_plain_planner_finish_without_responder_call():
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
    assert len(model.messages) == 2


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
            _finish_message("Подтверждено."),
        ]
    )

    result = run_worker_graph(
        task="Получи значение",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=2,
    )

    assert result.answer == "Подтверждено."
    assert [item.name for item in result.display_items] == ["lookup"]
    repair_prompt = " ".join(str(model.messages[1][-1].content).split())
    assert "Предыдущий обычный текст не выполняет task" in repair_prompt


def test_worker_falls_back_when_forced_first_tool_call_transport_fails():
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
    assert model.forced_lookup_calls == 2
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
        ]
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
                mismatches=["Правило преобразования ещё не подтверждено."],
                important_facts=["Источник: source_contracts."],
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

    assert result.goal_satisfied is True
    assert calls == ["source", "rule"]
    assert len(model.observer.messages) == 2
    second_observer_prompt = model.observer.messages[1]
    second_payload = json.loads(second_observer_prompt[-1].content)
    assert len(second_payload["prior_state"]) == 1
    assert second_payload["prior_state"][0]["summary"] == (
        "Источник: source_contracts."
    )
    observer_system_prompt = " ".join(
        str(second_observer_prompt[0].content).split()
    )
    assert "Оцени выполнение task по совокупности prior_state" in (
        observer_system_prompt
    )
    assert "не по последнему tool call изолированно" in observer_system_prompt


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
                mismatches=["Task ожидала значение, но tool вернул пустой результат."],
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

    assert result.goal_satisfied is False
    assert result.mismatches == [
        "Task ожидала значение, но tool вернул пустой результат."
    ]


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
        ]
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
    assert [item.name for item in result.display_items] == ["run_sql", "run_sql"]
    assert "t_rate_rule_param" in result.display_items[0].content


def test_worker_cannot_finish_after_observer_reports_semantic_mismatch():
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
            _finish_message("Подтверждено поле target_table."),
        ],
        observer_responses=[
            Observation(
                summary="Tool получил значение поля source_table.",
                goal_satisfied=False,
                mismatches=[
                    "Task просит target_table, но tool получил source_table."
                ],
            ),
            Observation(
                summary="Tool получил требуемое поле target_table.",
                goal_satisfied=True,
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

    assert result.answer == "Подтверждено поле target_table."
    assert selected_fields == ["source_table", "target_table"]
    assert len(model.observer.messages) == 2
    repair_prompt = " ".join(str(model.messages[2][-1].content).split())
    full_repair_context = " ".join(str(model.messages[2]).split())
    assert "Предыдущая попытка завершить worker запрещена" in repair_prompt
    assert "Что выполнено неправильно" in repair_prompt
    assert (
        "Task просит target_table, но tool получил source_table"
        in full_repair_context
    )


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
            _finish_message("Текущего результата достаточно."),
            _finish_message("Не могу исправить вызов текущими tools."),
        ],
        observer_responses=[
            Observation(
                summary="Получен только список имён без агрегирования.",
                goal_satisfied=False,
                mismatches=[
                    "Task требует сравнить количества строк, но текущий tool "
                    "возвращает только имена."
                ],
                reroute_required=True,
                reroute_reason="Нужен tool для произвольной агрегации данных.",
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

    assert result.reroute_required is True
    assert result.reroute_reason == (
        "Нужен tool для произвольной агрегации данных."
    )
    assert result.goal_satisfied is False
    assert result.display_items == []
    observer_payload = str(model.observer.messages[0][-1].content)
    assert '"available_tools"' in observer_payload
    assert '"name": "list_names"' in observer_payload


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
        ]
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


def test_public_worker_contract_exposes_bounded_cycles_and_opaque_result_refs(
    caplog,
):
    from agents.worker import (
        WorkerAnswer,
        resolve_worker_display_refs,
        worker_chat,
    )

    hidden_full_result = "FULL_RESULT_MUST_NOT_LEAVE_WORKER"
    graph_result = WorkerRunResult(
        answer="Готово.",
        display_items=[
            WorkerDisplayItem(name="list_files", content=hidden_full_result)
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
                    mismatches=["Нужна дополнительная проверка."],
                    important_facts=["Найдено 3 файла."],
                ),
            )
        ],
        goal_satisfied=False,
        mismatches=["Нужна дополнительная проверка."],
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

    assert isinstance(result, WorkerAnswer)
    assert result.answer == graph_result.answer
    assert result.goal_satisfied is False
    assert result.mismatches == graph_result.mismatches
    assert result.cycle_history == graph_result.cycle_history
    assert len(result.result_refs) == 1
    assert result.result_refs[0].name == "list_files"
    assert hidden_full_result not in result.model_dump_json()
    refs = [item.ref for item in result.result_refs]
    assert resolve_worker_display_refs(refs) == graph_result.display_items
    assert resolve_worker_display_refs(refs) == []
    assert router.call_args.args == ("Покажи файлы",)
    assert "history" not in router.call_args.kwargs
    assert router.call_args.kwargs["available_tools"] == get_tools()
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
        reroute_reason=None,
    )
    observation_recorder.assert_called_once()
    observation_call = observation_recorder.call_args.kwargs
    assert observation_call["worker_task"] == "Покажи файлы"
    assert observation_call["cycle"] == 1
    assert observation_call["routing_attempt"] == 1
    assert observation_call["observation"]["goal_satisfied"] is False
    assert observation_call["observation"]["mismatches"] == [
        "Нужна дополнительная проверка."
    ]
    assert "Worker route:" in caplog.text
    assert "Worker observation:" in caplog.text
    assert '"goal_satisfied": false' in caplog.text


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

    assert result.answer == graph_result.answer
    assert result.goal_satisfied is True
    assert result.cycle_history[0].observation == observation
    assert result.result_refs == []
    assert [
        tool.name for tool in run_graph.call_args.kwargs["tools"]
    ] == ["analyze_known_facts"]


def test_public_worker_reroutes_original_task_after_observer_request():
    from agents.worker import worker_chat

    routes = [
        ToolRoute(
            tools=["list_s2t_table_names"],
            skills=["S2T-строки"],
            schemas=[],
        ),
        ToolRoute(
            tools=["run_sql"],
            skills=["SQLite SQL"],
            schemas=["SQLite ETL"],
        ),
    ]
    graph_results = [
        WorkerRunResult(
            answer="Нужна другая палитра.",
            cycle_history=[
                WorkerCycleTrace(
                    cycle=1,
                    tool_calls=[
                        {"name": "list_s2t_table_names", "args": {}}
                    ],
                    tool_results=[],
                    observation=Observation(
                        summary="Агрегирование не выполнено.",
                        goal_satisfied=False,
                        mismatches=[
                            "Текущий tool не выполняет агрегирование."
                        ],
                        reroute_required=True,
                        reroute_reason="Нужна произвольная SQL-агрегация.",
                    ),
                )
            ],
            goal_satisfied=False,
            mismatches=["Текущий tool не выполняет агрегирование."],
            reroute_required=True,
            reroute_reason="Нужна произвольная SQL-агрегация.",
        ),
        WorkerRunResult(
            answer="Максимум: t_example, 55 строк.",
            cycle_history=[
                WorkerCycleTrace(
                    cycle=1,
                    tool_calls=[{"name": "run_sql", "args": {}}],
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

    assert result.answer == "Максимум: t_example, 55 строк."
    assert [cycle.cycle for cycle in result.cycle_history] == [1, 2]
    assert [cycle.routing_attempt for cycle in result.cycle_history] == [1, 2]
    assert result.goal_satisfied is True
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
        "reason": "Нужна произвольная SQL-агрегация.",
        "mismatches": ["Текущий tool не выполняет агрегирование."],
        "previous_tool_palettes": [["list_s2t_table_names"]],
        "attempt": 1,
    }
    assert [
        [tool.name for tool in item.kwargs["tools"]]
        for item in run_graph.call_args_list
    ] == [["list_s2t_table_names"], ["run_sql"]]


def test_public_worker_can_execute_repeated_reroute_palette():
    from agents.worker import worker_chat

    repeated_route = ToolRoute(
        tools=["list_s2t_table_names"],
        skills=["S2T-строки"],
        schemas=[],
    )
    graph_results = [
        WorkerRunResult(
            answer="Нужно исправить запрос.",
            goal_satisfied=False,
            mismatches=["SQL не учитывает нужный фильтр."],
            reroute_required=True,
            reroute_reason="Повтори SQL с правильным фильтром.",
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

    assert result.goal_satisfied is True
    assert result.answer == "Максимум: t_example, 55 строк."
    assert router.call_count == 2
    assert run_graph.call_count == 2
    assert [
        [tool.name for tool in item.kwargs["tools"]]
        for item in run_graph.call_args_list
    ] == [["list_s2t_table_names"], ["list_s2t_table_names"]]
    second_system_prompt = run_graph.call_args_list[1].kwargs["system_prompt"]
    assert "<reroute_feedback>" in second_system_prompt
    assert "Повтори SQL с правильным фильтром." in second_system_prompt
    assert "SQL не учитывает нужный фильтр." in second_system_prompt
    assert "палитра может совпадать с предыдущей" in second_system_prompt
    assert router.call_args_list[1].kwargs["reroute_context"] == {
        "reason": "Повтори SQL с правильным фильтром.",
        "mismatches": ["SQL не учитывает нужный фильтр."],
        "previous_tool_palettes": [["list_s2t_table_names"]],
        "attempt": 1,
    }


def test_worker_reroutes_when_first_data_tool_is_omitted_twice():
    def lookup():
        return {"value": 1}

    model = _WorkerModel(
        [
            AIMessage(content="Сначала поясню действие."),
            AIMessage(content="Теперь вызову инструмент."),
        ]
    )

    result = run_worker_graph(
        task="Получи значение через lookup",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(lookup),),
        max_steps=1,
    )

    assert result.goal_satisfied is False
    assert result.reroute_required is True
    assert "дважды не сформировал" in str(result.reroute_reason)
    assert "не вызвал ни один" in result.mismatches[0]
