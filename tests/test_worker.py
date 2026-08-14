from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from agents.chat_graph import (
    Observation,
    WorkerDisplayItem,
    WorkerResponseError,
    WorkerRunResult,
    run_worker_graph,
)
from agents.tools import get_tools, load_skills
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
    assert "answer должен содержать ровно их" in _WORKER_PLANNER_PROMPT
    assert "без вступления, заключения" in _WORKER_PLANNER_PROMPT
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
    assert "Предыдущий обычный текст не выполняет task" in str(
        model.messages[1][-1].content
    )


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


def test_public_worker_contract_exposes_only_answer_and_opaque_result_refs():
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
        goal_satisfied=False,
        mismatches=["Нужна дополнительная проверка."],
    )
    route = ToolRoute(tools=["list_files"], skills=["Excel и описания"])
    with (
        patch("agents.worker.select_chat_route", return_value=route) as router,
        patch(
            "agents.worker.run_worker_graph",
            return_value=graph_result,
        ) as run_graph,
        patch("agents.worker.load_skills", wraps=load_skills),
    ):
        result = worker_chat("  Покажи файлы  ")

    assert isinstance(result, WorkerAnswer)
    assert result.answer == graph_result.answer
    assert result.goal_satisfied is False
    assert result.mismatches == graph_result.mismatches
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


def test_public_worker_reroutes_original_task_after_observer_request():
    from agents.worker import worker_chat

    routes = [
        ToolRoute(tools=["list_s2t_table_names"], skills=["S2T-строки"]),
        ToolRoute(tools=["run_sql"], skills=["SQLite SQL"]),
    ]
    graph_results = [
        WorkerRunResult(
            answer="Нужна другая палитра.",
            goal_satisfied=False,
            mismatches=["Текущий tool не выполняет агрегирование."],
            reroute_required=True,
            reroute_reason="Нужна произвольная SQL-агрегация.",
        ),
        WorkerRunResult(
            answer="Максимум: t_example, 55 строк.",
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


def test_worker_rejects_positive_sql_observation_with_missing_requested_role():
    def fake_sql(query: str):
        return {
            "query": query,
            "columns": ["source_table", "row_count"],
            "rows": [{"source_table": "wrong_role", "row_count": 292}],
        }

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_sql",
                        "args": {
                            "query": (
                                "SELECT source_table, COUNT(*) AS row_count "
                                "FROM s2t_transformations GROUP BY source_table"
                            )
                        },
                        "id": "wrong-role",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Максимальная таблица найдена."),
        ],
        observer_responses=[
            Observation(
                summary="Максимальная таблица найдена.",
                goal_satisfied=True,
            )
        ],
    )

    result = run_worker_graph(
        task="Найди target_table с максимальным числом строк.",
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(fake_sql, name="run_sql"),),
        max_steps=1,
    )

    assert result.goal_satisfied is False
    assert any("`target_table`" in item for item in result.mismatches)


def test_worker_rejects_sql_that_omits_named_source_and_exact_role_value():
    def fake_sql(query: str):
        return {
            "query": query,
            "columns": ["target_table", "row_count"],
            "rows": [{"target_table": "Source columns", "row_count": 130}],
        }

    model = _WorkerModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_sql",
                        "args": {
                            "query": (
                                "SELECT table_name AS target_table, row_num AS "
                                "row_count FROM target_tables ORDER BY row_num "
                                "DESC LIMIT 1"
                            )
                        },
                        "id": "wrong-source",
                        "type": "tool_call",
                    }
                ],
            ),
            _finish_message("Source columns, 130."),
        ],
        observer_responses=[
            Observation(
                summary="Source columns, 130.",
                goal_satisfied=True,
            )
        ],
    )

    result = run_worker_graph(
        task=(
            "В s2t_transformations для source_table = "
            "b7000000250039_loans_productregister::union::1::branch::2 "
            "найди target_table с максимальным числом строк."
        ),
        system_prompt="Системный контекст",
        model=model,
        tools=(_as_tool(fake_sql, name="run_sql"),),
        max_steps=1,
    )

    assert result.goal_satisfied is False
    mismatch_text = " ".join(result.mismatches)
    assert "`s2t_transformations`" in mismatch_text
    assert "source_table = b7000000250039" in mismatch_text
