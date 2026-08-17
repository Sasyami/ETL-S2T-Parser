import json
from contextlib import nullcontext
from unittest.mock import patch

from langchain_core.messages import AIMessage

from agents.chat_graph import WorkerDisplayItem, WorkerRunResult
from agents.coordinator import CoordinatorAnswer


class _SupervisorModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tools = None
        self.messages = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages, **kwargs):
        del kwargs
        self.messages.append(list(messages))
        return self.responses.pop(0)


def _delegate_message(task, *, context="", call_id="delegate-1", extra_args=None):
    args = {"task": task, "context": context}
    args.update(extra_args or {})
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "delegate_to_coordinator",
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _supervisor_patches(model):
    return (
        patch("agents.supervisor.chat_model", model),
        patch("agents.supervisor.get_callback_handler", return_value=None),
        patch(
            "agents.supervisor.langfuse_trace_context",
            return_value=nullcontext(),
        ),
    )


def _payload(model):
    return json.loads(model.messages[0][1].content)


def test_supervisor_graph_routes_coordinator_directly_to_end():
    from agents.supervisor import build_supervisor_graph

    model = _SupervisorModel([])
    graph = build_supervisor_graph(model)
    graph_view = graph.get_graph()

    assert {"supervisor", "coordinator"}.issubset(graph_view.nodes)
    assert "limit" not in graph_view.nodes
    edges = {(edge.source, edge.target) for edge in graph_view.edges}
    assert ("__start__", "supervisor") in edges
    assert ("coordinator", "__end__") in edges
    assert ("coordinator", "supervisor") not in edges


def test_supervisor_prompt_keeps_decision_and_handoff_llm_driven():
    from agents.supervisor import _SUPERVISOR_PROMPT

    normalized_prompt = " ".join(_SUPERVISOR_PROMPT.split()).lower()
    assert "реши, ответить сразу или вызвать" in normalized_prompt
    assert "конкретное самодостаточное поручение" in normalized_prompt
    assert "не добавляй новых целей" in normalized_prompt.lower()
    assert "не придумывай требования" in normalized_prompt
    assert "не требуй json" in normalized_prompt
    assert "только когда его явно задал пользователь" in normalized_prompt
    assert "именно исполнимое поручение" in normalized_prompt
    assert "дословно сохраняй значимые идентификаторы" in normalized_prompt
    assert "при единственном однозначном референте" in normalized_prompt
    assert "никогда не передавай такую неразрешённую ссылку" in normalized_prompt
    assert "посчитай строки в таблице x" in normalized_prompt
    assert "является частью task, а не context" in normalized_prompt
    assert "только компактные устойчивые правила" in normalized_prompt
    assert "в context остались только повторно применимые договорённости" in normalized_prompt


def test_supervisor_answers_directly_when_coordinator_is_not_needed():
    from agents.supervisor import supervisor_chat

    model = _SupervisorModel([AIMessage(content="Здравствуйте!")])
    model_patch, callback_patch, trace_patch = _supervisor_patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch("agents.supervisor.coordinator_chat") as coordinator,
    ):
        result = supervisor_chat("Привет")

    assert result == WorkerRunResult(answer="Здравствуйте!", display_items=[])
    coordinator.assert_not_called()
    assert len(model.bound_tools) == 1
    parameters = model.bound_tools[0]["function"]["parameters"]
    assert set(parameters["properties"]) == {"task", "context"}
    assert parameters["required"] == ["task", "context"]
    assert parameters["properties"]["context"]["maxLength"] == 4000
    task_description = parameters["properties"]["task"]["description"]
    context_description = parameters["properties"]["context"]["description"]
    assert "Исполнимое самодостаточное поручение" in task_description
    assert "разовые объекты, ID, числа, результаты" in context_description
    assert parameters["additionalProperties"] is False


def test_supervisor_delegates_whole_goal_and_returns_coordinator_result():
    from agents.supervisor import supervisor_chat

    task = "Для файла с file_id=42 перечисли все листы и их число."
    common_context = (
        "Под словом «листы» в этом диалоге всегда понимаются листы Excel."
    )
    model = _SupervisorModel(
        [_delegate_message(task, context=common_context)]
    )
    coordinator_result = CoordinatorAnswer(
        answer="В файле два листа: S2T и Дополнительные объекты.",
        display_refs=["ref-sheets"],
    )
    resolved_items = [
        WorkerDisplayItem(
            name="list_sheets",
            content='["S2T", "Дополнительные объекты"]',
        )
    ]
    history = [
        {"role": "user", "content": "Открой файл 42"},
        {"role": "assistant", "content": "Файл выбран."},
    ]
    model_patch, callback_patch, trace_patch = _supervisor_patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch(
            "agents.supervisor.coordinator_chat",
            return_value=coordinator_result,
        ) as coordinator,
        patch(
            "agents.supervisor.resolve_worker_display_refs",
            return_value=resolved_items,
        ) as resolve_refs,
    ):
        result = supervisor_chat(
            "Какие в нём листы и сколько их?",
            history=history,
            session_id="session-1",
        )

    assert result == WorkerRunResult(
        answer=coordinator_result.answer,
        display_items=resolved_items,
    )
    coordinator.assert_called_once_with(task, context=common_context)
    resolve_refs.assert_called_once_with(["ref-sheets"])
    assert _payload(model) == {
        "current_query": "Какие в нём листы и сколько их?",
        "recent_history": history,
    }
    assert len(model.messages) == 1


def test_supervisor_passes_native_task_without_validation_or_repair():
    from agents.supervisor import supervisor_chat

    decision = _delegate_message(
        "Покажи файлы",
        extra_args={"tools": ["list_files"]},
    )
    model = _SupervisorModel([decision])
    model_patch, callback_patch, trace_patch = _supervisor_patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch(
            "agents.supervisor.coordinator_chat",
            return_value=CoordinatorAnswer(
                answer="Найдено три файла.",
                display_refs=[],
            ),
        ) as coordinator,
    ):
        result = supervisor_chat("Покажи файлы")

    assert result.answer == "Найдено три файла."
    coordinator.assert_called_once_with("Покажи файлы", context="")
    assert len(model.messages) == 1


def test_supervisor_uses_llm_delegated_task_without_python_rewrite():
    from agents.supervisor import supervisor_chat

    decision = _delegate_message(
        "Проверь две таблицы через SQLite,",
    )
    model = _SupervisorModel([decision])
    model_patch, callback_patch, trace_patch = _supervisor_patches(model)
    query = "Через Neo4j найди путь от source_a до target_b. Не используй SQLite."
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch(
            "agents.supervisor.coordinator_chat",
            return_value=CoordinatorAnswer(answer="Путь найден.", display_refs=[]),
        ) as coordinator,
    ):
        result = supervisor_chat(
            query,
            history=[{"role": "assistant", "content": "Прошлый ответ."}],
        )

    assert result.answer == "Путь найден."
    coordinator.assert_called_once_with(
        "Проверь две таблицы через SQLite,",
        context="",
    )


def test_supervisor_prompt_is_generic_and_preserves_semantics():
    from agents.supervisor import _SUPERVISOR_PROMPT, _delegate_tool_schema

    normalized_prompt = " ".join(_SUPERVISOR_PROMPT.split()).lower()
    assert "дословно сохраняй" in normalized_prompt
    assert "не заменяй их похожими" in normalized_prompt
    assert "устойчивые правила и устоявшиеся идеи" in normalized_prompt
    assert "ссылки на историю уже разрешены" in str(_delegate_tool_schema()).lower()
    assert "не помещай в context" in normalized_prompt
    assert "текущую task или её сокращённый пересказ" in normalized_prompt
    assert "конкретные объекты, id, имена, числа и результаты" in normalized_prompt
    for domain_detail in ("Neo4j", "SQLite", "s2t_transformations", "file_id"):
        assert domain_detail not in _SUPERVISOR_PROMPT
