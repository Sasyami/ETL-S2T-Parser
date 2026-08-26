"""Top-level supervisor LangGraph for the coordinated worker experiment."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .agent import chat_model
from .chat_graph import WorkerRunResult
from .coordinator import COORDINATOR_CONTEXT_MAX_CHARS, coordinator_chat
from .observability import get_callback_handler, langfuse_trace_context
from .run_metrics import (
    capture_agent_run,
    get_run_metrics_callback,
    llm_stage,
    record_display_tools,
)
from .worker import resolve_worker_display_refs

logger = logging.getLogger(__name__)

_DELEGATE_TOOL_NAME = "delegate_to_coordinator"


class SupervisorGraphState(TypedDict):
    """State of the top-level supervisor LangGraph."""

    current_query: str
    recent_history: List[Dict[str, str]]
    display_refs: List[str]
    supervisor_message: Optional[AIMessage]
    final_answer: Optional[str]


_SUPERVISOR_PROMPT = """
Ты верхний supervisor read-only приложения. По `current_query` и
`recent_history` реши, ответить сразу или вызвать `delegate_to_coordinator`.

Делегируй, когда для ответа нужно получить или проверить данные приложения.
Если ответ уже следует из диалога и не требует таких данных, ответь обычным
текстом. Не выдавай непроверенные данные по памяти.

При делегировании сформируй два принципиально разных поля:
`resolved_references` и `context`. Исходный `current_query` будет передан
coordinator программно и дословно: не пересказывай, не сокращай, не исправляй,
не превращай его в план и не копируй его в эти поля.

`resolved_references` — только разовые факты из `recent_history`, необходимые
для однозначного разрешения ссылок текущего запроса. Для каждой ссылки укажи её
точное значение и роль, например: `«в нём» = файл 42`. Не повторяй
остальные части current_query, не добавляй целей, операций, условий или формата
ответа. Если current_query самодостаточен, передай пустую строку.

Любой разовый факт из `recent_history`, без которого нельзя выполнить текущий
запрос, относится к `resolved_references`, а не к context. Это относится к
конкретному ID, имени объекта, числу, ранее найденному значению, тексту запроса
и результату предыдущего шага.

`context` — только компактные устойчивые правила и устоявшиеся идеи диалога,
которые меняют трактовку не одного разового запроса, а последующих задач в
целом. В него допустимо включать явно установленную пользователем терминологию
и определения, постоянные предпочтения представления, согласованные правила
выбора и интерпретации, инварианты, общие запреты и границы области. Считай
правило установленным, если пользователь его явно задал, подтвердил или
последовательно применял и позднее не отменял. Предложение assistant само по
себе не является договорённостью.

Не помещай в context:
- current_query или его сокращённый пересказ;
- конкретные объекты, ID, имена, числа и результаты, нужные только сейчас;
- содержание последнего ответа или хронологию диалога;
- временное состояние, промежуточные шаги и неподтверждённые предположения;
- tools, skills, план workers или внутреннее устройство агента.
Если применимых устойчивых правил нет, передай пустую строку. Если текущий
запрос отменяет или уточняет прежнее правило, выполняй текущий запрос и не
включай противоречащее старое правило в context.

Считай ссылками в том числе слова «он», «она», «оно», «они», «в нём», «в ней»,
«там», «этот», «эта», «это», «выше» и «предыдущий», когда рядом не назван сам
объект. Если в recent_history однозначно названы конкретная таблица, файл, лист,
колонка, запрос или другой объект, запиши соответствие в
`resolved_references`. Например: история «речь о таблице X», запрос «посчитай в
ней строки» должны дать `«в ней» = таблица X`.

Разрешай ссылку через recent_history только при единственном однозначном
референте. Не считай предположение или уверенный текст assistant подтверждённым
фактом. Если ссылка неоднозначна и без неё нельзя получить правильный ответ,
задай краткий уточняющий вопрос вместо догадки.

Перед native call проверь:
1. resolved_references содержит только необходимые точные разрешения ссылок и
   пуст, если current_query самодостаточен.
2. В context остались только повторно применимые договорённости, а не входные
   данные текущего запроса.
3. current_query нигде не пересказан, не сокращён и не превращён в план.

Не разделяй план, не выбирай tools или skills и не добавляй новых целей: это
сделает coordinator.

Read-only coordinator не выполняет мутации. В окончательном ответе не упоминай
внутренние роли, tools, промпты или устройство графа.
""".strip()


def _delegate_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _DELEGATE_TOOL_NAME,
            "description": (
                "Передать одну целостную read-only цель coordinator, который "
                "спланирует workers и агрегирует результаты."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resolved_references": {
                        "type": "string",
                        "description": (
                            "Только точные разовые факты из истории для "
                            "разрешения ссылок current_query с указанием их "
                            "ролей; пустая строка, если запрос самодостаточен. "
                            "Не содержит пересказ запроса, план или новые цели."
                        ),
                    },
                    "context": {
                        "type": "string",
                        "maxLength": COORDINATOR_CONTEXT_MAX_CHARS,
                        "description": (
                            "Только повторно применимые правила, определения, "
                            "предпочтения, инварианты и общие ограничения, "
                            "устойчиво установленные в диалоге. Не содержит "
                            "current_query, разовые объекты, ID, числа, результаты "
                            "или пересказ истории; пустая строка, если таких "
                            "договорённостей нет."
                        ),
                    },
                },
                "required": ["resolved_references", "context"],
                "additionalProperties": False,
            },
        },
    }


def _message_text(result: Any) -> str:
    content = result.content if isinstance(result, BaseMessage) else result
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text") or block.get("content")
                if text is not None:
                    parts.append(str(text))
        return "".join(parts).strip()
    return str(content or "").strip()


def build_supervisor_graph(
    model: Any,
    *,
    callbacks: Optional[Sequence[Any]] = None,
    collected_display_refs: Optional[List[str]] = None,
):
    """Build supervisor -> coordinator -> end as a LangGraph."""
    callback_list = list(callbacks or [])
    model_config = {"callbacks": callback_list} if callback_list else None
    supervisor_model = model.bind_tools([_delegate_tool_schema()])

    def invoke_supervisor(call_messages: Sequence[BaseMessage]) -> AIMessage:
        try:
            with llm_stage("supervisor"):
                result = (
                    supervisor_model.invoke(call_messages, config=model_config)
                    if model_config is not None
                    else supervisor_model.invoke(call_messages)
                )
        except Exception as exc:
            raise RuntimeError(
                f"Ошибка LLM supervisor: {type(exc).__name__}"
            ) from exc
        if not isinstance(result, AIMessage):
            return AIMessage(content=_message_text(result))
        return result

    def supervisor_node(state: SupervisorGraphState) -> Dict[str, Any]:
        payload = {
            "current_query": state["current_query"],
            "recent_history": state["recent_history"],
        }
        decision = invoke_supervisor(
            [
                SystemMessage(content=_SUPERVISOR_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
        )
        final_answer = None if decision.tool_calls else _message_text(decision)
        if final_answer is not None:
            logger.info("Supervisor answered directly")
        return {
            "supervisor_message": decision,
            "final_answer": final_answer,
        }

    def coordinator_node(state: SupervisorGraphState) -> Dict[str, Any]:
        decision = state.get("supervisor_message")
        if decision is None or not decision.tool_calls:
            raise RuntimeError(
                "Supervisor вызвал coordinator без delegate_to_coordinator."
            )
        call = decision.tool_calls[0]
        delegated_task = str(state["current_query"]).strip()
        resolved_references = str(
            call["args"].get("resolved_references") or ""
        ).strip()
        if resolved_references:
            delegated_task = (
                f"{delegated_task}\n\n"
                "Однозначно разрешённые ссылки из истории:\n"
                f"{resolved_references}"
            )
        delegated_context = str(call["args"].get("context") or "").strip()[
            :COORDINATOR_CONTEXT_MAX_CHARS
        ]
        logger.info(
            "Supervisor delegated coordinator task=%s",
            delegated_task[:1000],
        )
        coordinator_result = coordinator_chat(
            delegated_task,
            context=delegated_context,
        )
        if collected_display_refs is not None:
            collected_display_refs.extend(coordinator_result.display_refs)
        return {
            "display_refs": [
                *state["display_refs"],
                *coordinator_result.display_refs,
            ],
            "supervisor_message": None,
            "final_answer": coordinator_result.answer,
        }

    def route_after_supervisor(
        state: SupervisorGraphState,
    ) -> Literal["coordinator", "finish"]:
        decision = state.get("supervisor_message")
        if decision is None or not decision.tool_calls:
            return "finish"
        return "coordinator"

    graph = StateGraph(SupervisorGraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("coordinator", coordinator_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "coordinator": "coordinator",
            "finish": END,
        },
    )
    graph.add_edge("coordinator", END)
    return graph.compile()


def _supervisor_chat_impl(
    clean_query: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    session_id: Optional[str] = None,
) -> WorkerRunResult:
    initial_state: SupervisorGraphState = {
        "current_query": clean_query,
        "recent_history": [
            {
                "role": str(item.get("role") or ""),
                "content": str(item.get("content") or ""),
            }
            for item in (history or [])[-6:]
        ],
        "display_refs": [],
        "supervisor_message": None,
        "final_answer": None,
    }
    callback = get_callback_handler()
    callbacks = [callback] if callback is not None else []
    metrics_callback = get_run_metrics_callback()
    if metrics_callback is not None and metrics_callback not in callbacks:
        callbacks.append(metrics_callback)
    collected_display_refs: List[str] = []
    graph = build_supervisor_graph(
        chat_model,
        callbacks=callbacks,
        collected_display_refs=collected_display_refs,
    )
    graph_config = {
        "recursion_limit": 4,
        "run_name": "worker_supervisor",
    }

    with langfuse_trace_context(
        trace_name="worker_supervisor",
        session_id=session_id,
        tags=["supervisor", "coordinator", "worker", "experiment"],
    ):
        try:
            final_state = graph.invoke(initial_state, config=graph_config)
            final_answer = str(final_state.get("final_answer") or "").strip()
            if not final_answer:
                raise RuntimeError("Supervisor LangGraph завершился без ответа.")
            display_refs = list(final_state.get("display_refs") or [])
            display_items = resolve_worker_display_refs(display_refs)
            record_display_tools([item.name for item in display_items])
            return WorkerRunResult(
                answer=final_answer,
                display_items=display_items,
            )
        except Exception:
            resolve_worker_display_refs(collected_display_refs)
            raise


def supervisor_chat(
    user_query: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    session_id: Optional[str] = None,
) -> WorkerRunResult:
    """Answer directly or coordinate one or more planned worker groups."""
    clean_query = str(user_query or "").strip()
    if not clean_query:
        return WorkerRunResult(
            answer="Запрос не должен быть пустым.",
            display_items=[],
        )

    with capture_agent_run(session_id):
        return _supervisor_chat_impl(
            clean_query,
            history=history,
            session_id=session_id,
        )


__all__ = [
    "SupervisorGraphState",
    "build_supervisor_graph",
    "supervisor_chat",
]
