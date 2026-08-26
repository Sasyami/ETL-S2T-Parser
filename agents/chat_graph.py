"""LangGraph runtime for the read-only chat agent.

Architecture:

    planner (native tool calling)
        ├─ data tool -> ToolNode -> observer (structured output) -> planner
        ├─ worker finish_worker call -> END
        └─ legacy no tool_calls -> responder -> END

The legacy chat mode keeps raw ToolMessage content in the graph. The isolated
worker mode instead stores full tool results outside message history and puts a
single bounded text preview into each ToolMessage. The same planner completes a
worker with finish_worker(summary), without a separate responder call. A higher
level coordinator decides which complete results should be displayed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Sequence, TypedDict
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .contracts import (
    EvidenceFact,
    Observation,
    parse_worker_request,
)
from .observability import get_callback_handler, langfuse_trace_context
from .run_metrics import llm_stage

logger = logging.getLogger(__name__)

_VISUALIZATION_URL = re.compile(
    r"^/exports/(?:sql-lineage|s2t-graphs)/[A-Za-z0-9_.-]+\.html$"
)
_S2T_GRAPH_DATA_URL = re.compile(
    r"^/exports/s2t-graphs/[A-Za-z0-9_.-]+\.json$"
)
_OBSERVATION_GAP_MAX_CHARS = 1200
_OBSERVATION_FACT_MAX_CHARS = 300
_OBSERVATION_FACTS_MAX_COUNT = 8
_OBSERVATION_LIMITATIONS_MAX_COUNT = 4
_PLANNER_HANDOFF_MAX_CHARS = 12000
DEFAULT_TOOL_MESSAGE_PREVIEW_CHARS = 6000
_FINISH_WORKER_TOOL_NAME = "finish_worker"
_ANALYZE_KNOWN_FACTS_TOOL_NAME = "analyze_known_facts"
_OBSERVER_MAX_RETRIES = 5


@tool(parse_docstring=True)
def analyze_known_facts(answer: str) -> Dict[str, str]:
    """Передать observer анализ уже известных фактов без чтения данных.

    Это внутренний worker-tool для маршрута без внешних data tools. Он не
    получает новые факты, ничего не читает и только возвращает переданный ответ,
    чтобы обычный ToolMessage/observer-цикл оставался единым.

    Args:
        answer: Готовый ответ, построенный только по task, skills и schemas.
    """
    return {"answer": str(answer or "").strip()}


def ensure_worker_tools(
    tools: Mapping[str, BaseTool] | Sequence[BaseTool],
) -> tuple[BaseTool, ...]:
    """Guarantee one executable tool for every isolated worker cycle."""
    selected = tuple(tools.values()) if isinstance(tools, Mapping) else tuple(tools)
    return selected or (analyze_known_facts,)


_WORKER_PLANNER_PROMPT = """
Ты planner изолированного read-only worker. Доступные worker tools:
{{AVAILABLE_TOOLS}}.

Каждый твой ответ — ровно один или несколько native calls доступных worker
tools либо один native call `finish_worker`. Обычный текст без tool_calls
запрещён. `finish_worker` разрешён на любом шаге, в том числе до data tool.
Палитра worker никогда не пуста: если router не выбрал внешний data tool,
доступен внутренний `analyze_known_facts`. Передай ему готовый ответ, построенный
только по точным фактам из task, skills и schemas. Этот tool не подтверждает
новые данные, а создаёт обычный ToolMessage для проверки observer.
После каждого результата оцени task,
последний tool exchange и накопленную выжимку observer, затем либо вызови
следующий tool, либо заверши работу через finish_worker. Читай description и
схему выбранного tool, сохраняй смысл,
 ограничения и точные значения task. Не придумывай факты и не повторяй успешный
 вызов без новой причины. Не считай производный результат готовым входным фактом
 и не переименовывай заданную операцию. Не конструируй отсутствующий объект
 анализа только для заполнения обязательного аргумента tool: аргументы бери из
 task или подтверждённых результатов предыдущих tools. Квалифицированное имя
 `table.column` разделяй на `table_name` и `column_name`, если tool имеет такие
 отдельные аргументы. Если обязательного входа нет, этот tool не подходит.
 Перед data-tool call сопоставь его фактические
 аргументы с каждым условием task: объектом, scope, фильтрами, вычислением,
группировкой, порядком и правилом разрешения равенства. Не отправляй call,
который теряет хотя бы одно условие. После ошибки исправь действие по фактическому
результату либо честно укажи ограничение. В `finish_worker.summary` верни только
краткую внутреннюю отметку о выполнении или оставшемся ограничении. Evidence
сохраняется отдельно: не копируй результаты и не формулируй финальный ответ.
Если доступны несколько нужных `previous_results`, прочитай их одним вызовом
`read_previous_result(result_ids=[...])`, чтобы сохранить шаги для data-tools.

Следуй `status` последней Observation: при `continue` закрой `gap` следующим
worker-tool вызовом; при `complete` заверши работу через `finish_worker`;
`reroute` завершит текущий graph программно. Обычный текст запрещён.

""".strip()

_LEGACY_PLANNER_PROMPT = """
Ты planner read-only агента. Используй только доступные tools:
{{AVAILABLE_TOOLS}}. Читай description и схему выбранного tool. На каждом шаге
либо верни нужный native tool call, либо компактную выжимку подтверждённых
фактов для responder, если данных уже достаточно. Сохраняй смысл, ограничения
и точные значения запроса, не придумывай факты и не повторяй успешный вызов без
новой причины. Не конструируй отсутствующий объект анализа только для
заполнения обязательного аргумента tool: аргументы бери из запроса или
подтверждённых ToolMessage. После ошибки скорректируй действие по фактическому
ToolMessage.
""".strip()

_OBSERVER_PROMPT = """
Ты observer worker. Верни только structured output Observation по схеме, без Markdown
и пояснений. Сверь `user_request`, `prior_state`, текущие tool call и
result. Не выполняй производный анализ: upstream сделает его.

Выбери ровно один `status`:
- `complete` — принятые результаты содержат все исходные данные для task;
- `continue` — остаётся `gap`, который можно закрыть текущей палитрой tools;
- `reroute` — остаётся `gap`, но ни один available tool его не закрывает.

`gap` — одна консолидированная строка незакрытых требований; только при
`complete` верни JSON null, не строку `"null"`. Не повторяй одну причину и её
следствия.

`accepted_tool_call_ids` — накопительный список успешных tool results, которые
подтверждают task. Исключай ошибочные, нерелевантные и заменённые результаты.
В `facts` сохраняй только подтверждённые факты и их `evidence_ids`; ошибки,
предположения и аргументы вызова фактами не являются. `limitations` содержит
только ограничения. Результат внутреннего `analyze_known_facts` не является новым evidence.

Вызов не подтверждает task, если его аргументы потеряли или изменили объект,
scope, фильтр либо операцию. Нулевой результат подтверждает отсутствие данных только
при точных аргументах из task или принятого evidence. Объект, который
planner сам составил, не подтверждает исходную операцию: верни `gap`.
`previous_results` — только навигация до их чтения tool. `saved_result` —
служебная ссылка; при `truncated=true` полнота набора не подтверждена.
Если task зависит от прошлого результата, сначала прочитай нужный `result_id`
через `read_previous_result`; не заменяй его догадкой по description.

Выбирай `reroute`, только если ни один `available_tools` не закрывает gap. Если
достаточно изменить аргументы текущего tool, выбери `continue`.

{{PRIOR_STATE_RULE}}

Не выбирай следующий tool и не формулируй пользовательский ответ.
""".strip()

_LEGACY_RESPONDER_PROMPT = """
Сформируй окончательный ответ пользователю по выжимке planner и фактическим
результатам tools.

Не вызывай инструменты. Используй исходный запрос, пользовательскую историю
диалога, реальные ToolMessage и planner_handoff ниже. Внутренние observations
намеренно не передаются, чтобы не дублировать и не искажать результаты tools.
Сохраняй требуемую полноту, структуру и точные значения результата. Если данных
недостаточно либо инструмент завершился ошибкой, явно укажи ограничение. Не
придумывай отсутствующие факты и не упоминай внутреннее устройство графа.
""".strip()

_PLANNER_HANDOFF_PROMPT = """
Planner решил, что дополнительных инструментов больше не требуется, и
сформировал следующую выжимку проверенных фактов:

<planner_handoff>
{{PLANNER_TEXT}}
</planner_handoff>

Используй выжимку как навигацию по результатам, но точные значения и полный
запрошенный вывод бери из реальных ToolMessage. Верни полноценный окончательный
ответ, а не комментарий к выжимке.
""".strip()

_FINISH_ONLY_REPAIR_PROMPT = """
Лимит шагов исчерпан. Больше не вызывай data tools. По уже подтверждённым
результатам верни один native call finish_worker. Если задача завершена не
полностью, честно отрази это в summary.
""".strip()

_WORKER_NATIVE_CALL_REPAIR_PROMPT = """
Обычный текст planner недопустим и отброшен. Верни ровно один native call:
вызови подходящий доступный worker tool либо `finish_worker`. `finish_worker`
разрешён даже до получения data-tool результата. Не повторяй отброшенный текст
и не добавляй пояснения вне tool call.
""".strip()

_WORKER_CONTINUE_CALL_REPAIR_PROMPT = """
Observer вернул status=continue: завершать worker сейчас запрещено. Вызови ровно
один или несколько доступных data tools и закрой указанный gap. Сохрани точные
значения task; не возвращай обычный текст и не вызывай finish_worker.
""".strip()

_OBSERVER_REPAIR_PROMPT = """
Предыдущий observer-вызов не вернул валидный structured output Observation.
Повторно оцени тот же исходный user_request, prior_state, tool_calls и
tool_results из payload выше. Data tool уже выполнен: не требуй его повторного
вызова только из-за ошибки формата observer. Верни только один валидный
structured output Observation без Markdown и дополнительного текста.

Ошибка structured output: {validation_error}
""".strip()

class ChatHistoryMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class WorkerCycleTrace(BaseModel):
    """Bounded planner/tool/observer exchange retained for diagnostics."""

    model_config = ConfigDict(extra="forbid")

    cycle: int = Field(ge=1)
    routing_attempt: int = Field(default=1, ge=1)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    observation: Observation


class AgentGraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    system_prompt: str
    planner_message: Optional[AIMessage]
    observations: List[Observation]
    cycle_history: List[WorkerCycleTrace]
    tool_steps: int
    max_steps: int
    terminal_gap: Optional[str]


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif block.get("type") == "text" and "content" in block:
                    parts.append(str(block["content"]))
        if parts:
            return "".join(parts)
    return str(content)


def _message_text(message: BaseMessage) -> str:
    return _message_content_text(message.content).strip()


def _tool_content_text(content: Any) -> str:
    """Serialize one tool result to the single textual worker-message field."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def _tool_message_preview(content: Any, max_chars: int) -> str:
    text = _tool_content_text(content).strip()
    limit = max(1, int(max_chars))
    if len(text) <= limit:
        return text
    marker = f"… [preview обрезан: полный результат {len(text)} символов]"
    if len(marker) >= limit:
        return marker[:limit]
    return text[: limit - len(marker)].rstrip() + marker


def _normalize_tools(
    tools: Mapping[str, BaseTool] | Sequence[BaseTool],
) -> List[BaseTool]:
    if isinstance(tools, Mapping):
        tool_list = list(tools.values())
    else:
        tool_list = list(tools)

    names = [tool.name for tool in tool_list]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(
            "Имена инструментов должны быть уникальными: "
            + ", ".join(duplicate_names)
        )

    return tool_list


def _with_structured_output(model: Any, schema: Any) -> Any:
    """Prefer native calls and normalize only surrounding call-name spaces."""
    try:
        raw_model = model.with_structured_output(
            schema,
            method="function_calling",
            include_raw=True,
        )
    except TypeError:
        try:
            return model.with_structured_output(
                schema,
                method="function_calling",
            )
        except TypeError:
            return model.with_structured_output(schema)

    class _NormalizedStructuredOutput:
        def invoke(self, messages: Any, **kwargs: Any) -> Any:
            result = raw_model.invoke(messages, **kwargs)
            if not isinstance(result, Mapping):
                return result
            parsed = result.get("parsed")
            if parsed is not None:
                return parsed

            raw_message = result.get("raw")
            expected_name = str(
                getattr(schema, "model_config", {}).get("title")
                or getattr(schema, "__name__", "")
            )
            matching_calls = [
                call
                for call in getattr(raw_message, "tool_calls", []) or []
                if str(call.get("name") or "").strip() == expected_name
            ]
            if len(matching_calls) == 1:
                return schema.model_validate(
                    matching_calls[0].get("args") or {}
                )

            parsing_error = result.get("parsing_error")
            if isinstance(parsing_error, BaseException):
                raise parsing_error
            raise ValueError(
                "Structured output did not contain exactly one expected call "
                f"{expected_name}."
            )

    return _NormalizedStructuredOutput()


def _split_worker_request(
    value: Any,
) -> tuple[str, Optional[List[Dict[str, Any]]]]:
    """Expose only the current task and minimal previous result refs."""
    parts = parse_worker_request(value)
    previous_results = (
        [item.model_dump(mode="json") for item in parts.previous_results]
        if parts.previous_results is not None
        else None
    )
    return parts.current_task, previous_results


def _history_messages(
    history: Optional[List[ChatHistoryMessage]],
) -> List[BaseMessage]:
    result: List[BaseMessage] = []
    for item in history or []:
        if item["role"] == "user":
            result.append(HumanMessage(content=item["content"]))
        else:
            result.append(AIMessage(content=item["content"]))
    return result


def _clip_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _compact_observation(observation: Observation) -> Observation:
    return Observation(
        status=observation.status,
        gap=(
            _clip_text(observation.gap, _OBSERVATION_GAP_MAX_CHARS)
            if observation.gap is not None
            else None
        ),
        accepted_tool_call_ids=list(observation.accepted_tool_call_ids),
        facts=[
            EvidenceFact(
                text=_clip_text(item.text, _OBSERVATION_FACT_MAX_CHARS),
                evidence_ids=list(item.evidence_ids),
            )
            for item in observation.facts[
                :_OBSERVATION_FACTS_MAX_COUNT
            ]
        ],
        limitations=[
            _clip_text(item, _OBSERVATION_FACT_MAX_CHARS)
            for item in observation.limitations[
                :_OBSERVATION_LIMITATIONS_MAX_COUNT
            ]
        ],
    )


class WorkerFinishPayload(BaseModel):
    """Strict arguments of the worker's native finish call."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)

    @field_validator("summary")
    @classmethod
    def _summary_must_not_be_blank(cls, value: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("summary must not be blank")
        return clean_value

class WorkerDisplayItem(BaseModel):
    """Full successful tool output retained outside the worker LLM context."""

    model_config = ConfigDict(extra="forbid")

    name: str
    content: str
    evidence_id: str = Field(default="", exclude=True)
    tool_call_id: str = Field(default="", exclude=True)
    arguments: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    preview: str = Field(default="", exclude=True)
    truncated: bool = Field(default=False, exclude=True)


class WorkerRunResult(BaseModel):
    """Internal worker-graph result before the public handoff is built."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    display_items: List[WorkerDisplayItem] = Field(default_factory=list)
    cycle_history: List[WorkerCycleTrace] = Field(default_factory=list)
    status: Literal["complete", "reroute"] = Field(
        default="complete",
        exclude=True,
    )
    gap: Optional[str] = Field(default=None, exclude=True)
    facts: List[EvidenceFact] = Field(default_factory=list, exclude=True)
    accepted_tool_call_ids: List[str] = Field(
        default_factory=list,
        exclude=True,
    )

    @field_validator("gap", mode="before")
    @classmethod
    def _normalize_result_gap(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        clean_value = str(value).strip()
        return clean_value or None

    @model_validator(mode="after")
    def _validate_result_status(self) -> "WorkerRunResult":
        if self.status == "reroute" and self.gap is None:
            raise ValueError("reroute worker result must contain gap")
        return self


class WorkerResponseError(RuntimeError):
    """Raised when the worker violates its finish contract."""


def _finish_worker_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _FINISH_WORKER_TOOL_NAME,
            "description": (
                "Завершить worker после complete и вернуть краткую "
                "внутреннюю отметку. Evidence хранится отдельно."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "Краткая отметка о выполнении или ограничении без "
                            "копирования результатов и производного анализа"
                        ),
                    },
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    }


def _worker_finish_payload(
    call: Mapping[str, Any],
) -> WorkerFinishPayload:
    try:
        payload = WorkerFinishPayload.model_validate(call.get("args") or {})
    except (TypeError, ValueError, ValidationError) as exc:
        raise WorkerResponseError(
            "finish_worker вернул невалидные аргументы"
        ) from exc

    return payload


def _runtime_context(
    state: AgentGraphState,
    *,
    include_observations: bool = True,
) -> Optional[str]:
    parts: List[str] = []

    if include_observations:
        observations = list(state.get("observations") or [])
        latest_observations = (
            [(len(observations), observations[-1])] if observations else []
        )
        for index, observation in latest_observations:
            observation_parts = [f"Observation для шага {index}."]
            if observation.status == "complete":
                observation_parts.append(
                    "Статус: complete. Необходимые исходные данные получены; "
                    "заверши worker через finish_worker."
                )
            elif observation.status == "continue":
                observation_parts.append(
                    "Статус: continue. Закрой gap следующим вызовом доступного "
                    "worker tool:\n" + str(observation.gap or "")
                )
            else:
                observation_parts.append(
                    "Статус: reroute. Текущая палитра не закрывает gap:\n"
                    + str(observation.gap or "")
                )
            if observation.facts:
                observation_parts.append(
                    "Подтверждённые факты:\n- "
                    + "\n- ".join(
                        f"{fact.text} [evidence: "
                        f"{', '.join(fact.evidence_ids) or 'нет'}]"
                        for fact in observation.facts
                    )
                )
            if observation.limitations:
                observation_parts.append(
                    "Ограничения и неоднозначности:\n- "
                    + "\n- ".join(observation.limitations)
                )
            parts.append("\n".join(observation_parts))

    if not parts:
        return None

    return "\n\n".join(parts)


def _planner_instruction(
    available_tool_names: Sequence[str],
    *,
    worker_finish: bool = False,
) -> str:
    available = ", ".join(available_tool_names) or "нет"
    template = (
        _WORKER_PLANNER_PROMPT
        if worker_finish
        else _LEGACY_PLANNER_PROMPT
    )
    return template.replace("{{AVAILABLE_TOOLS}}", available)


def _planner_messages(
    state: AgentGraphState,
    available_tool_names: Sequence[str] = (),
    *,
    worker_finish: bool = False,
) -> List[BaseMessage]:
    limit_reached = state["tool_steps"] >= state["max_steps"]
    planner_instruction = _planner_instruction(
        available_tool_names,
        worker_finish=worker_finish,
    )

    if limit_reached:
        if worker_finish:
            planner_instruction += (
                "\n\nЛимит data-tool шагов исчерпан. Сейчас разрешён только "
                "finish_worker. Заверши задачу с подтверждёнными фактами и честно "
                "укажи недостающие данные."
            )
        else:
            planner_instruction += (
                "\n\nЛимит инструментальных шагов исчерпан. "
                "Не вызывай инструменты; верни обычное сообщение без tool_calls, "
                "чтобы граф перешёл к responder."
            )

    system_parts = [state["system_prompt"].strip(), planner_instruction]
    runtime_context = _runtime_context(state)
    if runtime_context is not None:
        system_parts.append(runtime_context)

    messages: List[BaseMessage] = [
        SystemMessage(content="\n\n".join(system_parts))
    ]

    if worker_finish:
        task_message = next(
            (
                message
                for message in state["messages"]
                if isinstance(message, HumanMessage)
            ),
            None,
        )
        if task_message is not None:
            messages.append(task_message)
        try:
            latest_call, latest_results = _latest_tool_exchange(
                state["messages"]
            )
        except RuntimeError:
            latest_call = None
            latest_results = []
        if latest_call is not None:
            messages.append(latest_call)
            messages.extend(latest_results)
    else:
        messages.extend(state["messages"])
    return messages


def _latest_tool_exchange(
    messages: Sequence[BaseMessage],
) -> tuple[AIMessage, List[ToolMessage]]:
    tool_results: List[ToolMessage] = []

    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            tool_results.append(message)
            continue

        if isinstance(message, AIMessage) and message.tool_calls:
            tool_results.reverse()
            if not tool_results:
                raise RuntimeError(
                    "После tool call отсутствует соответствующий ToolMessage."
                )
            return message, tool_results

    raise RuntimeError("В истории не найден последний инструментальный обмен.")


def _last_user_query(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    return ""


def _tool_message_payload(message: ToolMessage) -> Dict[str, Any]:
    return {
        "name": message.name,
        "tool_call_id": message.tool_call_id,
        "content": message.content,
        "status": getattr(message, "status", None),
        "is_error": _tool_message_has_error(message),
    }


def _tool_message_has_error(message: ToolMessage) -> bool:
    if getattr(message, "status", None) == "error":
        return True

    content = message.content
    if isinstance(content, dict):
        payload = content
    else:
        try:
            payload = json.loads(_message_content_text(content))
        except (json.JSONDecodeError, TypeError):
            return False

    return isinstance(payload, dict) and bool(payload.get("error"))


def _tool_result_truncated(message: ToolMessage) -> bool:
    content = message.content
    if isinstance(content, dict):
        payload = content
    else:
        try:
            payload = json.loads(_message_content_text(content))
        except (json.JSONDecodeError, TypeError):
            return False
    if not isinstance(payload, dict):
        return False
    saved_result = payload.get("saved_result")
    return bool(
        payload.get("truncated")
        or payload.get("input_truncated")
        or (
            isinstance(saved_result, dict)
            and saved_result.get("truncated")
        )
    )


def _visualization_urls(messages: Sequence[BaseMessage]) -> List[str]:
    urls: List[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(_message_content_text(message.content))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        url = payload.get("visualization_url")
        if (
            isinstance(url, str)
            and _VISUALIZATION_URL.fullmatch(url)
            and url not in urls
        ):
            urls.append(url)
    return urls


def _s2t_graph_data_urls(messages: Sequence[BaseMessage]) -> List[str]:
    urls: List[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(_message_content_text(message.content))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        url = payload.get("data_url")
        if (
            isinstance(url, str)
            and _S2T_GRAPH_DATA_URL.fullmatch(url)
            and url not in urls
        ):
            urls.append(url)
    return urls


def build_agent_graph(
    model: Any,
    tools: Mapping[str, BaseTool] | Sequence[BaseTool],
    *,
    raw_tool_results: Optional[Dict[str, ToolMessage]] = None,
    evidence_ids_by_tool_call: Optional[Dict[str, str]] = None,
    tool_message_preview_chars: Optional[int] = None,
    worker_finish: bool = False,
):
    """Build planner -> tools -> observer -> planner."""
    tool_list = _normalize_tools(tools)
    tool_names = tuple(tool.name for tool in tool_list)
    retained_tool_results = (
        raw_tool_results if raw_tool_results is not None else {}
    )
    evidence_ids = (
        evidence_ids_by_tool_call
        if evidence_ids_by_tool_call is not None
        else {}
    )
    finish_schema = _finish_worker_tool_schema()
    planner_model = (
        model.bind_tools([*tool_list, finish_schema])
        if worker_finish
        else (model.bind_tools(tool_list) if tool_list else model)
    )
    continuation_model = (
        model.bind_tools(tool_list)
        if worker_finish and tool_list
        else planner_model
    )
    finish_model = None
    if worker_finish:
        try:
            finish_model = model.bind_tools(
                [finish_schema],
                tool_choice=_FINISH_WORKER_TOOL_NAME,
            )
        except TypeError:
            finish_model = model.bind_tools([finish_schema])
    observer_model = _with_structured_output(model, Observation)
    tool_node = ToolNode(tool_list, handle_tool_errors=True)

    def evidence_payload(message: ToolMessage) -> Dict[str, Any]:
        payload = _tool_message_payload(message)
        evidence_id = evidence_ids.get(str(message.tool_call_id or ""))
        if evidence_id is not None:
            payload["evidence_id"] = evidence_id
        return payload

    def invoke_with_fallback(
        primary_model: Any,
        messages: Sequence[BaseMessage],
        *,
        stage: str,
        fallback_model: Optional[Any] = None,
    ) -> Any:
        with llm_stage(stage):
            try:
                return primary_model.invoke(messages)
            except Exception:
                if fallback_model is None or fallback_model is primary_model:
                    raise
                logger.warning(
                    "Forced tool-choice LLM call failed; retrying with the "
                    "regular planner tool palette",
                    exc_info=True,
                )
                return fallback_model.invoke(messages)

    def planner(state: AgentGraphState) -> Dict[str, Any]:
        limit_reached = state["tool_steps"] >= state["max_steps"]
        latest_observation = (
            (state.get("observations") or [])[-1]
            if state.get("observations")
            else None
        )
        finish_only = bool(
            limit_reached
            or (
                worker_finish
                and latest_observation is not None
                and latest_observation.status == "complete"
            )
        )
        must_continue = bool(
            worker_finish
            and not limit_reached
            and latest_observation is not None
            and latest_observation.status == "continue"
        )
        if finish_only and worker_finish:
            selected_model = finish_model
        elif finish_only:
            selected_model = model
        elif must_continue:
            selected_model = continuation_model
        else:
            selected_model = planner_model
        planner_messages = _planner_messages(
            state,
            tool_names,
            worker_finish=worker_finish,
        )

        selected_fallback = (
            planner_model
            if worker_finish and selected_model is finish_model
            else None
        )
        planner_stage = (
            "finish_worker"
            if worker_finish and finish_only
            else "worker_planner" if worker_finish else "legacy_planner"
        )
        try:
            reply = invoke_with_fallback(
                selected_model,
                planner_messages,
                stage=planner_stage,
                fallback_model=selected_fallback,
            )
        except Exception as exc:
            logger.exception("LLM error in planner")
            # Legacy mode can hand this text to its responder. Worker mode
            # rejects plain text below and requests one native call.
            reply = AIMessage(content=f"Planner error: {type(exc).__name__}")

        if not isinstance(reply, AIMessage):
            reply = AIMessage(content=_message_content_text(reply))

        invalid_continue_finish = bool(
            must_continue
            and any(
                call.get("name") == _FINISH_WORKER_TOOL_NAME
                for call in reply.tool_calls
            )
        )
        if worker_finish and (
            not reply.tool_calls
            or invalid_continue_finish
        ):
            logger.warning(
                "Worker planner returned %s; discarding it and requesting "
                "one allowed native call",
                (
                    "finish_worker while observer requires continue"
                    if invalid_continue_finish
                    else "plain text"
                ),
            )
            continue_repair_prompt = _WORKER_CONTINUE_CALL_REPAIR_PROMPT
            repaired_reply = invoke_with_fallback(
                selected_model,
                [
                    *planner_messages,
                    HumanMessage(
                        content=(
                            continue_repair_prompt
                            if must_continue
                            else _WORKER_NATIVE_CALL_REPAIR_PROMPT
                        )
                    ),
                ],
                stage=planner_stage,
                fallback_model=selected_fallback,
            )
            if not isinstance(repaired_reply, AIMessage):
                repaired_reply = AIMessage(
                    content=_message_content_text(repaired_reply)
                )
            repaired_continue_finish = bool(
                must_continue
                and any(
                    call.get("name") == _FINISH_WORKER_TOOL_NAME
                    for call in repaired_reply.tool_calls
                )
            )
            if (
                not repaired_reply.tool_calls
                or repaired_continue_finish
            ):
                repaired_reply = AIMessage(
                    content=(
                        "Worker planner после repair не вернул допустимый "
                        "native data-tool call."
                    )
                )
            reply = repaired_reply

        if finish_only and worker_finish and reply.tool_calls and not any(
            call.get("name") == _FINISH_WORKER_TOOL_NAME
            for call in reply.tool_calls
        ):
            logger.warning(
                "Worker requested another data tool after the step limit; "
                "asking the LLM to finish"
            )
            repaired_reply = invoke_with_fallback(
                finish_model,
                [
                    *planner_messages,
                    HumanMessage(content=_FINISH_ONLY_REPAIR_PROMPT),
                ],
                stage="finish_worker",
                fallback_model=planner_model,
            )
            if not isinstance(repaired_reply, AIMessage):
                repaired_reply = AIMessage(
                    content=_message_content_text(repaired_reply)
                )
            if repaired_reply.tool_calls and not any(
                call.get("name") == _FINISH_WORKER_TOOL_NAME
                for call in repaired_reply.tool_calls
            ):
                repaired_reply = AIMessage(
                    content=(
                        "Worker после повторного LLM-вызова не завершил "
                        "задачу native-вызовом."
                    )
                )
            reply = repaired_reply

        logger.info(
            "Agent planner after %s tool step(s): tool_calls=%s content=%s",
            state["tool_steps"],
            [call.get("name") for call in reply.tool_calls],
            _message_text(reply)[:1000],
        )

        return {"planner_message": reply}

    def prepare_tool_call(state: AgentGraphState) -> Dict[str, Any]:
        planner_message = state.get("planner_message")
        if planner_message is None or not planner_message.tool_calls:
            raise RuntimeError("Planner не выбрал инструмент.")

        return {
            "messages": [planner_message],
            "planner_message": None,
        }

    def execute_tools(state: AgentGraphState) -> Dict[str, Any]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            raise RuntimeError("ToolNode вызван без AIMessage.tool_calls.")

        logger.info(
            "Executing tool step %s: %s",
            state["tool_steps"] + 1,
            [
                {
                    "name": call.get("name"),
                    "args": call.get("args", {}),
                }
                for call in last_message.tool_calls
            ],
        )

        result = tool_node.invoke(state)
        raw_messages = [
            message.model_copy(update={"status": "error"})
            if isinstance(message, ToolMessage)
            and _tool_message_has_error(message)
            and getattr(message, "status", None) != "error"
            else message
            for message in result.get("messages", [])
        ]
        tool_messages: List[BaseMessage] = []
        for message in raw_messages:
            if not isinstance(message, ToolMessage):
                tool_messages.append(message)
                continue

            if not _tool_message_has_error(message):
                from .tools.saved_results import persist_sqlite_tool_message

                message = persist_sqlite_tool_message(message)

            tool_call_id = str(message.tool_call_id or "").strip()
            if (
                tool_call_id
                and not _tool_message_has_error(message)
                and str(message.name or "")
                != _ANALYZE_KNOWN_FACTS_TOOL_NAME
            ):
                evidence_ids.setdefault(
                    tool_call_id,
                    f"evidence_{uuid4().hex}",
                )

            retained_tool_results[message.tool_call_id] = message

            if tool_message_preview_chars is None:
                tool_messages.append(message)
            else:
                tool_messages.append(
                    message.model_copy(
                        update={
                            "content": _tool_message_preview(
                                message.content,
                                tool_message_preview_chars,
                            )
                        }
                    )
                )
        logger.info(
            "Tool step result: %s",
            json.dumps(
                [
                    evidence_payload(message)
                    if isinstance(message, ToolMessage)
                    else str(message)
                    for message in tool_messages
                ],
                ensure_ascii=False,
                default=str,
            )[:3000],
        )

        return {
            "messages": tool_messages,
            "tool_steps": state["tool_steps"] + len(last_message.tool_calls),
        }

    def observer(state: AgentGraphState) -> Dict[str, Any]:
        planner_message = state.get("planner_message")
        no_tool_cycle = bool(
            worker_finish
            and not tool_list
            and planner_message is not None
        )
        candidate_answer = ""
        if no_tool_cycle:
            assert planner_message is not None
            finish_calls = [
                call
                for call in planner_message.tool_calls
                if call.get("name") == _FINISH_WORKER_TOOL_NAME
            ]
            if len(finish_calls) == 1 and len(planner_message.tool_calls) == 1:
                candidate_answer = _worker_finish_payload(
                    finish_calls[0]
                ).summary
            elif not planner_message.tool_calls:
                candidate_answer = _message_text(planner_message)
            tool_call_message = AIMessage(content=candidate_answer)
            tool_results: List[ToolMessage] = []
        else:
            tool_call_message, tool_results = _latest_tool_exchange(
                state["messages"]
            )

        prior_observation = (
            (state.get("observations") or [])[-1]
            if state.get("observations")
            else None
        )
        prior_accepted_order = list(
            prior_observation.accepted_tool_call_ids
            if prior_observation is not None
            else []
        )
        prior_accepted_ids = set(prior_accepted_order)
        current_accepted_candidates = [
            str(message.tool_call_id or "").strip()
            for message in tool_results
            if str(message.tool_call_id or "").strip()
            and not _tool_message_has_error(message)
            and str(message.name or "") != _ANALYZE_KNOWN_FACTS_TOOL_NAME
        ]
        allowed_ids = prior_accepted_ids | set(current_accepted_candidates)
        prior_evidence = [
            evidence_payload(
                retained_tool_results[tool_call_id].model_copy(
                    update={
                        "content": _tool_message_preview(
                            retained_tool_results[tool_call_id].content,
                            1500,
                        )
                    }
                )
            )
            for tool_call_id in prior_accepted_order
            if tool_call_id in retained_tool_results
        ]

        current_user_request, previous_results = _split_worker_request(
            _last_user_query(state["messages"])
        )
        payload = {
            "user_request": current_user_request,
            "available_tools": [
                {
                    "name": tool.name,
                    "description": _clip_text(tool.description, 600),
                }
                for tool in tool_list
            ],
            "prior_state": (
                [
                    observation.model_dump()
                    for observation in (state.get("observations") or [])
                ]
                if worker_finish
                else []
            ),
            "accepted_evidence": prior_evidence,
            "tool_calls": tool_call_message.tool_calls,
            "tool_results": [
                evidence_payload(message) for message in tool_results
            ],
            "candidate_answer": candidate_answer,
        }
        if previous_results is not None:
            payload["previous_results"] = previous_results

        prior_state_rule = (
            "`prior_state` и `accepted_evidence` накоплены ранее. Верни "
            "накопительную Observation: сохрани принятые факты и IDs, не "
            "требуй их повторно и обнови `gap` после текущего результата."
            if worker_finish
            else ""
        )
        observer_messages: List[BaseMessage] = [
            SystemMessage(
                content="\n\n".join(
                    part
                    for part in (
                        state["system_prompt"].strip(),
                        _OBSERVER_PROMPT.replace(
                            "{{PRIOR_STATE_RULE}}",
                            prior_state_rule,
                        ),
                    )
                    if part
                )
            ),
            HumanMessage(
                content=json.dumps(payload, ensure_ascii=False, default=str)
            ),
        ]

        def parse_observation(result: Any) -> Observation:
            parsed_observation = (
                result
                if isinstance(result, Observation)
                else Observation.model_validate(result)
            )
            accepted_ids = set(parsed_observation.accepted_tool_call_ids)
            invalid_ids = sorted(accepted_ids - allowed_ids)
            if invalid_ids:
                raise ValueError(
                    "accepted_tool_call_ids содержит неизвестные, ошибочные "
                    "или внутренние результаты: " + ", ".join(invalid_ids)
                )
            if (
                parsed_observation.status == "complete"
                and allowed_ids
                and not accepted_ids
            ):
                raise ValueError(
                    "status=complete требует хотя бы один принятый "
                    "внешний tool result в accepted_tool_call_ids"
                )
            accepted_evidence_ids = {
                evidence_ids[tool_call_id]
                for tool_call_id in accepted_ids
                if tool_call_id in evidence_ids
            }
            fact_evidence_ids = {
                evidence_id
                for fact in parsed_observation.facts
                for evidence_id in fact.evidence_ids
            }
            unknown_evidence_ids = sorted(
                fact_evidence_ids - accepted_evidence_ids
            )
            if unknown_evidence_ids:
                raise ValueError(
                    "facts содержит неизвестные или непринятые evidence_ids: "
                    + ", ".join(unknown_evidence_ids)
                )
            if accepted_evidence_ids and any(
                not fact.evidence_ids for fact in parsed_observation.facts
            ):
                raise ValueError(
                    "Каждый подтверждённый fact должен ссылаться на evidence_id"
                )
            return parsed_observation

        observation: Optional[Observation] = None
        observer_error: Optional[Exception] = None
        attempt_messages = observer_messages
        total_attempts = _OBSERVER_MAX_RETRIES + 1
        for attempt_index in range(total_attempts):
            try:
                with llm_stage("observer"):
                    observation = parse_observation(
                        observer_model.invoke(attempt_messages)
                    )
                break
            except Exception as error:
                observer_error = error
                if attempt_index >= _OBSERVER_MAX_RETRIES:
                    break
                logger.warning(
                    "Structured observer output rejected on attempt %s/%s; "
                    "repeating observer without repeating data tools: %s",
                    attempt_index + 1,
                    total_attempts,
                    error,
                )
                attempt_messages = [
                    *observer_messages,
                    HumanMessage(
                        content=_OBSERVER_REPAIR_PROMPT.replace(
                            "{validation_error}",
                            f"{type(error).__name__}: {error}",
                        )
                    ),
                ]

        if observation is None:
            terminal_gap = (
                "Observer не смог вернуть валидную структуру после "
                f"{total_attempts} попыток; data tool не повторялся."
            )
            logger.error(
                "Structured observer repair failed after %s attempts; "
                "failing worker contract without repeating data tools: %s",
                total_attempts,
                observer_error,
            )
            return {"terminal_gap": terminal_gap}

        observation = _compact_observation(observation)
        logger.info(
            "Observer result: %s",
            observation.model_dump_json()[:2000],
        )
        update: Dict[str, Any] = {
            "observations": (
                [observation]
                if worker_finish
                else [
                    *(state.get("observations") or []),
                    observation,
                ]
            )
        }
        if worker_finish:
            prior_cycles = list(state.get("cycle_history") or [])
            update["cycle_history"] = [
                *prior_cycles,
                WorkerCycleTrace(
                    cycle=len(prior_cycles) + 1,
                    tool_calls=[
                        {
                            "name": str(call.get("name") or "unknown_tool"),
                            "args": call.get("args", {}),
                        }
                        for call in tool_call_message.tool_calls
                    ],
                    tool_results=[
                        evidence_payload(message)
                        for message in tool_results
                    ],
                    observation=observation,
                ),
            ]
        return update

    def responder(state: AgentGraphState) -> Dict[str, Any]:
        response_instruction = _LEGACY_RESPONDER_PROMPT

        planner_message = state.get("planner_message")
        planner_text = _clip_text(
            _message_text(planner_message),
            _PLANNER_HANDOFF_MAX_CHARS,
        ) if (
            planner_message is not None and not planner_message.tool_calls
        ) else ""
        if planner_text:
            response_instruction += "\n\n" + _PLANNER_HANDOFF_PROMPT.replace(
                "{{PLANNER_TEXT}}",
                planner_text,
            )

        system_parts = [state["system_prompt"].strip(), response_instruction]
        runtime_context = _runtime_context(
            state,
            include_observations=False,
        )
        if runtime_context is not None:
            system_parts.append(runtime_context)

        messages: List[BaseMessage] = [
            SystemMessage(content="\n\n".join(system_parts))
        ]

        messages.extend(state["messages"])

        try:
            with llm_stage("legacy_responder"):
                reply = model.invoke(messages)
            if not isinstance(reply, AIMessage):
                reply = AIMessage(content=_message_content_text(reply))
        except Exception as exc:
            logger.exception("LLM error in responder")
            reply = AIMessage(
                content=f"Не удалось сформировать ответ из-за ошибки связи с LLM: {exc}"
            )

        return {
            "messages": [reply],
            "planner_message": None,
        }

    def route_after_planner(
        state: AgentGraphState,
    ) -> Literal["prepare_tool", "observer", "responder", "finish"]:
        planner_message = state.get("planner_message")
        if worker_finish:
            if not tool_list:
                return "observer"
            if planner_message is not None and any(
                call.get("name") == _FINISH_WORKER_TOOL_NAME
                for call in planner_message.tool_calls
            ):
                return "finish"
            if planner_message is None or not planner_message.tool_calls:
                return "finish"
            return "prepare_tool"
        if state["tool_steps"] >= state["max_steps"]:
            return "responder"
        if planner_message is not None and planner_message.tool_calls:
            return "prepare_tool"
        return "responder"

    def route_after_observer(
        state: AgentGraphState,
    ) -> Literal["planner", "responder", "finish"]:
        if state.get("terminal_gap"):
            return "finish" if worker_finish else "responder"
        latest_observation = (
            (state.get("observations") or [])[-1]
            if state.get("observations")
            else None
        )
        if (
            worker_finish
            and latest_observation is not None
            and latest_observation.status == "reroute"
        ):
            return "finish"
        if worker_finish and not tool_list:
            return "finish"
        return "planner"

    graph = StateGraph(AgentGraphState)
    graph.add_node("planner", planner)
    graph.add_node("prepare_tool", prepare_tool_call)
    graph.add_node("tools", execute_tools)
    graph.add_node("observer", observer)
    graph.add_node("responder", responder)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "prepare_tool": "prepare_tool",
            "observer": "observer",
            "responder": "responder",
            "finish": END,
        },
    )
    graph.add_edge("prepare_tool", "tools")
    graph.add_edge("tools", "observer")
    graph.add_conditional_edges(
        "observer",
        route_after_observer,
        {
            "planner": "planner",
            "responder": "responder",
            "finish": END,
        },
    )
    graph.add_edge("responder", END)

    return graph.compile()


def run_agent_graph(
    user_query: str,
    system_prompt: str,
    model: Any,
    tools: Mapping[str, BaseTool] | Sequence[BaseTool],
    max_steps: int = 5,
    history: Optional[List[ChatHistoryMessage]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    trace_tags: Optional[List[str]] = None,
    trace_metadata: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List[Any]] = None,
) -> str:
    """Run the graph and return the final responder message."""
    clean_query = user_query.strip()
    if not clean_query:
        return "Запрос не должен быть пустым."

    bounded_steps = max(1, int(max_steps))
    graph = build_agent_graph(model, tools)

    initial_messages = [
        *_history_messages(history),
        HumanMessage(content=clean_query),
    ]

    initial_state: AgentGraphState = {
        "messages": initial_messages,
        "system_prompt": system_prompt,
        "planner_message": None,
        "observations": [],
        "cycle_history": [],
        "tool_steps": 0,
        "max_steps": bounded_steps,
        "terminal_gap": None,
    }

    config: Dict[str, Any] = {
        "recursion_limit": bounded_steps * 4 + 8,
        "run_name": "agent_chat",
    }

    callback_list = list(callbacks or [])
    handler = get_callback_handler()
    if handler is not None and handler not in callback_list:
        callback_list.append(handler)
    if callback_list:
        config["callbacks"] = callback_list

    with langfuse_trace_context(
        trace_name="agent_chat",
        session_id=session_id,
        user_id=user_id,
        metadata=trace_metadata,
        tags=trace_tags or ["chat"],
    ):
        final_state = graph.invoke(initial_state, config=config)

    messages = final_state.get("messages") or []
    if messages and isinstance(messages[-1], AIMessage):
        answer = _message_text(messages[-1])
        if answer:
            for url in _visualization_urls(messages):
                label = (
                    "Открыть интерактивный граф связей S2T-таблиц"
                    if url.startswith("/exports/s2t-graphs/")
                    else "Открыть интерактивный SQL lineage-граф"
                )
                link = f"[{label}]({url})"
                if url not in answer:
                    answer += f"\n\n{link}"
            for url in _s2t_graph_data_urls(messages):
                link = f"[Открыть данные графа в JSON]({url})"
                if link not in answer:
                    answer += f"\n\n{link}"
            logger.info(
                "Agent final response (%d chars):\n%s",
                len(answer),
                answer,
            )
            return answer

    logger.warning("Agent finished without a final AIMessage response")
    return "Модель не вернула финальный ответ."


def run_worker_graph(
    task: str,
    system_prompt: str,
    model: Any,
    tools: Mapping[str, BaseTool] | Sequence[BaseTool],
    max_steps: int = 5,
    *,
    tool_message_preview_chars: int = DEFAULT_TOOL_MESSAGE_PREVIEW_CHARS,
    callbacks: Optional[List[Any]] = None,
) -> WorkerRunResult:
    """Run the internal worker graph and retain its selected UI results locally."""
    clean_task = str(task or "").strip()
    if not clean_task:
        raise WorkerResponseError("Worker получил пустую task.")

    bounded_steps = max(1, int(max_steps))
    preview_chars = max(1, int(tool_message_preview_chars))
    raw_tool_results: Dict[str, ToolMessage] = {}
    evidence_ids_by_tool_call: Dict[str, str] = {}
    worker_tools = ensure_worker_tools(tools)
    graph = build_agent_graph(
        model,
        worker_tools,
        raw_tool_results=raw_tool_results,
        evidence_ids_by_tool_call=evidence_ids_by_tool_call,
        tool_message_preview_chars=preview_chars,
        worker_finish=True,
    )

    initial_state: AgentGraphState = {
        "messages": [HumanMessage(content=clean_task)],
        "system_prompt": system_prompt,
        "planner_message": None,
        "observations": [],
        "cycle_history": [],
        "tool_steps": 0,
        "max_steps": bounded_steps,
        "terminal_gap": None,
    }
    config: Dict[str, Any] = {
        "recursion_limit": bounded_steps * 4 + 8,
        "run_name": "isolated_worker",
    }

    callback_list = list(callbacks or [])
    handler = get_callback_handler()
    if handler is not None and handler not in callback_list:
        callback_list.append(handler)
    if callback_list:
        config["callbacks"] = callback_list

    with langfuse_trace_context(
        trace_name="isolated_worker",
        metadata={"tool_message_preview_chars": preview_chars},
        tags=["worker", "experiment"],
    ):
        final_state = graph.invoke(initial_state, config=config)

    latest_observation = (
        (final_state.get("observations") or [])[-1]
        if final_state.get("observations")
        else None
    )
    if latest_observation is not None and latest_observation.status == "reroute":
        reroute_gap = str(latest_observation.gap or "").strip()
        logger.info(
            "Worker graph finished with reroute request: gap=%s",
            reroute_gap,
        )
        return WorkerRunResult(
            answer=reroute_gap,
            display_items=[],
            cycle_history=list(final_state.get("cycle_history") or []),
            status="reroute",
            gap=reroute_gap,
            facts=list(latest_observation.facts),
            accepted_tool_call_ids=(
                list(latest_observation.accepted_tool_call_ids)
                if latest_observation is not None
                else []
            ),
        )

    accepted_tool_call_ids = (
        list(latest_observation.accepted_tool_call_ids)
        if latest_observation is not None
        else []
    )
    accepted_id_set = set(accepted_tool_call_ids)
    successful_messages = [
        message
        for message in raw_tool_results.values()
        if message.tool_call_id in accepted_id_set
        and not _tool_message_has_error(message)
        and message.name != _ANALYZE_KNOWN_FACTS_TOOL_NAME
    ]
    tool_arguments: Dict[str, Dict[str, Any]] = {}
    for message in final_state.get("messages") or []:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            call_id = str(call.get("id") or "").strip()
            if call_id:
                tool_arguments[call_id] = dict(call.get("args") or {})

    display_items: List[WorkerDisplayItem] = []
    for message in successful_messages:
        full_content = _tool_content_text(message.content)
        display_items.append(
            WorkerDisplayItem(
                name=str(message.name or "unknown_tool"),
                content=full_content,
                evidence_id=evidence_ids_by_tool_call.get(
                    str(message.tool_call_id or ""),
                    "",
                ),
                tool_call_id=str(message.tool_call_id or ""),
                arguments=tool_arguments.get(message.tool_call_id, {}),
                preview=_tool_message_preview(message.content, preview_chars),
                truncated=(
                    len(full_content.strip()) > preview_chars
                    or _tool_result_truncated(message)
                ),
            )
        )

    terminal_gap = str(final_state.get("terminal_gap") or "").strip()
    if terminal_gap:
        raise WorkerResponseError(terminal_gap)

    planner_message = final_state.get("planner_message")
    if planner_message is None:
        raise WorkerResponseError("Воркер завершился без finish_worker.")
    finish_calls = [
        call
        for call in planner_message.tool_calls
        if call.get("name") == _FINISH_WORKER_TOOL_NAME
    ]
    if len(finish_calls) == 1 and len(planner_message.tool_calls) == 1:
        payload = _worker_finish_payload(finish_calls[0])
    else:
        planner_contract_gap = (
            _message_text(planner_message).strip()
            or "Воркер должен завершиться ровно одним вызовом finish_worker."
        )
        raise WorkerResponseError(planner_contract_gap)
    gap = (
        latest_observation.gap
        if latest_observation is not None
        and latest_observation.status != "complete"
        else None
    )
    answer = payload.summary

    logger.info(
        "Worker final response (%d chars), display tools=%s",
        len(answer),
        [item.name for item in display_items],
    )
    return WorkerRunResult(
        answer=answer,
        display_items=display_items,
        cycle_history=list(final_state.get("cycle_history") or []),
        status="complete",
        gap=gap,
        facts=(
            list(latest_observation.facts)
            if latest_observation is not None
            else []
        ),
        accepted_tool_call_ids=accepted_tool_call_ids,
    )
