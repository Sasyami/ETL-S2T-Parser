"""LangGraph runtime for the read-only chat agent.

Architecture:

    planner (native tool calling)
        ├─ tool_calls -> ToolNode -> observer (structured output) -> planner
        ├─ worker finish_worker call -> END
        └─ legacy no tool_calls -> responder -> END

The legacy chat mode keeps raw ToolMessage content in the graph. The isolated
worker mode instead stores full tool results outside message history and puts a
single bounded text preview into each ToolMessage. The same planner completes a
worker with finish_worker(answer), without a separate responder call. A higher
level coordinator decides which complete results should be displayed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
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

from .observability import get_callback_handler, langfuse_trace_context

logger = logging.getLogger(__name__)

_VISUALIZATION_URL = re.compile(
    r"^/exports/(?:sql-lineage|s2t-graphs)/[A-Za-z0-9_.-]+\.html$"
)
_S2T_GRAPH_DATA_URL = re.compile(
    r"^/exports/s2t-graphs/[A-Za-z0-9_.-]+\.json$"
)
_OBSERVATION_SUMMARY_MAX_CHARS = 1200
_OBSERVATION_FACT_MAX_CHARS = 300
_OBSERVATION_FACTS_MAX_COUNT = 8
_OBSERVATION_MISMATCHES_MAX_COUNT = 8
_OBSERVATION_LIMITATIONS_MAX_COUNT = 4
_PLANNER_HANDOFF_MAX_CHARS = 12000
DEFAULT_TOOL_MESSAGE_PREVIEW_CHARS = 6000
_FINISH_WORKER_TOOL_NAME = "finish_worker"
_WORKER_PLANNER_PROMPT = """
Ты planner изолированного read-only worker. Решай переданную task только по
подтверждённым результатам доступных data tools: {{AVAILABLE_TOOLS}}.

До первого успешного результата обязательно вызови подходящий data tool. На
каждом следующем шаге оцени task, последний tool exchange и накопленную выжимку
observer, затем либо вызови следующий tool, либо заверши работу через
finish_worker. Читай description и схему выбранного tool, сохраняй смысл,
ограничения и точные значения task. Не придумывай факты и не повторяй успешный
вызов без новой причины. Не считай производный результат готовым входным фактом
и не переименовывай заданную операцию. После ошибки исправь действие по
фактическому результату либо честно укажи ограничение. Полные результаты не
копируй в answer: внешний coordinator сам решит, что показывать отдельно.
Если task требует вернуть «только» конкретные поля, значения или элементы,
answer должен содержать ровно их: без вступления, заключения, подтверждения,
интерпретации и дополнительных пояснений.

Если structured observation содержит `goal_satisfied=false`, task ещё не
выполнена: не вызывай finish_worker и не возвращай обычный финальный текст.
Прочитай отдельный блок «Что выполнено неправильно», исправь каждый указанный
пункт и вызови подходящий data tool. Завершай работу только когда observer
вернул `goal_satisfied=true` либо лимит data-tool шагов уже исчерпан.
""".strip()

_LEGACY_PLANNER_PROMPT = """
Ты planner read-only агента. Используй только доступные tools:
{{AVAILABLE_TOOLS}}. Читай description и схему выбранного tool. На каждом шаге
либо верни нужный native tool call, либо компактную выжимку подтверждённых
фактов для responder, если данных уже достаточно. Сохраняй смысл, ограничения
и точные значения запроса, не придумывай факты и не повторяй успешный вызов без
новой причины. После ошибки скорректируй действие по фактическому ToolMessage.
""".strip()

_OBSERVER_PROMPT = """
Ты observer многошагового агента. Сопоставь исходную `user_request`, фактические
аргументы последнего tool call и его результат. Проверь, выполнена ли именно
запрошенная операция над теми же сущностями, ролями, полями, условиями и
ограничениями. Успешное выполнение tool само по себе не подтверждает task.

Верни structured output по переданной схеме Observation. Поле `goal_satisfied`
равно true только когда результат подтверждает выполнение всей task. Если
результат ошибочен, частичен, неоднозначен или относится к другой сущности,
роли, колонке, условию либо операции, верни false и отдельно перечисли в
`mismatches` все конкретные отличия фактического действия от task. Не прячь эти
отличия только в `summary`, `limitations` или `important_facts`. При
`goal_satisfied=false` список `mismatches` не может быть пустым; при true он
должен быть пустым.

В `summary` дай компактную фактическую выжимку результата. Например, если task
просит `target_table`, а запрос группирует или возвращает `source_table`,
`goal_satisfied` обязан быть false, даже если tool успешно вернул строки. Не
выбирай следующий инструмент и не формулируй ответ пользователю. Не придумывай
отсутствующие данные.

Поле `available_tools` содержит только текущую палитру worker. Установи
`reroute_required=true` только если task нельзя исправить следующим вызовом ни
одного из этих tools и требуется другая инструментальная возможность. В
`reroute_reason` кратко опиши недостающую возможность, но не выбирай конкретный
tool: новый выбор выполнит отдельный router. Если достаточно исправить аргументы
или запрос уже доступного tool, оставь `reroute_required=false`. При
`goal_satisfied=true` reroute запрещён. Ошибка предыдущего вызова сама по себе
не требует reroute: если тот же tool можно вызвать с исправленными аргументами,
схемой или запросом, текущая палитра достаточна.

Проверяй все точные имена из task посимвольно. Отдельно сопоставь:
- источник данных из task с фактическим источником tool call;
- требуемые сущности и роли с выбранными, группируемыми и фильтруемыми полями;
- каждое точное значение из task с фактическим условием, в котором оно должно
  использоваться; отсутствие значения или фильтра назови отдельным mismatch;
- требуемую операцию с фактической операцией;
- ограничения и форму новых фактов, которые должен получить этот worker.
Используй переданный runtime-контекст и схему, чтобы отличать физический источник
данных от логического имени или значения колонки. Не переопределяй их смысл по
одной разговорной формулировке task. Для каждого ошибочного или лишнего условия
назови фактическое условие и требуемое условие отдельно.
`source_table`, `source_field`, `target_table` и `target_field` — разные роли и
никогда не подтверждают друг друга. Если task просит считать `source_table`, а
tool call считает `source_field`, обязательно назови это главным
несоответствием, добавь его в `mismatches` и верни `goal_satisfied=false`.
Перечисли все обнаруженные несоответствия, чтобы planner мог исправить их одним
следующим вызовом.

Если task передаёт уже подтверждённый факт как условие или контекст для нового
вычисления, не требуй повторно получать этот факт. Проверяй, что tool использует
его как заданное условие, а статус нового шага определяй по запрошенному новому
результату.

Если task просит полный направленный путь между двумя узлами, промежуточные узлы
в корректном пути являются обязательной частью результата, а не лишними
сущностями только потому, что пользователь не перечислил их заранее. Наличие
нескольких одинаковых строк пути и ограничение числа возвращённых строк не
делают один найденный полный путь неполным, если task просит сам путь, а не все
уникальные пути. Считай такой результат выполненным, когда хотя бы одна строка
содержит оба заданных конца, все узлы между ними по порядку и глубину пути.
{{PRIOR_STATE_RULE}}

Перед возвратом structured output выполни финальную посимвольную сверку всех
названных в task источников, полей, значений, фильтров и операций с фактическими
аргументами tool call. Идентификаторы с разным написанием не являются
синонимами. Если task требует `source_table`, а SELECT, COUNT, GROUP BY или
фильтр использует вместо него `source_field`, обязательно верни
`goal_satisfied=false` и отдельный mismatch вида «требовалось ...; фактически
выполнено ...». Одного такого отличия достаточно для false независимо от
успешного статуса tool и правдоподобия результата.
Не вызывай tools. Верни только structured output Observation.
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

_FIRST_TOOL_REPAIR_PROMPT = """
Ты ещё не получил ни одного результата data tool. Предыдущий обычный текст не выполняет task.
Верни сейчас ровно один native call одного из доступных data
tools, сохрани смысл и точные значения task. Не пиши ответ или намерение
словами.
""".strip()

_FINISH_ONLY_REPAIR_PROMPT = """
Лимит шагов исчерпан. Больше не вызывай data tools. По уже подтверждённым
результатам верни один native call finish_worker. Если задача завершена не
полностью, честно отрази это в answer.
""".strip()

_UNSATISFIED_REPAIR_PROMPT = """
Последний structured observer вернул `goal_satisfied=false`. Предыдущая попытка
завершить worker запрещена: фактический tool call не подтвердил task. Исправь
каждый пункт из отдельного блока «Что выполнено неправильно» и верни сейчас
native call одного из доступных data tools. Используй точный источник,
сущности, роли, поля, условия и операцию из исходной task. Не заменяй
одноимённые роли близкими, не проверяй заново уже подтверждённые входные факты
и не повторяй тот же семантически неверный вызов. Не вызывай finish_worker и не
отвечай обычным текстом.
""".strip()


class ChatHistoryMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class Observation(BaseModel):
    """Structured reflection over the latest tool execution result."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        description=(
            "Краткий фактический вывод из последнего результата инструмента. "
            "Не добавляй факты, которых нет в результате."
        )
    )
    goal_satisfied: bool = Field(
        description=(
            "Подтверждает ли фактический результат выполнение всей исходной "
            "task без смысловых подмен."
        ),
    )
    mismatches: List[str] = Field(
        default_factory=list,
        description=(
            "Все конкретные отличия фактического tool call и результата от "
            "исходной task. Пусто только при goal_satisfied=true."
        ),
    )
    has_error: bool = Field(
        default=False,
        description="Есть ли в результате ошибка выполнения или некорректные данные.",
    )
    important_facts: List[str] = Field(
        default_factory=list,
        description="Факты из результата, важные для следующего шага planner.",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description=(
            "Ограничения, неоднозначности и непроверенные предположения результата. "
            "Не выбирай следующий инструмент."
        ),
    )
    reroute_required: bool = Field(
        default=False,
        description=(
            "Нужна ли новая палитра tools, потому что текущими tools исправить "
            "несоответствие невозможно."
        ),
    )
    reroute_reason: Optional[str] = Field(
        default=None,
        description=(
            "Какая инструментальная возможность отсутствует в текущей палитре. "
            "Не выбирай конкретный tool."
        ),
    )

    @field_validator(
        "mismatches",
        "important_facts",
        "limitations",
        mode="before",
    )
    @classmethod
    def _remove_blank_list_items(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("observation list fields must be arrays")
        return [
            clean_item
            for item in value
            if (clean_item := str(item or "").strip())
        ]

    @model_validator(mode="after")
    def _mismatches_match_goal_status(self) -> "Observation":
        if self.goal_satisfied and self.mismatches:
            raise ValueError(
                "mismatches must be empty when goal_satisfied is true"
            )
        if not self.goal_satisfied and not self.mismatches:
            raise ValueError(
                "mismatches must describe why goal_satisfied is false"
            )
        if self.goal_satisfied and self.reroute_required:
            raise ValueError(
                "reroute_required must be false when goal_satisfied is true"
            )
        if self.reroute_required and not str(self.reroute_reason or "").strip():
            raise ValueError(
                "reroute_reason is required when reroute_required is true"
            )
        return self


class AgentGraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    system_prompt: str
    planner_message: Optional[AIMessage]
    observations: List[Observation]
    tool_steps: int
    max_steps: int


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
        summary=_clip_text(
            observation.summary,
            _OBSERVATION_SUMMARY_MAX_CHARS,
        ),
        goal_satisfied=observation.goal_satisfied,
        mismatches=[
            _clip_text(item, _OBSERVATION_FACT_MAX_CHARS)
            for item in observation.mismatches[
                :_OBSERVATION_MISMATCHES_MAX_COUNT
            ]
        ],
        has_error=observation.has_error,
        important_facts=[
            _clip_text(item, _OBSERVATION_FACT_MAX_CHARS)
            for item in observation.important_facts[
                :_OBSERVATION_FACTS_MAX_COUNT
            ]
        ],
        limitations=[
            _clip_text(item, _OBSERVATION_FACT_MAX_CHARS)
            for item in observation.limitations[
                :_OBSERVATION_LIMITATIONS_MAX_COUNT
            ]
        ],
        reroute_required=observation.reroute_required,
        reroute_reason=(
            _clip_text(observation.reroute_reason, _OBSERVATION_FACT_MAX_CHARS)
            if observation.reroute_reason
            else None
        ),
    )


class WorkerFinishPayload(BaseModel):
    """Strict arguments of the worker's native finish call."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)

    @field_validator("answer")
    @classmethod
    def _answer_must_not_be_blank(cls, value: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("answer must not be blank")
        return clean_value

class WorkerDisplayItem(BaseModel):
    """Full successful tool output retained outside the worker LLM context."""

    model_config = ConfigDict(extra="forbid")

    name: str
    content: str


class WorkerRunResult(BaseModel):
    """Internal graph result or top-level chat result with UI display data."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    display_items: List[WorkerDisplayItem] = Field(default_factory=list)
    goal_satisfied: bool = Field(default=True, exclude=True)
    mismatches: List[str] = Field(default_factory=list, exclude=True)
    reroute_required: bool = Field(default=False, exclude=True)
    reroute_reason: Optional[str] = Field(default=None, exclude=True)


class WorkerResponseError(RuntimeError):
    """Raised when the worker violates its finish contract."""


def _finish_worker_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _FINISH_WORKER_TOOL_NAME,
            "description": (
                "Завершить подзадачу worker и вернуть краткий точный ответ. "
                "Результаты tools для UI выбирает внешний coordinator."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Краткий итог по подтверждённым данным.",
                    },
                },
                "required": ["answer"],
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
        for index, observation in enumerate(
            state.get("observations") or [],
            start=1,
        ):
            observation_parts = [
                f"Выжимка observer для шага {index}:\n{observation.summary}"
            ]
            if observation.goal_satisfied is False:
                observation_parts.append(
                    "Статус выполнения: goal_satisfied=false. Исходная task "
                    "ещё не подтверждена; не завершай worker."
                )
                observation_parts.append(
                    "Что выполнено неправильно:\n- "
                    + "\n- ".join(observation.mismatches)
                    + "\nИсправь каждый пункт следующим data-tool вызовом."
                )
            else:
                observation_parts.append(
                    "Статус выполнения: goal_satisfied=true. Результат "
                    "подтверждает исходную task."
                )
            if observation.important_facts:
                observation_parts.append(
                    "Важные факты:\n- "
                    + "\n- ".join(observation.important_facts)
                )
            if observation.limitations:
                observation_parts.append(
                    "Ограничения и неоднозначности:\n- "
                    + "\n- ".join(observation.limitations)
                )
            if observation.has_error:
                observation_parts.append(
                    "Этот инструментальный шаг содержал ошибку. Реши, нужно ли "
                    "исправить аргументы, выбрать другой инструмент или завершить "
                    "работу с честным указанием ограничения."
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


def _fallback_observation(
    tool_call_message: AIMessage,
    tool_results: Sequence[ToolMessage],
    error: Exception,
) -> Observation:
    raw_output = getattr(error, "llm_output", None)
    if raw_output:
        return Observation(
            summary=str(raw_output),
            goal_satisfied=False,
            mismatches=[
                "Observer не вернул валидный structured output, поэтому "
                "соответствие результата исходной task не подтверждено."
            ],
            limitations=[f"Ошибка observer: {type(error).__name__}"],
        )

    names = [call.get("name", "unknown_tool") for call in tool_call_message.tool_calls]
    result_preview = "\n".join(
        f"{message.name or 'unknown_tool'}: {_message_content_text(message.content)[:1500]}"
        for message in tool_results
    )

    return Observation(
        summary=(
            "Observer не смог получить текстовую выжимку. "
            f"Выполнены инструменты: {', '.join(names)}. "
            f"Сырой результат:\n{result_preview}"
        ),
        has_error=True,
        goal_satisfied=False,
        mismatches=[
            "Observer не смог проверить соответствие tool-вызова исходной "
            "task; результат нельзя считать подтверждённым."
        ],
        important_facts=[],
        limitations=[f"Ошибка observer: {type(error).__name__}"],
    )


def _enforce_required_sql_roles(
    observation: Observation,
    user_request: str,
    tool_call_message: AIMessage,
    tool_results: Sequence[ToolMessage],
) -> Observation:
    """Reject a positive audit when run_sql omitted an explicitly named role."""
    if not observation.goal_satisfied:
        return observation
    sql_calls = [
        call
        for call in tool_call_message.tool_calls
        if call.get("name") == "run_sql"
    ]
    if not sql_calls:
        return observation

    request_text = str(user_request or "").lower()
    evidence_text = "\n".join(
        [
            *(
                json.dumps(
                    call.get("args") or {},
                    ensure_ascii=False,
                    default=str,
                )
                for call in sql_calls
            ),
            *(
                _message_content_text(message.content)
                for message in tool_results
            ),
        ]
    ).lower()
    role_names = (
        "source_table",
        "source_field",
        "target_table",
        "target_field",
    )
    missing_roles = [
        role
        for role in role_names
        if re.search(rf"(?<!\w){re.escape(role)}(?!\w)", request_text)
        and not re.search(
            rf"(?<!\w){re.escape(role)}(?!\w)",
            evidence_text,
        )
    ]
    missing_sources = [
        source_name
        for source_name in ("s2t_transformations",)
        if re.search(
            rf"(?<!\w){re.escape(source_name)}(?!\w)",
            request_text,
        )
        and not re.search(
            rf"(?<!\w){re.escape(source_name)}(?!\w)",
            evidence_text,
        )
    ]
    named_role_values = [
        (match.group(1).lower(), match.group(2).strip("`'\".()"))
        for match in re.finditer(
            r"(?<!\w)(source_table|target_table)\s*=\s*"
            r"[`'\"]?([^\s,;`'\"]+)",
            request_text,
        )
    ]
    missing_role_values = [
        (role, value)
        for role, value in named_role_values
        if value and value not in evidence_text
    ]
    if not missing_roles and not missing_sources and not missing_role_values:
        return observation

    return Observation(
        summary=observation.summary,
        goal_satisfied=False,
        mismatches=[
            *observation.mismatches,
            *(
                f"Требовалось поле `{role}`, но оно отсутствует и в "
                "аргументах run_sql, и в его результате."
                for role in missing_roles
            ),
            *(
                f"Требовался источник `{source_name}`, но run_sql не "
                "использовал его и не вернул из него результат."
                for source_name in missing_sources
            ),
            *(
                f"Требовалось условие `{role} = {value}`, но точное значение "
                "отсутствует и в аргументах run_sql, и в его результате."
                for role, value in missing_role_values
            ),
        ],
        has_error=observation.has_error,
        important_facts=observation.important_facts,
        limitations=observation.limitations,
        reroute_required=False,
        reroute_reason=None,
    )


def build_agent_graph(
    model: Any,
    tools: Mapping[str, BaseTool] | Sequence[BaseTool],
    *,
    raw_tool_results: Optional[Dict[str, ToolMessage]] = None,
    tool_message_preview_chars: Optional[int] = None,
    worker_finish: bool = False,
):
    """Build the planner -> tools -> observer -> planner graph."""
    tool_list = _normalize_tools(tools)
    tool_names = tuple(tool.name for tool in tool_list)
    finish_schema = _finish_worker_tool_schema()
    planner_model = model.bind_tools(
        [*tool_list, finish_schema] if worker_finish else tool_list
    )
    first_data_model = model.bind_tools(tool_list) if worker_finish else None
    first_tool_model = first_data_model
    if worker_finish and len(tool_list) == 1:
        try:
            first_tool_model = model.bind_tools(
                tool_list,
                tool_choice=tool_names[0],
            )
        except TypeError:
            first_tool_model = first_data_model
    finish_model = None
    if worker_finish:
        try:
            finish_model = model.bind_tools(
                [finish_schema],
                tool_choice=_FINISH_WORKER_TOOL_NAME,
            )
        except TypeError:
            finish_model = model.bind_tools([finish_schema])
    observer_model = model.with_structured_output(Observation)
    tool_node = ToolNode(tool_list, handle_tool_errors=True)

    def invoke_with_fallback(
        primary_model: Any,
        messages: Sequence[BaseMessage],
        *,
        fallback_model: Optional[Any] = None,
    ) -> Any:
        try:
            return primary_model.invoke(messages)
        except Exception:
            if fallback_model is None or fallback_model is primary_model:
                raise
            logger.warning(
                "Forced tool-choice LLM call failed; retrying with the regular "
                "planner tool palette",
                exc_info=True,
            )
            return fallback_model.invoke(messages)

    def planner(state: AgentGraphState) -> Dict[str, Any]:
        limit_reached = state["tool_steps"] >= state["max_steps"]
        finish_only = limit_reached
        if finish_only and worker_finish:
            selected_model = finish_model
        elif finish_only:
            selected_model = model
        elif state["tool_steps"] == 0 and first_tool_model is not None:
            selected_model = first_tool_model
        else:
            selected_model = planner_model
        planner_messages = _planner_messages(
            state,
            tool_names,
            worker_finish=worker_finish,
        )

        selected_fallback = (
            first_data_model
            if selected_model is first_tool_model
            and first_tool_model is not first_data_model
            else (
                planner_model
                if worker_finish and selected_model is finish_model
                else None
            )
        )
        try:
            reply = invoke_with_fallback(
                selected_model,
                planner_messages,
                fallback_model=selected_fallback,
            )
        except Exception as exc:
            logger.exception("LLM error in planner")
            # A plain message routes to the legacy responder or becomes the
            # worker's honest final text without another LLM call.
            reply = AIMessage(content=f"Planner error: {type(exc).__name__}")

        if not isinstance(reply, AIMessage):
            reply = AIMessage(content=_message_content_text(reply))

        if (
            worker_finish
            and state["tool_steps"] == 0
            and not any(
                call.get("name") in tool_names
                for call in reply.tool_calls
            )
        ):
            logger.warning(
                "Worker planner omitted the required first data tool call; repairing"
            )
            repair_messages = [
                *planner_messages,
                HumanMessage(
                    content=_FIRST_TOOL_REPAIR_PROMPT
                ),
            ]
            repaired_reply = invoke_with_fallback(
                selected_model,
                repair_messages,
                fallback_model=(
                    first_data_model
                    if selected_model is first_tool_model
                    and first_tool_model is not first_data_model
                    else None
                ),
            )
            if not isinstance(repaired_reply, AIMessage):
                repaired_reply = AIMessage(
                    content=_message_content_text(repaired_reply)
                )
            if not any(
                call.get("name") in tool_names
                for call in repaired_reply.tool_calls
            ):
                reroute_reason = (
                    "Planner дважды не сформировал обязательный первый "
                    "data-tool вызов для текущей task."
                )
                reroute_observation = Observation(
                    summary=reroute_reason,
                    goal_satisfied=False,
                    mismatches=[
                        "До получения данных planner не вызвал ни один "
                        "доступный data tool даже после repair-вызова."
                    ],
                    has_error=True,
                    limitations=[
                        "Текущий запуск worker завершён до выполнения data tool."
                    ],
                    reroute_required=True,
                    reroute_reason=reroute_reason,
                )
                logger.warning(
                    "Worker requests reroute after missing the first data "
                    "tool twice: tools=%s",
                    list(tool_names),
                )
                return {
                    "planner_message": AIMessage(content=""),
                    "observations": [reroute_observation],
                }
            reply = repaired_reply

        latest_observation = (
            (state.get("observations") or [])[-1]
            if state.get("observations")
            else None
        )
        semantic_retry_required = (
            worker_finish
            and not finish_only
            and latest_observation is not None
            and latest_observation.goal_satisfied is False
            and (
                not any(
                    call.get("name") in tool_names
                    for call in reply.tool_calls
                )
                or any(
                    call.get("name") == _FINISH_WORKER_TOOL_NAME
                    for call in reply.tool_calls
                )
            )
        )
        if semantic_retry_required:
            logger.warning(
                "Worker tried to finish after observer marked task "
                "goal_satisfied=false; requesting a corrected data tool call"
            )
            retry_model = (
                first_tool_model
                if len(tool_list) == 1 and first_tool_model is not None
                else first_data_model
            )
            repaired_reply = invoke_with_fallback(
                retry_model,
                [
                    *planner_messages,
                    HumanMessage(content=_UNSATISFIED_REPAIR_PROMPT),
                ],
                fallback_model=(
                    first_data_model
                    if retry_model is first_tool_model
                    and first_tool_model is not first_data_model
                    else None
                ),
            )
            if not isinstance(repaired_reply, AIMessage):
                repaired_reply = AIMessage(
                    content=_message_content_text(repaired_reply)
                )
            if (
                not any(
                    call.get("name") in tool_names
                    for call in repaired_reply.tool_calls
                )
                or any(
                    call.get("name") == _FINISH_WORKER_TOOL_NAME
                    for call in repaired_reply.tool_calls
                )
            ):
                reroute_reason = str(
                    latest_observation.reroute_reason
                    or (
                        "Текущая палитра tools не позволила planner "
                        "сформировать исправленный data-tool вызов после "
                        "замечаний observer."
                    )
                ).strip()
                reroute_observation = latest_observation.model_copy(
                    update={
                        "reroute_required": True,
                        "reroute_reason": reroute_reason,
                    }
                )
                logger.warning(
                    "Worker requests reroute after failed semantic repair: "
                    "tools=%s mismatches=%s",
                    list(tool_names),
                    latest_observation.mismatches,
                )
                return {
                    "planner_message": AIMessage(content=""),
                    "observations": [reroute_observation],
                }
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
                raise WorkerResponseError(
                    "Worker после повторного LLM-вызова не завершил задачу."
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

            if raw_tool_results is not None:
                raw_tool_results[message.tool_call_id] = message

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
                    _tool_message_payload(message)
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
        tool_call_message, tool_results = _latest_tool_exchange(state["messages"])

        payload = {
            "user_request": _last_user_query(state["messages"]),
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
            "tool_calls": tool_call_message.tool_calls,
            "tool_results": [
                _tool_message_payload(message) for message in tool_results
            ],
        }

        prior_state_rule = (
            "Поле prior_state содержит прошлую компактную выжимку: верни "
            "обновлённую самодостаточную выжимку, сохранив из неё только "
            "факты, ещё нужные для исходной task."
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

        try:
            result = observer_model.invoke(observer_messages)
            observation = (
                result
                if isinstance(result, Observation)
                else Observation.model_validate(result)
            )
            tool_has_error = any(
                _tool_message_has_error(message)
                for message in tool_results
            )
            if tool_has_error:
                observation_payload = observation.model_dump()
                observation_payload["has_error"] = True
                observation = Observation.model_validate(observation_payload)
            observation = _enforce_required_sql_roles(
                observation,
                payload["user_request"],
                tool_call_message,
                tool_results,
            )
        except Exception as exc:
            logger.exception("Structured observer failed")
            observation = _fallback_observation(
                tool_call_message,
                tool_results,
                exc,
            )

        observation = _compact_observation(observation)
        logger.info(
            "Observer result: %s",
            observation.model_dump_json()[:2000],
        )
        return {
            "observations": (
                [observation]
                if worker_finish
                else [
                    *(state.get("observations") or []),
                    observation,
                ]
            )
        }

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
    ) -> Literal["prepare_tool", "responder", "finish"]:
        planner_message = state.get("planner_message")
        if worker_finish:
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
            "responder": "responder",
            "finish": END,
        },
    )
    graph.add_edge("prepare_tool", "tools")
    graph.add_edge("tools", "observer")
    graph.add_edge("observer", "planner")
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
        "tool_steps": 0,
        "max_steps": bounded_steps,
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
        return WorkerRunResult(
            answer="Подзадача воркера не должна быть пустой.",
            display_items=[],
            goal_satisfied=False,
            mismatches=["Worker получил пустую task."],
        )

    bounded_steps = max(1, int(max_steps))
    preview_chars = max(1, int(tool_message_preview_chars))
    raw_tool_results: Dict[str, ToolMessage] = {}
    graph = build_agent_graph(
        model,
        tools,
        raw_tool_results=raw_tool_results,
        tool_message_preview_chars=preview_chars,
        worker_finish=True,
    )

    initial_state: AgentGraphState = {
        "messages": [HumanMessage(content=clean_task)],
        "system_prompt": system_prompt,
        "planner_message": None,
        "observations": [],
        "tool_steps": 0,
        "max_steps": bounded_steps,
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
    if latest_observation is not None and latest_observation.reroute_required:
        reroute_reason = str(
            latest_observation.reroute_reason
            or "Текущей палитры tools недостаточно для выполнения task."
        ).strip()
        logger.info(
            "Worker graph finished with reroute request: reason=%s",
            reroute_reason,
        )
        return WorkerRunResult(
            answer=reroute_reason,
            display_items=[],
            goal_satisfied=False,
            mismatches=list(latest_observation.mismatches),
            reroute_required=True,
            reroute_reason=reroute_reason,
        )

    planner_message = final_state.get("planner_message")
    if planner_message is None:
        raise WorkerResponseError(
            "Воркер завершился без finish_worker"
        )
    finish_calls = [
        call
        for call in planner_message.tool_calls
        if call.get("name") == _FINISH_WORKER_TOOL_NAME
    ]
    if len(finish_calls) == 1 and len(planner_message.tool_calls) == 1:
        payload = _worker_finish_payload(finish_calls[0])
    elif not planner_message.tool_calls and _message_text(planner_message):
        payload = WorkerFinishPayload(
            answer=_message_text(planner_message),
        )
        logger.info(
            "Worker planner returned plain final text without another LLM call"
        )
    else:
        raise WorkerResponseError(
            "Воркер должен завершиться ровно одним вызовом finish_worker"
        )
    successful_messages = [
        message
        for message in raw_tool_results.values()
        if not _tool_message_has_error(message)
    ]
    goal_satisfied = bool(
        latest_observation is not None
        and latest_observation.goal_satisfied
    )
    mismatches = (
        list(latest_observation.mismatches)
        if latest_observation is not None
        else ["Worker завершился без structured observation результата."]
    )
    answer = payload.answer

    display_items = [
        WorkerDisplayItem(
            name=str(message.name or "unknown_tool"),
            content=_tool_content_text(message.content),
        )
        for message in successful_messages
    ]
    logger.info(
        "Worker final response (%d chars), display tools=%s",
        len(answer),
        [item.name for item in display_items],
    )
    return WorkerRunResult(
        answer=answer,
        display_items=display_items,
        goal_satisfied=goal_satisfied,
        mismatches=mismatches,
    )
