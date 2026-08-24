"""LangGraph runtime for the read-only chat agent.

Architecture:

    planner (native tool calling)
        ├─ data tool -> ToolNode -> observer (structured output) -> planner
        ├─ analyze -> analyzer LLM -> observer -> planner
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

from .observability import get_callback_handler, langfuse_trace_context

logger = logging.getLogger(__name__)

_VISUALIZATION_URL = re.compile(
    r"^/exports/(?:sql-lineage|s2t-graphs)/[A-Za-z0-9_.-]+\.html$"
)
_S2T_GRAPH_DATA_URL = re.compile(
    r"^/exports/s2t-graphs/[A-Za-z0-9_.-]+\.json$"
)
_OBSERVATION_SUMMARY_MAX_CHARS = 1200
_OBSERVATION_PROBLEM_MAX_CHARS = 1200
_OBSERVATION_FACT_MAX_CHARS = 300
_OBSERVATION_FACTS_MAX_COUNT = 8
_OBSERVATION_LIMITATIONS_MAX_COUNT = 4
_ANALYSIS_SUMMARY_MAX_CHARS = 1600
_ANALYSIS_FACT_MAX_CHARS = 400
_ANALYSIS_FACTS_MAX_COUNT = 10
_ANALYSIS_LIMITATIONS_MAX_COUNT = 4
_PLANNER_HANDOFF_MAX_CHARS = 12000
DEFAULT_TOOL_MESSAGE_PREVIEW_CHARS = 6000
_FINISH_WORKER_TOOL_NAME = "finish_worker"
_ANALYZE_TOOL_NAME = "analyze"
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

До первого успешного результата обязательно вызови подходящий worker tool.
Палитра worker никогда не пуста: если router не выбрал внешний data tool,
доступен внутренний `analyze_known_facts`. Передай ему готовый ответ, построенный
только по точным фактам из task, skills и schemas. Этот tool не подтверждает
новые данные, а создаёт обычный ToolMessage для проверки observer.
На каждом следующем шаге оцени task,
последний tool exchange и накопленную выжимку observer, затем либо вызови
следующий tool, либо заверши работу через finish_worker. Читай description и
схему выбранного tool, сохраняй смысл,
ограничения и точные значения task. Не придумывай факты и не повторяй успешный
вызов без новой причины. Не считай производный результат готовым входным фактом
и не переименовывай заданную операцию. Не конструируй отсутствующий объект
анализа только для заполнения обязательного аргумента tool: аргументы бери из
task или подтверждённых результатов предыдущих tools. Если обязательного входа
нет, этот tool не подходит. После ошибки исправь действие по фактическому
результату либо честно укажи ограничение. Полные результаты не копируй в answer:
внешний coordinator сам решит, что показывать отдельно.
После появления успешного полного результата worker делает доступным внутреннее
действие `analyze`. Если task требует интерпретации, вызови его с ID нужных
полных результатов из системного каталога. До получения источника `analyze`
недоступен; не повторяй data tool только ради анализа сохранённого результата.
Если observer сообщил, что исходные данные получены, но производный вывод ещё
не выполнен, обязательно вызови `analyze` для соответствующих tool_result_ids.
До успешного анализа не вызывай `finish_worker`.
Если task требует вернуть «только» конкретные поля, значения или элементы,
answer должен содержать ровно их: без вступления, заключения, подтверждения,
интерпретации и дополнительных пояснений.

Если structured observation содержит `goal_satisfied=false`, совокупных
подтверждённых фактов ещё недостаточно для task: не вызывай finish_worker и не
возвращай обычный финальный текст.
Прочитай отдельный блок «Что выполнено неправильно», исправь описанную проблему
и вызови подходящий worker tool. Завершай работу только когда observer
вернул `goal_satisfied=true` либо лимит data-tool шагов уже исчерпан.

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
Ты observer многошагового worker. Верни только structured output Observation по
переданной схеме. Не пиши Markdown, заголовки, пояснения до или после структуры
и не дублируй поля схемы обычным текстом.

Сопоставь исходную `user_request` с `prior_state`, аргументами текущего tool call
и его фактическим результатом. `candidate_analysis` — вывод отдельного analyzer,
а `analysis_sources` — выбранные полные результаты, доступные для display.
Критикуй вывод по этим источникам и не выполняй анализ заново.
Успешный статус сам по себе не подтверждает task. Проверяй точные сущности,
роли, значения, условия, ограничения и операцию.

Если task требует производного вывода, которого нет как явного факта в
результатах data tools, этот вывод должен быть получен через
`candidate_analysis`. При `candidate_analysis=null` не выполняй интерпретацию
сам и не ставь `goal_satisfied=true`: укажи в `problem`, что исходные данные
получены, но требуемый анализ ещё не выполнен. Простой перенос явно возвращённых
значений, строк или списков не является производным выводом и отдельного
`candidate_analysis` не требует.

Правила результата:
- `goal_satisfied=true`, только если совокупность подтверждённых фактов из
  `prior_state` и текущего результата закрывает всю task. Последний tool call не
  обязан один повторять уже подтверждённые части;
- при `goal_satisfied=false` `problem` — одна краткая консолидированная строка
  обо всех незакрытых требованиях task из prior_state и текущего результата.
  Не разбивай её на список, не повторяй одну причину через её следствия и
  обновляй строку после исправления части требований. При true верни null;
- `summary` — компактная накопительная выжимка подтверждённых фактов, нужных для
  ответа. Не превращай ошибки, ограничения и предположения в факты;
- `important_facts` содержит только подтверждённые факты для следующего шага,
  `limitations` — текущие ограничения и неоднозначности;
- если обязательный факт, значение, фильтр или операция отсутствует, явно назови
  это в `problem`. Одного неисправленного смыслового отличия достаточно для
  `goal_satisfied=false`.

Результат внутреннего `analyze_known_facts` не является новым внешним фактом.
Проверяй его `answer` только по фактам из task и выбранного системного контекста.

Служебный блок `saved_result` внутри ToolMessage содержит только result_ref,
схему и полноту временно сохранённых строк. Не считай его отдельным бизнес-
фактом и не включай result_ref в пользовательскую выжимку. Если
`truncated=true`, сохранённые строки не подтверждают вывод о полном наборе.

Объект анализа, который planner сам составил только ради обязательного аргумента
tool и который отсутствует в task и подтверждённых результатах, не подтверждает
исходную операцию. Отрази это в `problem`; если доступная палитра не умеет
прочитать требуемый сохранённый объект, запроси reroute.

Установи `reroute_required=true` только когда оставшуюся задачу невозможно
исправить новым вызовом ни одного `available_tools`. Если достаточно изменить
аргументы текущего tool, reroute не нужен. При true включи недостающую
инструментальную возможность в ту же строку `problem`; отдельного поля причины
нет.

{{PRIOR_STATE_RULE}}

Не выбирай следующий tool и не формулируй пользовательский ответ. Верни только
structured output Observation без Markdown.
""".strip()

_ANALYZER_PROMPT = """
Ты analyzer выбранных результатов read-only tools. Примени runtime skills к
полному содержимому `source_results` и выполни `analysis_instruction`. Верни
только structured output WorkerAnalysis.

Твоя роль — извлечь и интерпретировать факты, а не оценивать завершённость task:
- `summary` — самодостаточный результат анализа текущего tool exchange;
- `facts` — подтверждённые результатом факты и корректные выводы из них;
- `limitations` — ошибки, неполнота результата и то, что он не доказывает.

Сохраняй точные объекты, роли, выражения и условия. Не вызывай tools, не выбирай
следующий шаг, не выставляй goal_satisfied/reroute и не оформляй финальный ответ.
Не считай аргументы task найденными фактами. При ошибке или отсутствии нужных
данных зафиксируй ограничение вместо выдуманного анализа.
""".strip()

_ANALYZER_REPAIR_PROMPT = """
Предыдущий ответ analyzer не соответствует WorkerAnalysis. Data-tool уже
выполнен: не вызывай и не имитируй его повторно. Исправь только structured
output с полями summary, facts и limitations.

Ошибка: {validation_error}
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
полностью, честно отрази это в answer.
""".strip()

_UNSATISFIED_REPAIR_PROMPT = """
Последний structured observer вернул `goal_satisfied=false`. Предыдущая попытка
завершить worker запрещена: совокупность подтверждённых фактов ещё не закрывает
task. Исправь описанную в отдельном блоке «Что выполнено неправильно» проблему и
верни сейчас native call одного из доступных worker tools. Если не хватает
интерпретации уже сохранённого результата, вызови `analyze` с его ID; иначе
используй точный источник,
сущности, роли, поля, условия и операцию из исходной task. Не заменяй
одноимённые роли близкими, не проверяй заново уже подтверждённые входные факты
и не повторяй тот же семантически неверный вызов. Не вызывай finish_worker и не
отвечай обычным текстом.
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


class Observation(BaseModel):
    """Structured cumulative reflection over a multi-step worker run."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        description=(
            "Самодостаточная накопительная выжимка подтверждённых фактов из "
            "prior_state и текущего результата. Не добавляй неподтверждённые "
            "факты."
        )
    )
    goal_satisfied: bool = Field(
        description=(
            "Подтверждает ли совокупность prior_state и текущего результата "
            "выполнение всей исходной task без смысловых подмен."
        ),
    )
    problem: Optional[str] = Field(
        default=None,
        description=(
            "Одна краткая консолидированная строка обо всех незакрытых "
            "требованиях исходной task из текущего результата и prior_state. "
            "Не перечисляй одну причину и её следствия как разные проблемы. "
            "Null только при goal_satisfied=true."
        ),
    )
    has_error: bool = Field(
        default=False,
        description="Есть ли в результате ошибка выполнения или некорректные данные.",
    )
    important_facts: List[str] = Field(
        default_factory=list,
        description=(
            "Подтверждённые накопительные факты, важные для следующего шага "
            "planner."
        ),
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

    @field_validator(
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

    @field_validator("problem", mode="before")
    @classmethod
    def _normalize_problem(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        clean_value = str(value).strip()
        return clean_value or None

    @model_validator(mode="after")
    def _problem_matches_goal_status(self) -> "Observation":
        if self.goal_satisfied and self.problem is not None:
            raise ValueError(
                "problem must be null when goal_satisfied is true"
            )
        if not self.goal_satisfied and self.problem is None:
            raise ValueError(
                "problem must describe why goal_satisfied is false"
            )
        if self.goal_satisfied and self.reroute_required:
            raise ValueError(
                "reroute_required must be false when goal_satisfied is true"
            )
        return self


class WorkerAnalysis(BaseModel):
    """Structured interpretation produced before the observer critique."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    facts: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def _summary_must_not_be_blank(cls, value: str) -> str:
        clean_value = str(value or "").strip()
        if not clean_value:
            raise ValueError("analysis summary must not be blank")
        return clean_value

    @field_validator("facts", "limitations", mode="before")
    @classmethod
    def _clean_analysis_items(cls, values: Any) -> List[str]:
        if values is None:
            return []
        if not isinstance(values, (list, tuple)):
            values = [values]
        return [
            clean_item
            for item in values
            if (clean_item := str(item or "").strip())
        ]


class WorkerCycleTrace(BaseModel):
    """Bounded planner/tool/observer exchange exposed to the coordinator."""

    model_config = ConfigDict(extra="forbid")

    cycle: int = Field(ge=1)
    routing_attempt: int = Field(default=1, ge=1)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    analysis: Optional[WorkerAnalysis] = None
    observation: Observation


class AgentGraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    system_prompt: str
    planner_message: Optional[AIMessage]
    analysis: Optional[WorkerAnalysis]
    analysis_sources: List[Dict[str, Any]]
    observations: List[Observation]
    cycle_history: List[WorkerCycleTrace]
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
        problem=(
            _clip_text(observation.problem, _OBSERVATION_PROBLEM_MAX_CHARS)
            if observation.problem is not None
            else None
        ),
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
    )


def _compact_worker_analysis(analysis: WorkerAnalysis) -> WorkerAnalysis:
    return WorkerAnalysis(
        summary=_clip_text(analysis.summary, _ANALYSIS_SUMMARY_MAX_CHARS),
        facts=[
            _clip_text(item, _ANALYSIS_FACT_MAX_CHARS)
            for item in analysis.facts[:_ANALYSIS_FACTS_MAX_COUNT]
        ],
        limitations=[
            _clip_text(item, _ANALYSIS_FACT_MAX_CHARS)
            for item in analysis.limitations[:_ANALYSIS_LIMITATIONS_MAX_COUNT]
        ],
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
    cycle_history: List[WorkerCycleTrace] = Field(default_factory=list)
    goal_satisfied: bool = Field(default=True, exclude=True)
    problem: Optional[str] = Field(default=None, exclude=True)
    reroute_required: bool = Field(default=False, exclude=True)


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


def _analyze_tool_schema() -> Dict[str, Any]:
    """Return the native schema for the worker's internal analysis action."""
    return {
        "type": "function",
        "function": {
            "name": _ANALYZE_TOOL_NAME,
            "description": (
                "Проанализировать выбранные полные результаты уже выполненных "
                "data tools отдельным structured analyzer LLM."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_result_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": (
                            "Точные ID результатов из доступного системного "
                            "каталога без изменения."
                        ),
                    },
                    "instruction": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Что именно нужно установить или объяснить по "
                            "выбранным результатам."
                        ),
                    },
                },
                "required": ["tool_result_ids", "instruction"],
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
            observation_parts = [
                f"Выжимка observer для шага {index}:\n{observation.summary}"
            ]
            if observation.goal_satisfied is False:
                observation_parts.append(
                    "Статус выполнения: goal_satisfied=false. Исходная task "
                    "ещё не подтверждена; не завершай worker."
                )
                observation_parts.append(
                    "Что выполнено неправильно:\n"
                    + str(observation.problem or "")
                    + "\nИсправь проблему следующим worker-tool вызовом."
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


def _available_analysis_results(
    results: Mapping[str, ToolMessage],
) -> List[Dict[str, str]]:
    """List retained successful data results selectable by internal analyze."""
    return [
        {
            "tool_result_id": str(result_id),
            "tool_name": str(message.name or "unknown_tool"),
        }
        for result_id, message in results.items()
        if not _tool_message_has_error(message)
        and message.name not in {
            _ANALYZE_TOOL_NAME,
            _ANALYZE_KNOWN_FACTS_TOOL_NAME,
        }
    ]


def _tool_calls_by_id(
    messages: Sequence[BaseMessage],
) -> Dict[str, Dict[str, Any]]:
    calls: Dict[str, Dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            call_id = str(call.get("id") or "").strip()
            if call_id:
                calls[call_id] = {
                    "name": str(call.get("name") or "unknown_tool"),
                    "args": call.get("args", {}),
                }
    return calls


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
            problem=(
                "Observer не вернул валидный structured output, поэтому "
                "соответствие результата исходной task не подтверждено."
            ),
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
        problem=(
            "Observer не смог проверить соответствие tool-вызова исходной "
            "task; результат нельзя считать подтверждённым."
        ),
        important_facts=[],
        limitations=[f"Ошибка observer: {type(error).__name__}"],
    )


def build_agent_graph(
    model: Any,
    tools: Mapping[str, BaseTool] | Sequence[BaseTool],
    *,
    raw_tool_results: Optional[Dict[str, ToolMessage]] = None,
    tool_message_preview_chars: Optional[int] = None,
    worker_finish: bool = False,
):
    """Build planner -> tools/analyzer -> observer -> planner."""
    tool_list = _normalize_tools(tools)
    tool_names = tuple(tool.name for tool in tool_list)
    retained_tool_results = (
        raw_tool_results if raw_tool_results is not None else {}
    )
    analyze_available = worker_finish
    analyze_schema = _analyze_tool_schema()
    finish_schema = _finish_worker_tool_schema()
    planner_model = (
        model.bind_tools([*tool_list, analyze_schema, finish_schema])
        if worker_finish
        else (model.bind_tools(tool_list) if tool_list else model)
    )
    first_data_model = (
        model.bind_tools(tool_list)
        if worker_finish and tool_list
        else None
    )
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
    analyzer_model: Any = None

    def get_analyzer_model() -> Any:
        nonlocal analyzer_model
        if not analyze_available:
            raise RuntimeError("analyze вызван вне worker graph")
        if analyzer_model is None:
            analyzer_model = model.with_structured_output(WorkerAnalysis)
            if hasattr(analyzer_model, "with_retry"):
                analyzer_model = analyzer_model.with_retry(
                    retry_if_exception_type=(ValidationError,),
                    wait_exponential_jitter=False,
                    stop_after_attempt=2,
                )
        return analyzer_model
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
                and latest_observation.goal_satisfied
            )
        )
        if finish_only and worker_finish:
            selected_model = finish_model
        elif finish_only:
            selected_model = model
        elif state["tool_steps"] == 0 and first_tool_model is not None:
            selected_model = first_tool_model
        else:
            selected_model = planner_model
        analysis_result_catalog = _available_analysis_results(
            retained_tool_results
        )
        planner_tool_names = [*tool_names]
        if analyze_available and analysis_result_catalog:
            planner_tool_names.append(_ANALYZE_TOOL_NAME)
        planner_messages = _planner_messages(
            state,
            planner_tool_names,
            worker_finish=worker_finish,
        )
        if analyze_available and analysis_result_catalog:
            catalog_text = json.dumps(
                analysis_result_catalog,
                ensure_ascii=False,
                default=str,
            )
            first_message = planner_messages[0]
            planner_messages[0] = first_message.model_copy(
                update={
                    "content": (
                        f"{_message_content_text(first_message.content)}\n\n"
                        "Доступные для внутреннего `analyze` полные результаты "
                        "data tools:\n"
                        f"{catalog_text}\n"
                        "Передай нужные tool_result_ids без изменения."
                    )
                }
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

        semantic_retry_required = (
            worker_finish
            and not finish_only
            and latest_observation is not None
            and latest_observation.goal_satisfied is False
            and (
                not any(
                    call.get("name") in planner_tool_names
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

        analyze_calls = [
            call
            for call in planner_message.tool_calls
            if call.get("name") == _ANALYZE_TOOL_NAME
        ]
        if analyze_calls and len(planner_message.tool_calls) != 1:
            raise WorkerResponseError(
                "analyze должен вызываться отдельно от остальных tools"
            )

        return {
            "messages": [planner_message],
            "planner_message": None,
            "analysis": None,
            "analysis_sources": [],
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

    def analyzer(state: AgentGraphState) -> Dict[str, Any]:
        active_analyzer_model = get_analyzer_model()

        analyze_message = state["messages"][-1]
        if (
            not isinstance(analyze_message, AIMessage)
            or len(analyze_message.tool_calls) != 1
            or analyze_message.tool_calls[0].get("name") != _ANALYZE_TOOL_NAME
        ):
            raise RuntimeError("Analyzer вызван без одиночного analyze tool call")

        analyze_call = analyze_message.tool_calls[0]
        analyze_args = analyze_call.get("args") or {}
        requested_ids = list(
            dict.fromkeys(
                str(item or "").strip()
                for item in (analyze_args.get("tool_result_ids") or [])
                if str(item or "").strip()
            )
        )
        instruction = str(analyze_args.get("instruction") or "").strip()
        calls_by_id = _tool_calls_by_id(state["messages"][:-1])
        available_results = {
            item["tool_result_id"]: retained_tool_results[
                item["tool_result_id"]
            ]
            for item in _available_analysis_results(retained_tool_results)
        }
        missing_ids = [
            result_id
            for result_id in requested_ids
            if result_id not in available_results
        ]
        analysis_sources = [
            {
                "tool_result_id": result_id,
                "tool_call": calls_by_id.get(
                    result_id,
                    {
                        "name": str(
                            available_results[result_id].name
                            or "unknown_tool"
                        ),
                        "args": {},
                    },
                ),
                "tool_result": _tool_message_payload(
                    available_results[result_id]
                ),
            }
            for result_id in requested_ids
            if result_id in available_results
        ]

        analysis_error = None
        if not requested_ids:
            analysis_error = "analyze не получил ни одного tool_result_id."
        elif missing_ids:
            analysis_error = (
                "Полные результаты недоступны для анализа: "
                + ", ".join(missing_ids)
            )
        elif not instruction:
            analysis_error = "analyze не получил непустую instruction."

        payload = {
            "user_request": _last_user_query(state["messages"]),
            "analysis_instruction": instruction,
            "source_results": analysis_sources,
        }
        analyzer_messages: List[BaseMessage] = [
            SystemMessage(
                content="\n\n".join(
                    part
                    for part in (
                        state["system_prompt"].strip(),
                        _ANALYZER_PROMPT,
                    )
                    if part
                )
            ),
            HumanMessage(
                content=json.dumps(payload, ensure_ascii=False, default=str)
            ),
        ]

        def parse_analysis(result: Any) -> WorkerAnalysis:
            return (
                result
                if isinstance(result, WorkerAnalysis)
                else WorkerAnalysis.model_validate(result)
            )

        analyzer_failed = False
        if analysis_error is not None:
            analyzer_failed = True
            analysis = WorkerAnalysis(
                summary="Анализ не выполнен: нет выбранного полного результата.",
                limitations=[analysis_error],
            )
        else:
            try:
                analysis = parse_analysis(
                    active_analyzer_model.invoke(analyzer_messages)
                )
            except Exception as first_error:
                logger.warning(
                    "Structured analyzer output rejected; repeating analyzer "
                    "without repeating data tools: %s",
                    first_error,
                )
                repair_messages: List[BaseMessage] = [
                    *analyzer_messages,
                    HumanMessage(
                        content=_ANALYZER_REPAIR_PROMPT.replace(
                            "{validation_error}",
                            f"{type(first_error).__name__}: {first_error}",
                        )
                    ),
                ]
                try:
                    analysis = parse_analysis(
                        active_analyzer_model.invoke(repair_messages)
                    )
                except Exception as repair_error:
                    analyzer_failed = True
                    logger.exception(
                        "Structured analyzer repair failed; preserving raw "
                        "tool results for observer critique"
                    )
                    analysis = WorkerAnalysis(
                        summary=(
                            "Analyzer не вернул валидный structured результат."
                        ),
                        limitations=[
                            "Ошибка analyzer: "
                            f"{type(repair_error).__name__}"
                        ],
                    )

        analysis = _compact_worker_analysis(analysis)
        logger.info(
            "Analyzer result: %s",
            analysis.model_dump_json()[:3000],
        )
        analysis_message = ToolMessage(
            content=analysis.model_dump_json(),
            tool_call_id=str(analyze_call.get("id") or "analyze"),
            name=_ANALYZE_TOOL_NAME,
            status="error" if analyzer_failed else "success",
        )
        return {
            "messages": [analysis_message],
            "analysis": analysis,
            "analysis_sources": analysis_sources,
            "tool_steps": state["tool_steps"] + 1,
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
                ).answer
            elif not planner_message.tool_calls:
                candidate_answer = _message_text(planner_message)
            tool_call_message = AIMessage(content=candidate_answer)
            tool_results: List[ToolMessage] = []
        else:
            tool_call_message, tool_results = _latest_tool_exchange(
                state["messages"]
            )

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
            "candidate_answer": candidate_answer,
            "candidate_analysis": (
                state["analysis"].model_dump()
                if state.get("analysis") is not None
                else None
            ),
            "analysis_sources": list(state.get("analysis_sources") or []),
        }

        prior_state_rule = (
            "Поле prior_state содержит накопленные компактные observations "
            "предыдущих циклов. Оцени выполнение task по совокупности "
            "prior_state и текущего результата, а не по последнему tool call "
            "изолированно. Уже подтверждённые части task не должны повторно "
            "присутствовать в текущем результате. Верни обновлённую "
            "самодостаточную выжимку, сохранив только подтверждённые факты, "
            "ещё нужные для исходной task; прошлую problem не считай фактом "
            "и обновляй после исправления части требований."
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
            observation = (
                result
                if isinstance(result, Observation)
                else Observation.model_validate(result)
            )
            if any(
                _tool_message_has_error(message)
                for message in tool_results
            ):
                observation_payload = observation.model_dump()
                observation_payload["has_error"] = True
                return Observation.model_validate(observation_payload)
            return observation

        observation: Optional[Observation] = None
        observer_error: Optional[Exception] = None
        attempt_messages = observer_messages
        total_attempts = _OBSERVER_MAX_RETRIES + 1
        for attempt_index in range(total_attempts):
            try:
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
            logger.exception(
                "Structured observer repair failed after %s attempts; "
                "stopping worker without repeating data tools",
                total_attempts,
                exc_info=observer_error,
            )
            raise WorkerResponseError(
                "Observer не вернул валидный structured output после "
                f"{total_attempts} попыток; data tool повторно не вызван."
            ) from observer_error

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
                        _tool_message_payload(message)
                        for message in tool_results
                    ],
                    analysis=state.get("analysis"),
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

    def route_after_prepare(
        state: AgentGraphState,
    ) -> Literal["tools", "analyzer"]:
        message = state["messages"][-1]
        if (
            analyze_available
            and isinstance(message, AIMessage)
            and len(message.tool_calls) == 1
            and message.tool_calls[0].get("name") == _ANALYZE_TOOL_NAME
        ):
            return "analyzer"
        return "tools"

    def route_after_observer(
        state: AgentGraphState,
    ) -> Literal["planner", "finish"]:
        latest_observation = (
            (state.get("observations") or [])[-1]
            if state.get("observations")
            else None
        )
        if (
            worker_finish
            and latest_observation is not None
            and latest_observation.reroute_required
        ):
            return "finish"
        if worker_finish and not tool_list:
            return "finish"
        return "planner"

    graph = StateGraph(AgentGraphState)
    graph.add_node("planner", planner)
    graph.add_node("prepare_tool", prepare_tool_call)
    graph.add_node("tools", execute_tools)
    if analyze_available:
        graph.add_node("analyzer", analyzer)
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
    if analyze_available:
        graph.add_conditional_edges(
            "prepare_tool",
            route_after_prepare,
            {
                "tools": "tools",
                "analyzer": "analyzer",
            },
        )
    else:
        graph.add_edge("prepare_tool", "tools")
    graph.add_edge("tools", "observer")
    if analyze_available:
        graph.add_edge("analyzer", "observer")
    graph.add_conditional_edges(
        "observer",
        route_after_observer,
        {
            "planner": "planner",
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
        "analysis": None,
        "analysis_sources": [],
        "observations": [],
        "cycle_history": [],
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
            problem="Worker получил пустую task.",
        )

    bounded_steps = max(1, int(max_steps))
    preview_chars = max(1, int(tool_message_preview_chars))
    raw_tool_results: Dict[str, ToolMessage] = {}
    worker_tools = ensure_worker_tools(tools)
    graph = build_agent_graph(
        model,
        worker_tools,
        raw_tool_results=raw_tool_results,
        tool_message_preview_chars=preview_chars,
        worker_finish=True,
    )

    initial_state: AgentGraphState = {
        "messages": [HumanMessage(content=clean_task)],
        "system_prompt": system_prompt,
        "planner_message": None,
        "analysis": None,
        "analysis_sources": [],
        "observations": [],
        "cycle_history": [],
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
        reroute_problem = str(latest_observation.problem or "").strip()
        logger.info(
            "Worker graph finished with reroute request: problem=%s",
            reroute_problem,
        )
        return WorkerRunResult(
            answer=reroute_problem,
            display_items=[],
            cycle_history=list(final_state.get("cycle_history") or []),
            goal_satisfied=False,
            problem=reroute_problem,
            reroute_required=True,
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
        and message.name not in {
            _ANALYZE_TOOL_NAME,
            _ANALYZE_KNOWN_FACTS_TOOL_NAME,
        }
    ]
    goal_satisfied = bool(
        latest_observation is not None
        and latest_observation.goal_satisfied
    )
    problem = (
        latest_observation.problem
        if latest_observation is not None
        else "Worker завершился без structured observation результата."
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
        cycle_history=list(final_state.get("cycle_history") or []),
        goal_satisfied=goal_satisfied,
        problem=problem,
    )
