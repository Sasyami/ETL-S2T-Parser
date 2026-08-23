"""LLM-driven workers with separate upstream and downstream coordination."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .agent import chat_model
from .observability import get_callback_handler, langfuse_trace_context
from .run_metrics import (
    get_run_metrics_callback,
    record_coordinator_plan,
    record_upstream_answer,
    record_upstream_evidence,
)
from .worker import discard_worker_result_refs, worker_chat
from .tools.saved_results import saved_result_store_scope

logger = logging.getLogger(__name__)

COORDINATOR_MAX_WORKERS = 8
COORDINATOR_CONTEXT_MAX_CHARS = 4000
_PLAN_TOOL_NAME = "submit_worker_plan"
_DISPATCH_TOOL_NAME = "dispatch_worker"
_UPSTREAM_EVIDENCE_TOOL_NAME = "submit_upstream_evidence"
_UPSTREAM_ANSWER_TOOL_NAME = "finish_upstream_answer"


class WorkerPlanStep(BaseModel):
    """One semantic worker step selected by the planner LLM."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(
        min_length=1,
        description=(
            "Одна пользовательская проверка или один самостоятельно "
            "проверяемый результат. Несколько чтений и атрибутов остаются "
            "одним шагом, если все их входы уже известны. Отдельный шаг нужен "
            "только для операции с аргументом из предыдущего результата."
        ),
    )
    presentation: Literal["answer_only", "full_results"]

    @field_validator("goal")
    @classmethod
    def _strip_goal(cls, value: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("goal must not be blank")
        return clean_value


class WorkerPlan(BaseModel):
    """Planner-selected sequence of worker steps."""

    model_config = ConfigDict(extra="forbid")

    steps: List[WorkerPlanStep] = Field(
        min_length=1,
        max_length=COORDINATOR_MAX_WORKERS,
        description=(
            "По умолчанию один шаг на всю пользовательскую проверку. "
            "Несколько шагов допустимы только для истинной зависимости, когда "
            "аргумент следующего вызова отсутствует во входе и должен быть найден."
        ),
    )


class WorkerDispatch(BaseModel):
    """Self-contained task sent to one isolated worker."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(
        min_length=1,
        description=(
            "Самодостаточная задача ровно с одной операцией текущего шага; "
            "не включает цели или требуемые результаты соседних шагов."
        ),
    )

    @field_validator("task")
    @classmethod
    def _strip_task(cls, value: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("task must not be blank")
        return clean_value


class UpstreamEvidence(BaseModel):
    """Verified evidence assembled from completed worker runs."""

    model_config = ConfigDict(extra="forbid")

    confirmed_facts: List[str] = Field(
        default_factory=list,
        description=(
            "Самодостаточные подтверждённые факты с точными объектами, "
            "ролями и значениями; без пользовательского оформления."
        ),
    )
    unresolved_requirements: List[str] = Field(
        default_factory=list,
        description=(
            "Все требования исходной task, которые не подтверждены или "
            "противоречат друг другу."
        ),
    )

    @field_validator("confirmed_facts", "unresolved_requirements")
    @classmethod
    def _clean_items(cls, values: List[str]) -> List[str]:
        result: List[str] = []
        for value in values:
            clean_value = str(value or "").strip()
            if not clean_value:
                raise ValueError("upstream evidence must not contain blanks")
            if clean_value not in result:
                result.append(clean_value)
        return result


class UpstreamAnswer(BaseModel):
    """Final user-facing answer returned along the upstream path."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)

    @field_validator("answer", mode="before")
    @classmethod
    def _serialize_structured_answer(cls, value: Any) -> Any:
        if isinstance(value, (Mapping, list, tuple, int, float, bool)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return value

    @field_validator("answer")
    @classmethod
    def _strip_answer(cls, value: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("answer must not be blank")
        return clean_value


class CoordinatorAnswer(BaseModel):
    """Coordinator output consumed by the top-level supervisor."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    display_refs: List[str] = Field(default_factory=list)


class CoordinatorResultRef(TypedDict):
    result_key: str
    ref: str
    name: str


class CoordinatorWorkerRun(TypedDict):
    step: int
    goal: str
    task: str
    answer: str
    cycle_history: List[Dict[str, Any]]
    result_refs: List[CoordinatorResultRef]
    saved_results: List[Dict[str, Any]]
    goal_satisfied: bool
    mismatches: List[str]


class CoordinatorGraphState(TypedDict):
    task: str
    context: str
    plan: List[Dict[str, str]]
    next_step: int
    pending_task: Optional[str]
    worker_runs: List[CoordinatorWorkerRun]
    upstream_result: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    selected_display_refs: List[str]


class CoordinatorResponseError(RuntimeError):
    """Raised when an LLM response violates a structural coordinator contract."""


_DOWNSTREAM_PLAN_PROMPT = f"""
Ты downstream planner coordinator. Проведи задачу сверху вниз: раздели её на упорядоченный
план для изолированных generic workers и верни ровно один native call
`{_PLAN_TOOL_NAME}` со списком `steps` длиной от 1 до
{COORDINATOR_MAX_WORKERS}. Каждый step содержит только `goal` и `presentation`.

Правила декомпозиции:
- по умолчанию верни ровно один step, содержащий всю пользовательскую проверку;
- создавай несколько steps только при истинной зависимости: конкретный аргумент
  следующей операции отсутствует в исходной task, и без результата предыдущего
  step следующий вызов данных сформировать невозможно;
- один step описывает одну пользовательскую проверку или один самостоятельно
  проверяемый результат;
- несколько запрошенных атрибутов одного найденного объекта или записи являются
  одним результатом и остаются в одном step; не разделяй их получение,
  перечисление или прямое объяснение, если новой операции с данными не требуется;
- одна проверка, сравнение или объяснение остаётся одним step, даже если требует
  нескольких чтений из явно названных источников или нескольких tools, когда
  все идентификаторы и фильтры уже заданы. Не разделяй получение опорных фактов,
  их сравнение и прямой вывод на отдельные steps;
- несколько источников доказательств для одного итогового вывода остаются одним
  step, даже если task вводит источник словом «отдельно» или «дополнительно»;
- один и тот же итоговый вывод, сравнение или требуемое поле ответа не может
  повторяться в goals разных steps;
- разные источники, атрибуты, проверки, части анализа и разделы итогового ответа
  сами по себе не являются зависимостью и не создают отдельные steps;
- не создавай отдельные steps для получения опорных фактов, оформления,
  пересказа или повторного показа тех же данных;
- покрой каждым запрошенным фактом, источником, ограничением и элементом формы
  ответа ровно один goal. Опциональный дополнительный источник включай в ту же
  проверку, а не превращай в блокирующий предварительный step;
- сохраняй порядок, точные сущности, идентификаторы, значения, условия,
  ограничения и запреты исходной задачи без переименования;
- используй `context` только для однозначного разрешения ссылок и общих
  ограничений. Если данных недостаточно, не угадывай и не достраивай предметную
  схему.

`presentation` — только технический режим передачи результата и не влияет на
декомпозицию. Не выбирай tools и skills, не выполняй задачу и не формулируй
ответ пользователю.
""".strip()

_DOWNSTREAM_PLAN_REPAIR_PROMPT = f"""
Предыдущий native call `{_PLAN_TOOL_NAME}` не соответствует технической схеме.
Верни исправленный native call ровно один раз. Массив `steps` должен содержать
от 1 до {COORDINATOR_MAX_WORKERS} элементов; каждый элемент должен иметь только
непустой `goal` и `presentation` со значением `answer_only` или `full_results`.
Исправь только формат native call. По умолчанию используй один step; несколько
допустимы лишь для аргумента, который должен быть найден предыдущим step. Не
отбрасывай требования, не добавляй новые факты и не вызывай другие tools.
""".strip()

_DOWNSTREAM_DISPATCH_PROMPT = f"""
Ты downstream dispatcher coordinator. Материализуй `current_step` в одну
самодостаточную задачу для изолированного generic worker.

Верни ровно один native call `{_DISPATCH_TOOL_NAME}` с полем `task`. Worker
получит только эту строку.

Главные правила:
- `current_step.goal` — единственная операция текущего worker. Не заменяй её
  операцией прошлого или соседнего step и не возвращай worker к уже завершённой
  проверке;
- перенеси все относящиеся к current_step источники, сущности, роли, фильтры,
  область данных, запреты и форму ответа. `original_task` используй только для
  восстановления этих ограничений, не добавляя цели соседних steps;
- `completed_workers` используй только если аргумент current_step зависит от их
  результата. Не проси повторно получать или проверять завершённые факты;
- подтверждёнными считай только факты из финального `cycle_history`, где
  observation имеет `goal_satisfied=true`. Точные идентификаторы копируй
  посимвольно, сохраняя их роль; неоднозначному значению роль не назначай;
- если текущей операции нужен полный `saved_results` с `truncated=false`, включи
  его точный result_ref. При `truncated=true` preview не доказывает полный набор;
- неразрешённую ссылку замени однозначным названием из `context`. Если его нет,
  явно сохрани необходимость уточнения, не выбирая объект самостоятельно;
- не добавляй tools, skills, план, служебные поля, рассуждения, показ или экспорт.

Перед native call сверь task с `current_step.goal`: каждый требуемый текущим
шагом факт и идентификатор присутствует, завершённая операция не повторена, а
цели предыдущего и следующего steps отсутствуют.
""".strip()

_DOWNSTREAM_DISPATCH_REPAIR_PROMPT = f"""
Предыдущий native call `{_DISPATCH_TOOL_NAME}` не соответствует технической
схеме. Верни исправленный native call ровно один раз с единственным непустым
строковым полем `task`. Сохрани текущую операцию, точные идентификаторы,
источники, условия и ограничения из входного payload. Не добавляй соседние
шаги, tools, служебные поля или новые факты.
""".strip()

_UPSTREAM_EVIDENCE_PROMPT = f"""
Ты upstream evidence coordinator. Проведи результаты worker снизу вверх:
проверь и объедини их,
но не формулируй пользовательский ответ и не занимайся его оформлением.

Верни ровно один native call `{_UPSTREAM_EVIDENCE_TOOL_NAME}`:
- `confirmed_facts` — самодостаточные подтверждённые факты, необходимые
  исходной task;
- `unresolved_requirements` — все требования task, которые не подтверждены,
  противоречат друг другу или потеряны.

Каждый worker result содержит `goal`, `task`, краткий `answer`, ordered
`cycle_history`, `goal_satisfied` и `mismatches`. Подтверждённым считай только
факт из шага с `goal_satisfied=true`, который поддержан его финальной
structured observation и фактическим tool preview. Ошибочный промежуточный
вызов, намерение worker и неподтверждённый текст answer фактами не являются.

Используй `original_task` только для точных названий объектов, ролей,
идентификаторов и входных условий. Не превращай утверждение из task в найденный
факт. В каждом confirmed fact дословно сохрани объект, роль, значение, порядок
и связь между ними; значение без субъекта или стороны сравнения не является
самодостаточным. Не смешивай evidence разных объектов. При конфликте ничего не
угадывай: перенеси требование в unresolved_requirements.

Проверь покрытие всей original_task. Если worker имеет goal_satisfied=false,
перенеси все его актуальные mismatches в unresolved_requirements и не продолжай
зависимое вычисление мысленно. Не добавляй result refs, UI-решения, формат
ответа, вступление или вывод для пользователя.
""".strip()

_UPSTREAM_EVIDENCE_REPAIR_PROMPT = f"""
Предыдущий native call `{_UPSTREAM_EVIDENCE_TOOL_NAME}` не соответствует технической
схеме. Верни его ровно один раз с двумя массивами непустых строк:
`confirmed_facts` и `unresolved_requirements`. Исправь только структуру, не
добавляй факты и не формулируй пользовательский ответ.
""".strip()

_UPSTREAM_ANSWER_PROMPT = f"""
Ты upstream answer coordinator. Заверши обратный путь результатов к supervisor:
сформируй окончательный ответ только
по `original_task`, устойчивому `context` и проверенному `upstream_evidence`.
Не обращайся к workers и не переоценивай доказательства.

Верни ровно один native call `{_UPSTREAM_ANSWER_TOOL_NAME}` с единственным строковым
полем `answer`. Используй только `confirmed_facts`; каждое
`unresolved_requirements` кратко отрази как неподтверждённое требование, не
додумывая ответ.

Сохрани точные идентификаторы, роли, значения, связи и порядок. Имена и входные
условия можно дословно брать из original_task для подписывания подтверждённых
фактов, но нельзя считать их результатами проверки. Соблюдай запрошенный формат.
Если пользователь потребовал «только» конкретные элементы, не добавляй
вступление, заключение или описание внутренних действий. Если задан буквальный
шаблон вида `имя=<значение>`, сохрани имя и знак `=` дословно. JSON, список,
число или объект сериализуй в строку поля `answer`.

Не упоминай upstream/downstream coordinator, workers, tools, observations,
result refs и внутреннюю схему.
""".strip()

_UPSTREAM_ANSWER_REPAIR_PROMPT = f"""
Предыдущий native call `{_UPSTREAM_ANSWER_TOOL_NAME}` не соответствует технической
схеме. Верни его ровно один раз с единственным непустым строковым полем
`answer`. Исправь только транспортный формат, не добавляя и не удаляя факты.
""".strip()


def _plan_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _PLAN_TOOL_NAME,
            "description": (
                "Зафиксировать план отдельных операций получения или проверки "
                "данных; формат выдачи хранится в presentation."
            ),
            "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": COORDINATOR_MAX_WORKERS,
                    "description": (
                        "По умолчанию один шаг на всю проверку; несколько "
                        "только когда аргумент следующего вызова отсутствует "
                        "во входе и должен быть найден предыдущим шагом"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                            "description": (
                                    "Одна пользовательская проверка или один "
                                    "самостоятельно проверяемый результат с "
                                    "точными сущностями и условиями. Несколько "
                                    "чтений остаются одним шагом, если все "
                                    "входы известны; отдельный шаг нужен только "
                                    "для аргумента из предыдущего результата"
                                ),
                            },
                            "presentation": {
                                "type": "string",
                                "enum": ["answer_only", "full_results"],
                                "description": (
                                    "Технический режим передачи результата: "
                                    "answer_only — текстовый ответ; full_results "
                                    "— ответ вместе с полным результатом"
                                ),
                            },
                        },
                        "required": ["goal", "presentation"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["steps"],
            "additionalProperties": False,
            },
        },
    }


def _dispatch_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _DISPATCH_TOOL_NAME,
            "description": "Передать одну самодостаточную задачу worker.",
            "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Самодостаточная пользовательская задача ровно с одной "
                        "операцией текущего step, без целей соседних steps и "
                        "служебной разметки coordinator"
                    ),
                }
            },
            "required": ["task"],
            "additionalProperties": False,
            },
        },
    }


def _upstream_evidence_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _UPSTREAM_EVIDENCE_TOOL_NAME,
            "description": (
                "Зафиксировать проверенные факты workers отдельно от "
                "пользовательского представления."
            ),
            "parameters": {
            "type": "object",
            "properties": {
                "confirmed_facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Самодостаточные подтверждённые факты с точными "
                        "объектами, ролями, связями и значениями"
                    ),
                },
                "unresolved_requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Все требования исходной task, которые не удалось "
                        "подтвердить без догадки"
                    ),
                },
            },
            "required": ["confirmed_facts", "unresolved_requirements"],
            "additionalProperties": False,
            },
        },
    }


def _upstream_answer_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _UPSTREAM_ANSWER_TOOL_NAME,
            "description": (
                "Оформить проверенные upstream-факты в окончательный ответ."
            ),
            "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": (
                        "Пользовательский ответ только из подтверждённых "
                        "upstream-фактов и явно указанных ограничений"
                    ),
                },
            },
            "required": ["answer"],
            "additionalProperties": False,
            },
        },
    }


def _native_payload(
    message: Any,
    tool_name: str,
    payload_model: type[BaseModel],
) -> BaseModel:
    if not isinstance(message, AIMessage):
        raise CoordinatorResponseError(
            f"Coordinator ожидал AIMessage с native call {tool_name}."
        )
    matching_calls = [
        call for call in message.tool_calls if call.get("name") == tool_name
    ]
    if len(matching_calls) != 1:
        raise CoordinatorResponseError(
            f"Coordinator должен вернуть ровно один native call {tool_name}."
        )
    try:
        return payload_model.model_validate(matching_calls[0].get("args") or {})
    except (TypeError, ValueError, ValidationError) as exc:
        raise CoordinatorResponseError(
            f"Coordinator вернул невалидную структуру {tool_name}."
        ) from exc


def build_coordinator_graph(
    model: Any,
    *,
    callbacks: Optional[Sequence[Any]] = None,
    collected_result_refs: Optional[List[str]] = None,
):
    """Build downstream task flow and upstream result flow around workers."""
    callback_list = list(callbacks or [])
    model_config = {"callbacks": callback_list} if callback_list else None
    def bind_required_tool(schema: Dict[str, Any], tool_name: str) -> Any:
        try:
            return model.bind_tools([schema], tool_choice=tool_name)
        except TypeError:
            return model.bind_tools([schema])

    plan_model = bind_required_tool(_plan_tool_schema(), _PLAN_TOOL_NAME)
    dispatch_model = bind_required_tool(
        _dispatch_tool_schema(),
        _DISPATCH_TOOL_NAME,
    )
    upstream_evidence_model = bind_required_tool(
        _upstream_evidence_tool_schema(),
        _UPSTREAM_EVIDENCE_TOOL_NAME,
    )
    upstream_answer_model = bind_required_tool(
        _upstream_answer_tool_schema(),
        _UPSTREAM_ANSWER_TOOL_NAME,
    )

    def invoke(selected_model: Any, messages: Sequence[BaseMessage]) -> Any:
        try:
            return (
                selected_model.invoke(messages, config=model_config)
                if model_config is not None
                else selected_model.invoke(messages)
            )
        except Exception as exc:
            raise CoordinatorResponseError(
                f"Ошибка LLM coordinator: {type(exc).__name__}"
            ) from exc

    def downstream_plan_node(state: CoordinatorGraphState) -> Dict[str, Any]:
        plan_messages: List[BaseMessage] = [
            SystemMessage(content=_DOWNSTREAM_PLAN_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {"task": state["task"], "context": state["context"]},
                    ensure_ascii=False,
                )
            ),
        ]
        result = invoke(plan_model, plan_messages)
        try:
            plan = _native_payload(result, _PLAN_TOOL_NAME, WorkerPlan)
        except CoordinatorResponseError:
            logger.warning(
                "Coordinator plan call violated plan schema; requesting one "
                "LLM repair"
            )
            repaired_result = invoke(
                plan_model,
                [
                    *plan_messages,
                    result,
                    HumanMessage(content=_DOWNSTREAM_PLAN_REPAIR_PROMPT),
                ],
            )
            plan = _native_payload(
                repaired_result,
                _PLAN_TOOL_NAME,
                WorkerPlan,
            )
        assert isinstance(plan, WorkerPlan)
        plan_payload = [
            {
                "step": index,
                "goal": step.goal,
                "presentation": step.presentation,
            }
            for index, step in enumerate(plan.steps, start=1)
        ]
        logger.info(
            "Coordinator planned worker_steps=%s plan=%s",
            len(plan.steps),
            json.dumps(plan_payload, ensure_ascii=False),
        )
        record_coordinator_plan(plan_payload)
        return {
            "plan": [step.model_dump() for step in plan.steps],
            "next_step": 0,
        }

    def downstream_materialize_node(
        state: CoordinatorGraphState,
    ) -> Dict[str, Any]:
        step_index = state["next_step"]
        current_step = state["plan"][step_index]
        completed_workers = [
            {
                "step": run["step"],
                "status": "completed",
                "goal": run["goal"],
                "task": run["task"],
                "answer": run["answer"],
                "cycle_history": list(run["cycle_history"]),
                "saved_results": list(run["saved_results"]),
            }
            for run in state["worker_runs"]
        ]
        dispatch_messages: List[BaseMessage] = [
            SystemMessage(content=_DOWNSTREAM_DISPATCH_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {
                        "original_task": state["task"],
                        "context": state["context"],
                        "current_step": {
                            "step": step_index + 1,
                            "goal": current_step["goal"],
                        },
                        "completed_workers": completed_workers,
                    },
                    ensure_ascii=False,
                )
            ),
        ]
        result = invoke(dispatch_model, dispatch_messages)
        try:
            dispatch = _native_payload(
                result,
                _DISPATCH_TOOL_NAME,
                WorkerDispatch,
            )
        except CoordinatorResponseError:
            logger.warning(
                "Coordinator dispatch call violated dispatch schema; "
                "requesting one LLM repair"
            )
            repaired_result = invoke(
                dispatch_model,
                [
                    *dispatch_messages,
                    result,
                    HumanMessage(content=_DOWNSTREAM_DISPATCH_REPAIR_PROMPT),
                ],
            )
            dispatch = _native_payload(
                repaired_result,
                _DISPATCH_TOOL_NAME,
                WorkerDispatch,
            )
        assert isinstance(dispatch, WorkerDispatch)
        worker_task = dispatch.task
        if len(state["plan"]) == 1:
            worker_task = state["task"].strip()
            context = state["context"].strip()
            if context:
                worker_task = (
                    f"{worker_task}\n\nУстойчивые правила контекста:\n{context}"
                )
        logger.info(
            "Coordinator materialized worker step=%s task=%s",
            step_index + 1,
            worker_task[:1000],
        )
        return {"pending_task": worker_task}

    def worker_node(state: CoordinatorGraphState) -> Dict[str, Any]:
        worker_task = str(state.get("pending_task") or "").strip()
        if not worker_task:
            raise CoordinatorResponseError(
                "Coordinator вызвал worker без материализованной task."
            )
        step_index = state["next_step"]
        worker_result = worker_chat(worker_task)
        if not worker_result.goal_satisfied:
            failed_refs = [item.ref for item in worker_result.result_refs]
            if failed_refs:
                discard_worker_result_refs(failed_refs)
            mismatch_text = "; ".join(worker_result.mismatches) or (
                "worker не указал конкретные несоответствия"
            )
            logger.warning(
                "Worker step=%s did not satisfy task; stopping plan: %s",
                step_index + 1,
                mismatch_text,
            )
            failed_run: CoordinatorWorkerRun = {
                "step": step_index + 1,
                "goal": state["plan"][step_index]["goal"],
                "task": worker_task,
                "answer": worker_result.answer,
                "cycle_history": [
                    cycle.model_dump(mode="json")
                    for cycle in worker_result.cycle_history
                ],
                "result_refs": [],
                "saved_results": [
                    item.model_dump(mode="json")
                    for item in worker_result.saved_results
                ],
                "goal_satisfied": False,
                "mismatches": list(worker_result.mismatches),
            }
            return {
                "worker_runs": [*state["worker_runs"], failed_run],
                "next_step": len(state["plan"]),
                "pending_task": None,
            }

        result_refs: List[CoordinatorResultRef] = []
        for result_index, result_ref in enumerate(worker_result.result_refs, 1):
            if collected_result_refs is not None:
                collected_result_refs.append(result_ref.ref)
            result_refs.append(
                {
                    "result_key": f"step-{step_index + 1}:result-{result_index}",
                    "ref": result_ref.ref,
                    "name": result_ref.name,
                }
            )
        run: CoordinatorWorkerRun = {
            "step": step_index + 1,
            "goal": state["plan"][step_index]["goal"],
            "task": worker_task,
            "answer": worker_result.answer,
            "cycle_history": [
                cycle.model_dump(mode="json")
                for cycle in worker_result.cycle_history
            ],
            "result_refs": result_refs,
            "saved_results": [
                item.model_dump(mode="json")
                for item in worker_result.saved_results
            ],
            "goal_satisfied": worker_result.goal_satisfied,
            "mismatches": list(worker_result.mismatches),
        }
        return {
            "worker_runs": [*state["worker_runs"], run],
            "next_step": step_index + 1,
            "pending_task": None,
        }

    def upstream_evidence_node(state: CoordinatorGraphState) -> Dict[str, Any]:
        worker_payload = [
            {
                "step": run["step"],
                "status": (
                    "completed" if run["goal_satisfied"] else "failed"
                ),
                "goal": run["goal"],
                "task": run["task"],
                "answer": run["answer"],
                "cycle_history": list(run["cycle_history"]),
                "goal_satisfied": run["goal_satisfied"],
                "mismatches": list(run["mismatches"]),
            }
            for run in state["worker_runs"]
        ]
        upstream_messages: List[BaseMessage] = [
            SystemMessage(content=_UPSTREAM_EVIDENCE_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {
                        "original_task": state["task"],
                        "context": state["context"],
                        "plan": state["plan"],
                        "worker_results": worker_payload,
                    },
                    ensure_ascii=False,
                )
            ),
        ]
        result = invoke(upstream_evidence_model, upstream_messages)
        try:
            evidence = _native_payload(
                result,
                _UPSTREAM_EVIDENCE_TOOL_NAME,
                UpstreamEvidence,
            )
        except CoordinatorResponseError:
            logger.warning(
                "Upstream coordinator violated evidence schema; "
                "requesting one LLM repair"
            )
            repaired_result = invoke(
                upstream_evidence_model,
                [
                    *upstream_messages,
                    result,
                    HumanMessage(content=_UPSTREAM_EVIDENCE_REPAIR_PROMPT),
                ],
            )
            evidence = _native_payload(
                repaired_result,
                _UPSTREAM_EVIDENCE_TOOL_NAME,
                UpstreamEvidence,
            )
        assert isinstance(evidence, UpstreamEvidence)
        upstream_result = evidence.model_dump()
        selected_display_refs = [
            result_ref["ref"]
            for run in state["worker_runs"]
            if run["goal_satisfied"]
            and state["plan"][run["step"] - 1]["presentation"]
            == "full_results"
            for result_ref in run["result_refs"]
        ]
        record_upstream_evidence(upstream_result)
        logger.info(
            "Upstream coordinator evidence: %s",
            json.dumps(upstream_result, ensure_ascii=False)[:8000],
        )
        return {
            "upstream_result": upstream_result,
            "selected_display_refs": selected_display_refs,
        }

    def upstream_answer_node(state: CoordinatorGraphState) -> Dict[str, Any]:
        upstream_result = state.get("upstream_result")
        if not upstream_result:
            raise CoordinatorResponseError(
                "Upstream answer coordinator вызван без upstream evidence."
            )
        upstream_answer_messages: List[BaseMessage] = [
            SystemMessage(content=_UPSTREAM_ANSWER_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {
                        "original_task": state["task"],
                        "context": state["context"],
                        "upstream_evidence": upstream_result,
                    },
                    ensure_ascii=False,
                )
            ),
        ]
        result = invoke(upstream_answer_model, upstream_answer_messages)
        try:
            upstream_answer = _native_payload(
                result,
                _UPSTREAM_ANSWER_TOOL_NAME,
                UpstreamAnswer,
            )
        except CoordinatorResponseError:
            logger.warning(
                "Upstream answer coordinator violated answer schema; requesting "
                "one LLM repair"
            )
            repaired_result = invoke(
                upstream_answer_model,
                [
                    *upstream_answer_messages,
                    result,
                    HumanMessage(content=_UPSTREAM_ANSWER_REPAIR_PROMPT),
                ],
            )
            upstream_answer = _native_payload(
                repaired_result,
                _UPSTREAM_ANSWER_TOOL_NAME,
                UpstreamAnswer,
            )
        assert isinstance(upstream_answer, UpstreamAnswer)
        final_answer = upstream_answer.answer
        record_upstream_answer(final_answer)
        logger.info(
            "Upstream coordinator answer: %s",
            json.dumps(
                {"answer": final_answer},
                ensure_ascii=False,
            )[:8000],
        )
        return {"final_answer": final_answer}

    def route_after_worker(
        state: CoordinatorGraphState,
    ) -> Literal["downstream_materialize", "upstream_evidence"]:
        if state["next_step"] < len(state["plan"]):
            return "downstream_materialize"
        return "upstream_evidence"

    graph = StateGraph(CoordinatorGraphState)
    graph.add_node("downstream_plan", downstream_plan_node)
    graph.add_node("downstream_materialize", downstream_materialize_node)
    graph.add_node("worker", worker_node)
    graph.add_node("upstream_evidence", upstream_evidence_node)
    graph.add_node("upstream_answer", upstream_answer_node)
    graph.add_edge(START, "downstream_plan")
    graph.add_edge("downstream_plan", "downstream_materialize")
    graph.add_edge("downstream_materialize", "worker")
    graph.add_conditional_edges(
        "worker",
        route_after_worker,
        {
            "downstream_materialize": "downstream_materialize",
            "upstream_evidence": "upstream_evidence",
        },
    )
    graph.add_edge("upstream_evidence", "upstream_answer")
    graph.add_edge("upstream_answer", END)
    return graph.compile()


def coordinator_chat(task: str, *, context: str = "") -> CoordinatorAnswer:
    """Send tasks downstream to workers and return verified results upstream."""
    clean_task = str(task or "").strip()
    clean_context = str(context or "").strip()[:COORDINATOR_CONTEXT_MAX_CHARS]
    if not clean_task:
        return CoordinatorAnswer(
            answer="Задача coordinator не должна быть пустой.",
            display_refs=[],
        )

    callback = get_callback_handler()
    callbacks = [callback] if callback is not None else []
    metrics_callback = get_run_metrics_callback()
    if metrics_callback is not None and metrics_callback not in callbacks:
        callbacks.append(metrics_callback)
    collected_result_refs: List[str] = []
    graph = build_coordinator_graph(
        chat_model,
        callbacks=callbacks,
        collected_result_refs=collected_result_refs,
    )
    initial_state: CoordinatorGraphState = {
        "task": clean_task,
        "context": clean_context,
        "plan": [],
        "next_step": 0,
        "pending_task": None,
        "worker_runs": [],
        "upstream_result": None,
        "final_answer": None,
        "selected_display_refs": [],
    }
    config = {
        "recursion_limit": COORDINATOR_MAX_WORKERS * 2 + 6,
        "run_name": "worker_coordinator",
    }

    with (
        saved_result_store_scope(),
        langfuse_trace_context(
            trace_name="worker_coordinator",
            metadata={"max_workers": COORDINATOR_MAX_WORKERS},
            tags=["coordinator", "worker", "experiment"],
        ),
    ):
        try:
            final_state = graph.invoke(initial_state, config=config)
            final_answer = str(final_state.get("final_answer") or "").strip()
            if not final_answer:
                raise CoordinatorResponseError(
                    "Coordinator LangGraph завершился без ответа."
                )
            selected_refs = list(final_state.get("selected_display_refs") or [])
            selected_set = set(selected_refs)
            unselected_refs = [
                ref for ref in collected_result_refs if ref not in selected_set
            ]
            if unselected_refs:
                discard_worker_result_refs(unselected_refs)
            return CoordinatorAnswer(
                answer=final_answer,
                display_refs=selected_refs,
            )
        except Exception:
            if collected_result_refs:
                discard_worker_result_refs(collected_result_refs)
            raise


__all__ = [
    "COORDINATOR_MAX_WORKERS",
    "COORDINATOR_CONTEXT_MAX_CHARS",
    "UpstreamEvidence",
    "UpstreamAnswer",
    "CoordinatorAnswer",
    "CoordinatorGraphState",
    "CoordinatorResponseError",
    "WorkerDispatch",
    "WorkerPlan",
    "WorkerPlanStep",
    "build_coordinator_graph",
    "coordinator_chat",
]
