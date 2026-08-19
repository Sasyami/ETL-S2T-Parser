"""LLM-driven worker coordinator with an isolated final aggregator."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .agent import chat_model
from .observability import get_callback_handler, langfuse_trace_context
from .run_metrics import get_run_metrics_callback, record_coordinator_plan
from .worker import discard_worker_result_refs, worker_chat

logger = logging.getLogger(__name__)

COORDINATOR_MAX_WORKERS = 8
COORDINATOR_CONTEXT_MAX_CHARS = 4000
_PLAN_TOOL_NAME = "submit_worker_plan"
_DISPATCH_TOOL_NAME = "dispatch_worker"
_COORDINATE_TOOL_NAME = "submit_coordination_result"
_FINISH_TOOL_NAME = "finish_coordination"


class WorkerPlanStep(BaseModel):
    """One semantic worker step selected by the planner LLM."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(
        min_length=1,
        description=(
            "Одна операция с данными, дающая один самостоятельно проверяемый "
            "результат. Зависимая операция должна быть отдельным шагом."
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


class CoordinationFinish(BaseModel):
    """Coordinator synthesis over worker runs and selected UI results."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    display_result_keys: List[str] = Field(default_factory=list)

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

    @field_validator("display_result_keys")
    @classmethod
    def _deduplicate_keys(cls, values: List[str]) -> List[str]:
        result: List[str] = []
        for value in values:
            clean_value = str(value or "").strip()
            if not clean_value:
                raise ValueError("display_result_keys must not contain blanks")
            if clean_value not in result:
                result.append(clean_value)
        return result


class FinalAggregation(BaseModel):
    """Final user-facing answer produced only from coordinator synthesis."""

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
    goal_satisfied: bool
    mismatches: List[str]


class CoordinatorGraphState(TypedDict):
    task: str
    context: str
    plan: List[Dict[str, str]]
    next_step: int
    pending_task: Optional[str]
    worker_runs: List[CoordinatorWorkerRun]
    coordinator_result: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    selected_display_refs: List[str]


class CoordinatorResponseError(RuntimeError):
    """Raised when an LLM response violates a structural coordinator contract."""


_PLAN_PROMPT = f"""
Ты planner промежуточного coordinator. Раздели входную задачу на упорядоченный
план для изолированных generic workers и верни ровно один native call
`{_PLAN_TOOL_NAME}` со списком `steps` длиной от 1 до
{COORDINATOR_MAX_WORKERS}. Каждый step содержит только `goal` и `presentation`.

Правила декомпозиции:
- один step описывает одну операцию с данными и один самостоятельно проверяемый
  результат;
- если операция использует результат предыдущей операции, создай отдельный
  следующий step; нужный результат позднее подставит dispatcher;
- не объединяй независимо проверяемые операции в одном goal и не создавай
  отдельные steps для оформления, пересказа или повторного показа тех же данных;
- сохраняй порядок, точные сущности, идентификаторы, значения, условия,
  ограничения и запреты исходной задачи без переименования;
- используй `context` только для однозначного разрешения ссылок и общих
  ограничений. Если данных недостаточно, не угадывай и не достраивай предметную
  схему.

`presentation` — только технический режим передачи результата и не влияет на
декомпозицию. Не выбирай tools и skills, не выполняй задачу и не формулируй
ответ пользователю.
""".strip()

_PLAN_REPAIR_PROMPT = f"""
Предыдущий native call `{_PLAN_TOOL_NAME}` не соответствует технической схеме.
Верни исправленный native call ровно один раз. Массив `steps` должен содержать
от 1 до {COORDINATOR_MAX_WORKERS} элементов; каждый элемент должен иметь только
непустой `goal` и `presentation` со значением `answer_only` или `full_results`.
Исправь только формат native call: не объединяй независимо проверяемые операции,
не отбрасывай требования, не добавляй новые факты и не вызывай другие tools.
""".strip()

_DISPATCH_PROMPT = f"""
Ты dispatcher промежуточного coordinator. Материализуй `current_step` в одну
самодостаточную задачу для изолированного generic worker.

Верни ровно один native call `{_DISPATCH_TOOL_NAME}` с полем `task`. Worker
получит только эту строку. `original_task` задаёт ограничения и точные значения,
а `current_step.goal` — единственную операцию этого worker. Используй общий
`context` и краткие ответы `completed_workers` лишь там, где они нужны текущей
цели. Подставляй подтверждённые предыдущим worker значения в зависимый шаг, не
заставляя worker повторять завершённую работу.
Не добавляй в task отдельные результаты, которые нужны только соседним шагам:
текущий worker должен выполнить ровно свою одну операцию.

Каждый completed worker содержит краткий answer и ordered `cycle_history`:
вызовы tools, ограниченные текстовые preview их результатов и structured
observation, а поле `status=completed` явно отмечает завершённый пункт плана.
Используй историю, чтобы сохранить точный смысл подтверждённых фактов и
зависимостей. Это не полные результаты tools. Ошибочные промежуточные циклы
нужны только для понимания исправлений; подтверждёнными считай факты из цикла,
observation которого завершает goal с `goal_satisfied=true`.

Факты completed_workers являются подтверждёнными входными условиями зависимого
шага, а не новой целью. Не проси текущий worker повторно получать, проверять,
пересчитывать или возвращать их, если current_step.goal этого явно не требует.
Task должна запрашивать только новые факты текущей операции; итоговый aggregator
сам объединит их с ответами и историей предыдущих workers.

Если original_task или current_step.goal содержит неразрешённую ссылку на
объект, используй точное однозначное название из `context` и обязательно замени
им ссылку в task. Не передавай worker слова «в нём», «в ней», «там», «этот»,
«эта» или «это» вместо названия объекта. Если context содержит факт «речь о
таблице X», worker должен получить «таблица X». Если однозначного названия нет,
не выбирай произвольный объект и сформулируй невозможность выполнения без
уточнения.

Считай значение подтверждённым для текущей цели только если предыдущий ответ
явно связывает его с требуемой сущностью, ролью или вычислением. Не присваивай
общему или неоднозначному значению нужный смысл самостоятельно. Если нужный
факт не подтверждён, не выдумывай его и явно сохрани это ограничение в task.
Точные идентификаторы из предыдущего ответа копируй посимвольно, не исправляй,
не сокращай и не создавай их варианты. Все упоминания одного идентификатора
внутри task должны совпадать.

Сохраняй относящиеся к шагу источники, запреты, идентификаторы, числа, связи и
операцию дословно по смыслу. Не превращай вычисляемую величину в якобы готовое
поле и не придумывай отсутствующие во входе сущности, поля или значения. Не
опускай явно названный исходный набор данных, хранилище или систему: повтори их
в task, даже если они кажутся очевидными из original_task. Подставляя результат
предыдущего шага, сохраняй его исходную роль: значение роли остаётся условием
внутри исходного набора данных и не становится новым источником или объектом.
Worker не должен восстанавливать источник по догадке.

Не
добавляй цели соседних шагов, tools, skills, план, служебные поля или
рассуждения. Отдельный показ, экспорт и выбор полного результата выполняются
за пределами worker. Не добавляй эти требования в task и не отвлекай worker от
получения фактов для правильного текстового ответа. При этом сохраняй
ограничения на форму самого текстового ответа, например просьбу вернуть только
число или перечислить значения в заданном порядке.

Перед native call молча проверь, что task явно содержит источник, требуемую
сущность или роль, операцию, условия и факты для ответа из current_step и
original_task, а все повторения точных идентификаторов совпадают с источником.
""".strip()

_DISPATCH_REPAIR_PROMPT = f"""
Предыдущий native call `{_DISPATCH_TOOL_NAME}` не соответствует технической
схеме. Верни исправленный native call ровно один раз с единственным непустым
строковым полем `task`. Сохрани текущую операцию, точные идентификаторы,
источники, условия и ограничения из входного payload. Не добавляй соседние
шаги, tools, служебные поля или новые факты.
""".strip()

_COORDINATE_PROMPT = f"""
Ты coordinator результатов workers. Сформируй самодостаточный результат по исходной
задаче, плану и кратким ответам workers, затем выбери полезные полные результаты
для отдельного показа в UI.

Главный приоритет — правильный текстовый ответ. Перед завершением молча сверь
его с исходной задачей: все запрошенные факты, связи, вычисления, порядок,
ограничения и форма ответа должны быть соблюдены. Не добавляй сведения, которых
пользователь не просил, особенно если он потребовал «только» конкретный итог.
Не повторяй в answer отрицательные ограничения и названия источников, способов
или действий, которые требовалось не использовать: молча соблюдай такие запреты.
Упоминай их только если пользователь отдельно попросил подтвердить соблюдение.
При требовании «только» верни ровно перечисленные пользователем элементы без
вступления, заключения, подтверждения, интерпретации и иных пояснений.

Верни ровно один native call `{_COORDINATE_TOOL_NAME}`:
- `answer`: всегда строка с самодостаточным ответом пользователю без внутренних
  терминов. Даже если пользователь запросил JSON, объект или список, сериализуй
  требуемое представление внутрь строки `answer`; никогда не передавай объект,
  массив, число или null как значение этого технического поля;
- `display_result_keys`: ключи только тех доступных результатов, которые нужно
  показать полностью отдельно.

Полные tool results намеренно не передаются. Каждый worker result содержит
ordered `cycle_history`: точные tool calls, ограниченные текстовые preview и
structured observations, а `status` явно отмечает пункт как `completed` или
`failed`. Используй историю вместе с кратким answer, чтобы не терять роль и
происхождение фактов. Ошибочный промежуточный вызов не является фактом для
итогового ответа; учитывай его только как исправленную попытку. Опирайся на
important_facts и summary observation того цикла, который подтверждает goal.
Не придумывай отсутствующие факты или ключи. Сохраняй связи между значениями и
объединяй результаты зависимых шагов по смыслу исходной задачи. Сохраняй в
answer точные идентификаторы, порядок и числовые значения из подтверждённых
worker-результатов, не заменяя их общими описаниями. Считай шаг выполненным только если
worker answer и его финальная observation подтверждают требуемый goal. При
неоднозначности, противоречии или ошибке не назначай значению нужную
роль и честно укажи, какой факт не подтверждён.
Если у worker указан `goal_satisfied=false`, план остановлен на этом шаге:
используй его `mismatches` для краткого пользовательского ограничения, не
продолжай зависимые вычисления мысленно и не выбирай display results этого шага.

Выбор display results вторичен и не должен влиять на `answer`; при сомнении
можно вернуть пустой список. Выбирай display result только для шага с
`presentation=full_results`, если его
полный результат доступен. Для `answer_only` не выбирай result keys. Показ в UI
не меняет полноту текстового ответа и не является отдельным шагом. Если worker
не выполнил часть плана, честно укажи это и не выдавай описание намерения за
полученный результат.
""".strip()

_COORDINATE_REPAIR_PROMPT = f"""
Предыдущий native call `{_COORDINATE_TOOL_NAME}` не соответствует его технической
схеме. Верни исправленный native call ровно один раз. Поле `answer` обязательно
должно быть строкой. Если пользовательский ответ имеет форму JSON-объекта,
массива, числа или другой структуры, сериализуй её в текст внутри `answer`, не
меняя подтверждённые факты. `display_result_keys` должен оставаться массивом
доступных строковых ключей. Не добавляй новые факты и не вызывай другие tools.
""".strip()

_AGGREGATE_PROMPT = f"""
Ты финальный aggregator. На входе находится только самодостаточный
`coordinator_result`, уже собранный coordinator по результатам workers.

Верни ровно один native call `{_FINISH_TOOL_NAME}` с единственным полем
`answer`. Сохрани все факты, точные идентификаторы, числа, связи, порядок и
форму ответа из `coordinator_result.answer`. Ничего не добавляй, не удаляй, не
переоценивай и не пытайся восстановить исходную задачу или внутреннюю историю.
Не упоминай coordinator, workers, tools и observations. Если answer уже готов
для пользователя, перенеси его без смысловых изменений. Даже если это JSON,
объект, массив или число, значение технического поля `answer` должно быть
строкой.
""".strip()

_AGGREGATE_REPAIR_PROMPT = f"""
Предыдущий native call `{_FINISH_TOOL_NAME}` не соответствует технической
схеме. Верни исправленный native call с единственным строковым полем `answer`.
Сохрани `coordinator_result.answer` без добавления и потери фактов.
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
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": (
                                    "Одна операция с данными и один "
                                    "самостоятельно проверяемый результат, с "
                                    "точными сущностями и условиями"
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


def _coordination_result_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _COORDINATE_TOOL_NAME,
            "description": (
                "Собрать результат coordinator и выбрать полные результаты "
                "workers для UI."
            ),
            "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": (
                        "Правильный ответ в запрошенной форме, содержащий "
                        "только явно подтверждённые worker-ответами факты"
                    ),
                },
                "display_result_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["answer", "display_result_keys"],
            "additionalProperties": False,
            },
        },
    }


def _finish_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _FINISH_TOOL_NAME,
            "description": (
                "Вернуть окончательный текст только из результата coordinator."
            ),
            "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": (
                        "Пользовательский ответ без добавления или потери "
                        "фактов результата coordinator"
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
    """Build plan -> workers -> coordinate -> final aggregate graph."""
    callback_list = list(callbacks or [])
    model_config = {"callbacks": callback_list} if callback_list else None
    plan_model = model.bind_tools([_plan_tool_schema()])
    dispatch_model = model.bind_tools([_dispatch_tool_schema()])
    coordinate_model = model.bind_tools([_coordination_result_tool_schema()])
    aggregate_model = model.bind_tools([_finish_tool_schema()])

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

    def plan_node(state: CoordinatorGraphState) -> Dict[str, Any]:
        plan_messages: List[BaseMessage] = [
            SystemMessage(content=_PLAN_PROMPT),
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
                    HumanMessage(content=_PLAN_REPAIR_PROMPT),
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

    def materialize_node(state: CoordinatorGraphState) -> Dict[str, Any]:
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
            }
            for run in state["worker_runs"]
        ]
        dispatch_messages: List[BaseMessage] = [
            SystemMessage(content=_DISPATCH_PROMPT),
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
                    HumanMessage(content=_DISPATCH_REPAIR_PROMPT),
                ],
            )
            dispatch = _native_payload(
                repaired_result,
                _DISPATCH_TOOL_NAME,
                WorkerDispatch,
            )
        assert isinstance(dispatch, WorkerDispatch)
        logger.info(
            "Coordinator materialized worker step=%s task=%s",
            step_index + 1,
            dispatch.task[:1000],
        )
        return {"pending_task": dispatch.task}

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
            "goal_satisfied": worker_result.goal_satisfied,
            "mismatches": list(worker_result.mismatches),
        }
        return {
            "worker_runs": [*state["worker_runs"], run],
            "next_step": step_index + 1,
            "pending_task": None,
        }

    def coordinate_node(state: CoordinatorGraphState) -> Dict[str, Any]:
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
                "available_results": [
                    {
                        "result_key": result_ref["result_key"],
                        "tool_name": result_ref["name"],
                    }
                    for result_ref in run["result_refs"]
                ],
            }
            for run in state["worker_runs"]
        ]
        refs_by_key = {
            result_ref["result_key"]: result_ref["ref"]
            for run in state["worker_runs"]
            for result_ref in run["result_refs"]
        }
        coordinate_messages: List[BaseMessage] = [
            SystemMessage(content=_COORDINATE_PROMPT),
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
        result = invoke(coordinate_model, coordinate_messages)
        try:
            finish = _native_payload(
                result,
                _COORDINATE_TOOL_NAME,
                CoordinationFinish,
            )
        except CoordinatorResponseError:
            logger.warning(
                "Coordinator synthesis call violated result schema; "
                "requesting one LLM repair"
            )
            repaired_result = invoke(
                coordinate_model,
                [
                    *coordinate_messages,
                    result,
                    HumanMessage(content=_COORDINATE_REPAIR_PROMPT),
                ],
            )
            finish = _native_payload(
                repaired_result,
                _COORDINATE_TOOL_NAME,
                CoordinationFinish,
            )
        assert isinstance(finish, CoordinationFinish)
        unknown_keys = [
            key for key in finish.display_result_keys if key not in refs_by_key
        ]
        if unknown_keys:
            raise CoordinatorResponseError(
                "Coordinator выбрал неизвестные result keys: "
                + ", ".join(unknown_keys)
            )
        return {
            "coordinator_result": {
                "answer": finish.answer,
                "display_result_keys": list(finish.display_result_keys),
            },
            "selected_display_refs": [
                refs_by_key[key] for key in finish.display_result_keys
            ],
        }

    def aggregate_node(state: CoordinatorGraphState) -> Dict[str, Any]:
        coordinator_result = state.get("coordinator_result")
        if not coordinator_result:
            raise CoordinatorResponseError(
                "Финальный aggregator вызван без результата coordinator."
            )
        aggregate_messages: List[BaseMessage] = [
            SystemMessage(content=_AGGREGATE_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {"coordinator_result": coordinator_result},
                    ensure_ascii=False,
                )
            ),
        ]
        result = invoke(aggregate_model, aggregate_messages)
        try:
            aggregation = _native_payload(
                result,
                _FINISH_TOOL_NAME,
                FinalAggregation,
            )
        except CoordinatorResponseError:
            logger.warning(
                "Final aggregate call violated finish schema; requesting one "
                "LLM repair"
            )
            repaired_result = invoke(
                aggregate_model,
                [
                    *aggregate_messages,
                    result,
                    HumanMessage(content=_AGGREGATE_REPAIR_PROMPT),
                ],
            )
            aggregation = _native_payload(
                repaired_result,
                _FINISH_TOOL_NAME,
                FinalAggregation,
            )
        assert isinstance(aggregation, FinalAggregation)
        return {"final_answer": aggregation.answer}

    def route_after_worker(
        state: CoordinatorGraphState,
    ) -> Literal["materialize", "coordinate"]:
        if state["next_step"] < len(state["plan"]):
            return "materialize"
        return "coordinate"

    graph = StateGraph(CoordinatorGraphState)
    graph.add_node("plan", plan_node)
    graph.add_node("materialize", materialize_node)
    graph.add_node("worker", worker_node)
    graph.add_node("coordinate", coordinate_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "materialize")
    graph.add_edge("materialize", "worker")
    graph.add_conditional_edges(
        "worker",
        route_after_worker,
        {"materialize": "materialize", "coordinate": "coordinate"},
    )
    graph.add_edge("coordinate", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()


def coordinator_chat(task: str, *, context: str = "") -> CoordinatorAnswer:
    """Plan workers, execute them, aggregate answers, and select UI results."""
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
        "coordinator_result": None,
        "final_answer": None,
        "selected_display_refs": [],
    }
    config = {
        "recursion_limit": COORDINATOR_MAX_WORKERS * 2 + 6,
        "run_name": "worker_coordinator",
    }

    with langfuse_trace_context(
        trace_name="worker_coordinator",
        metadata={"max_workers": COORDINATOR_MAX_WORKERS},
        tags=["coordinator", "worker", "experiment"],
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
    "CoordinationFinish",
    "FinalAggregation",
    "CoordinatorAnswer",
    "CoordinatorGraphState",
    "CoordinatorResponseError",
    "WorkerDispatch",
    "WorkerPlan",
    "WorkerPlanStep",
    "build_coordinator_graph",
    "coordinator_chat",
]
