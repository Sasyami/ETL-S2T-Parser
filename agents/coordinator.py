"""LLM-driven workers with separate upstream and downstream coordination."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from .agent import chat_model
from .contracts import (
    MAX_PLAN_STEPS,
    PlanStep,
    UpstreamDecision,
    UpstreamOutput,
    WORKER_STABLE_CONTEXT_MARKER,
    WorkerOutcome,
    WorkerPlan,
)
from .observability import get_callback_handler, langfuse_trace_context
from .run_metrics import (
    get_run_metrics_callback,
    llm_stage,
    record_coordinator_plan,
    record_upstream_output,
)
from .tools.context import load_upstream_analysis_context
from .worker import discard_worker_display_refs, worker_chat
from .tools.saved_results import saved_result_store_scope

logger = logging.getLogger(__name__)

COORDINATOR_MAX_WORKERS = MAX_PLAN_STEPS
COORDINATOR_MAX_CYCLES = 2
COORDINATOR_CONTEXT_MAX_CHARS = 4000
_PLAN_TOOL_NAME = "submit_worker_plan"
_UPSTREAM_ANSWER_TOOL_NAME = "submit_upstream_answer"
_UPSTREAM_DATA_DECISION_TOOL_NAME = "submit_upstream_data_decision"
_UPSTREAM_ANALYSIS_CONTEXT = load_upstream_analysis_context()


class CoordinatorAnswer(BaseModel):
    """Coordinator output consumed by the top-level supervisor."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    display_refs: List[str] = Field(default_factory=list)


class CoordinatorWorkerRun(TypedDict):
    cycle: int
    step: int
    outcome: WorkerOutcome


class CoordinatorGraphState(TypedDict):
    task: str
    context: str
    cycle: int
    plan: List[Dict[str, Any]]
    next_step: int
    worker_runs: List[CoordinatorWorkerRun]
    upstream_problem: Optional[str]
    upstream_output: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    selected_display_refs: List[str]


class CoordinatorResponseError(RuntimeError):
    """Raised when an LLM response violates a structural coordinator contract."""


_DOWNSTREAM_PLAN_PROMPT = f"""
Ты downstream planner coordinator. Верни один native call `{_PLAN_TOOL_NAME}`
со списком `steps` от 1 до {COORDINATOR_MAX_WORKERS}. Каждый step содержит
одну самодостаточную `task` на получение исходных данных одним worker.

Выдели из `original_task` только необходимые чтения исходных фактов. Сохрани в
tasks точные объекты, роли, scope, фильтры и поля результата. Не переноси в
worker-задачи сравнение, оценку, объяснение, вывод или оформление ответа — всё
это выполняет upstream после получения evidence.
Роли `source`/`target`, направление связи и тип каждой сущности — файл, таблица
или колонка/поле — сохраняй явно; не заменяй их общим либо неоднозначным
обозначением.
Явно указанные идентификаторы уже являются входами: не создавай отдельные чтения,
чтобы найти их повторно. Если одной операции нужны несколько таких входов, сохрани
их вместе в одной task. Разделяй только независимые чтения, результаты которых
upstream сможет объединить без передачи данных между workers.

Workers выполняются последовательно и не получают результаты друг друга,
поэтому каждая task должна содержать все входы своего чтения. Не ссылайся на
«найденный» объект, предыдущий step или результат другого worker. Копируй полный
идентификатор, но не включай в него пунктуацию, отделяющую последующий текст.
Не выбирай tools или skills здесь и не придумывай evidence_id или факты. `context`
используй только для разрешения ссылок и устойчивых ограничений.

В повторном цикле прошлые результаты недоступны. По `original_task` и одной
строке `problem` построй полный исправленный план чтения заново.
""".strip()

_DOWNSTREAM_PLAN_REPAIR_PROMPT = f"""
Предыдущий native call `{_PLAN_TOOL_NAME}` нарушает схему или смысловой контракт.
Верни исправленный native call ровно один раз. Массив `steps` должен содержать
от 1 до {COORDINATOR_MAX_WORKERS} элементов; каждый элемент должен иметь
только одну непустую самодостаточную `task`.
Совокупность tasks обязана сохранить каждый явно запрошенный исходный результат,
точный идентификатор и условие из original task. Явные идентификаторы считай
готовыми входами, а несколько входов одной операции сохраняй в одной task.
Не ссылайся на предыдущий step, его результат или «найденный» объект: каждая task
должна содержать точные значения всех своих входов. Разделяй только независимые
чтения, не требующие передачи результатов между workers.
Не добавляй шаг сравнения, оценки, объяснения, вывода или оформления ответа:
производный анализ выполняет upstream. Не добавляй tools и факты.

Причина отклонения: {{validation_error}}
""".strip()

_UPSTREAM_DATA_DECISION_PROMPT = f"""
Ты проверяешь достаточность данных перед upstream answer. Вход содержит только
`original_task` и принятые `evidence`. Верни ровно один native call
`{_UPSTREAM_DATA_DECISION_TOOL_NAME}`:

- `decision="pass"`, если evidence достаточно для формирования ответа;
- `decision="reroute"`, если требуется новый цикл чтения.

`problem` необязателен. При reroute он может кратко описать, каких именно данных
не хватает новому downstream-плану. Не формируй пользовательский ответ и не
выбирай display-results.

Evidence содержит `evidence_id`, `tool_name`, точные `args`, фактический
`preview`, `truncated` и безопасный `display_id`. Аргументы подтверждают область
чтения, preview — найденные данные. Не додумывай отсутствующее; при
`truncated=true` не считай полный набор подтверждённым.

Сопоставь каждый запрошенный исходный результат и его scope с прямым
подтверждением в evidence. Нельзя считать значение одной метрики подтверждением
другой.
""".strip()

_UPSTREAM_ANSWER_PROMPT = f"""
Ты upstream answer coordinator. Предварительная проверка уже вернула `pass`.
Вход содержит `original_task` и принятые `evidence`. Сам выполни запрошенный
анализ и верни ровно один native call `{_UPSTREAM_ANSWER_TOOL_NAME}` с готовым
`answer`. При наличии подтверждающих evidence передай `used_evidence_ids` и
нужные `display_evidence_ids`.

Evidence содержит `evidence_id`, `tool_name`, точные `args`, фактический
`preview`, `truncated` и безопасный `display_id`. Аргументы подтверждают область
чтения, preview — найденные данные. Не додумывай отсутствующее; при
`truncated=true` не утверждай полноту набора. `display_evidence_ids` выбирай
только из непустых `display_id` и включай также в `used_evidence_ids`.

Перед `answer` сопоставь каждый запрошенный результат и его scope с прямым
подтверждением в evidence. Нельзя повторять значение одной метрики вместо
отсутствующей другой. Если данных всё же недостаточно, явно укажи это в ответе:
на этом линейном этапе возврата к чтению уже нет.

Соблюдай запрошенный формат. Если буквальный компактный формат не задан, ответ
должен быть самодостаточным: подпиши смысл каждого значения и не возвращай
безымянную CSV-последовательность. Если пользователь потребовал «только»
конкретные элементы, не добавляй вступление и заключение. Для шаблона вида
`имя=<значение>` сохрани имя и знак `=` дословно. Не упоминай coordinators,
workers, tools, previews, result refs и внутреннюю схему.
""".strip()

_UPSTREAM_DATA_DECISION_REPAIR_PROMPT = f"""
Предыдущий native call решения о данных не соответствует схеме. Верни ровно один
`{_UPSTREAM_DATA_DECISION_TOOL_NAME}` с обязательным `decision`: `pass` или
`reroute`. `problem` опционален. Не формируй ответ и не выбирай evidence.
""".strip()

_UPSTREAM_ANSWER_REPAIR_PROMPT = f"""
Предыдущий upstream answer call не соответствует схеме. Верни ровно один
`{_UPSTREAM_ANSWER_TOOL_NAME}` с обязательным `answer` и опциональными evidence
IDs. Используй только доступные evidence_id и не добавляй факты.
""".strip()

def _plan_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _PLAN_TOOL_NAME,
            "description": (
                "Зафиксировать последовательность готовых worker tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": COORDINATOR_MAX_WORKERS,
                        "description": (
                            "Необходимые независимые чтения исходных данных; "
                            "без отдельных шагов производного анализа"
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": (
                                        "Готовая задача одного worker только "
                                        "на получение необходимых исходных "
                                        "данных; производный анализ выполняется "
                                        "upstream"
                                    ),
                                },
                            },
                            "required": ["task"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["steps"],
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
                "Вернуть готовый итоговый ответ по достаточным evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Непустой готовый пользовательский ответ.",
                    },
                    "used_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Все evidence_id, использованные в ответе.",
                    },
                    "display_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Подмножество used evidence для отдельного display."
                        ),
                    },
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }


def _upstream_data_decision_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _UPSTREAM_DATA_DECISION_TOOL_NAME,
            "description": (
                "Решить, перейти к upstream answer или повторить чтение данных."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["pass", "reroute"],
                        "description": (
                            "pass продолжает к ответу; reroute повторяет чтение."
                        ),
                    },
                    "problem": {
                        "type": "string",
                        "description": (
                            "Необязательное уточнение нехватки данных для нового плана."
                        ),
                    }
                },
                "required": ["decision"],
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
    except ValidationError as exc:
        details: List[str] = []
        for item in exc.errors():
            path = ".".join(str(part) for part in item.get("loc") or ())
            message_text = str(item.get("msg") or "validation failed")
            details.append(
                f"{path}: {message_text}" if path else message_text
            )
        detail_text = "; ".join(details)[:1200]
        raise CoordinatorResponseError(
            f"Coordinator вернул невалидную структуру {tool_name}: "
            + detail_text
        ) from exc
    except (TypeError, ValueError) as exc:
        raise CoordinatorResponseError(
            f"Coordinator вернул невалидную структуру {tool_name}: "
            + str(exc)[:1200]
        ) from exc


def _native_upstream_decision(message: Any) -> UpstreamDecision:
    """Parse the data decision made before upstream answer generation."""
    decision = _native_payload(
        message,
        _UPSTREAM_DATA_DECISION_TOOL_NAME,
        UpstreamDecision,
    )
    assert isinstance(decision, UpstreamDecision)
    return decision


def _native_upstream_answer(message: Any) -> UpstreamOutput:
    """Parse the final answer after the data decision returned pass."""
    output = _native_payload(
        message,
        _UPSTREAM_ANSWER_TOOL_NAME,
        UpstreamOutput,
    )
    assert isinstance(output, UpstreamOutput)
    return output


def _repair_messages(
    base_messages: Sequence[BaseMessage],
    invalid_result: Any,
    repair_prompt: str,
) -> List[BaseMessage]:
    """Build provider-valid history after rejecting a native tool call."""
    messages = list(base_messages)
    if isinstance(invalid_result, AIMessage):
        tool_calls = list(invalid_result.tool_calls)
        call_ids = [
            str(call.get("id") or "").strip() for call in tool_calls
        ]
        if not tool_calls or all(call_ids):
            messages.append(invalid_result)
            for call, call_id in zip(tool_calls, call_ids):
                messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "status": "rejected",
                                "reason": "native call failed validation",
                            },
                            ensure_ascii=False,
                        ),
                        tool_call_id=call_id,
                        name=str(call.get("name") or "invalid_call"),
                    )
                )
    messages.append(HumanMessage(content=repair_prompt))
    return messages


def build_coordinator_graph(
    model: Any,
    *,
    callbacks: Optional[Sequence[Any]] = None,
    collected_display_refs: Optional[List[str]] = None,
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
    upstream_data_decision_model = bind_required_tool(
        _upstream_data_decision_tool_schema(),
        _UPSTREAM_DATA_DECISION_TOOL_NAME,
    )
    upstream_answer_model = bind_required_tool(
        _upstream_answer_tool_schema(),
        _UPSTREAM_ANSWER_TOOL_NAME,
    )

    def invoke(
        selected_model: Any,
        messages: Sequence[BaseMessage],
        *,
        stage: str,
    ) -> Any:
        try:
            with llm_stage(stage):
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
        plan_payload: Dict[str, Any] = {
            "original_task": state["task"],
            "context": state["context"],
        }
        if state["upstream_problem"] is not None:
            plan_payload["problem"] = state["upstream_problem"]
        plan_messages: List[BaseMessage] = [
            SystemMessage(content=_DOWNSTREAM_PLAN_PROMPT),
            HumanMessage(
                content=json.dumps(
                    plan_payload,
                    ensure_ascii=False,
                )
            ),
        ]
        plan_result = invoke(
            plan_model,
            plan_messages,
            stage="downstream_plan",
        )
        try:
            plan = _native_payload(
                plan_result,
                _PLAN_TOOL_NAME,
                WorkerPlan,
            )
        except CoordinatorResponseError as first_error:
            logger.warning(
                "Coordinator plan call violated plan schema; requesting one "
                "LLM repair: %s",
                first_error,
            )
            repaired_result = invoke(
                plan_model,
                _repair_messages(
                    plan_messages,
                    plan_result,
                    _DOWNSTREAM_PLAN_REPAIR_PROMPT.replace(
                        "{validation_error}",
                        str(first_error),
                    ),
                ),
                stage="downstream_plan",
            )
            plan = _native_payload(
                repaired_result,
                _PLAN_TOOL_NAME,
                WorkerPlan,
            )
            plan_result = repaired_result
            assert isinstance(plan, WorkerPlan)
        assert isinstance(plan, WorkerPlan)

        recorded_plan = [
            {
                "cycle": state["cycle"],
                "step": index,
                "task": step.task,
            }
            for index, step in enumerate(plan.steps, start=1)
        ]
        logger.info(
            "Coordinator planned worker_steps=%s plan=%s",
            len(plan.steps),
            json.dumps(recorded_plan, ensure_ascii=False),
        )
        record_coordinator_plan(recorded_plan)
        return {
            "plan": [step.model_dump() for step in plan.steps],
            "next_step": 0,
        }

    def worker_node(state: CoordinatorGraphState) -> Dict[str, Any]:
        step_index = state["next_step"]
        plan_step = state["plan"][step_index]
        planned_task = str(plan_step["task"] or "").strip()
        if not planned_task:
            raise CoordinatorResponseError(
                "Coordinator вызвал worker с пустой task из плана."
            )
        worker_task = planned_task
        context = state["context"].strip()
        if context:
            worker_task += WORKER_STABLE_CONTEXT_MARKER + context
        logger.info(
            "Coordinator dispatches planned worker step=%s task=%s",
            step_index + 1,
            worker_task[:1000],
        )
        outcome = worker_chat(worker_task)
        for artifact in outcome.evidence:
            if artifact.display_ref and collected_display_refs is not None:
                collected_display_refs.append(artifact.display_ref)
        run: CoordinatorWorkerRun = {
            "cycle": state["cycle"],
            "step": step_index + 1,
            "outcome": outcome,
        }
        return {
            "worker_runs": [*state["worker_runs"], run],
            "next_step": step_index + 1,
        }

    def validate_upstream_decision(
        message: Any,
        *,
        can_reroute: bool,
    ) -> UpstreamDecision:
        decision = _native_upstream_decision(message)
        if decision.decision == "reroute" and not can_reroute:
            raise CoordinatorResponseError(
                "На последнем цикле data decision должен быть pass."
            )
        return decision

    def validate_upstream_answer(
        message: Any,
        *,
        available_evidence_ids: set[str],
        available_display_refs: Dict[str, str],
    ) -> UpstreamOutput:
        output = _native_upstream_answer(message)
        unknown_ids = sorted(
            (
                set(output.used_evidence_ids)
                | set(output.display_evidence_ids)
            )
            - available_evidence_ids
        )
        undisplayable_ids = sorted(
            set(output.display_evidence_ids) - set(available_display_refs)
        )
        if unknown_ids or undisplayable_ids:
            raise CoordinatorResponseError(
                "Upstream coordinator выбрал неизвестные evidence_id: "
                + ", ".join([*unknown_ids, *undisplayable_ids])
            )
        return output

    def upstream_node(state: CoordinatorGraphState) -> Dict[str, Any]:
        available_evidence_ids: set[str] = set()
        available_display_refs: Dict[str, str] = {}
        evidence_payload: List[Dict[str, Any]] = []
        for run in state["worker_runs"]:
            evidence_payload.extend(
                run["outcome"].upstream_payload()["evidence"]
            )
            for artifact in run["outcome"].evidence:
                if artifact.evidence_id in available_evidence_ids:
                    raise CoordinatorResponseError(
                        "Workers вернули дублирующий evidence_id: "
                        + artifact.evidence_id
                    )
                available_evidence_ids.add(artifact.evidence_id)
                if artifact.display_ref is not None:
                    available_display_refs[
                        artifact.evidence_id
                    ] = artifact.display_ref
        upstream_payload = {
            "original_task": state["task"],
            "evidence": evidence_payload,
        }
        decision_messages: List[BaseMessage] = [
            SystemMessage(
                content="\n\n".join(
                    part
                    for part in (
                        _UPSTREAM_DATA_DECISION_PROMPT,
                        (
                            "Разрешён ещё один полный цикл чтения: при "
                            "нехватке данных верни decision=reroute."
                            if state["cycle"] < COORDINATOR_MAX_CYCLES
                            else (
                                "Это последний цикл: верни decision=pass. "
                                "Возможную нехватку данных кратко укажи в problem."
                            )
                        ),
                    )
                    if part
                )
            ),
            HumanMessage(
                content=json.dumps(
                    upstream_payload,
                    ensure_ascii=False,
                )
            ),
        ]
        evidence_context = (
            "\nДоступные used_evidence_ids (копируй дословно): "
            + json.dumps(
                sorted(available_evidence_ids),
                ensure_ascii=False,
            )
            + "\nДоступные display_evidence_ids: "
            + json.dumps(
                sorted(available_display_refs),
                ensure_ascii=False,
            )
        )

        def invoke_decision(
            messages: Sequence[BaseMessage],
        ) -> tuple[Any, UpstreamDecision]:
            can_reroute = state["cycle"] < COORDINATOR_MAX_CYCLES
            result = invoke(
                upstream_data_decision_model,
                messages,
                stage="upstream",
            )
            try:
                decision = validate_upstream_decision(
                    result,
                    can_reroute=can_reroute,
                )
            except CoordinatorResponseError as first_error:
                logger.warning(
                    "Upstream data decision violated schema; "
                    "requesting one LLM repair: %s",
                    first_error,
                )
                result = invoke(
                    upstream_data_decision_model,
                    _repair_messages(
                        messages,
                        result,
                        _UPSTREAM_DATA_DECISION_REPAIR_PROMPT
                        + "\nОшибка: "
                        + str(first_error),
                    ),
                    stage="upstream",
                )
                decision = validate_upstream_decision(
                    result,
                    can_reroute=can_reroute,
                )
            return result, decision

        def invoke_answer(
            messages: Sequence[BaseMessage],
        ) -> tuple[Any, UpstreamOutput]:
            result = invoke(
                upstream_answer_model,
                messages,
                stage="upstream",
            )
            try:
                output = validate_upstream_answer(
                    result,
                    available_evidence_ids=available_evidence_ids,
                    available_display_refs=available_display_refs,
                )
            except CoordinatorResponseError as first_error:
                logger.warning(
                    "Upstream answer violated schema; requesting one LLM "
                    "repair: %s",
                    first_error,
                )
                result = invoke(
                    upstream_answer_model,
                    _repair_messages(
                        messages,
                        result,
                        _UPSTREAM_ANSWER_REPAIR_PROMPT
                        + "\nОшибка: "
                        + str(first_error)
                        + evidence_context,
                    ),
                    stage="upstream",
                )
                output = validate_upstream_answer(
                    result,
                    available_evidence_ids=available_evidence_ids,
                    available_display_refs=available_display_refs,
                )
            return result, output

        def data_request_update(problem: str) -> Dict[str, Any]:
            if state["cycle"] >= COORDINATOR_MAX_CYCLES:
                raise CoordinatorResponseError(
                    "Последний upstream-цикл не может запросить новые данные."
                )
            logger.info(
                "Upstream requests clean data cycle=%s problem=%s",
                state["cycle"] + 1,
                problem,
            )
            return {
                "cycle": state["cycle"] + 1,
                "plan": [],
                "next_step": 0,
                "worker_runs": [],
                "upstream_problem": problem,
                "upstream_output": None,
                "final_answer": None,
                "selected_display_refs": [],
            }

        _, decision = invoke_decision(decision_messages)
        if decision.decision == "reroute":
            return data_request_update(decision.problem)

        answer_payload = dict(upstream_payload)
        if decision.problem:
            answer_payload["data_problem"] = decision.problem
        answer_messages: List[BaseMessage] = [
            SystemMessage(
                content="\n\n".join(
                    part
                    for part in (
                        _UPSTREAM_ANSWER_PROMPT,
                        _UPSTREAM_ANALYSIS_CONTEXT,
                    )
                    if part
                )
            ),
            HumanMessage(
                content=json.dumps(answer_payload, ensure_ascii=False)
            ),
        ]
        _, evidence = invoke_answer(answer_messages)

        upstream_output = evidence.model_dump()
        selected_display_refs = [
            available_display_refs[evidence_id]
            for evidence_id in evidence.display_evidence_ids
        ]
        record_upstream_output(upstream_output)
        logger.info(
            "Upstream coordinator result: %s",
            json.dumps(upstream_output, ensure_ascii=False)[:8000],
        )
        return {
            "upstream_output": upstream_output,
            "final_answer": evidence.answer,
            "selected_display_refs": selected_display_refs,
        }

    def route_after_worker(
        state: CoordinatorGraphState,
    ) -> Literal["worker", "upstream"]:
        if state["next_step"] < len(state["plan"]):
            return "worker"
        return "upstream"

    def route_after_upstream(
        state: CoordinatorGraphState,
    ) -> Literal["downstream_plan", "end"]:
        if str(state.get("final_answer") or "").strip():
            return "end"
        if state.get("upstream_problem") is not None:
            return "downstream_plan"
        raise CoordinatorResponseError(
            "Upstream не вернул ни ответ, ни запрос дополнительных данных."
        )

    graph = StateGraph(CoordinatorGraphState)
    graph.add_node("downstream_plan", downstream_plan_node)
    graph.add_node("worker", worker_node)
    graph.add_node("upstream", upstream_node)
    graph.add_edge(START, "downstream_plan")
    graph.add_edge("downstream_plan", "worker")
    graph.add_conditional_edges(
        "worker",
        route_after_worker,
        {
            "worker": "worker",
            "upstream": "upstream",
        },
    )
    graph.add_conditional_edges(
        "upstream",
        route_after_upstream,
        {
            "downstream_plan": "downstream_plan",
            "end": END,
        },
    )
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
    collected_display_refs: List[str] = []
    graph = build_coordinator_graph(
        chat_model,
        callbacks=callbacks,
        collected_display_refs=collected_display_refs,
    )
    initial_state: CoordinatorGraphState = {
        "task": clean_task,
        "context": clean_context,
        "cycle": 1,
        "plan": [],
        "next_step": 0,
        "worker_runs": [],
        "upstream_problem": None,
        "upstream_output": None,
        "final_answer": None,
        "selected_display_refs": [],
    }
    config = {
        "recursion_limit": (
            COORDINATOR_MAX_CYCLES * (COORDINATOR_MAX_WORKERS + 2) + 4
        ),
        "run_name": "worker_coordinator",
    }

    with (
        saved_result_store_scope(),
        langfuse_trace_context(
            trace_name="worker_coordinator",
            metadata={
                "max_workers": COORDINATOR_MAX_WORKERS,
                "max_cycles": COORDINATOR_MAX_CYCLES,
            },
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
                ref for ref in collected_display_refs if ref not in selected_set
            ]
            if unselected_refs:
                discard_worker_display_refs(unselected_refs)
            return CoordinatorAnswer(
                answer=final_answer,
                display_refs=selected_refs,
            )
        except Exception:
            if collected_display_refs:
                discard_worker_display_refs(collected_display_refs)
            raise


__all__ = [
    "COORDINATOR_MAX_CYCLES",
    "COORDINATOR_MAX_WORKERS",
    "COORDINATOR_CONTEXT_MAX_CHARS",
    "PlanStep",
    "UpstreamOutput",
    "UpstreamDecision",
    "CoordinatorAnswer",
    "CoordinatorGraphState",
    "CoordinatorResponseError",
    "WorkerPlan",
    "build_coordinator_graph",
    "coordinator_chat",
]
