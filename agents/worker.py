"""Isolated generic read-only worker experiment.

The public worker contract accepts only one self-contained task. Tool, skill,
and schema selection and the planner/tool/observer loop remain internal to the
worker.
The worker exposes its concise answer, a bounded internal cycle trace, and
opaque references to every successful tool result; a higher-level coordinator
selects UI results.
"""

from __future__ import annotations

import json
import logging
from threading import Lock
from typing import Any, Dict, List, Sequence, Tuple
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .agent import build_chat_system_prompt, chat_model
from .chat_graph import (
    DEFAULT_TOOL_MESSAGE_PREVIEW_CHARS,
    WorkerCycleTrace,
    WorkerDisplayItem,
    ensure_worker_tools,
    run_worker_graph,
)
from .observability import get_callback_handler
from .run_metrics import (
    get_run_metrics_callback,
    record_worker_observation,
    record_worker_route,
    record_worker_task,
)
from .tools import get_tools, load_schemas, load_skills
from .tools.saved_results import (
    SavedResultDescriptor,
    bind_saved_result_schemas,
    get_active_saved_result_store,
)
from .tools.routing import select_chat_route

logger = logging.getLogger(__name__)

WORKER_MAX_STEPS = 5
WORKER_MAX_REROUTES = 5
WORKER_TOOL_MESSAGE_PREVIEW_CHARS = DEFAULT_TOOL_MESSAGE_PREVIEW_CHARS
_REROUTE_FEEDBACK_MAX_CHARS = 4000
_DISPLAY_RESULTS: Dict[str, WorkerDisplayItem] = {}
_DISPLAY_RESULTS_LOCK = Lock()


class WorkerResultRef(BaseModel):
    """Opaque full-result reference plus minimal coordinator metadata."""

    model_config = ConfigDict(extra="forbid")

    ref: str
    name: str


class WorkerAnswer(BaseModel):
    """Worker output without full tool results."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    result_refs: List[WorkerResultRef] = Field(default_factory=list)
    saved_results: List[SavedResultDescriptor] = Field(default_factory=list)
    cycle_history: List[WorkerCycleTrace] = Field(default_factory=list)
    goal_satisfied: bool = True
    problem: str | None = None


def _store_display_items(items: Sequence[WorkerDisplayItem]) -> List[WorkerResultRef]:
    refs: List[WorkerResultRef] = []
    with _DISPLAY_RESULTS_LOCK:
        for item in items:
            ref = uuid4().hex
            _DISPLAY_RESULTS[ref] = item
            refs.append(WorkerResultRef(ref=ref, name=item.name))
    return refs


def resolve_worker_display_refs(refs: Sequence[str]) -> List[WorkerDisplayItem]:
    """Consume selected full results after coordination has finished."""
    with _DISPLAY_RESULTS_LOCK:
        return [
            item
            for ref in refs
            if (item := _DISPLAY_RESULTS.pop(ref, None)) is not None
        ]


def discard_worker_result_refs(refs: Sequence[str]) -> None:
    """Discard full results that the coordinator did not select for the UI."""
    with _DISPLAY_RESULTS_LOCK:
        for ref in refs:
            _DISPLAY_RESULTS.pop(ref, None)


def _planner_reroute_feedback(context: Dict[str, Any]) -> str:
    payload = {
        "problem": str(context.get("problem") or "").strip(),
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    if len(serialized) > _REROUTE_FEEDBACK_MAX_CHARS:
        serialized = serialized[: _REROUTE_FEEDBACK_MAX_CHARS - 1] + "…"
    return (
        "Повторный запуск worker после неуспешной попытки. Ниже только "
        "диагностическая выжимка предыдущего запуска, а не новая task. "
        "Учти её при первом следующем вызове data tool и исправь описанную "
        "проблему с помощью расширенной палитры.\n"
        f"<reroute_feedback>{serialized}</reroute_feedback>"
    )


def worker_chat(task: str) -> WorkerAnswer:
    """Execute one self-contained task in an isolated generic worker."""
    clean_task = str(task or "").strip()
    if not clean_task:
        return WorkerAnswer(
            answer="Подзадача воркера не должна быть пустой.",
            result_refs=[],
            goal_satisfied=False,
            problem="Worker получил пустую task.",
        )

    record_worker_task(clean_task)

    callback = get_callback_handler()
    callbacks = [callback] if callback is not None else []
    metrics_callback = get_run_metrics_callback()
    if metrics_callback is not None and metrics_callback not in callbacks:
        callbacks.append(metrics_callback)
    saved_store = get_active_saved_result_store()
    initial_saved_refs = {
        item.result_ref for item in saved_store.descriptors()
    } if saved_store is not None else set()

    def newly_saved_results() -> List[SavedResultDescriptor]:
        if saved_store is None:
            return []
        return [
            item
            for item in saved_store.descriptors()
            if item.result_ref not in initial_saved_refs
        ]

    attempted_palettes: List[Tuple[str, ...]] = []
    cycle_history: List[WorkerCycleTrace] = []
    reroute_context: Dict[str, Any] | None = None
    reroute_count = 0

    while True:
        available_tools = bind_saved_result_schemas(get_tools(), clean_task)
        route_kwargs: Dict[str, Any] = {
            "model": chat_model,
            "available_tools": available_tools,
            "callbacks": callbacks,
        }
        if reroute_context is not None:
            route_kwargs["reroute_context"] = reroute_context
        route = select_chat_route(clean_task, **route_kwargs)
        palette = tuple(sorted(dict.fromkeys(route.tools)))
        attempted_palettes.append(palette)
        selected_names = set(route.tools)
        selected_tools = tuple(
            item for item in available_tools if item.name in selected_names
        )
        worker_tools = ensure_worker_tools(selected_tools)
        selected_skills = load_skills(tuple(route.skills))
        selected_schemas = load_schemas(tuple(route.schemas))
        reroute_problem = (
            str(reroute_context.get("problem") or "").strip()
            if reroute_context is not None
            else ""
        )
        record_worker_route(
            worker_task=clean_task,
            routing_attempt=reroute_count + 1,
            tools=[tool.name for tool in worker_tools],
            skills=list(route.skills),
            schemas=list(route.schemas),
            problem=reroute_problem or None,
        )

        logger.info(
            "Worker route: %s",
            json.dumps(
                {
                    "task": clean_task,
                    "routing_attempt": reroute_count + 1,
                    "tools": [tool.name for tool in selected_tools],
                    "worker_tools": [tool.name for tool in worker_tools],
                    "skills": list(route.skills),
                    "schemas": list(route.schemas),
                    "problem": reroute_problem or None,
                },
                ensure_ascii=False,
            )[:8000],
        )

        system_prompt = build_chat_system_prompt(
            selected_skills,
            selected_schemas,
        )
        if reroute_context is not None:
            system_prompt = (
                f"{system_prompt}\n\n"
                f"{_planner_reroute_feedback(reroute_context)}"
            )

        graph_result = run_worker_graph(
            task=clean_task,
            system_prompt=system_prompt,
            model=chat_model,
            tools=worker_tools,
            max_steps=WORKER_MAX_STEPS,
            tool_message_preview_chars=WORKER_TOOL_MESSAGE_PREVIEW_CHARS,
            callbacks=callbacks,
        )
        first_cycle_number = len(cycle_history) + 1
        new_cycles = [
            cycle.model_copy(
                update={
                    "cycle": first_cycle_number + index,
                    "routing_attempt": reroute_count + 1,
                }
            )
            for index, cycle in enumerate(graph_result.cycle_history)
        ]
        cycle_history.extend(new_cycles)
        for cycle in new_cycles:
            observation_payload = cycle.observation.model_dump()
            record_worker_observation(
                worker_task=clean_task,
                cycle=cycle.cycle,
                routing_attempt=cycle.routing_attempt,
                observation=observation_payload,
                analysis=(
                    cycle.analysis.model_dump()
                    if cycle.analysis is not None
                    else None
                ),
            )
            logger.info(
                "Worker observation: %s",
                json.dumps(
                    {
                        "task": clean_task,
                        "cycle": cycle.cycle,
                        "routing_attempt": cycle.routing_attempt,
                        "analysis": (
                            cycle.analysis.model_dump()
                            if cycle.analysis is not None
                            else None
                        ),
                        "observation": observation_payload,
                    },
                    ensure_ascii=False,
                )[:8000],
            )
        if not graph_result.reroute_required:
            return WorkerAnswer(
                answer=graph_result.answer,
                result_refs=_store_display_items(graph_result.display_items),
                saved_results=newly_saved_results(),
                cycle_history=cycle_history,
                goal_satisfied=graph_result.goal_satisfied,
                problem=graph_result.problem,
            )

        if reroute_count >= WORKER_MAX_REROUTES:
            return WorkerAnswer(
                answer=str(graph_result.problem or ""),
                result_refs=[],
                saved_results=newly_saved_results(),
                cycle_history=cycle_history,
                goal_satisfied=False,
                problem=graph_result.problem,
            )

        reroute_count += 1
        reroute_context = {
            "problem": str(graph_result.problem or ""),
            "previous_tool_palettes": [
                list(item) for item in attempted_palettes
            ],
            "attempt": reroute_count,
        }
        logger.info(
            "Worker returns to tool-router: attempt=%s problem=%s",
            reroute_count,
            graph_result.problem,
        )


__all__ = [
    "WORKER_MAX_STEPS",
    "WORKER_MAX_REROUTES",
    "WORKER_TOOL_MESSAGE_PREVIEW_CHARS",
    "WorkerAnswer",
    "WorkerResultRef",
    "discard_worker_result_refs",
    "resolve_worker_display_refs",
    "worker_chat",
]
