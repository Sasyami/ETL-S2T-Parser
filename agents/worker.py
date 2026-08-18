"""Isolated generic read-only worker experiment.

The public worker contract accepts only one self-contained task. Tool and skill
selection and the planner/tool/observer loop remain internal to the worker.
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
    run_worker_graph,
)
from .observability import get_callback_handler
from .run_metrics import get_run_metrics_callback, record_worker_task
from .tools import get_tools, get_tools_for_names, load_skills
from .tools.routing import select_chat_route

logger = logging.getLogger(__name__)

WORKER_MAX_STEPS = 5
WORKER_MAX_REROUTES = 2
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
    cycle_history: List[WorkerCycleTrace] = Field(default_factory=list)
    goal_satisfied: bool = True
    mismatches: List[str] = Field(default_factory=list)


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
        "reason": str(context.get("reason") or "").strip(),
        "mismatches": [
            str(item).strip()
            for item in context.get("mismatches", [])
            if str(item).strip()
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    if len(serialized) > _REROUTE_FEEDBACK_MAX_CHARS:
        serialized = serialized[: _REROUTE_FEEDBACK_MAX_CHARS - 1] + "…"
    return (
        "Повторный запуск worker после неуспешной попытки. Ниже только "
        "диагностическая выжимка предыдущего запуска, а не новая task. "
        "Учти её при первом следующем вызове data tool и исправь перечисленные "
        "ошибки. Доступная палитра может совпадать с предыдущей.\n"
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
            mismatches=["Worker получил пустую task."],
        )

    record_worker_task(clean_task)

    callback = get_callback_handler()
    callbacks = [callback] if callback is not None else []
    metrics_callback = get_run_metrics_callback()
    if metrics_callback is not None and metrics_callback not in callbacks:
        callbacks.append(metrics_callback)
    available_tools = get_tools()
    attempted_palettes: List[Tuple[str, ...]] = []
    cycle_history: List[WorkerCycleTrace] = []
    reroute_context: Dict[str, Any] | None = None
    reroute_count = 0

    while True:
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
        selected_tools = get_tools_for_names(route.tools)
        selected_skills = load_skills(tuple(route.skills))

        logger.info(
            "Worker routed tools=%s skills=%s reroute=%s",
            [tool.name for tool in selected_tools],
            route.skills,
            reroute_count,
        )

        system_prompt = build_chat_system_prompt(
            selected_skills,
            [tool.name for tool in selected_tools],
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
            tools=selected_tools,
            max_steps=WORKER_MAX_STEPS,
            tool_message_preview_chars=WORKER_TOOL_MESSAGE_PREVIEW_CHARS,
            callbacks=callbacks,
        )
        first_cycle_number = len(cycle_history) + 1
        cycle_history.extend(
            [
                cycle.model_copy(
                    update={
                        "cycle": first_cycle_number + index,
                        "routing_attempt": reroute_count + 1,
                    }
                )
                for index, cycle in enumerate(graph_result.cycle_history)
            ]
        )
        if not graph_result.reroute_required:
            return WorkerAnswer(
                answer=graph_result.answer,
                result_refs=_store_display_items(graph_result.display_items),
                cycle_history=cycle_history,
                goal_satisfied=graph_result.goal_satisfied,
                mismatches=graph_result.mismatches,
            )

        reroute_reason = str(
            graph_result.reroute_reason
            or "Текущей палитры tools недостаточно для выполнения task."
        ).strip()
        if reroute_count >= WORKER_MAX_REROUTES:
            return WorkerAnswer(
                answer=reroute_reason,
                result_refs=[],
                cycle_history=cycle_history,
                goal_satisfied=False,
                mismatches=list(graph_result.mismatches),
            )

        reroute_count += 1
        reroute_context = {
            "reason": reroute_reason,
            "mismatches": list(graph_result.mismatches),
            "previous_tool_palettes": [
                list(item) for item in attempted_palettes
            ],
            "attempt": reroute_count,
        }
        logger.info(
            "Worker returns to tool-router: attempt=%s reason=%s",
            reroute_count,
            reroute_reason,
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
