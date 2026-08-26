"""Isolated generic read-only worker experiment.

The public worker contract accepts one self-contained task. Tool, skill and
schema selection and the planner/tool/observer loop remain internal to the
worker.
The worker exposes one typed outcome with facts and bounded evidence artifacts;
later workers receive only lazy run-scoped result references, while a higher-
level coordinator performs the analysis and selects UI results.
"""

from __future__ import annotations

import json
import logging
from threading import Lock
from typing import Any, Dict, List, Sequence, Tuple
from uuid import uuid4

from .agent import build_chat_system_prompt, chat_model
from .chat_graph import (
    DEFAULT_TOOL_MESSAGE_PREVIEW_CHARS,
    WorkerCycleTrace,
    WorkerDisplayItem,
    WorkerResponseError,
    ensure_worker_tools,
    run_worker_graph,
)
from .contracts import (
    EvidenceArtifact,
    PreviousResultReference,
    SavedResultDescriptor,
    WorkerOutcome,
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
    bind_saved_result_schemas,
    get_active_saved_result_store,
)
from .tools.routing import select_chat_route

logger = logging.getLogger(__name__)

WORKER_MAX_STEPS = 5
WORKER_MAX_REROUTES = 5
WORKER_TOOL_MESSAGE_PREVIEW_CHARS = DEFAULT_TOOL_MESSAGE_PREVIEW_CHARS
_REROUTE_FEEDBACK_MAX_CHARS = 4000
_TOOL_ARGUMENTS_MAX_CHARS = 2000
_HANDOFF_DESCRIPTION_MAX_CHARS = 600
_READ_PREVIOUS_RESULT_TOOL_NAME = "read_previous_result"
_DISPLAY_RESULTS: Dict[str, WorkerDisplayItem] = {}
_DISPLAY_RESULTS_LOCK = Lock()


def _compact_tool_arguments(
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        serialized = json.dumps(
            arguments,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        serialized = str(arguments)
    if len(serialized) <= _TOOL_ARGUMENTS_MAX_CHARS:
        return dict(arguments)
    marker = "… [аргументы обрезаны]"
    preview = serialized[: _TOOL_ARGUMENTS_MAX_CHARS - len(marker)].rstrip()
    return {
        "_truncated": True,
        "json_preview": preview + marker,
    }


def _store_evidence_items(
    items: Sequence[WorkerDisplayItem],
    datasets: Sequence[SavedResultDescriptor],
) -> List[EvidenceArtifact]:
    artifacts: List[EvidenceArtifact] = []
    dataset_by_tool_call = {
        str(item.source_tool_call_id or ""): item.result_ref
        for item in datasets
        if str(item.source_tool_call_id or "").strip()
    }
    with _DISPLAY_RESULTS_LOCK:
        for item in items:
            display_ref = None
            if item.name != _READ_PREVIOUS_RESULT_TOOL_NAME:
                display_ref = uuid4().hex
                _DISPLAY_RESULTS[display_ref] = item
            artifacts.append(
                EvidenceArtifact(
                    evidence_id=(
                        item.evidence_id or f"evidence_{uuid4().hex}"
                    ),
                    tool_name=item.name,
                    compact_args=_compact_tool_arguments(item.arguments),
                    preview=item.preview,
                    truncated=item.truncated,
                    display_ref=display_ref,
                    dataset_ref=dataset_by_tool_call.get(item.tool_call_id),
                )
            )
    return artifacts


def _handoff_description(
    item: WorkerDisplayItem,
) -> str:
    """Build a deterministic label from the executed tool call only."""
    compact_args = _compact_tool_arguments(item.arguments)
    serialized_args = json.dumps(
        compact_args,
        ensure_ascii=False,
        default=str,
        sort_keys=True,
        separators=(",", ":"),
    )
    description = f"{item.name}: args={serialized_args}"
    if len(description) <= _HANDOFF_DESCRIPTION_MAX_CHARS:
        return description
    marker = "…"
    return (
        description[: _HANDOFF_DESCRIPTION_MAX_CHARS - len(marker)].rstrip()
        + marker
    )


def _register_previous_results(
    items: Sequence[WorkerDisplayItem],
    datasets: Sequence[SavedResultDescriptor],
) -> List[PreviousResultReference]:
    """Persist accepted full results and return only opaque refs for handoff."""
    store = get_active_saved_result_store()
    if store is None:
        return []
    dataset_by_tool_call = {
        str(item.source_tool_call_id or ""): item.result_ref
        for item in datasets
        if str(item.source_tool_call_id or "").strip()
    }
    return [
        store.register_previous_result(
            source_tool=item.name,
            source_tool_call_id=item.tool_call_id,
            content=item.content,
            description=_handoff_description(item),
            dataset_ref=dataset_by_tool_call.get(item.tool_call_id),
        )
        for item in items
        if item.name != _READ_PREVIOUS_RESULT_TOOL_NAME
    ]


def resolve_worker_display_refs(refs: Sequence[str]) -> List[WorkerDisplayItem]:
    """Consume selected full results after coordination has finished."""
    with _DISPLAY_RESULTS_LOCK:
        return [
            item
            for ref in refs
            if (item := _DISPLAY_RESULTS.pop(ref, None)) is not None
        ]


def discard_worker_display_refs(refs: Sequence[str]) -> None:
    """Discard full results that the coordinator did not select for the UI."""
    with _DISPLAY_RESULTS_LOCK:
        for ref in refs:
            _DISPLAY_RESULTS.pop(ref, None)


def _planner_reroute_feedback(context: Dict[str, Any]) -> str:
    payload = {
        "gap": str(context.get("gap") or "").strip(),
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


def _final_outcome_summary(
    answer: Any,
    *,
    internal_gap: Any = None,
) -> str:
    """Fold internal failure feedback into the sole public summary field."""
    clean_answer = str(answer or "").strip()
    clean_gap = str(internal_gap or "").strip()
    if not clean_gap:
        return clean_answer or "Worker завершился без текстовой выжимки."
    if not clean_answer:
        return clean_gap
    if clean_gap.casefold() in clean_answer.casefold():
        return clean_answer
    return f"{clean_answer}\nПричина незавершённости: {clean_gap}"


def worker_chat(
    task: str,
) -> WorkerOutcome:
    """Execute one self-contained task in an isolated generic worker."""
    clean_task = str(task or "").strip()
    if not clean_task:
        return WorkerOutcome(
            summary="Worker получил пустую task.",
        )
    worker_request = clean_task

    record_worker_task(worker_request)

    callback = get_callback_handler()
    callbacks = [callback] if callback is not None else []
    metrics_callback = get_run_metrics_callback()
    if metrics_callback is not None and metrics_callback not in callbacks:
        callbacks.append(metrics_callback)
    saved_store = get_active_saved_result_store()
    initial_saved_refs = {
        item.result_ref for item in saved_store.descriptors()
    } if saved_store is not None else set()

    def newly_saved_datasets(
        accepted_tool_call_ids: Sequence[str] | None = None,
    ) -> List[SavedResultDescriptor]:
        if saved_store is None:
            return []
        descriptors = [
            item
            for item in saved_store.descriptors()
            if item.result_ref not in initial_saved_refs
        ]
        if accepted_tool_call_ids is None:
            return descriptors
        accepted_ids = set(accepted_tool_call_ids)
        return [
            item
            for item in descriptors
            if item.source_tool_call_id in accepted_ids
        ]

    attempted_palettes: List[Tuple[str, ...]] = []
    cycle_history: List[WorkerCycleTrace] = []
    reroute_context: Dict[str, Any] | None = None
    reroute_count = 0

    while True:
        available_tools = bind_saved_result_schemas(get_tools(), worker_request)
        route_kwargs: Dict[str, Any] = {
            "model": chat_model,
            "available_tools": available_tools,
            "callbacks": callbacks,
        }
        if reroute_context is not None:
            route_kwargs["reroute_context"] = reroute_context
        route = select_chat_route(worker_request, **route_kwargs)
        palette = tuple(sorted(dict.fromkeys(route.tools)))
        attempted_palettes.append(palette)
        selected_names = set(route.tools)
        selected_tools = tuple(
            item for item in available_tools if item.name in selected_names
        )
        worker_tools = ensure_worker_tools(selected_tools)
        selected_skills = load_skills(tuple(route.skills))
        selected_schemas = load_schemas(tuple(route.schemas))
        reroute_gap = (
            str(reroute_context.get("gap") or "").strip()
            if reroute_context is not None
            else ""
        )
        record_worker_route(
            worker_task=clean_task,
            routing_attempt=reroute_count + 1,
            tools=[tool.name for tool in worker_tools],
            skills=list(route.skills),
            schemas=list(route.schemas),
            gap=reroute_gap or None,
        )

        logger.info(
            "Worker route: %s",
            json.dumps(
                {
                    "task": worker_request,
                    "routing_attempt": reroute_count + 1,
                    "tools": [tool.name for tool in selected_tools],
                    "worker_tools": [tool.name for tool in worker_tools],
                    "skills": list(route.skills),
                    "schemas": list(route.schemas),
                    "gap": reroute_gap or None,
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

        try:
            graph_result = run_worker_graph(
                task=worker_request,
                system_prompt=system_prompt,
                model=chat_model,
                tools=worker_tools,
                max_steps=WORKER_MAX_STEPS,
                tool_message_preview_chars=WORKER_TOOL_MESSAGE_PREVIEW_CHARS,
                callbacks=callbacks,
            )
        except WorkerResponseError as exc:
            logger.warning("Worker contract failed: %s", exc)
            return WorkerOutcome(summary=str(exc))
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
                worker_task=worker_request,
                cycle=cycle.cycle,
                routing_attempt=cycle.routing_attempt,
                observation=observation_payload,
            )
            logger.info(
                "Worker observation: %s",
                json.dumps(
                    {
                        "task": worker_request,
                        "cycle": cycle.cycle,
                        "routing_attempt": cycle.routing_attempt,
                        "observation": observation_payload,
                    },
                    ensure_ascii=False,
                )[:8000],
            )
        if graph_result.status != "reroute":
            datasets = newly_saved_datasets(
                graph_result.accepted_tool_call_ids
            )
            summary = _final_outcome_summary(
                graph_result.answer,
                internal_gap=graph_result.gap,
            )
            previous_results = _register_previous_results(
                graph_result.display_items,
                datasets,
            )
            return WorkerOutcome(
                summary=summary,
                facts=list(graph_result.facts),
                evidence=_store_evidence_items(
                    graph_result.display_items,
                    datasets,
                ),
                datasets=datasets,
                previous_results=previous_results,
            )

        if reroute_count >= WORKER_MAX_REROUTES:
            return WorkerOutcome(
                summary=str(
                    graph_result.gap
                    or "Worker исчерпал лимит reroute без результата."
                ),
            )

        reroute_count += 1
        reroute_context = {
            "gap": str(graph_result.gap or ""),
            "previous_tool_palettes": [
                list(item) for item in attempted_palettes
            ],
            "attempt": reroute_count,
        }
        logger.info(
            "Worker returns to tool-router: attempt=%s gap=%s",
            reroute_count,
            graph_result.gap,
        )


__all__ = [
    "WORKER_MAX_STEPS",
    "WORKER_MAX_REROUTES",
    "WORKER_TOOL_MESSAGE_PREVIEW_CHARS",
    "EvidenceArtifact",
    "WorkerOutcome",
    "discard_worker_display_refs",
    "resolve_worker_display_refs",
    "worker_chat",
]
