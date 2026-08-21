"""Opt-in passive metrics for real agent runs and live scenarios."""

from __future__ import annotations

import os
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from time import perf_counter
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, ConfigDict, Field


_METRICS_REGISTRY_LIMIT = 100
_VALUE_PREVIEW_CHARS = 2000
_ACTIVE_RUN: ContextVar[Optional["_RunCollector"]] = ContextVar(
    "agent_run_metrics",
    default=None,
)
_COMPLETED_RUNS: "OrderedDict[str, AgentRunMetrics]" = OrderedDict()
_COMPLETED_RUNS_LOCK = Lock()


class LLMCallMetric(BaseModel):
    """One real model request observed through LangChain callbacks."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    model: str = ""
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    has_error: bool = False


class ToolCallMetric(BaseModel):
    """One executed data-tool call without its potentially large result."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    name: str
    input_preview: str = ""
    elapsed_seconds: float = 0.0
    has_error: bool = False


class ObservationMetric(BaseModel):
    """One structured worker observation retained without full tool results."""

    model_config = ConfigDict(extra="forbid")

    worker_task: str
    cycle: int
    routing_attempt: int
    summary: str
    goal_satisfied: bool
    mismatches: List[str] = Field(default_factory=list)
    has_error: bool = False
    important_facts: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    reroute_required: bool = False
    reroute_reason: Optional[str] = None


class WorkerRouteMetric(BaseModel):
    """One router selection for a worker task and reroute attempt."""

    model_config = ConfigDict(extra="forbid")

    worker_task: str
    routing_attempt: int
    tools: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    schemas: List[str] = Field(default_factory=list)
    reroute_reason: Optional[str] = None


class AgentRunMetrics(BaseModel):
    """Completed metrics snapshot retained by session id for test inspection."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    elapsed_seconds: float
    llm_calls: List[LLMCallMetric] = Field(default_factory=list)
    tool_calls: List[ToolCallMetric] = Field(default_factory=list)
    worker_tasks: List[str] = Field(default_factory=list)
    coordinator_plan: List[Dict[str, Any]] = Field(default_factory=list)
    worker_routes: List[WorkerRouteMetric] = Field(default_factory=list)
    observations: List[ObservationMetric] = Field(default_factory=list)
    coordinate_result: Optional[Dict[str, Any]] = None
    aggregate_result: Optional[str] = None
    display_tools: List[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    error: Optional[str] = None


def _clip(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) <= _VALUE_PREVIEW_CHARS:
        return text
    return text[: _VALUE_PREVIEW_CHARS - 1].rstrip() + "…"


def _metrics_enabled() -> bool:
    for name in ("AGENT_RUN_METRICS_ENABLED", "RUN_LIVE_AGENT_SCENARIOS"):
        if (os.getenv(name) or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True
    return False


def _usage_values(response: Any) -> tuple[int, int, int, int]:
    llm_output = getattr(response, "llm_output", None)
    usage: Mapping[str, Any] = {}
    if isinstance(llm_output, Mapping):
        candidate = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(candidate, Mapping):
            usage = candidate

    if not usage:
        for generation_group in getattr(response, "generations", None) or []:
            for generation in (
                generation_group
                if isinstance(generation_group, list)
                else [generation_group]
            ):
                message_usage = getattr(
                    getattr(generation, "message", None),
                    "usage_metadata",
                    None,
                )
                if isinstance(message_usage, Mapping):
                    usage = message_usage
                    break
            if usage:
                break

    input_tokens = int(
        usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    )
    output_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    total_tokens = int(usage.get("total_tokens") or 0)
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    cache_read_tokens = int(usage.get("precached_prompt_tokens") or 0)
    details = usage.get("input_token_details")
    if isinstance(details, Mapping):
        cache_read_tokens = int(details.get("cache_read") or cache_read_tokens)
    return input_tokens, output_tokens, total_tokens, cache_read_tokens


class _RunCollector:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.started_at = perf_counter()
        self.lock = Lock()
        self.llm_calls: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.tool_calls: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.worker_tasks: List[str] = []
        self.coordinator_plan: List[Dict[str, Any]] = []
        self.worker_routes: List[WorkerRouteMetric] = []
        self.observations: List[ObservationMetric] = []
        self.coordinate_result: Optional[Dict[str, Any]] = None
        self.aggregate_result: Optional[str] = None
        self.display_tools: List[str] = []
        self.error: Optional[str] = None

    def start_llm(self, run_id: Any, serialized: Any) -> None:
        key = str(run_id)
        model = ""
        if isinstance(serialized, Mapping):
            kwargs = serialized.get("kwargs")
            if isinstance(kwargs, Mapping):
                model = str(
                    kwargs.get("model") or kwargs.get("model_name") or ""
                )
            if not model:
                identifier = serialized.get("id")
                if isinstance(identifier, list) and identifier:
                    model = str(identifier[-1])
        with self.lock:
            self.llm_calls.setdefault(
                key,
                {
                    "run_id": key,
                    "model": model,
                    "started_at": perf_counter(),
                },
            )

    def finish_llm(self, run_id: Any, response: Any, *, error: bool) -> None:
        key = str(run_id)
        with self.lock:
            item = self.llm_calls.setdefault(
                key,
                {"run_id": key, "model": "", "started_at": perf_counter()},
            )
            item["elapsed_seconds"] = max(
                0.0,
                perf_counter() - float(item.get("started_at") or perf_counter()),
            )
            item["has_error"] = error
            if not error:
                (
                    item["input_tokens"],
                    item["output_tokens"],
                    item["total_tokens"],
                    item["cache_read_tokens"],
                ) = _usage_values(response)

    def start_tool(self, run_id: Any, serialized: Any, input_value: Any) -> None:
        key = str(run_id)
        name = "unknown_tool"
        if isinstance(serialized, Mapping):
            name = str(serialized.get("name") or name)
        with self.lock:
            self.tool_calls.setdefault(
                key,
                {
                    "run_id": key,
                    "name": name,
                    "input_preview": _clip(input_value),
                    "started_at": perf_counter(),
                },
            )

    def finish_tool(self, run_id: Any, *, error: bool) -> None:
        key = str(run_id)
        with self.lock:
            item = self.tool_calls.get(key)
            if item is None:
                return
            item["elapsed_seconds"] = max(
                0.0,
                perf_counter() - float(item.get("started_at") or perf_counter()),
            )
            item["has_error"] = error

    def snapshot(self) -> AgentRunMetrics:
        with self.lock:
            llm_calls = [
                LLMCallMetric.model_validate(
                    {
                        key: value
                        for key, value in item.items()
                        if key != "started_at"
                    }
                )
                for item in self.llm_calls.values()
            ]
            tool_calls = [
                ToolCallMetric.model_validate(
                    {
                        key: value
                        for key, value in item.items()
                        if key != "started_at"
                    }
                )
                for item in self.tool_calls.values()
            ]
            return AgentRunMetrics(
                session_id=self.session_id,
                elapsed_seconds=max(0.0, perf_counter() - self.started_at),
                llm_calls=llm_calls,
                tool_calls=tool_calls,
                worker_tasks=list(self.worker_tasks),
                coordinator_plan=[dict(item) for item in self.coordinator_plan],
                worker_routes=list(self.worker_routes),
                observations=list(self.observations),
                coordinate_result=(
                    dict(self.coordinate_result)
                    if self.coordinate_result is not None
                    else None
                ),
                aggregate_result=self.aggregate_result,
                display_tools=list(self.display_tools),
                input_tokens=sum(item.input_tokens for item in llm_calls),
                output_tokens=sum(item.output_tokens for item in llm_calls),
                total_tokens=sum(item.total_tokens for item in llm_calls),
                cache_read_tokens=sum(item.cache_read_tokens for item in llm_calls),
                error=self.error,
            )


class _RunMetricsCallback(BaseCallbackHandler):
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        del messages, kwargs
        if collector := _ACTIVE_RUN.get():
            collector.start_llm(run_id, serialized)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        del prompts, kwargs
        if collector := _ACTIVE_RUN.get():
            collector.start_llm(run_id, serialized)

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        del kwargs
        if collector := _ACTIVE_RUN.get():
            collector.finish_llm(run_id, response, error=False)

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        del error, kwargs
        if collector := _ACTIVE_RUN.get():
            collector.finish_llm(run_id, None, error=True)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if collector := _ACTIVE_RUN.get():
            collector.start_tool(run_id, serialized, input_str)

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        del output, kwargs
        if collector := _ACTIVE_RUN.get():
            collector.finish_tool(run_id, error=False)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        del error, kwargs
        if collector := _ACTIVE_RUN.get():
            collector.finish_tool(run_id, error=True)


_CALLBACK = _RunMetricsCallback()


@contextmanager
def capture_agent_run(session_id: Optional[str]) -> Iterator[None]:
    """Capture one real supervisor run when metrics are explicitly enabled."""
    clean_session_id = str(session_id or "").strip()
    if not clean_session_id or not _metrics_enabled():
        yield
        return

    collector = _RunCollector(clean_session_id)
    token = _ACTIVE_RUN.set(collector)
    try:
        yield
    except Exception as exc:
        collector.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        snapshot = collector.snapshot()
        _ACTIVE_RUN.reset(token)
        with _COMPLETED_RUNS_LOCK:
            _COMPLETED_RUNS[clean_session_id] = snapshot
            _COMPLETED_RUNS.move_to_end(clean_session_id)
            while len(_COMPLETED_RUNS) > _METRICS_REGISTRY_LIMIT:
                _COMPLETED_RUNS.popitem(last=False)


def get_run_metrics_callback() -> Optional[BaseCallbackHandler]:
    """Return the passive callback only while a captured run is active."""
    return _CALLBACK if _ACTIVE_RUN.get() is not None else None


def record_worker_task(task: str) -> None:
    if collector := _ACTIVE_RUN.get():
        with collector.lock:
            collector.worker_tasks.append(_clip(task))


def record_coordinator_plan(steps: List[Dict[str, Any]]) -> None:
    if collector := _ACTIVE_RUN.get():
        with collector.lock:
            collector.coordinator_plan = [dict(item) for item in steps]


def record_worker_route(
    *,
    worker_task: str,
    routing_attempt: int,
    tools: List[str],
    skills: List[str],
    schemas: List[str],
    reroute_reason: Optional[str] = None,
) -> None:
    """Retain an already selected router palette for live diagnostics."""
    if collector := _ACTIVE_RUN.get():
        metric = WorkerRouteMetric(
            worker_task=_clip(worker_task),
            routing_attempt=max(1, int(routing_attempt)),
            tools=[str(item) for item in tools],
            skills=[str(item) for item in skills],
            schemas=[str(item) for item in schemas],
            reroute_reason=_clip(reroute_reason) or None,
        )
        with collector.lock:
            collector.worker_routes.append(metric)


def record_worker_observation(
    *,
    worker_task: str,
    cycle: int,
    routing_attempt: int,
    observation: Mapping[str, Any],
) -> None:
    """Retain a bounded structured observation for live-run diagnostics."""
    if collector := _ACTIVE_RUN.get():
        metric = ObservationMetric(
            worker_task=_clip(worker_task),
            cycle=max(1, int(cycle)),
            routing_attempt=max(1, int(routing_attempt)),
            summary=_clip(observation.get("summary")),
            goal_satisfied=bool(observation.get("goal_satisfied")),
            mismatches=[
                _clip(item) for item in observation.get("mismatches", [])
            ],
            has_error=bool(observation.get("has_error")),
            important_facts=[
                _clip(item)
                for item in observation.get("important_facts", [])
            ],
            limitations=[
                _clip(item) for item in observation.get("limitations", [])
            ],
            reroute_required=bool(observation.get("reroute_required")),
            reroute_reason=(
                _clip(observation.get("reroute_reason")) or None
            ),
        )
        with collector.lock:
            collector.observations.append(metric)


def record_coordinate_result(result: Mapping[str, Any]) -> None:
    """Retain the coordinator synthesis before final aggregation."""
    if collector := _ACTIVE_RUN.get():
        payload = {
            "answer": _clip(result.get("answer")),
            "display_result_keys": [
                str(item) for item in result.get("display_result_keys", [])
            ],
        }
        with collector.lock:
            collector.coordinate_result = payload


def record_aggregate_result(answer: str) -> None:
    """Retain the final answer produced by the isolated aggregator."""
    if collector := _ACTIVE_RUN.get():
        with collector.lock:
            collector.aggregate_result = _clip(answer)


def record_display_tools(names: List[str]) -> None:
    if collector := _ACTIVE_RUN.get():
        with collector.lock:
            collector.display_tools = [str(name) for name in names]


def consume_agent_run_metrics(session_id: str) -> Optional[AgentRunMetrics]:
    """Consume a completed metrics snapshot, primarily from live tests."""
    with _COMPLETED_RUNS_LOCK:
        return _COMPLETED_RUNS.pop(str(session_id), None)


__all__ = [
    "AgentRunMetrics",
    "LLMCallMetric",
    "ObservationMetric",
    "ToolCallMetric",
    "WorkerRouteMetric",
    "capture_agent_run",
    "consume_agent_run_metrics",
    "get_run_metrics_callback",
    "record_aggregate_result",
    "record_coordinate_result",
    "record_coordinator_plan",
    "record_display_tools",
    "record_worker_observation",
    "record_worker_route",
    "record_worker_task",
]
