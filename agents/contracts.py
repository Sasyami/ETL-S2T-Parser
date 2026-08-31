"""Typed contracts exchanged between workers and their coordinator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ObservationStatus = Literal["complete", "continue", "reroute"]
UpstreamAction = Literal["pass", "reroute"]
MAX_PLAN_STEPS = 8
WORKER_STABLE_CONTEXT_MARKER = "\n\nУстойчивые правила контекста:\n"
WORKER_PREVIOUS_RESULTS_MARKER = "\n\nРезультаты прошлых workers."
WORKER_OPERATION_EXECUTION_MARKER = "\n\nOperation-skill текущей задачи:\n"
WORKER_OPERATION_COMPLETENESS_MARKER = (
    "\n\nOperation-skill проверки полноты:\n"
)


class SavedResultColumn(BaseModel):
    """One physical column exposed by a run-scoped dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str
    sqlite_type: str


class PreviousResultSchema(BaseModel):
    """Compact table schema exposed with a lazy previous-result reference."""

    model_config = ConfigDict(extra="forbid")

    result_ref: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    truncated: bool = False
    columns: List[SavedResultColumn] = Field(default_factory=list)


class PreviousResultReference(BaseModel):
    """Minimal lazy reference passed from one worker to later workers."""

    model_config = ConfigDict(extra="forbid")

    result_id: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=600)
    result_schema: Optional[PreviousResultSchema] = None

    @field_validator("result_id", "description")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        clean_value = str(value or "").strip()
        if not clean_value:
            raise ValueError("previous result fields must not be blank")
        return clean_value


@dataclass(frozen=True)
class WorkerRequestParts:
    """Programmatic envelope around the current worker task."""

    current_task: str
    stable_context: str = ""
    operation_execution_context: str = ""
    operation_completeness_context: str = ""
    previous_results: Optional[List[PreviousResultReference]] = None


def parse_worker_request(value: Any) -> WorkerRequestParts:
    """Separate coordinator-owned context from the worker's current task."""
    full_text = str(value or "").strip()
    task_and_context = full_text
    previous_results: Optional[List[PreviousResultReference]] = None

    if WORKER_PREVIOUS_RESULTS_MARKER in task_and_context:
        task_and_context, handoff_text = task_and_context.split(
            WORKER_PREVIOUS_RESULTS_MARKER,
            1,
        )
        json_start = handoff_text.find("{")
        if json_start >= 0:
            try:
                decoded = json.loads(handoff_text[json_start:])
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, Mapping):
                raw_results = decoded.get("previous_results")
                if isinstance(raw_results, list):
                    try:
                        previous_results = [
                            PreviousResultReference.model_validate(item)
                            for item in raw_results
                        ]
                    except (TypeError, ValueError):
                        previous_results = None

    current_task = task_and_context
    stable_context = ""
    if WORKER_STABLE_CONTEXT_MARKER in current_task:
        current_task, stable_context = current_task.split(
            WORKER_STABLE_CONTEXT_MARKER,
            1,
        )

    operation_completeness_context = ""
    if WORKER_OPERATION_COMPLETENESS_MARKER in current_task:
        current_task, operation_completeness_context = current_task.split(
            WORKER_OPERATION_COMPLETENESS_MARKER,
            1,
        )

    operation_execution_context = ""
    if WORKER_OPERATION_EXECUTION_MARKER in current_task:
        current_task, operation_execution_context = current_task.split(
            WORKER_OPERATION_EXECUTION_MARKER,
            1,
        )

    return WorkerRequestParts(
        current_task=current_task.strip(),
        stable_context=stable_context.strip(),
        operation_execution_context=operation_execution_context.strip(),
        operation_completeness_context=operation_completeness_context.strip(),
        previous_results=previous_results,
    )


class EvidenceFact(BaseModel):
    """One compact fact with explicit provenance."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    evidence_ids: List[str] = Field(default_factory=list, max_length=20)

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        clean_value = str(value or "").strip()
        if not clean_value:
            raise ValueError("fact text must not be blank")
        return clean_value

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _clean_evidence_ids(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("evidence_ids must be an array")
        return list(
            dict.fromkeys(
                clean_item
                for item in value
                if (clean_item := str(item or "").strip())
            )
        )


class Observation(BaseModel):
    """Cumulative worker assessment over accepted evidence."""

    model_config = ConfigDict(extra="forbid")

    status: ObservationStatus = Field(
        description=(
            "complete — нужные исходные данные получены; continue — нужен "
            "ещё один вызов текущей палитры; reroute — нужна новая палитра."
        )
    )
    gap: Optional[str] = Field(
        default=None,
        description=(
            "Одна краткая консолидированная строка обо всех незакрытых "
            "требованиях исходной task из текущего результата и prior_state. "
            "Не перечисляй одну причину и её следствия как разные проблемы. "
            "Null только при status=complete."
        ),
    )
    accepted_tool_call_ids: List[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Накопительный список идентификаторов успешных tool results, "
            "которые семантически подтверждают исходную task."
        ),
    )
    facts: List[EvidenceFact] = Field(
        default_factory=list,
        description=(
            "Подтверждённые накопительные факты. Каждый факт явно ссылается "
            "на evidence_id из принятых tool results."
        ),
    )
    limitations: List[str] = Field(
        default_factory=list,
        description=(
            "Ограничения, неоднозначности и непроверенные предположения "
            "результата. Не выбирай следующий инструмент."
        ),
    )

    @field_validator("accepted_tool_call_ids", "limitations", mode="before")
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

    @field_validator("accepted_tool_call_ids")
    @classmethod
    def _deduplicate_accepted_tool_call_ids(
        cls,
        values: List[str],
    ) -> List[str]:
        return list(dict.fromkeys(values))

    @field_validator("gap", mode="before")
    @classmethod
    def _normalize_gap(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        clean_value = str(value).strip()
        if clean_value.casefold() == "null":
            return None
        return clean_value or None

    @model_validator(mode="after")
    def _gap_matches_status(self) -> "Observation":
        if self.status == "complete" and self.gap is not None:
            raise ValueError("gap must be null when status is complete")
        if self.status != "complete" and self.gap is None:
            raise ValueError(
                "gap must describe why observation is not complete"
            )
        return self


class EvidenceArtifact(BaseModel):
    """Accepted bounded tool evidence plus runtime-only references."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    compact_args: Dict[str, Any] = Field(default_factory=dict)
    preview: str = ""
    truncated: bool = False
    display_ref: Optional[str] = Field(default=None, exclude=True)
    dataset_ref: Optional[str] = Field(default=None, exclude=True)

    @field_validator("evidence_id", "tool_name")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        clean_value = str(value or "").strip()
        if not clean_value:
            raise ValueError("evidence identifiers must not be blank")
        return clean_value

    @field_validator("display_ref", "dataset_ref", mode="before")
    @classmethod
    def _normalize_optional_ref(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        clean_value = str(value).strip()
        return clean_value or None


class SavedResultDescriptor(BaseModel):
    """Internal run-scoped descriptor of a materialized tabular result."""

    model_config = ConfigDict(extra="forbid")

    result_ref: str
    source_tool: str
    source_tool_call_id: Optional[str] = Field(default=None, exclude=True)
    row_count: int = Field(ge=0)
    source_total: Optional[int] = Field(default=None, ge=0)
    truncated: bool = False
    columns: List[SavedResultColumn] = Field(default_factory=list)


class WorkerOutcome(BaseModel):
    """Final worker result for upstream plus runtime-only lazy references."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    facts: List[EvidenceFact] = Field(default_factory=list)
    evidence: List[EvidenceArtifact] = Field(default_factory=list)
    datasets: List[SavedResultDescriptor] = Field(
        default_factory=list,
        exclude=True,
    )
    previous_results: List[PreviousResultReference] = Field(
        default_factory=list,
        exclude=True,
    )

    @field_validator("summary")
    @classmethod
    def _strip_summary(cls, value: str) -> str:
        clean_value = str(value or "").strip()
        if not clean_value:
            raise ValueError("worker summary must not be blank")
        return clean_value

    @model_validator(mode="after")
    def _validate_status_and_provenance(self) -> "WorkerOutcome":
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("worker evidence_id values must be unique")
        known_evidence_ids = set(evidence_ids)
        unknown_fact_ids = sorted(
            {
                evidence_id
                for fact in self.facts
                for evidence_id in fact.evidence_ids
                if evidence_id not in known_evidence_ids
            }
        )
        if unknown_fact_ids:
            raise ValueError(
                "worker facts reference unknown evidence_id values: "
                + ", ".join(unknown_fact_ids)
            )

        dataset_refs = [item.result_ref for item in self.datasets]
        if len(dataset_refs) != len(set(dataset_refs)):
            raise ValueError("worker dataset refs must be unique")
        known_dataset_refs = set(dataset_refs)
        unknown_dataset_refs = sorted(
            {
                item.dataset_ref
                for item in self.evidence
                if item.dataset_ref is not None
                and item.dataset_ref not in known_dataset_refs
            }
        )
        if unknown_dataset_refs:
            raise ValueError(
                "worker evidence references unknown dataset_ref values: "
                + ", ".join(unknown_dataset_refs)
            )
        result_ids = [item.result_id for item in self.previous_results]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("worker previous result ids must be unique")
        return self

    def upstream_payload(self) -> Dict[str, Any]:
        """Serialize accepted evidence without worker interpretations."""
        return {
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "tool_name": item.tool_name,
                    "args": item.compact_args,
                    "preview": item.preview,
                    "truncated": item.truncated,
                    "display_id": (
                        item.evidence_id
                        if item.display_ref is not None
                        else None
                    ),
                }
                for item in self.evidence
            ],
        }

    def handoff_payload(self) -> Dict[str, Any]:
        """Serialize only lazy result references for later workers."""
        return {
            "previous_results": [
                item.model_dump(mode="json", exclude_none=True)
                for item in self.previous_results
            ],
        }


class PlanStep(BaseModel):
    """One ready-to-run worker task selected by the downstream planner."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(
        min_length=1,
        description=(
            "Готовая задача одного worker на получение исходных данных. "
            "Производный анализ выполняет upstream coordinator."
        ),
    )
    @field_validator("task")
    @classmethod
    def _strip_task(cls, value: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("task must not be blank")
        return clean_value

class WorkerPlan(BaseModel):
    """Validated linear sequence of downstream worker steps."""

    model_config = ConfigDict(extra="forbid")

    steps: List[PlanStep] = Field(
        min_length=1,
        max_length=MAX_PLAN_STEPS,
        description=(
            "Последовательность worker tasks на получение данных; следующая "
            "task может лениво использовать принятые результаты предыдущих."
        ),
    )


class UpstreamOutput(BaseModel):
    """Final answer and evidence selection assembled by upstream."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=1,
        description=(
            "Готовый пользовательский ответ, сформированный по фактическим "
            "preview результатов tools."
        ),
    )
    used_evidence_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Все evidence_id, факты которых использованы при формировании answer."
        ),
    )
    display_evidence_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Evidence_id результатов, которые нужно показать отдельно и "
            "которые поддерживают сформированный ответ."
        ),
    )

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

    @field_validator("used_evidence_ids", "display_evidence_ids")
    @classmethod
    def _clean_output_evidence_ids(cls, values: List[str]) -> List[str]:
        result: List[str] = []
        for value in values:
            clean_value = str(value or "").strip()
            if not clean_value:
                raise ValueError("evidence id lists must not contain blanks")
            if clean_value not in result:
                result.append(clean_value)
        return result

    @model_validator(mode="after")
    def _display_ids_are_used(self) -> "UpstreamOutput":
        unused_display_ids = sorted(
            set(self.display_evidence_ids) - set(self.used_evidence_ids)
        )
        if unused_display_ids:
            raise ValueError(
                "display_evidence_ids must be included in used_evidence_ids"
            )
        return self


class UpstreamDecision(BaseModel):
    """Data-sufficiency decision made before the upstream answer."""

    model_config = ConfigDict(extra="forbid")

    decision: UpstreamAction
    problem: str = ""

    @field_validator("problem", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @field_validator("problem")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        clean_value = value.strip()
        return "" if clean_value.casefold() == "null" else clean_value


__all__ = [
    "EvidenceArtifact",
    "EvidenceFact",
    "MAX_PLAN_STEPS",
    "Observation",
    "ObservationStatus",
    "PlanStep",
    "PreviousResultReference",
    "PreviousResultSchema",
    "SavedResultColumn",
    "SavedResultDescriptor",
    "UpstreamOutput",
    "UpstreamAction",
    "UpstreamDecision",
    "WorkerRequestParts",
    "WorkerOutcome",
    "WorkerOutcomeStatus",
    "WorkerPlan",
    "WORKER_PREVIOUS_RESULTS_MARKER",
    "WORKER_OPERATION_COMPLETENESS_MARKER",
    "WORKER_OPERATION_EXECUTION_MARKER",
    "WORKER_STABLE_CONTEXT_MARKER",
    "parse_worker_request",
]
