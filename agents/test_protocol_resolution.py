"""Resolve literal validation-protocol mentions before deterministic readers."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .entity_resolution import (
    EntityMention,
    EntityResolution,
    resolve_entity,
    verify_exact_entity,
)
from .test_protocol import (
    EntityResolutionMetadata,
    ProtocolIssue,
    RawTestProtocolContract,
    ResolvedTestProtocolContract,
    TestProtocolLoad,
)


ProtocolResolutionStatus = Literal[
    "resolved",
    "unresolved_entity",
    "ambiguous_entity",
]


class TestProtocolResolutionResult(BaseModel):
    """Structured boundary between user wording and canonical protocol input."""

    model_config = ConfigDict(extra="forbid")

    status: ProtocolResolutionStatus
    contract: Optional[ResolvedTestProtocolContract] = None
    issues: List[ProtocolIssue] = Field(default_factory=list)
    resolutions: List[EntityResolution] = Field(default_factory=list)
    exact_bypass_count: int = Field(default=0, ge=0)


def _metadata(result: EntityResolution) -> EntityResolutionMetadata:
    method = "ambiguous" if result.status == "ambiguous" else result.method
    return EntityResolutionMetadata(
        entity_type=(
            "file"
            if result.entity_type == "file"
            else f"{result.role}_table"
        ),
        mention=result.mention,
        method=method,
        status=result.status,
        error_code=result.error_code,
        canonical=result.canonical_name,
        candidates=[
            candidate.canonical_name
            for candidate in result.candidate_set.candidates
        ],
    )


def _issue(result: EntityResolution, *, load_index: int | None) -> ProtocolIssue:
    return ProtocolIssue(
        code=(
            "ambiguous_entity"
            if result.status == "ambiguous"
            else "unresolved_entity"
        ),
        message=result.reason,
        load_index=load_index,
        candidates=[
            candidate.canonical_name
            for candidate in result.candidate_set.candidates
        ],
    )


def resolve_test_protocol_contract(
    raw: RawTestProtocolContract | Mapping[str, Any],
    *,
    callbacks: Sequence[Any] = (),
) -> TestProtocolResolutionResult:
    """Resolve only non-canonical mentions and preserve every ambiguity.

    The exact verifier is intentionally separate from approximate resolution:
    a canonical table goes straight through the exact role namespace, whereas
    a typo/partial/semantic mention enters the shared resolver.  No candidate
    is selected from an ambiguous result.
    """

    contract = (
        raw
        if isinstance(raw, RawTestProtocolContract)
        else RawTestProtocolContract.model_validate(raw)
    )
    resolutions: List[EntityResolution] = []
    metadata: List[EntityResolutionMetadata] = []
    issues: List[ProtocolIssue] = []
    exact_bypass_count = 0
    file_id = contract.file_id

    def resolve_one(
        request: EntityMention,
        *,
        load_index: int | None,
    ) -> EntityResolution:
        nonlocal exact_bypass_count
        exact = verify_exact_entity(request, callbacks=callbacks)
        if exact is not None:
            exact_bypass_count += 1
            result = exact
        else:
            result = resolve_entity(
                request,
                callbacks=callbacks,
                exact_checked=True,
            )
        resolutions.append(result)
        metadata.append(_metadata(result))
        if result.status != "resolved":
            issues.append(_issue(result, load_index=load_index))
        return result

    file_mention = str(contract.file_mention or "").strip()
    if file_mention:
        file_result = resolve_one(
            EntityMention(
                mention=file_mention,
                entity_type="file",
                role="file",
            ),
            load_index=None,
        )
        if file_result.status == "resolved":
            file_id = file_result.file_id

    resolved_loads: List[TestProtocolLoad] = []
    cache: Dict[tuple[str, str], EntityResolution] = {}
    for load_index, raw_load in enumerate(contract.loads, start=1):
        sources: List[str] = []
        for mention in raw_load.source_mentions:
            key = ("source", mention.casefold())
            result = cache.get(key)
            if result is None:
                result = resolve_one(
                    EntityMention(
                        mention=mention,
                        entity_type="table",
                        role="source",
                        file_id=file_id,
                    ),
                    load_index=load_index,
                )
                cache[key] = result
            if result.status == "resolved" and result.canonical_name:
                sources.append(result.canonical_name)

        target_key = ("target", raw_load.target_mention.casefold())
        target_result = cache.get(target_key)
        if target_result is None:
            target_result = resolve_one(
                EntityMention(
                    mention=raw_load.target_mention,
                    entity_type="table",
                    role="target",
                    file_id=file_id,
                ),
                load_index=load_index,
            )
            cache[target_key] = target_result
        if (
            len(sources) == len(raw_load.source_mentions)
            and target_result.status == "resolved"
            and target_result.canonical_name
        ):
            resolved_loads.append(
                TestProtocolLoad(
                    sources=sources,
                    target=target_result.canonical_name,
                    checks=raw_load.requested_checks,
                    explicit_key=raw_load.explicit_key,
                )
            )

    if issues:
        status: ProtocolResolutionStatus = (
            "ambiguous_entity"
            if any(issue.code == "ambiguous_entity" for issue in issues)
            else "unresolved_entity"
        )
        return TestProtocolResolutionResult(
            status=status,
            issues=issues,
            resolutions=resolutions,
            exact_bypass_count=exact_bypass_count,
        )

    return TestProtocolResolutionResult(
        status="resolved",
        contract=ResolvedTestProtocolContract(
            file_id=file_id,
            loads=resolved_loads,
            checks=contract.requested_checks,
            mode=contract.mode,
            explicit_key=contract.explicit_key,
            resolution_metadata=metadata,
        ),
        resolutions=resolutions,
        exact_bypass_count=exact_bypass_count,
    )


def validate_raw_contract_origin(
    contract: RawTestProtocolContract,
    original_task: str,
) -> None:
    """Reject identifiers introduced by extraction before resolution starts."""

    task = str(original_task or "")
    folded = task.casefold()
    values = [
        *contract.source_mentions,
        *contract.target_mentions,
    ]
    if contract.file_mention:
        values.append(contract.file_mention)
    missing = [value for value in values if value.casefold() not in folded]
    if contract.file_id is not None and (
        f"file_id={contract.file_id}" not in task.replace(" ", "").casefold()
    ):
        missing.append(f"file_id={contract.file_id}")
    for key in contract.explicit_key or []:
        if key.casefold() not in folded:
            missing.append(key)
    for load in contract.loads:
        for key in load.explicit_key or []:
            if key.casefold() not in folded:
                missing.append(key)
    if missing:
        raise ValueError(
            "Raw contract содержит значение не из original_task: "
            + ", ".join(dict.fromkeys(missing))
        )


__all__ = [
    "ProtocolResolutionStatus",
    "TestProtocolResolutionResult",
    "resolve_test_protocol_contract",
    "validate_raw_contract_origin",
]
