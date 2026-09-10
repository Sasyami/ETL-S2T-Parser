"""Deterministic nullable-constraint analysis over accepted saved evidence."""

import json

import pytest

from agents.constraint_rejection_analysis import (
    MAX_CONSTRAINT_REJECTION_PAYLOAD_CHARS,
    ConstraintRejectionFact,
    constraint_rejection_payload,
    derive_constraint_rejection_facts,
    is_exclusive_constraint_rejection_request,
    render_constraint_rejection_answer,
)
from agents.contracts import EvidenceArtifact
from agents.tools.saved_results import SavedResultStore


TASK = (
    "Для file_id=9101 оцени только SQL-риск constraint rejection из-за "
    "nullable-ограничений src_np.id → tgt_np.id. Верни "
    "source_not_null=<0|1>, target_not_null=<0|1> и вывод."
)
MAPPING_ARGS = {"source_table": "src_np", "target_table": "tgt_np"}
METADATA_ARGS = {
    "file_id": 9101,
    "source_table": "src_np",
    "source_column": "id",
    "target_table": "tgt_np",
    "target_column": "id",
}


def _mapping_row(
    *,
    source_field: str = "id",
    target_field: str = "id",
    source_table: str = "src_np",
    target_table: str = "tgt_np",
) -> dict:
    return {
        "source_table": source_table,
        "source_field": source_field,
        "target_table": target_table,
        "target_field": target_field,
        "transformation_rule": "SELECT s.id AS id FROM src_np AS s",
    }


def _metadata_row(
    role: str,
    not_null: object,
    *,
    file_id: int = 9101,
    table_name: str | None = None,
    column_name: str = "id",
) -> dict:
    return {
        "column_role": role,
        "file_id": file_id,
        "table_name": table_name
        or ("src_np" if role == "source" else "tgt_np"),
        "column_name": column_name,
        "not_null": not_null,
    }


def _artifact(
    store: SavedResultStore,
    *,
    tool_name: str,
    args: dict,
    rows: list[dict],
    evidence_id: str,
    total_matches: int | None = None,
    source_truncated: bool = False,
    artifact_truncated: bool = False,
) -> EvidenceArtifact:
    payload = {
        "columns": (
            [
                "source_table",
                "source_field",
                "target_table",
                "target_field",
                "transformation_rule",
            ]
            if tool_name == "read_s2t_source_to_target"
            else [
                "column_role",
                "file_id",
                "table_name",
                "column_name",
                "not_null",
            ]
        ),
        "rows": rows,
        "total_matches": (
            len(rows) if total_matches is None else total_matches
        ),
        "truncated": source_truncated,
    }
    descriptor = store.save_payload(
        source_tool=tool_name,
        source_tool_call_id=evidence_id,
        payload=payload,
    )
    assert descriptor is not None
    return EvidenceArtifact(
        evidence_id=evidence_id,
        tool_name=tool_name,
        compact_args=args,
        truncated=artifact_truncated,
        dataset_ref=descriptor.result_ref,
    )


def _complete_artifacts(
    store: SavedResultStore,
    *,
    source_not_null: object = 0,
    target_not_null: object = 1,
    metadata_rows: list[dict] | None = None,
) -> list[EvidenceArtifact]:
    mapping = _artifact(
        store,
        tool_name="read_s2t_source_to_target",
        args=MAPPING_ARGS,
        rows=[_mapping_row()],
        evidence_id="evidence-mapping",
    )
    metadata = _artifact(
        store,
        tool_name="get_source_target_column_pair",
        args=METADATA_ARGS,
        rows=(
            metadata_rows
            if metadata_rows is not None
            else [
                _metadata_row("source", source_not_null),
                _metadata_row("target", target_not_null),
            ]
        ),
        evidence_id="evidence-metadata",
    )
    return [mapping, metadata]


def _derive(
    *,
    source_not_null: object = 0,
    target_not_null: object = 1,
    metadata_rows: list[dict] | None = None,
) -> ConstraintRejectionFact:
    store = SavedResultStore()
    try:
        return derive_constraint_rejection_facts(
            TASK,
            _complete_artifacts(
                store,
                source_not_null=source_not_null,
                target_not_null=target_not_null,
                metadata_rows=metadata_rows,
            ),
            store,
        )[0]
    finally:
        store.close()


def test_live_wording_is_an_exclusive_constraint_rejection_request():
    assert is_exclusive_constraint_rejection_request(TASK) is True


@pytest.mark.parametrize(
    "task",
    [
        "Оцени constraint rejection src_np.id → tgt_np.id.",
        (
            "Для file_id=9101 оцени constraint rejection "
            "src_np.id → tgt_np.id и row filtering."
        ),
        (
            "Для file_id=9101 оцени только constraint rejection "
            "src_np.id → tgt_np.id и покажи все строки mapping."
        ),
        (
            "Для file_id=9101 оцени только nullable constraint rejection "
            "src_np.id → tgt_np.id и верни весь SQL."
        ),
        (
            "Для file_id=9101 оцени только constraint rejection из-за "
            "nullable и уникальности src_np.id → tgt_np.id."
        ),
        (
            "Для file_id=9101 оцени только constraint rejection из-за "
            "nullable и CHECK constraint src_np.id → tgt_np.id."
        ),
        (
            "Для file_id=9101 оцени только constraint rejection из-за "
            "nullable и DEFAULT constraint src_np.id → tgt_np.id."
        ),
        (
            "Для file_id=9101 оцени только constraint rejection из-за "
            "nullable и ссылочной целостности src_np.id → tgt_np.id."
        ),
        (
            "Для file_id=9101 оцени только constraint rejection из-за "
            "nullable и изменения данных src_np.id → tgt_np.id."
        ),
        (
            "Для file_id=9101 оцени только constraint rejection из-за "
            "nullable и фактических NULL в данных src_np.id → tgt_np.id."
        ),
        (
            "Для file_id=9101 оцени только nullable constraint rejection "
            "src_np.id → tgt_np.id и покажи весь каталог колонок."
        ),
        (
            "Для file_id=9101 оцени только nullable constraint rejection "
            "src_np.id → tgt_np.id и укажи типы данных обеих колонок."
        ),
        (
            "Для file_id=9101 оцени только nullable constraint rejection "
            "src_np.id → tgt_np.id и описание target-колонки."
        ),
        (
            "Для file_id=9101 оцени только nullable constraint rejection "
            "src_np.id → tgt_np.id и покажи mapping."
        ),
        (
            "Для file_id=9101 оцени только nullable constraint rejection "
            "src_np.id → tgt_np.id и укажи transformation_rule."
        ),
        (
            "Для file_id=9101 оцени только nullable constraint rejection "
            "с внешней ссылкой src_np.id → tgt_np.id."
        ),
        (
            "Для file_id=9101 оцени только nullable constraint rejection и "
            "риск переполнения длины src_np.id → tgt_np.id."
        ),
        (
            "Для file_id=9101 оцени только SQL-риск constraint rejection "
            "из-за nullable-ограничений src_np.id → tgt_np.id. Дай "
            "рекомендации по исправлению. Верни source_not_null=<0|1>, "
            "target_not_null=<0|1> и вывод."
        ),
        (
            "Для file_id=9101 оцени только SQL-риск constraint rejection "
            "из-за nullable-ограничений src_np.id → tgt_np.id. Составь DDL. "
            "Верни source_not_null=<0|1>, target_not_null=<0|1> и вывод."
        ),
        (
            "Для file_id=9101 оцени только SQL-риск constraint rejection "
            "из-за nullable-ограничений src_np.id → tgt_np.id. Сравни с "
            "другой загрузкой. Верни source_not_null=<0|1>, "
            "target_not_null=<0|1> и вывод."
        ),
        (
            "For file_id=9101 assess only constraint rejection caused by "
            "incompatible data types for source field "
            "src_np.id → tgt_np.id."
        ),
        (
            "Для file_id=9101 оцени только constraint rejection "
            "src_np.id -> tgt_np.id."
        ),
        (
            "Для file_id=9101 оцени только constraint rejection "
            "src_np.id → tgt_np.id и a.id → b.id."
        ),
    ],
)
def test_exclusive_predicate_fails_closed_for_broader_or_ambiguous_task(task):
    assert is_exclusive_constraint_rejection_request(task) is False


def test_derives_conditional_risk_and_renders_exact_requested_values():
    fact = _derive()

    assert fact.model_dump() == {
        "file_id": 9101,
        "source_table": "src_np",
        "source_field": "id",
        "target_table": "tgt_np",
        "target_field": "id",
        "source_not_null": 0,
        "target_not_null": 1,
        "conclusion": "conditional_rejection_risk",
        "mechanism": "nullable_source_to_not_null_target",
        "mapping_rows": 1,
        "exact_field_rows": 1,
        "source_metadata_rows": 1,
        "target_metadata_rows": 1,
        "evidence_ids": ["evidence-mapping", "evidence-metadata"],
    }
    answer = render_constraint_rejection_answer([fact])
    assert answer.startswith("source_not_null=0\ntarget_not_null=1\n")
    assert "`src_np.id → tgt_np.id`" in answer
    assert "условный риск constraint rejection" in answer
    assert "только если" in answer


@pytest.mark.parametrize(
    ("source_not_null", "target_not_null", "mechanism"),
    [
        (0, 0, "target_allows_null"),
        (1, 0, "target_allows_null"),
        (1, 1, "both_not_null"),
    ],
)
def test_known_non_conflicting_compatible_flags_are_narrowly_not_detected(
    source_not_null,
    target_not_null,
    mechanism,
):
    fact = _derive(
        source_not_null=source_not_null,
        target_not_null=target_not_null,
    )

    assert fact.conclusion == "nullable_mismatch_not_detected"
    assert fact.mechanism == mechanism
    assert "других constraints" in render_constraint_rejection_answer([fact])


@pytest.mark.parametrize(
    ("metadata_rows", "mechanism"),
    [
        ([], "missing_both_metadata"),
        ([_metadata_row("target", 1)], "missing_source_metadata"),
        ([_metadata_row("source", 0)], "missing_target_metadata"),
    ],
)
def test_empty_or_one_role_catalog_is_not_assessed(metadata_rows, mechanism):
    fact = _derive(metadata_rows=metadata_rows)

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == mechanism
    assert fact.source_not_null is None
    assert fact.target_not_null is None


@pytest.mark.parametrize(
    ("metadata_rows", "mechanism"),
    [
        (
            [
                _metadata_row("source", 0),
                _metadata_row("source", 1),
                _metadata_row("target", 1),
            ],
            "conflicting_source_metadata",
        ),
        (
            [
                _metadata_row("source", 0),
                _metadata_row("target", 0),
                _metadata_row("target", 1),
            ],
            "conflicting_target_metadata",
        ),
    ],
)
def test_conflicting_duplicate_catalog_values_are_never_guessed(
    metadata_rows,
    mechanism,
):
    fact = _derive(metadata_rows=metadata_rows)

    assert fact.conclusion == "conflicting"
    assert fact.mechanism == mechanism
    assert fact.source_not_null is None
    assert fact.target_not_null is None


def test_identical_duplicate_catalog_rows_remain_valid():
    fact = _derive(
        metadata_rows=[
            _metadata_row("source", 0),
            _metadata_row("source", 0),
            _metadata_row("target", 1),
            _metadata_row("target", 1),
        ]
    )

    assert fact.conclusion == "conditional_rejection_risk"
    assert fact.source_metadata_rows == 2
    assert fact.target_metadata_rows == 2


@pytest.mark.parametrize(
    ("source_not_null", "target_not_null", "mechanism"),
    [
        (None, 1, "unknown_source_not_null"),
        (0, None, "unknown_target_not_null"),
    ],
)
def test_unknown_not_null_is_not_normalized_or_guessed(
    source_not_null,
    target_not_null,
    mechanism,
):
    fact = _derive(
        source_not_null=source_not_null,
        target_not_null=target_not_null,
    )

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == mechanism
    assert "unknown" in render_constraint_rejection_answer([fact])


@pytest.mark.parametrize(
    ("metadata_rows", "unknown_field"),
    [
        (
            [
                _metadata_row("source", None),
                _metadata_row("source", 0),
                _metadata_row("target", 1),
            ],
            "source_not_null",
        ),
        (
            [
                _metadata_row("source", 0),
                _metadata_row("target", None),
                _metadata_row("target", 1),
            ],
            "target_not_null",
        ),
    ],
)
def test_mixed_known_and_unknown_duplicates_render_unknown(
    metadata_rows,
    unknown_field,
):
    fact = _derive(metadata_rows=metadata_rows)

    assert fact.conclusion == "not_assessed"
    assert getattr(fact, unknown_field) is None
    assert f"{unknown_field}=unknown" in render_constraint_rejection_answer(
        [fact]
    )


def test_missing_exact_field_mapping_is_not_assessed():
    store = SavedResultStore()
    try:
        mapping = _artifact(
            store,
            tool_name="read_s2t_source_to_target",
            args=MAPPING_ARGS,
            rows=[_mapping_row(source_field="other", target_field="other")],
            evidence_id="evidence-mapping",
        )
        metadata = _complete_artifacts(store)[1]
        fact = derive_constraint_rejection_facts(
            TASK,
            [mapping, metadata],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "no_exact_field_mapping"
    assert fact.exact_field_rows == 0


def test_wrong_mapping_scope_is_rejected_instead_of_cherry_picked():
    store = SavedResultStore()
    try:
        mapping = _artifact(
            store,
            tool_name="read_s2t_source_to_target",
            args=MAPPING_ARGS,
            rows=[
                _mapping_row(),
                _mapping_row(source_table="unrelated_source"),
            ],
            evidence_id="evidence-mapping",
        )
        metadata = _complete_artifacts(store)[1]
        fact = derive_constraint_rejection_facts(
            TASK,
            [mapping, metadata],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "scope_mismatch"


def test_wrong_metadata_file_or_role_is_scope_mismatch():
    fact = _derive(
        metadata_rows=[
            _metadata_row("source", 0),
            _metadata_row("target", 1, file_id=9102),
        ]
    )

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "scope_mismatch"


def test_null_metadata_identifier_is_scope_mismatch():
    invalid_target = _metadata_row("target", 1)
    invalid_target["table_name"] = None
    fact = _derive(
        metadata_rows=[
            _metadata_row("source", 0),
            invalid_target,
        ]
    )

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "scope_mismatch"


def test_noncanonical_not_null_value_is_incomplete_not_coerced():
    fact = _derive(
        metadata_rows=[
            _metadata_row("source", "false"),
            _metadata_row("target", 1),
        ]
    )

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "incomplete_catalog_evidence"
    assert fact.source_not_null is None


@pytest.mark.parametrize("kind", ["missing_dataset", "source_truncated"])
def test_incomplete_mapping_dataset_is_not_assessed(kind):
    store = SavedResultStore()
    try:
        if kind == "missing_dataset":
            mapping = EvidenceArtifact(
                evidence_id="evidence-mapping",
                tool_name="read_s2t_source_to_target",
                compact_args=MAPPING_ARGS,
            )
        else:
            mapping = _artifact(
                store,
                tool_name="read_s2t_source_to_target",
                args=MAPPING_ARGS,
                rows=[_mapping_row()],
                evidence_id="evidence-mapping",
                source_truncated=True,
            )
        metadata = _complete_artifacts(store)[1]
        fact = derive_constraint_rejection_facts(
            TASK,
            [mapping, metadata],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "incomplete_mapping_evidence"


def test_source_total_mismatch_is_incomplete_catalog_evidence():
    store = SavedResultStore()
    try:
        mapping = _complete_artifacts(store)[0]
        metadata = _artifact(
            store,
            tool_name="get_source_target_column_pair",
            args=METADATA_ARGS,
            rows=[_metadata_row("source", 0), _metadata_row("target", 1)],
            evidence_id="evidence-metadata",
            total_matches=3,
        )
        fact = derive_constraint_rejection_facts(
            TASK,
            [mapping, metadata],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "incomplete_catalog_evidence"


def test_preview_clipping_does_not_override_complete_saved_relations():
    store = SavedResultStore()
    try:
        mapping = _artifact(
            store,
            tool_name="read_s2t_source_to_target",
            args=MAPPING_ARGS,
            rows=[_mapping_row()],
            evidence_id="evidence-mapping",
            artifact_truncated=True,
        )
        metadata = _artifact(
            store,
            tool_name="get_source_target_column_pair",
            args=METADATA_ARGS,
            rows=[_metadata_row("source", 0), _metadata_row("target", 1)],
            evidence_id="evidence-metadata",
            artifact_truncated=True,
        )
        fact = derive_constraint_rejection_facts(
            TASK,
            [mapping, metadata],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "conditional_rejection_risk"


def test_wrong_exact_arguments_do_not_satisfy_provenance():
    store = SavedResultStore()
    try:
        artifacts = _complete_artifacts(store)
        artifacts[0] = artifacts[0].model_copy(
            update={
                "compact_args": {
                    "source_table": "src_np",
                    "target_table": "another_target",
                }
            }
        )
        fact = derive_constraint_rejection_facts(TASK, artifacts, store)[0]
    finally:
        store.close()

    assert fact.mechanism == "incomplete_mapping_evidence"
    assert fact.evidence_ids == []


def test_task_without_safe_exact_contract_produces_no_fact():
    store = SavedResultStore()
    try:
        facts = derive_constraint_rejection_facts(
            "Оцени nullable src_np.id → tgt_np.id без file scope.",
            [],
            store,
        )
    finally:
        store.close()

    assert facts == []


def test_payload_is_bounded_and_contains_no_source_rows_or_sql():
    facts = [
        ConstraintRejectionFact(
            file_id=index + 1,
            source_table="s" * 200,
            source_field="f" * 200,
            target_table="t" * 200,
            target_field="g" * 200,
            source_not_null=0,
            target_not_null=1,
            conclusion="conditional_rejection_risk",
            mechanism="nullable_source_to_not_null_target",
            mapping_rows=1,
            exact_field_rows=1,
            source_metadata_rows=1,
            target_metadata_rows=1,
            evidence_ids=["e" * 120 for _ in range(8)],
        )
        for index in range(8)
    ]

    payload = constraint_rejection_payload(facts)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert len(serialized) <= MAX_CONSTRAINT_REJECTION_PAYLOAD_CHARS
    assert payload["authority"] == (
        "deterministic_exact_s2t_and_catalog_full_saved_results"
    )
    assert "transformation_rule" not in serialized
    assert "SELECT" not in serialized
    assert payload.get("payload_truncated") is True


def test_payload_clips_externally_constructed_evidence_ids():
    fact = ConstraintRejectionFact(
        file_id=1,
        source_table="src",
        source_field="id",
        target_table="tgt",
        target_field="id",
        conclusion="not_assessed",
        mechanism="incomplete_mapping_evidence",
        evidence_ids=["e" * 1_000_000],
    )

    payload = constraint_rejection_payload([fact])
    serialized = json.dumps(payload, ensure_ascii=False)

    assert len(serialized) <= MAX_CONSTRAINT_REJECTION_PAYLOAD_CHARS
    assert len(payload["facts"][0]["evidence_ids"][0]) == 120


def test_fact_rejects_conclusion_that_contradicts_public_flags():
    with pytest.raises(ValueError, match="conditional rejection"):
        ConstraintRejectionFact(
            file_id=1,
            source_table="src",
            source_field="id",
            target_table="tgt",
            target_field="id",
            source_not_null=1,
            target_not_null=0,
            conclusion="conditional_rejection_risk",
            mechanism="nullable_source_to_not_null_target",
        )
