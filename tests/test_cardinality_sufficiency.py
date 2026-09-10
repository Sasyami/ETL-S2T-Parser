"""Pure tests for complete conditional-cardinality mapping evidence."""

from __future__ import annotations

import pytest

from agents.cardinality_sufficiency import (
    complete_cardinality_mapping_evidence_ids,
)
from agents.contracts import EvidenceArtifact
from agents.tools.saved_results import SavedResultStore


CARDINALITY_TASK = (
    "Оцени риск появления дубликатов при сохранённой "
    "S2T-трансформации src_alpha → tgt_beta. Назови фактический JOIN "
    "и явно отдели подтверждённый механизм от условия по уникальности."
)


@pytest.fixture
def saved_result_store():
    store = SavedResultStore()
    try:
        yield store
    finally:
        store.close()


def _saved_mapping_artifact(
    store: SavedResultStore,
    rows: list[dict[str, object]],
    *,
    evidence_id: str,
) -> EvidenceArtifact:
    descriptor = store.save_payload(
        source_tool="read_s2t_source_to_target",
        source_tool_call_id="call-cardinality",
        payload={
            "columns": [
                "source_table",
                "target_table",
                "transformation_rule",
            ],
            "rows": rows,
            "total_matches": len(rows),
            "truncated": False,
        },
    )
    assert descriptor is not None
    return EvidenceArtifact(
        evidence_id=evidence_id,
        tool_name="read_s2t_source_to_target",
        compact_args={
            "source_table": "src_alpha",
            "target_table": "tgt_beta",
        },
        dataset_ref=descriptor.result_ref,
    )


def test_complete_exact_saved_mapping_is_sufficient(saved_result_store):
    artifact = _saved_mapping_artifact(
        saved_result_store,
        [
            {
                "source_table": "src_alpha",
                "target_table": "tgt_beta",
                "transformation_rule": (
                    "SELECT s.id FROM src_alpha AS s "
                    "JOIN dim_alpha AS d ON d.id = s.id"
                ),
            }
        ],
        evidence_id="evidence-complete",
    )

    assert complete_cardinality_mapping_evidence_ids(
        CARDINALITY_TASK,
        [artifact],
        saved_result_store,
    ) == ["evidence-complete"]


def test_wrong_scope_saved_rows_are_not_sufficient(saved_result_store):
    artifact = _saved_mapping_artifact(
        saved_result_store,
        [
            {
                "source_table": "src_other",
                "target_table": "tgt_beta",
                "transformation_rule": (
                    "SELECT s.id FROM src_other AS s "
                    "JOIN dim_alpha AS d ON d.id = s.id"
                ),
            }
        ],
        evidence_id="evidence-wrong-scope",
    )

    assert complete_cardinality_mapping_evidence_ids(
        CARDINALITY_TASK,
        [artifact],
        saved_result_store,
    ) == []


@pytest.mark.parametrize("transformation_rule", ["   ", "-"])
def test_blank_or_dash_transformation_rule_is_not_sufficient(
    saved_result_store,
    transformation_rule,
):
    artifact = _saved_mapping_artifact(
        saved_result_store,
        [
            {
                "source_table": "src_alpha",
                "target_table": "tgt_beta",
                "transformation_rule": transformation_rule,
            }
        ],
        evidence_id="evidence-missing-rule",
    )

    assert complete_cardinality_mapping_evidence_ids(
        CARDINALITY_TASK,
        [artifact],
        saved_result_store,
    ) == []
