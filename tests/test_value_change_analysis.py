"""Deterministic field-scoped SQL value-change analysis."""

import pytest

from agents.contracts import EvidenceArtifact
from agents.tools.saved_results import SavedResultStore
from agents.transformation_ast import (
    analyze_field_value_change,
    normalize_transformation,
)
from agents.value_change_analysis import (
    derive_field_value_change_facts,
    extract_exact_field_pairs,
    field_value_change_payload,
    render_field_value_change_answer,
)


SQL_WITH_NEIGHBOUR_EXPRESSION = (
    "SELECT s.id AS id, COALESCE(s.value, 0) AS value "
    "FROM src_np AS s"
)


def _saved_artifact(
    store: SavedResultStore,
    rows: list[dict],
    *,
    evidence_id: str = "evidence-exact",
    artifact_truncated: bool = False,
    source_truncated: bool = False,
) -> EvidenceArtifact:
    descriptor = store.save_payload(
        source_tool="read_s2t_source_to_target",
        source_tool_call_id=evidence_id,
        payload={"rows": rows, "truncated": source_truncated},
    )
    assert descriptor is not None
    return EvidenceArtifact(
        evidence_id=evidence_id,
        tool_name="read_s2t_source_to_target",
        compact_args={"source_table": "src_np", "target_table": "tgt_np"},
        truncated=artifact_truncated,
        dataset_ref=descriptor.result_ref,
    )


def _mapping_row(rule: object) -> dict:
    return {
        "source_table": "src_np",
        "source_field": "id",
        "target_table": "tgt_np",
        "target_field": "id",
        "transformation_rule": rule,
    }


def test_exact_direct_projection_ignores_expression_on_other_alias():
    normalized = normalize_transformation(SQL_WITH_NEIGHBOUR_EXPRESSION)

    analysis = analyze_field_value_change(
        normalized,
        source_table="src_np",
        source_field="id",
        target_field="id",
    )

    assert normalized.outer_source_aliases["s"] == "src_np"
    assert analysis.status == "direct_projection"
    assert analysis.may_change_value is False
    assert analysis.target_expression == "s.id"
    assert analysis.source_columns == ["s.id"]


def test_expression_is_attached_only_to_its_exact_target_alias():
    normalized = normalize_transformation(SQL_WITH_NEIGHBOUR_EXPRESSION)

    analysis = analyze_field_value_change(
        normalized,
        source_table="src_np",
        source_field="value",
        target_field="value",
    )

    assert analysis.status == "value_expression"
    assert analysis.may_change_value is True
    assert analysis.target_expression == "COALESCE(s.value, 0)"


def test_same_column_name_from_another_alias_is_not_direct():
    normalized = normalize_transformation(
        "SELECT d.id AS id FROM src_np AS s JOIN dim_np AS d ON d.id = s.id"
    )

    analysis = analyze_field_value_change(
        normalized,
        source_table="src_np",
        source_field="id",
        target_field="id",
    )

    assert analysis.status == "source_provenance_unknown"
    assert analysis.may_change_value is None


def test_unqualified_column_requires_one_unambiguous_physical_source():
    direct = analyze_field_value_change(
        normalize_transformation("SELECT id FROM src_np"),
        source_table="src_np",
        source_field="id",
        target_field="id",
    )
    ambiguous = analyze_field_value_change(
        normalize_transformation(
            "SELECT id FROM src_np JOIN dim_np ON src_np.id = dim_np.id"
        ),
        source_table="src_np",
        source_field="id",
        target_field="id",
    )

    assert direct.status == "direct_projection"
    assert ambiguous.status == "source_provenance_unknown"


def test_missing_target_projection_is_unavailable_not_safe():
    normalized = normalize_transformation("SELECT s.id FROM src_np AS s")

    analysis = analyze_field_value_change(
        normalized,
        source_table="src_np",
        source_field="value",
        target_field="value",
    )

    assert analysis.status == "target_projection_missing"
    assert analysis.may_change_value is None


def test_set_operation_is_not_classified_from_only_its_first_branch():
    rule = (
        "SELECT s.id AS id FROM src_np AS s "
        "UNION ALL SELECT s.id AS id FROM src_np AS s"
    )
    normalized = normalize_transformation(rule)

    analysis = analyze_field_value_change(
        normalized,
        source_table="src_np",
        source_field="id",
        target_field="id",
    )

    assert normalized.has_set_operation is True
    assert analysis.status == "set_operation_unsupported"
    assert analysis.may_change_value is None


def test_extracts_backticked_and_multiple_exact_pairs():
    pairs = extract_exact_field_pairs(
        "Проверь `src_np.id` → `tgt_np.id` и a.value -> b.value."
    )

    assert [pair.model_dump() for pair in pairs] == [
        {
            "source_table": "src_np",
            "source_field": "id",
            "target_table": "tgt_np",
            "target_field": "id",
        },
        {
            "source_table": "a",
            "source_field": "value",
            "target_table": "b",
            "target_field": "value",
        },
    ]


def test_derives_authoritative_fact_from_full_accepted_saved_result():
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [
                _mapping_row(SQL_WITH_NEIGHBOUR_EXPRESSION),
                {
                    **_mapping_row(SQL_WITH_NEIGHBOUR_EXPRESSION),
                    "source_field": "value",
                    "target_field": "value",
                },
            ],
        )

        facts = derive_field_value_change_facts(
            "Оцени `src_np.id` → `tgt_np.id`.",
            [artifact],
            store,
        )
    finally:
        store.close()

    assert len(facts) == 1
    fact = facts[0]
    assert fact.conclusion == "not_detected"
    assert fact.mechanism == "direct_column"
    assert fact.matching_rows == 1
    assert fact.target_expressions == ["s.id"]
    assert fact.evidence_ids == ["evidence-exact"]
    payload = field_value_change_payload(facts)
    assert payload["authority"] == "deterministic_sqlglot_full_saved_result"
    assert "neighbouring output aliases are excluded" in payload["scope_rule"]
    answer = render_field_value_change_answer(facts)
    assert "механизм изменения значения не обнаружен" in answer
    assert "соседних output aliases" in answer
    assert "COALESCE" not in answer


@pytest.mark.parametrize("rule", [None, "", "   ", "-"])
def test_direct_rule_markers_are_not_detected_only_for_exact_s2t_pair(rule):
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [_mapping_row(rule)],
            evidence_id="direct-marker",
        )
        fact = derive_field_value_change_facts(
            "src_np.id → tgt_np.id",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_detected"
    assert fact.mechanism == "direct_mapping_rule"
    assert fact.matching_rows == 1


def test_direct_marker_on_another_field_does_not_prove_requested_pair():
    store = SavedResultStore()
    try:
        row = _mapping_row("-")
        row["source_field"] = "other"
        artifact = _saved_artifact(store, [row])
        fact = derive_field_value_change_facts(
            "src_np.id → tgt_np.id",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "no_exact_mapping"


def test_clipped_preview_flag_does_not_override_complete_saved_dataset():
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [_mapping_row(SQL_WITH_NEIGHBOUR_EXPRESSION)],
            artifact_truncated=True,
        )
        fact = derive_field_value_change_facts(
            "src_np.id → tgt_np.id",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_detected"
    assert fact.mechanism == "direct_column"


def test_source_dataset_truncation_remains_not_assessed():
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [_mapping_row(SQL_WITH_NEIGHBOUR_EXPRESSION)],
            source_truncated=True,
        )
        fact = derive_field_value_change_facts(
            "src_np.id → tgt_np.id",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "incomplete_evidence"


def test_more_than_100_duplicate_rows_use_complete_distinct_rule_aggregate():
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [_mapping_row(SQL_WITH_NEIGHBOUR_EXPRESSION) for _ in range(150)],
        )
        fact = derive_field_value_change_facts(
            "src_np.id → tgt_np.id",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_detected"
    assert fact.matching_rows == 150


def test_one_rule_with_duplicate_target_alias_is_not_assessed():
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [
                _mapping_row(
                    "SELECT s.id AS id, COALESCE(s.id, 0) AS id "
                    "FROM src_np AS s"
                )
            ],
        )
        fact = derive_field_value_change_facts(
            "src_np.id → tgt_np.id",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "ambiguous_target_projection"


def test_set_operation_fact_is_not_assessed():
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [
                _mapping_row(
                    "SELECT s.id AS id FROM src_np AS s "
                    "UNION SELECT s.id AS id FROM src_np AS s"
                )
            ],
        )
        fact = derive_field_value_change_facts(
            "src_np.id → tgt_np.id",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "set_operation"


def test_distinct_rules_with_conflicting_conclusions_are_explicit():
    store = SavedResultStore()
    try:
        conflicting = _saved_artifact(
            store,
            [
                _mapping_row(SQL_WITH_NEIGHBOUR_EXPRESSION),
                _mapping_row("SELECT COALESCE(s.id, 0) AS id FROM src_np AS s"),
            ],
            evidence_id="conflicting",
        )
        conflict_fact = derive_field_value_change_facts(
            "src_np.id → tgt_np.id",
            [conflicting],
            store,
        )[0]
    finally:
        store.close()

    assert conflict_fact.conclusion == "conflicting"
    assert conflict_fact.mechanism == "conflicting_rules"
