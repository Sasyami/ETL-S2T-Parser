"""Unit tests for deterministic conditional-cardinality facts."""

from __future__ import annotations

import json

import pytest

from agents.cardinality_analysis import (
    MAX_CARDINALITY_PAYLOAD_CHARS,
    CardinalityFact,
    CardinalityJoinFact,
    analyze_cardinality_rule,
    cardinality_payload,
    derive_cardinality_facts,
    render_cardinality_answer,
)
from agents.contracts import EvidenceArtifact
from agents.sql_risk_scope_contract import (
    LiteralSqlRiskScope,
    ReadS2TSourceToTargetRequirement,
    SqlRiskScopeContract,
)
from agents.tools.saved_results import SavedResultStore


CONTRACT = SqlRiskScopeContract(
    scope=LiteralSqlRiskScope(
        source="src_np",
        target="tgt_np",
        source_table="src_np",
        target_table="tgt_np",
    ),
    aspects=("cardinality",),
    requirements=(
        ReadS2TSourceToTargetRequirement(
            source_table="src_np",
            target_table="tgt_np",
        ),
    ),
    execution_mode="conditional_cardinality",
)
JOIN_RULE = (
    "SELECT s.id AS id, COALESCE(s.value, d.value) AS value "
    "FROM src_np AS s JOIN aux_np AS d ON d.id = s.id "
    "WHERE s.ok = TRUE"
)


@pytest.fixture
def saved_result_store():
    store = SavedResultStore()
    try:
        yield store
    finally:
        store.close()


def _artifact(
    store: SavedResultStore,
    rows: list[dict[str, object]],
    *,
    evidence_id: str = "evidence-cardinality",
    total: int | None = None,
    truncated: bool = False,
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
            "total_matches": len(rows) if total is None else total,
            "truncated": truncated,
        },
    )
    assert descriptor is not None
    return EvidenceArtifact(
        evidence_id=evidence_id,
        tool_name="read_s2t_source_to_target",
        compact_args={"source_table": "src_np", "target_table": "tgt_np"},
        dataset_ref=descriptor.result_ref,
    )


def _row(rule: str, *, source: str = "src_np", target: str = "tgt_np"):
    return {
        "source_table": source,
        "target_table": target,
        "transformation_rule": rule,
    }


def test_rule_analysis_keeps_only_actual_outer_join():
    analysis = analyze_cardinality_rule(JOIN_RULE)

    assert analysis.status == "join_detected"
    assert len(analysis.joins) == 1
    join = analysis.joins[0]
    assert join.join_type == "JOIN"
    assert join.relation == "aux_np AS d"
    assert join.predicate == "d.id = s.id"
    assert join.join_key_equalities == ["d.id = s.id"]
    assert join.uniqueness_condition == "full_join_key_uniqueness_unknown"
    assert "WHERE" not in join.model_dump_json()
    assert "COALESCE" not in join.model_dump_json()


def test_composite_join_key_is_kept_as_one_unknown_uniqueness_condition():
    analysis = analyze_cardinality_rule(
        "SELECT s.id FROM src_np s JOIN aux_np d "
        "ON d.id = s.id AND d.region_id = s.region_id"
    )

    assert analysis.status == "join_detected"
    assert analysis.joins[0].join_key_equalities == [
        "d.id = s.id",
        "d.region_id = s.region_id",
    ]
    assert (
        analysis.joins[0].uniqueness_condition
        == "full_join_key_uniqueness_unknown"
    )


def test_unrelated_column_equality_is_not_treated_as_full_join_key():
    analysis = analyze_cardinality_rule(
        "SELECT s.id FROM src_np s JOIN aux_np d ON s.left_id = s.right_id"
    )

    assert analysis.status == "join_detected"
    assert analysis.joins[0].join_key_equalities == []
    assert (
        analysis.joins[0].uniqueness_condition
        == "join_match_multiplicity_unknown"
    )


@pytest.mark.parametrize("side", ["LEFT", "RIGHT", "FULL"])
def test_outer_join_is_supported_with_exact_join_type(side):
    analysis = analyze_cardinality_rule(
        f"SELECT s.id FROM src_np s {side} OUTER JOIN aux_np d "
        "ON d.id = s.id"
    )

    assert analysis.status == "join_detected"
    assert analysis.joins[0].join_type == f"{side} OUTER JOIN"


def test_where_and_coalesce_without_join_are_not_mechanisms():
    analysis = analyze_cardinality_rule(
        "SELECT COALESCE(s.value, 0) AS value FROM src_np s "
        "WHERE s.ok = TRUE"
    )

    assert analysis.status == "no_join"
    assert analysis.joins == []


@pytest.mark.parametrize(
    ("rule", "status"),
    [
        ("definitely not a complete select", "unparseable_rule"),
        (
            "SELECT id FROM src_np UNION ALL SELECT id FROM aux_np",
            "set_operation",
        ),
        (
            "SELECT s.id FROM src_np s LEFT SEMI JOIN aux_np d "
            "ON d.id = s.id",
            "unsupported_join",
        ),
        ("SELECT * FROM src_np s JOIN aux_np d", "unsupported_join"),
        (
            "SELECT * FROM src_np s FULL INNER JOIN aux_np d "
            "ON d.id = s.id",
            "unsupported_join",
        ),
        (
            "SELECT * FROM src_np s CROSS JOIN aux_np d ON d.id = s.id",
            "unsupported_join",
        ),
        (
            "SELECT * FROM src_np s NATURAL JOIN aux_np d ON d.id = s.id",
            "unsupported_join",
        ),
        (
            "SELECT * FROM src_np s JOIN aux_np d ON FALSE",
            "constant_false_predicate",
        ),
    ],
)
def test_rule_analysis_fails_closed(rule, status):
    assert analyze_cardinality_rule(rule).status == status


@pytest.mark.parametrize(
    "rule",
    [
        pytest.param(
            "SELECT s.id FROM src_np s JOIN aux_np d "
            "ON d.id = s.id AND FALSE",
            id="false-join-conjunct",
        ),
        pytest.param(
            "SELECT s.id FROM src_np s JOIN aux_np d "
            "ON d.id = s.id WHERE FALSE",
            id="false-final-filter",
        ),
    ],
)
def test_constant_false_predicate_is_not_assessed(saved_result_store, rule):
    analysis = analyze_cardinality_rule(rule)
    artifact = _artifact(saved_result_store, [_row(rule)])

    fact = derive_cardinality_facts(CONTRACT, [artifact], saved_result_store)[0]

    assert analysis.status == "constant_false_predicate"
    assert analysis.joins == []
    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "constant_false_predicate"
    assert fact.joins == []


@pytest.mark.parametrize(
    "rule",
    [
        pytest.param(
            "WITH scoped AS (SELECT * FROM src_np) "
            "SELECT s.id FROM scoped s JOIN aux_np d ON d.id = s.id",
            id="cte",
        ),
        pytest.param(
            "SELECT s.id, (SELECT MAX(x.id) FROM aux_other x) AS last_id "
            "FROM src_np s JOIN aux_np d ON d.id = s.id",
            id="nested-select",
        ),
        pytest.param(
            "SELECT DISTINCT s.id FROM src_np s JOIN aux_np d ON d.id = s.id",
            id="distinct",
        ),
        pytest.param(
            "SELECT s.id FROM src_np s JOIN aux_np d ON d.id = s.id "
            "GROUP BY s.id",
            id="group",
        ),
        pytest.param(
            "SELECT s.id FROM src_np s JOIN aux_np d ON d.id = s.id "
            "GROUP BY s.id HAVING COUNT(*) > 1",
            id="having",
        ),
        pytest.param(
            "SELECT s.id, ROW_NUMBER() OVER (PARTITION BY s.id) AS rn "
            "FROM src_np s JOIN aux_np d ON d.id = s.id QUALIFY rn = 1",
            id="qualify",
        ),
        pytest.param(
            "SELECT s.id FROM src_np s JOIN aux_np d ON d.id = s.id LIMIT 1",
            id="limit",
        ),
        pytest.param(
            "SELECT COUNT(*) FROM src_np s JOIN aux_np d ON d.id = s.id",
            id="aggregate-without-group",
        ),
    ],
)
def test_cardinality_reshaping_query_shapes_are_not_assessed(rule):
    assert analyze_cardinality_rule(rule).status == "unsupported_join"


def test_complete_duplicate_s2t_rows_produce_one_conditional_fact(
    saved_result_store,
):
    artifact = _artifact(
        saved_result_store,
        [_row(JOIN_RULE), _row(JOIN_RULE)],
    )

    facts = derive_cardinality_facts(CONTRACT, [artifact], saved_result_store)

    assert len(facts) == 1
    fact = facts[0]
    assert fact.conclusion == "conditional_duplicate_risk"
    assert fact.mechanism == "join_fanout"
    assert fact.condition == "full_join_key_uniqueness_unknown"
    assert fact.matching_rows == 2
    assert fact.evidence_ids == ["evidence-cardinality"]
    assert fact.joins[0].predicate == "d.id = s.id"


def test_fact_preserves_opaque_evidence_id_for_coordinator_provenance(
    saved_result_store,
):
    evidence_id = "evidence-" + "x" * 200
    artifact = _artifact(
        saved_result_store,
        [_row(JOIN_RULE)],
        evidence_id=evidence_id,
    )

    fact = derive_cardinality_facts(CONTRACT, [artifact], saved_result_store)[0]

    assert fact.evidence_ids == [evidence_id]


def test_projection_and_filter_differences_do_not_create_false_conflict(
    saved_result_store,
):
    artifact = _artifact(
        saved_result_store,
        [
            _row(JOIN_RULE),
            _row(
                "SELECT s.id, s.value FROM src_np s JOIN aux_np d "
                "ON d.id = s.id WHERE s.ok = FALSE"
            ),
        ],
    )

    fact = derive_cardinality_facts(CONTRACT, [artifact], saved_result_store)[0]

    assert fact.conclusion == "conditional_duplicate_risk"
    assert fact.mechanism == "join_fanout"


def test_conflicting_join_structures_fail_closed(saved_result_store):
    artifact = _artifact(
        saved_result_store,
        [
            _row(JOIN_RULE),
            _row("SELECT s.id FROM src_np s"),
        ],
    )

    fact = derive_cardinality_facts(CONTRACT, [artifact], saved_result_store)[0]

    assert fact.conclusion == "conflicting"
    assert fact.mechanism == "conflicting_rules"
    assert fact.joins == []


def test_empty_complete_mapping_is_not_assessed(saved_result_store):
    artifact = _artifact(saved_result_store, [])

    fact = derive_cardinality_facts(CONTRACT, [artifact], saved_result_store)[0]

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "no_exact_mapping"
    assert fact.matching_rows == 0


def test_wrong_scope_saved_rows_fail_closed(saved_result_store):
    artifact = _artifact(
        saved_result_store,
        [_row(JOIN_RULE, source="src_other")],
    )

    fact = derive_cardinality_facts(CONTRACT, [artifact], saved_result_store)[0]

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "scope_mismatch"


@pytest.mark.parametrize("rule", ["", "   ", "-"])
def test_missing_transformation_rule_fails_closed(saved_result_store, rule):
    artifact = _artifact(saved_result_store, [_row(rule)])

    fact = derive_cardinality_facts(CONTRACT, [artifact], saved_result_store)[0]

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "missing_transformation_rule"


def test_incomplete_saved_relation_fails_closed(saved_result_store):
    artifact = _artifact(saved_result_store, [_row(JOIN_RULE)], total=2)

    fact = derive_cardinality_facts(CONTRACT, [artifact], saved_result_store)[0]

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "incomplete_evidence"


def test_invalid_saved_count_fails_closed(saved_result_store, monkeypatch):
    artifact = _artifact(saved_result_store, [_row(JOIN_RULE)])
    monkeypatch.setattr(
        saved_result_store,
        "query",
        lambda **_kwargs: {
            "rows": [
                {
                    "saved_rows": "not-an-int",
                    "exact_rows": 1,
                    "missing_rule_rows": 0,
                }
            ],
            "truncated": False,
        },
    )

    fact = derive_cardinality_facts(CONTRACT, [artifact], saved_result_store)[0]

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "incomplete_evidence"


def test_renderer_names_join_and_unknown_condition_but_not_filter_or_projection(
    saved_result_store,
):
    artifact = _artifact(saved_result_store, [_row(JOIN_RULE)])
    fact = derive_cardinality_facts(CONTRACT, [artifact], saved_result_store)[0]

    answer = render_cardinality_answer([fact])
    payload = cardinality_payload([fact])

    assert "src_np → tgt_np" in answer
    assert "JOIN aux_np AS d ON d.id = s.id" in answer
    assert "JOIN может размножить строки" in answer
    assert "Уникальность полного JOIN-ключа" in answer
    assert "не установлена" in answer
    assert "условен" in answer
    assert "фактических дубликатов не установлено" in answer
    assert "Фактический JOIN" not in answer
    assert "Подтверждённый механизм" not in answer
    assert "Условие по уникальности" not in answer
    assert "where" not in answer.casefold()
    assert "coalesce" not in answer.casefold()
    assert payload["authority"] == "deterministic_sqlglot_full_saved_result"
    assert "SELECT s.id" not in json.dumps(payload, ensure_ascii=False)


def test_no_join_renderer_keeps_complete_requested_output_contract(
    saved_result_store,
):
    artifact = _artifact(
        saved_result_store,
        [_row("SELECT s.id FROM src_np AS s")],
    )
    fact = derive_cardinality_facts(CONTRACT, [artifact], saved_result_store)[0]

    answer = render_cardinality_answer([fact])

    assert "не применим" in answer
    assert "не доказывает отсутствие дубликатов" in answer
    assert "Фактический JOIN" not in answer


def test_payload_is_hard_bounded_for_maximal_valid_facts():
    join = CardinalityJoinFact(
        join_type="NATURAL FULL INNER JOIN",
        relation="r" * 240,
        predicate="p" * 600,
        join_key_equalities=["k" * 240] * 8,
        uniqueness_condition="full_join_key_uniqueness_unknown",
    )
    facts = [
        CardinalityFact(
            source_table="s" * 200,
            target_table="t" * 200,
            conclusion="conditional_duplicate_risk",
            mechanism="join_fanout",
            condition="full_join_key_uniqueness_unknown",
            matching_rows=1,
            joins=[join.model_copy(deep=True) for _ in range(4)],
            evidence_ids=["e" * 120 for _ in range(8)],
        )
        for _ in range(4)
    ]

    payload = cardinality_payload(facts)

    assert len(json.dumps(payload, ensure_ascii=False)) <= (
        MAX_CARDINALITY_PAYLOAD_CHARS
    )
    assert payload["payload_truncated"] is True
