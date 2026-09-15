"""Focused tests for neutral SQLGlot scope structure."""

from __future__ import annotations

import pytest

from agents.sql_risk_scope_contract import (
    GetSourceTargetColumnPairRequirement,
    ListS2TFieldMappingRequirement,
    SqlRiskExactScope,
    ReadS2TSourceToTargetRequirement,
    SqlRiskScopeContract,
)
from agents.sql_risk_assessment import MAX_SQL_RISK_ASSESSMENT_INPUT_CHARS
from agents.sql_risk_structure import (
    SQL_RISK_ASSESSMENT_ENVELOPE_RESERVE_CHARS,
    SQL_RISK_STRUCTURE_MAX_CHARS,
    SqlRiskStructuralEvidence,
    build_sql_risk_structure_bundle,
    sql_risk_structure_assessment_payload,
)


MAPPING_ARGS = {
    "source_table": "landing.orders",
    "target_table": "mart.orders",
}
METADATA_ARGS = {
    "file_id": 42,
    "source_table": "landing.orders",
    "source_column": "customer_id",
    "target_table": "mart.orders",
    "target_column": "customer_id",
}


def _table_contract(mode: str = "conditional_cardinality") -> SqlRiskScopeContract:
    aspect = {
        "row_filtering": "row_filtering",
        "conditional_cardinality": "cardinality",
        "write_semantics": "write_semantics",
    }[mode]
    return SqlRiskScopeContract(
        scope=SqlRiskExactScope(
            source="landing.orders",
            target="mart.orders",
            source_table="landing.orders",
            target_table="mart.orders",
        ),
        aspects=(aspect,),  # type: ignore[arg-type]
        requirements=(ReadS2TSourceToTargetRequirement(**MAPPING_ARGS),),
        execution_mode=mode,  # type: ignore[arg-type]
    )


def _nullable_contract() -> SqlRiskScopeContract:
    return SqlRiskScopeContract(
        scope=SqlRiskExactScope(
            source="landing.orders.customer_id",
            target="mart.orders.customer_id",
            source_table="landing.orders",
            target_table="mart.orders",
            source_field="customer_id",
            target_field="customer_id",
            file_id=42,
        ),
        aspects=("constraint_rejection",),
        requirements=(
            ListS2TFieldMappingRequirement(
                source_table="landing.orders",
                source_field="customer_id",
                target_table="mart.orders",
                target_field="customer_id",
            ),
            GetSourceTargetColumnPairRequirement(**METADATA_ARGS),
        ),
        execution_mode="nullable_constraint",
    )


def _mapping_evidence(rows: list[dict]) -> SqlRiskStructuralEvidence:
    return SqlRiskStructuralEvidence(
        evidence_id="mapping_evidence",
        tool_name="read_s2t_source_to_target",
        arguments=MAPPING_ARGS,
        payload={
            "rows": rows,
            "total_matches": len(rows),
            "returned_rows": len(rows),
            "truncated": False,
        },
    )


def _field_mapping_evidence(rows: list[dict]) -> SqlRiskStructuralEvidence:
    return SqlRiskStructuralEvidence(
        evidence_id="mapping_evidence",
        tool_name="list_s2t_field_mapping",
        arguments={
            "source_table": "landing.orders",
            "source_field": "customer_id",
            "target_table": "mart.orders",
            "target_field": "customer_id",
        },
        payload={
            "rows": rows,
            "total_matches": len(rows),
            "returned_rows": len(rows),
            "truncated": False,
        },
    )


def _mapping_row(sql: str | None, field: str = "customer_id") -> dict:
    return {
        "source_table": "landing.orders",
        "source_field": field,
        "target_table": "mart.orders",
        "target_field": field,
        "transformation_rule": sql,
    }


def test_composite_join_keeps_full_predicate_and_each_equality():
    sql = (
        "SELECT s.customer_id, d.segment AS segment "
        "FROM landing.orders AS s "
        "LEFT JOIN reference.customers AS d "
        "ON s.customer_id = d.customer_id "
        "AND s.region_id = d.region_id "
        "WHERE s.deleted_at IS NULL "
        "GROUP BY s.customer_id, d.segment "
        "HAVING COUNT(*) > 1 ORDER BY s.customer_id LIMIT 25 OFFSET 5"
    )

    bundle = build_sql_risk_structure_bundle(
        _table_contract(),
        [_mapping_evidence([_mapping_row(sql)])],
    )

    assert bundle.status == "ready"
    statement = bundle.rules[0].statements[0]
    assert statement.statement_kind == "SELECT"
    assert statement.joins[0].join_type == "LEFT JOIN"
    assert statement.joins[0].predicate == (
        "s.customer_id = d.customer_id AND s.region_id = d.region_id"
    )
    assert [item.sql for item in statement.joins[0].equalities] == [
        "s.customer_id = d.customer_id",
        "s.region_id = d.region_id",
    ]
    assert [item.sql for item in statement.filters] == [
        "s.deleted_at IS NULL"
    ]
    assert [item.sql for item in statement.having] == ["COUNT(*) > 1"]
    assert [item.sql for item in statement.group_by] == [
        "s.customer_id",
        "d.segment",
    ]
    assert [item.sql for item in statement.limits] == ["25"]
    assert [item.sql for item in statement.offsets] == ["5"]
    assert statement.projections[1].explicit_alias == "segment"
    assert statement.projections[1].referenced_columns == ["d.segment"]


def test_unrelated_insert_target_remains_visible_without_pair_attribution():
    sql = (
        "INSERT INTO operations.audit_log (order_id) "
        "SELECT order_id FROM landing.orders"
    )
    bundle = build_sql_risk_structure_bundle(
        _table_contract("write_semantics"),
        [_mapping_evidence([_mapping_row(sql)])],
    )

    statement = bundle.rules[0].statements[0]
    assert statement.statement_kind == "INSERT"
    assert statement.write_targets == ["operations.audit_log"]
    assert [source.name for source in statement.sources] == ["landing.orders"]
    serialized = bundle.model_dump(mode="json")
    assert "conclusion" not in str(serialized)
    assert "risk" not in str(serialized).casefold()


def test_parse_failure_is_unavailable_and_explicit_not_silently_dropped():
    bundle = build_sql_risk_structure_bundle(
        _table_contract("row_filtering"),
        [_mapping_evidence([_mapping_row("SELECT (")])],
    )

    assert bundle.status == "unavailable"
    assert bundle.rules[0].parse_status == "error"
    assert bundle.rules[0].raw_sql == "SELECT ("
    assert bundle.rules[0].parse_error
    assert [(issue.code, issue.rule_id) for issue in bundle.issues] == [
        ("sql_parse_error", "sql_rule_1")
    ]
    with pytest.raises(ValueError, match="cannot be assessed"):
        sql_risk_structure_assessment_payload(bundle)


def test_exact_duplicate_sql_is_one_rule_with_occurrence_and_mapping_counts():
    sql = "SELECT customer_id, amount FROM landing.orders"
    rows = [
        _mapping_row(sql, "customer_id"),
        _mapping_row(sql, "amount"),
        _mapping_row(" " + sql, "status"),
    ]
    bundle = build_sql_risk_structure_bundle(
        _table_contract(),
        [_mapping_evidence(rows)],
    )

    assert bundle.mapping_row_count == 3
    assert bundle.mapping_source_total == 3
    assert len(bundle.rules) == 2
    assert bundle.rules[0].raw_sql == sql
    assert bundle.rules[0].occurrence_count == 2
    assert [item.occurrence_count for item in bundle.rules[0].mappings] == [1, 1]
    assert bundle.rules[1].raw_sql == " " + sql
    assert bundle.rules[1].occurrence_count == 1
    assert bundle.rules[0].evidence_ids == ["mapping_evidence"]


def test_nullable_metadata_and_packed_mapping_are_preserved_as_neutral_values():
    sql = "SELECT customer_id FROM landing.orders"
    packed_mapping = SqlRiskStructuralEvidence(
        evidence_id="mapping_evidence",
        tool_name="read_s2t_source_to_target",
        arguments=MAPPING_ARGS,
        payload={
            "row_format": "arrays_in_column_order",
            "columns": [
                "source_table",
                "source_field",
                "target_table",
                "target_field",
                "transformation_rule",
            ],
            "dictionaries": {
                "source_table": ["landing.orders"],
                "target_table": ["mart.orders"],
                "transformation_rule": [sql],
            },
            "rows": [[0, "customer_id", 0, "customer_id", 0]],
            "total_matches": 1,
            "returned_rows": 1,
            "truncated": False,
        },
    )
    metadata = SqlRiskStructuralEvidence(
        evidence_id="metadata_evidence",
        tool_name="get_source_target_column_pair",
        arguments=METADATA_ARGS,
        payload={
            "rows": [
                {
                    "column_role": "source",
                    "file_id": 42,
                    "table_name": "landing.orders",
                    "column_name": "customer_id",
                    "data_type": "bigint",
                    "primary_key": 0,
                    "not_null": 0,
                    "description": "source customer",
                },
                {
                    "column_role": "target",
                    "file_id": 42,
                    "table_name": "mart.orders",
                    "column_name": "customer_id",
                    "data_type": "bigint",
                    "primary_key": 1,
                    "not_null": 1,
                    "description": "target customer",
                },
            ],
            "total_matches": 2,
            "returned_rows": 2,
            "truncated": False,
        },
    )

    bundle = build_sql_risk_structure_bundle(
        _nullable_contract(),
        [
            packed_mapping.model_copy(
                update={
                    "tool_name": "list_s2t_field_mapping",
                    "arguments": {
                        "source_table": "landing.orders",
                        "source_field": "customer_id",
                        "target_table": "mart.orders",
                        "target_field": "customer_id",
                    },
                }
            ),
            metadata,
        ],
    )

    assert bundle.status == "ready"
    assert bundle.mapping_row_count == 1
    assert bundle.metadata_row_count == 2
    assert [row.not_null for row in bundle.metadata_rows] == [0, 1]
    assert [row.column_role for row in bundle.metadata_rows] == [
        "source",
        "target",
    ]
    assert [item.evidence_id for item in bundle.evidence] == [
        "mapping_evidence",
        "metadata_evidence",
    ]
    assert bundle.mapping_rows[0].values == {
        "source_table": "landing.orders",
        "source_field": "customer_id",
        "target_table": "mart.orders",
        "target_field": "customer_id",
        "transformation_rule": sql,
    }


def test_nullable_metadata_preserves_exact_duplicate_role_rows():
    sql = "SELECT customer_id FROM landing.orders"
    source, target = [
        {
            "column_role": role,
            "file_id": 42,
            "table_name": table,
            "column_name": "customer_id",
            "data_type": "bigint",
            "primary_key": 0,
            "not_null": not_null,
        }
        for role, table, not_null in (
            ("source", "landing.orders", 0),
            ("target", "mart.orders", 1),
        )
    ]
    metadata_rows = [source, dict(source), target, dict(target)]
    metadata = SqlRiskStructuralEvidence(
        evidence_id="metadata_evidence",
        tool_name="get_source_target_column_pair",
        arguments=METADATA_ARGS,
        payload={
            "rows": metadata_rows,
            "total_matches": len(metadata_rows),
            "returned_rows": len(metadata_rows),
            "truncated": False,
        },
    )

    bundle = build_sql_risk_structure_bundle(
        _nullable_contract(),
        [_field_mapping_evidence([_mapping_row(sql)]), metadata],
    )

    assert bundle.status == "ready"
    assert bundle.metadata_row_count == 4
    assert [row.column_role for row in bundle.metadata_rows] == [
        "source",
        "source",
        "target",
        "target",
    ]
    assert [row.not_null for row in bundle.metadata_rows] == [0, 0, 1, 1]


def test_nullable_metadata_rejects_any_foreign_row():
    sql = "SELECT customer_id FROM landing.orders"
    source = {
        "column_role": "source",
        "file_id": 42,
        "table_name": "landing.orders",
        "column_name": "customer_id",
        "not_null": 0,
    }
    target = {
        "column_role": "target",
        "file_id": 42,
        "table_name": "mart.orders",
        "column_name": "customer_id",
        "not_null": 1,
    }
    foreign = {**target, "table_name": "mart.other_orders"}
    metadata_rows = [source, target, foreign]
    metadata = SqlRiskStructuralEvidence(
        evidence_id="metadata_evidence",
        tool_name="get_source_target_column_pair",
        arguments=METADATA_ARGS,
        payload={
            "rows": metadata_rows,
            "total_matches": len(metadata_rows),
            "returned_rows": len(metadata_rows),
            "truncated": False,
        },
    )

    bundle = build_sql_risk_structure_bundle(
        _nullable_contract(),
        [_field_mapping_evidence([_mapping_row(sql)]), metadata],
    )

    assert bundle.status == "unavailable"
    assert [issue.code for issue in bundle.issues] == ["incomplete_payload"]
    assert bundle.metadata_rows == []


def test_set_distinct_qualify_and_referenced_columns_are_exposed():
    sql = (
        "SELECT DISTINCT order_id FROM landing.orders "
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at) = 1 "
        "UNION ALL SELECT order_id FROM archive.orders"
    )
    bundle = build_sql_risk_structure_bundle(
        _table_contract("row_filtering"),
        [_mapping_evidence([_mapping_row(sql)])],
    )

    assert bundle.status == "ready"
    statement = bundle.rules[0].statements[0]
    assert statement.set_operations[0].operation == "UNION"
    assert statement.set_operations[0].distinct is False
    assert statement.distinct_query_blocks == ["query_1"]
    assert [item.sql for item in statement.qualify] == [
        "ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at) = 1"
    ]
    assert {source.name for source in statement.sources} == {
        "landing.orders",
        "archive.orders",
    }
    assert {"order_id", "customer_id", "created_at"}.issubset(
        set(statement.referenced_columns)
    )


def test_truncation_and_size_overflow_are_explicit_unavailable_states():
    sql = "SELECT customer_id FROM landing.orders"
    truncated = _mapping_evidence([_mapping_row(sql)])
    truncated_payload = truncated.model_copy(
        update={"payload": {**truncated.payload, "truncated": True}}
    )

    incomplete = build_sql_risk_structure_bundle(
        _table_contract(),
        [truncated_payload],
    )
    oversized = build_sql_risk_structure_bundle(
        _table_contract(),
        [_mapping_evidence([_mapping_row(sql)])],
        max_bundle_chars=100,
    )

    assert incomplete.status == "unavailable"
    assert [issue.code for issue in incomplete.issues] == ["incomplete_payload"]
    assert oversized.status == "unavailable"
    assert [issue.code for issue in oversized.issues] == ["bundle_too_large"]
    assert oversized.rules == []
    with pytest.raises(ValueError, match="cannot be assessed"):
        sql_risk_structure_assessment_payload(oversized)


def test_structure_cap_reserves_space_for_the_complete_assessment_envelope():
    assert SQL_RISK_ASSESSMENT_ENVELOPE_RESERVE_CHARS == 45_000
    assert SQL_RISK_STRUCTURE_MAX_CHARS == (
        MAX_SQL_RISK_ASSESSMENT_INPUT_CHARS
        - SQL_RISK_ASSESSMENT_ENVELOPE_RESERVE_CHARS
    )
