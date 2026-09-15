from __future__ import annotations

from unittest.mock import patch

import pytest

import storage.database as db_storage
from agents.test_protocol import RawTestProtocolContract, RawTestProtocolLoad
from agents.test_protocol_resolution import (
    resolve_test_protocol_contract,
    validate_raw_contract_origin,
)
from storage.database import get_db_connection, init_db


@pytest.fixture(autouse=True)
def _resolution_db(tmp_path):
    original = db_storage.DB_PATH
    db_storage.DB_PATH = str(tmp_path / "protocol-resolution.db")
    init_db()
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO files (file_id, filename, upload_time, description)
        VALUES (7, 'Mapping.xlsx', '2026-01-01', 'S2T mapping')
        """
    )
    conn.executemany(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, source_table, source_field,
         target_table, target_field, transformation_rule)
        VALUES (?, 7, 'S2T', ?, ?, 'id', ?, 'id',
                'SELECT id FROM stage.orders')
        """,
        [
            (1, 1, "stage.orders", "mart.orders"),
            (2, 2, "stage.order_items", "mart.order_items"),
            (3, 3, "stage.order_lines", "mart.order_lines"),
        ],
    )
    conn.commit()
    conn.close()
    try:
        yield
    finally:
        db_storage.DB_PATH = original


def _raw(
    source: str,
    target: str,
    *,
    file_id: int | None = None,
    file_mention: str | None = None,
) -> RawTestProtocolContract:
    if file_id is not None:
        file_scope_kind = "file_id"
    elif file_mention is not None:
        file_scope_kind = "file_mention"
    else:
        file_scope_kind = "not_provided"
    return RawTestProtocolContract(
        file_scope_kind=file_scope_kind,
        file_id=file_id,
        file_mention=file_mention,
        loads=[
            RawTestProtocolLoad(
                source_mentions=[source],
                target_mention=target,
                requested_checks=["row_count"],
            )
        ],
        mode="explicit",
    )


def test_exact_tables_bypass_approximate_resolver_and_keep_roles():
    with patch(
        "agents.test_protocol_resolution.resolve_entity"
    ) as approximate:
        result = resolve_test_protocol_contract(
            _raw("stage.orders", "mart.orders", file_id=7)
        )

    assert result.status == "resolved"
    assert result.exact_bypass_count == 2
    assert result.contract is not None
    assert result.contract.file_id == 7
    assert result.contract.loads[0].sources == ["stage.orders"]
    assert result.contract.loads[0].target == "mart.orders"
    assert [item.role for item in result.resolutions] == ["source", "target"]
    approximate.assert_not_called()


def test_typo_enters_shared_resolver_then_materializes_canonical_contract():
    result = resolve_test_protocol_contract(
        _raw("stage.ordres", "mart.orders")
    )

    assert result.status == "resolved"
    assert result.contract is not None
    assert result.contract.loads[0].sources == ["stage.orders"]
    assert [item.method for item in result.resolutions] == ["fuzzy", "exact"]
    assert result.exact_bypass_count == 1


def test_ambiguous_partial_returns_all_candidates_without_contract():
    result = resolve_test_protocol_contract(
        _raw("stage.order_", "mart.orders")
    )

    assert result.status == "ambiguous_entity"
    assert result.contract is None
    issue = result.issues[0]
    assert issue.code == "ambiguous_entity"
    assert set(issue.candidates) == {
        "stage.orders",
        "stage.order_items",
        "stage.order_lines",
    }
    ambiguous = result.resolutions[0]
    assert ambiguous.canonical_name is None
    assert {
        item.canonical_name for item in ambiguous.candidate_set.candidates
    } == {"stage.orders", "stage.order_items", "stage.order_lines"}


def test_protocol_without_file_resolves_tables_and_preserves_none_scope():
    result = resolve_test_protocol_contract(
        _raw("stage.orders", "mart.orders")
    )

    assert result.status == "resolved"
    assert result.contract is not None
    assert result.contract.file_id is None


def test_exact_filename_reuses_file_resolver_and_materializes_only_file_id():
    result = resolve_test_protocol_contract(
        _raw(
            "stage.orders",
            "mart.orders",
            file_mention="mapping.XLSX",
        )
    )

    assert result.status == "resolved"
    assert result.contract is not None
    assert result.contract.file_id == 7
    assert result.contract.filename is None
    assert result.resolutions[0].entity_type == "file"
    assert result.resolutions[0].method == "exact"


def test_resolved_file_scopes_approximate_table_resolution_across_uploads():
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO files (file_id, filename, upload_time, description)
        VALUES (8, 'Other Mapping.xlsx', '2026-01-02', 'Other S2T mapping')
        """
    )
    conn.executemany(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, source_table, source_field,
         target_table, target_field, transformation_rule)
        VALUES (?, ?, 'S2T', ?, ?, 'id', ?, 'id', 'SELECT id FROM source')
        """,
        [
            (10, 7, 10, "stage.scope_orders", "mart.orders"),
            (11, 8, 11, "stage.scope_order_items", "mart.other_orders"),
        ],
    )
    conn.commit()
    conn.close()

    result = resolve_test_protocol_contract(
        _raw(
            "stage.scope_order",
            "mart.orders",
            file_mention="Mapping.xlsx",
        )
    )

    assert result.status == "resolved"
    assert result.contract is not None
    assert result.contract.file_id == 7
    assert result.contract.loads[0].sources == ["stage.scope_orders"]
    assert result.contract.loads[0].target == "mart.orders"
    source_resolution = next(
        item for item in result.resolutions if item.role == "source"
    )
    assert source_resolution.method == "partial"
    assert {
        row["file_id"]
        for candidate in source_resolution.candidate_set.candidates
        for row in candidate.provenance
    } == {7}


def test_raw_origin_validation_happens_before_canonical_resolution():
    raw = _raw("stage.ordres", "mart.orders", file_id=7)
    validate_raw_contract_origin(
        raw,
        "Для file_id=7 проверь stage.ordres → mart.orders",
    )

    with pytest.raises(ValueError, match="не из original_task"):
        validate_raw_contract_origin(
            raw,
            "Для file_id=7 проверь stage.orders → mart.orders",
        )
