from __future__ import annotations

from unittest.mock import patch

import pytest

import storage.database as db_storage
from agents.entity_resolution import (
    EntityMention,
    resolve_entity,
    resolve_entity_batch,
    verify_exact_entity,
)
from storage.database import get_db_connection, init_db


@pytest.fixture(autouse=True)
def _entity_db(tmp_path):
    original = db_storage.DB_PATH
    db_storage.DB_PATH = str(tmp_path / "entity-resolution.db")
    init_db()
    conn = get_db_connection()
    conn.executemany(
        """
        INSERT INTO files (file_id, filename, upload_time, description)
        VALUES (?, ?, ?, ?)
        """,
        [
            (1, "Orders Catalog.xlsx", "2026-01-01", "Каталог заказов"),
            (2, "Risk Register.xlsx", "2026-01-02", "Реестр рисков"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, source_table, source_field,
         target_table, target_field, transformation_rule)
        VALUES (?, ?, 'S2T', ?, ?, 'id', ?, 'id', 'SELECT id FROM source')
        """,
        [
            (1, 1, 1, "stage.customer_orders", "mart.customer_orders"),
            (2, 1, 2, "stage.customer_orders", "mart.customer_orders"),
            (3, 1, 3, "stage.payments", "mart.customer_payments"),
            (4, 1, 4, "shared.entity", "sink.one"),
            (5, 1, 5, "root.one", "shared.entity"),
            (6, 1, 6, "source_customer_current", "target.current"),
            (7, 1, 7, "source_customer_history", "target.history"),
            (8, 2, 8, "source.risk", "mart.risk_profile"),
            (9, 2, 9, "source.audit", "mart.audit_profile"),
            (10, 2, 10, "stage.customer_daily", "target.daily"),
            (11, 2, 11, "stage.customer_dairy", "target.dairy"),
        ],
    )
    conn.commit()
    conn.close()
    try:
        yield
    finally:
        db_storage.DB_PATH = original


def _request(
    mention: str,
    *,
    role: str = "source",
    strategy: str = "auto",
) -> EntityMention:
    return EntityMention(
        mention=mention,
        entity_type="file" if role == "file" else "table",
        role=role,
        strategy=strategy,
    )


def test_exact_table_short_circuits_without_semantic_call_and_keeps_provenance():
    with patch(
        "agents.entity_resolution._semantic_search"
    ) as semantic:
        result = resolve_entity(_request("STAGE.CUSTOMER_ORDERS"))

    assert result.status == "resolved"
    assert result.method == "exact"
    assert result.canonical_name == "stage.customer_orders"
    assert result.candidate_set.coverage == "complete"
    assert len(result.candidate_set.candidates) == 1
    assert len(result.candidate_set.candidates[0].provenance) == 2
    semantic.assert_not_called()


def test_exact_verifier_never_enters_approximate_candidate_search():
    with (
        patch("agents.entity_resolution._candidate_universe") as universe,
        patch("agents.entity_resolution._semantic_search") as semantic,
    ):
        exact = verify_exact_entity(_request("stage.customer_orders"))
        missing = verify_exact_entity(_request("stage.customer_ordres"))

    assert exact is not None and exact.method == "exact"
    assert missing is None
    universe.assert_not_called()
    semantic.assert_not_called()


def test_exact_filename_uses_existing_resolver_and_skips_semantic():
    with patch(
        "agents.entity_resolution._semantic_search"
    ) as semantic:
        result = resolve_entity(_request("orders catalog.XLSX", role="file"))

    assert result.status == "resolved"
    assert result.method == "exact"
    assert result.file_id == 1
    assert result.canonical_name == "Orders Catalog.xlsx"
    assert result.candidate_set.source == "resolve_file"
    semantic.assert_not_called()


def test_normalized_exact_resolves_quoted_qualified_table_name():
    with patch("agents.entity_resolution._semantic_search") as semantic:
        result = resolve_entity(
            _request(
                ' "stage" . "customer_orders" ',
                strategy="semantic",
            )
        )

    assert result.status == "resolved"
    assert result.method == "normalized_exact"
    assert result.canonical_name == "stage.customer_orders"
    semantic.assert_not_called()


def test_unique_partial_name_resolves_before_fuzzy_or_semantic():
    with patch(
        "agents.entity_resolution._semantic_search"
    ) as semantic:
        result = resolve_entity(_request("payments"))

    assert result.status == "resolved"
    assert result.method == "partial"
    assert result.canonical_name == "stage.payments"
    semantic.assert_not_called()


def test_typo_uses_fuzzy_threshold_and_gap():
    with patch(
        "agents.entity_resolution._semantic_search"
    ) as semantic:
        result = resolve_entity(_request("stage.customer_ordres"))

    assert result.status == "resolved"
    assert result.method == "fuzzy"
    assert result.canonical_name == "stage.customer_orders"
    assert result.candidate_set.threshold == pytest.approx(0.84)
    assert result.candidate_set.minimum_gap == pytest.approx(0.06)
    assert result.candidate_set.candidates[0].score > 0.94
    semantic.assert_not_called()


def test_ambiguous_partial_keeps_every_candidate_and_never_selects_one():
    result = resolve_entity(_request("source_customer"))

    assert result.status == "ambiguous"
    assert result.method == "partial"
    assert result.error_code == "ambiguous_entity"
    assert result.canonical_name is None
    assert {
        candidate.canonical_name for candidate in result.candidate_set.candidates
    } == {"source_customer_current", "source_customer_history"}


def test_fuzzy_gap_keeps_close_typo_candidates_ambiguous():
    with patch("agents.entity_resolution._semantic_search") as semantic:
        result = resolve_entity(_request("stage.customer_darly"))

    assert result.status == "ambiguous"
    assert result.method == "fuzzy"
    assert result.canonical_name is None
    assert {
        candidate.canonical_name for candidate in result.candidate_set.candidates
    } == {"stage.customer_daily", "stage.customer_dairy"}
    assert all(
        candidate.score >= result.candidate_set.threshold
        for candidate in result.candidate_set.candidates
    )
    semantic.assert_not_called()


def test_duplicate_exact_filenames_are_ambiguous_by_file_id():
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO files (file_id, filename, upload_time)
        VALUES (3, 'Orders Catalog.xlsx', '2026-01-03')
        """
    )
    conn.commit()
    conn.close()

    result = resolve_entity(_request("Orders Catalog.xlsx", role="file"))

    assert result.status == "ambiguous"
    assert result.method == "exact"
    assert result.file_id is None
    assert [candidate.file_id for candidate in result.candidate_set.candidates] == [
        3,
        1,
    ]


def test_semantic_file_resolution_reuses_ranked_file_search():
    semantic_payload = {
        "scope": "files",
        "total_candidates": 2,
        "returned_rows": 2,
        "rows": [
            {
                "scope": "files",
                "record_id": 2,
                "file_id": 2,
                "filename": "Risk Register.xlsx",
                "name": "Risk Register.xlsx",
                "description": "Реестр рисков",
                "score": 0.91,
            },
            {
                "scope": "files",
                "record_id": 1,
                "file_id": 1,
                "filename": "Orders Catalog.xlsx",
                "name": "Orders Catalog.xlsx",
                "description": "Каталог заказов",
                "score": 0.42,
            },
        ],
    }
    with patch(
        "agents.entity_resolution._semantic_search"
    ) as semantic:
        semantic.return_value = semantic_payload
        result = resolve_entity(
            _request(
                "файл с реестром операционных рисков",
                role="file",
                strategy="semantic",
            )
        )

    assert result.status == "resolved"
    assert result.method == "semantic"
    assert result.canonical_name == "Risk Register.xlsx"
    assert result.file_id == 2
    assert semantic.call_args.args[0]["scope"] == "files"


def test_semantic_table_resolution_is_role_scoped_and_preserves_duplicates():
    semantic_payload = {
        "scope": "target_tables",
        "total_candidates": 3,
        "returned_rows": 3,
        "rows": [
            {
                "scope": "target_tables",
                "record_id": 20,
                "file_id": 2,
                "name": "mart.risk_profile",
                "description": "Профиль риска",
                "score": 0.93,
            },
            {
                "scope": "target_tables",
                "record_id": 21,
                "file_id": 2,
                "name": "mart.risk_profile",
                "description": "Риски клиента",
                "score": 0.92,
            },
            {
                "scope": "target_tables",
                "record_id": 22,
                "file_id": 2,
                "name": "mart.audit_profile",
                "description": "Аудит",
                "score": 0.61,
            },
        ],
    }
    with patch(
        "agents.entity_resolution._semantic_search"
    ) as semantic:
        semantic.return_value = semantic_payload
        result = resolve_entity(
            _request("витрина профиля риска", role="target", strategy="semantic")
        )

    assert result.status == "resolved"
    assert result.canonical_name == "mart.risk_profile"
    assert result.role == "target"
    assert len(result.candidate_set.candidates) == 2
    assert len(result.candidate_set.candidates[0].provenance) == 2
    assert semantic.call_args.args[0]["scope"] == "target_tables"


def test_semantic_table_candidate_is_verified_in_role_specific_global_s2t():
    semantic_payload = {
        "scope": "target_tables",
        "total_candidates": 2,
        "returned_rows": 2,
        "rows": [
            {
                "scope": "target_tables",
                "record_id": 30,
                "file_id": 2,
                "name": "catalog.only_table",
                "score": 0.99,
            },
            {
                "scope": "target_tables",
                "record_id": 31,
                "file_id": 2,
                "name": "mart.risk_profile",
                "score": 0.90,
            },
        ],
    }
    with patch(
        "agents.entity_resolution._semantic_search",
        return_value=semantic_payload,
    ):
        result = resolve_entity(
            _request("профиль риска", role="target", strategy="semantic")
        )

    assert result.status == "resolved"
    assert result.canonical_name == "mart.risk_profile"
    assert [
        candidate.canonical_name for candidate in result.candidate_set.candidates
    ] == ["mart.risk_profile"]


def test_source_and_target_roles_resolve_independently_for_same_name():
    source, target = resolve_entity_batch(
        [
            _request("shared.entity", role="source"),
            _request("shared.entity", role="target"),
        ]
    )

    assert source.status == target.status == "resolved"
    assert source.canonical_name == target.canonical_name == "shared.entity"
    assert source.role == "source"
    assert target.role == "target"
    assert {
        row["record_id"] for row in source.candidate_set.candidates[0].provenance
    } == {4}
    assert {
        row["record_id"] for row in target.candidate_set.candidates[0].provenance
    } == {5}


def test_truncated_semantic_ambiguity_retains_all_returned_candidates():
    semantic_payload = {
        "scope": "files",
        "total_candidates": 7,
        "returned_rows": 3,
        "rows": [
            {
                "scope": "files",
                "record_id": 101,
                "file_id": 101,
                "filename": "Risk A.xlsx",
                "score": 0.91,
            },
            {
                "scope": "files",
                "record_id": 102,
                "file_id": 102,
                "filename": "Risk B.xlsx",
                "score": 0.89,
            },
            {
                "scope": "files",
                "record_id": 103,
                "file_id": 103,
                "filename": "Risk C.xlsx",
                "score": 0.70,
            },
        ],
    }
    with patch(
        "agents.entity_resolution._semantic_search"
    ) as semantic:
        semantic.return_value = semantic_payload
        result = resolve_entity(
            _request("файл о рисках", role="file", strategy="semantic")
        )

    assert result.status == "ambiguous"
    assert result.canonical_name is None
    assert result.candidate_set.coverage == "truncated"
    assert result.candidate_set.total_candidates == 7
    assert [
        candidate.canonical_name for candidate in result.candidate_set.candidates
    ] == ["Risk A.xlsx", "Risk B.xlsx", "Risk C.xlsx"]
