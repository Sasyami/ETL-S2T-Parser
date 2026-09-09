"""Deterministic write-semantics analysis over accepted saved evidence."""

import json

import pytest

from agents.contracts import EvidenceArtifact
from agents.tools.saved_results import SavedResultStore
from agents.write_semantics_analysis import (
    MAX_WRITE_SEMANTICS_PAYLOAD_CHARS,
    classify_write_semantics_rule,
    derive_write_semantics_facts,
    extract_exact_table_pairs,
    is_exclusive_write_semantics_request,
    render_terminal_write_semantics_negative,
    render_write_semantics_answer,
    write_semantics_payload,
)


SOURCE_TABLE = "src_np"
TARGET_TABLE = "tgt_np"


def _mapping_row(rule: object, **overrides: object) -> dict:
    row = {
        "source_table": SOURCE_TABLE,
        "source_field": "id",
        "target_table": TARGET_TABLE,
        "target_field": "id",
        "transformation_rule": rule,
    }
    row.update(overrides)
    return row


def _saved_artifact(
    store: SavedResultStore,
    rows: list[dict],
    *,
    artifact_tool: str = "read_s2t_source_to_target",
    saved_tool: str | None = None,
    source_table: str = SOURCE_TABLE,
    target_table: str = TARGET_TABLE,
    evidence_id: str = "evidence-write",
    preview: str = "",
    artifact_truncated: bool = False,
    source_truncated: bool = False,
) -> EvidenceArtifact:
    descriptor = store.save_payload(
        source_tool=saved_tool or artifact_tool,
        source_tool_call_id=evidence_id,
        payload={"rows": rows, "truncated": source_truncated},
    )
    assert descriptor is not None
    return EvidenceArtifact(
        evidence_id=evidence_id,
        tool_name=artifact_tool,
        compact_args={
            "source_table": source_table,
            "target_table": target_table,
        },
        preview=preview,
        truncated=artifact_truncated,
        dataset_ref=descriptor.result_ref,
    )


def test_extracts_quoted_schema_pairs_and_deduplicates_casefolded_values():
    pairs = extract_exact_table_pairs(
        "`src_np` → `tgt_np`; \"raw.sales\" -> mart.sales; "
        "SRC_NP => TGT_NP; $$raw.events -> $$mart.events"
    )

    assert [pair.model_dump() for pair in pairs] == [
        {"source_table": SOURCE_TABLE, "target_table": TARGET_TABLE},
        {"source_table": "raw.sales", "target_table": "mart.sales"},
        {"source_table": "$$raw.events", "target_table": "$$mart.events"},
    ]


def test_current_live_wording_is_an_exclusive_write_semantics_request():
    task = (
        "Оцени только SQL-аспект write semantics для сохранённой "
        f"S2T-загрузки {SOURCE_TABLE} → {TARGET_TABLE}: append, overwrite, "
        "MERGE/UPSERT или conflict handling. Если write statement не "
        "сохранён, честно отметь «не оценено» и не выводи режим из PK."
    )

    assert is_exclusive_write_semantics_request(task) is True


@pytest.mark.parametrize(
    "task",
    [
        f"Оцени write semantics для {SOURCE_TABLE} → {TARGET_TABLE}.",
        (
            f"Оцени только write semantics для {SOURCE_TABLE} → "
            f"{TARGET_TABLE} и покажи полный результат."
        ),
        (
            f"Оцени только write semantics и value changes для "
            f"{SOURCE_TABLE} → {TARGET_TABLE}."
        ),
        "Оцени только write semantics без точной пары.",
        "",
    ],
)
def test_exclusive_predicate_rejects_broad_or_display_requests(task):
    assert is_exclusive_write_semantics_request(task) is False


@pytest.mark.parametrize(
    "rule",
    [
        None,
        "",
        "   ",
        "-",
        "SELECT * FROM src_np",
        "WITH x AS (SELECT * FROM src_np) SELECT * FROM x",
        "SELECT 'MERGE INTO target' AS harmless_text FROM src_np",
        "-- INSERT INTO target\nSELECT * FROM src_np",
    ],
)
def test_select_with_and_empty_rules_have_no_write_statement(rule):
    result = classify_write_semantics_rule(rule)

    assert result.status == "absent"
    assert result.mechanism == "write_statement_absent"
    assert result.detected_mechanisms == []


def test_non_text_rule_fails_conservatively():
    result = classify_write_semantics_rule(0)

    assert result.status == "unknown"
    assert result.mechanism == "unparseable_rule"


@pytest.mark.parametrize(
    ("rule", "mechanism"),
    [
        ("INSERT INTO tgt_np SELECT * FROM src_np", "insert"),
        (
            "INSERT INTO tgt_np(id) VALUES (1) "
            "ON CONFLICT (id) DO NOTHING",
            "insert_on_conflict",
        ),
        (
            "INSERT OVERWRITE TABLE tgt_np SELECT * FROM src_np",
            "insert_overwrite",
        ),
        ("UPDATE tgt_np SET id = 1", "update"),
        (
            "MERGE INTO tgt_np USING src_np ON tgt_np.id=src_np.id "
            "WHEN MATCHED THEN UPDATE SET id=src_np.id "
            "WHEN NOT MATCHED THEN INSERT (id) VALUES(src_np.id)",
            "merge",
        ),
        ("DELETE FROM tgt_np WHERE id = 1", "delete"),
        ("CREATE TABLE tgt_np AS SELECT * FROM src_np", "ctas"),
        ("SELECT * INTO tgt_np FROM src_np", "ctas"),
    ],
)
def test_root_write_statement_classification(rule, mechanism):
    result = classify_write_semantics_rule(rule)

    assert result.status == "explicit"
    assert result.mechanism == mechanism
    assert result.detected_mechanisms == [mechanism]


@pytest.mark.parametrize(
    ("rule", "mechanism"),
    [
        ("SELECT (", "unparseable_rule"),
        ("DROP TABLE tgt_np", "unsupported_statement"),
        (
            "SELECT 1; INSERT INTO tgt_np VALUES (1)",
            "multiple_statements",
        ),
    ],
)
def test_unknown_statement_shapes_fail_conservatively(rule, mechanism):
    result = classify_write_semantics_rule(rule)

    assert result.status == "unknown"
    assert result.mechanism == mechanism


@pytest.mark.parametrize(
    ("rule", "mechanism"),
    [
        (
            "WITH moved AS (DELETE FROM src_np WHERE done RETURNING *) "
            "SELECT * FROM moved",
            "delete",
        ),
        (
            "WITH changed AS (UPDATE src_np SET id=1 RETURNING *) "
            "SELECT * FROM changed",
            "update",
        ),
        (
            "WITH added AS (INSERT INTO tgt_np SELECT * FROM src_np "
            "RETURNING *) SELECT * FROM added",
            "insert",
        ),
    ],
)
def test_data_modifying_cte_is_detected_structurally(rule, mechanism):
    result = classify_write_semantics_rule(rule)

    assert result.status == "explicit"
    assert result.mechanism == mechanism


def test_complete_select_only_mapping_is_terminal_not_assessed():
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [
                _mapping_row("SELECT s.id AS id FROM src_np AS s"),
                _mapping_row(
                    "WITH x AS (SELECT * FROM src_np) SELECT * FROM x",
                    source_field="value",
                    target_field="value",
                ),
            ],
            # Neither the clipped public preview nor its flag is authoritative.
            preview="MERGE INTO tgt_np USING src_np ...",
            artifact_truncated=True,
        )
        fact = derive_write_semantics_facts(
            f"{SOURCE_TABLE} → {TARGET_TABLE}",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "write_statement_absent"
    assert fact.matching_rows == 2
    assert fact.evidence_ids == ["evidence-write"]
    answer = render_terminal_write_semantics_negative([fact])
    assert "write semantics не оценено" in answer
    assert "нельзя выводить из PK/UNIQUE" in answer
    assert "MERGE INTO tgt_np USING" not in answer


def test_more_than_100_duplicate_rows_use_distinct_rule_aggregation():
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [_mapping_row("SELECT * FROM src_np") for _ in range(150)],
        )
        fact = derive_write_semantics_facts(
            f"{SOURCE_TABLE} → {TARGET_TABLE}",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "write_statement_absent"
    assert fact.matching_rows == 150


def test_one_explicit_write_among_select_rules_is_reported():
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [
                _mapping_row("SELECT * FROM src_np"),
                _mapping_row(
                    "INSERT INTO tgt_np SELECT * FROM src_np",
                    source_field="value",
                    target_field="value",
                ),
            ],
        )
        fact = derive_write_semantics_facts(
            f"{SOURCE_TABLE} → {TARGET_TABLE}",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "explicit_write"
    assert fact.mechanism == "insert"
    assert fact.detected_mechanisms == ["insert"]
    assert render_terminal_write_semantics_negative([fact]) == ""
    assert "`insert`" in render_write_semantics_answer([fact])


def test_different_explicit_write_modes_are_conflicting():
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [
                _mapping_row("INSERT INTO tgt_np SELECT * FROM src_np"),
                _mapping_row(
                    "MERGE INTO tgt_np USING src_np "
                    "ON tgt_np.id=src_np.id WHEN MATCHED THEN "
                    "UPDATE SET id=src_np.id",
                    source_field="value",
                    target_field="value",
                ),
            ],
        )
        fact = derive_write_semantics_facts(
            f"{SOURCE_TABLE} → {TARGET_TABLE}",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "conflicting"
    assert fact.mechanism == "multiple"
    assert fact.detected_mechanisms == ["insert", "merge"]


def test_conflicting_data_modifying_ctes_are_not_collapsed_to_select():
    rule = (
        "WITH changed AS (UPDATE src_np SET id=1 RETURNING *), "
        "removed AS (DELETE FROM src_np WHERE id=2 RETURNING *) "
        "SELECT * FROM changed"
    )

    result = classify_write_semantics_rule(rule)

    assert result.status == "conflicting"
    assert result.mechanism == "multiple"
    assert result.detected_mechanisms == ["delete", "update"]


@pytest.mark.parametrize(
    ("mutation", "expected_mechanism"),
    [
        ("missing_dataset", "incomplete_evidence"),
        ("source_tool_mismatch", "incomplete_evidence"),
        ("source_truncated", "incomplete_evidence"),
        ("missing_column", "incomplete_evidence"),
        ("unparseable_rule", "unparseable_rule"),
    ],
)
def test_invalid_or_unknown_full_evidence_is_incomplete(
    mutation,
    expected_mechanism,
):
    store = SavedResultStore()
    try:
        rows = [_mapping_row("SELECT * FROM src_np")]
        kwargs = {}
        if mutation == "source_tool_mismatch":
            kwargs["saved_tool"] = "read_s2t_mapping"
        elif mutation == "source_truncated":
            kwargs["source_truncated"] = True
        elif mutation == "missing_column":
            rows = [
                {
                    "source_table": SOURCE_TABLE,
                    "target_table": TARGET_TABLE,
                    "source_field": "id",
                }
            ]
        elif mutation == "unparseable_rule":
            rows = [_mapping_row("SELECT (")]
        artifact = _saved_artifact(store, rows, **kwargs)
        if mutation == "missing_dataset":
            artifact = artifact.model_copy(update={"dataset_ref": None})
        fact = derive_write_semantics_facts(
            f"{SOURCE_TABLE} → {TARGET_TABLE}",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "incomplete"
    assert fact.mechanism == expected_mechanism
    assert render_terminal_write_semantics_negative([fact]) == ""


def test_non_exact_tool_and_reversed_exact_artifact_do_not_count():
    store = SavedResultStore()
    try:
        non_exact = _saved_artifact(
            store,
            [_mapping_row("SELECT * FROM src_np")],
            artifact_tool="search_s2t_transformations",
            saved_tool="search_s2t_transformations",
            evidence_id="search",
        )
        reversed_artifact = _saved_artifact(
            store,
            [_mapping_row("SELECT * FROM src_np")],
            source_table=TARGET_TABLE,
            target_table=SOURCE_TABLE,
            evidence_id="reversed",
        )
        fact = derive_write_semantics_facts(
            f"{SOURCE_TABLE} → {TARGET_TABLE}",
            [non_exact, reversed_artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "incomplete"
    assert fact.mechanism == "incomplete_evidence"
    assert fact.evidence_ids == []


def test_complete_dataset_without_exact_rows_is_not_assessed_but_not_terminal():
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [
                _mapping_row(
                    "SELECT * FROM another_src",
                    source_table="another_src",
                )
            ],
        )
        fact = derive_write_semantics_facts(
            f"{SOURCE_TABLE} → {TARGET_TABLE}",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "not_assessed"
    assert fact.mechanism == "no_exact_mapping"
    assert fact.matching_rows == 0
    assert render_terminal_write_semantics_negative([fact]) == ""


def test_more_than_100_distinct_rules_is_incomplete():
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [
                _mapping_row(f"SELECT {number} AS id FROM src_np")
                for number in range(101)
            ],
        )
        fact = derive_write_semantics_facts(
            f"{SOURCE_TABLE} → {TARGET_TABLE}",
            [artifact],
            store,
        )[0]
    finally:
        store.close()

    assert fact.conclusion == "incomplete"
    assert fact.mechanism == "incomplete_evidence"


def test_payload_is_bounded_and_contains_no_transformation_rules():
    store = SavedResultStore()
    try:
        artifact = _saved_artifact(
            store,
            [_mapping_row("SELECT * FROM src_np")],
        )
        facts = derive_write_semantics_facts(
            f"{SOURCE_TABLE} → {TARGET_TABLE}",
            [artifact],
            store,
        )
        payload = write_semantics_payload(facts)
    finally:
        store.close()

    serialized = json.dumps(payload, ensure_ascii=False)
    assert len(serialized) <= MAX_WRITE_SEMANTICS_PAYLOAD_CHARS
    assert payload["authority"] == "deterministic_sqlglot_full_saved_result"
    assert "transformation_rule" not in serialized
    assert "SELECT * FROM" not in serialized


def test_payload_has_a_hard_bound_for_many_long_but_valid_facts():
    from agents.write_semantics_analysis import WriteSemanticsFact

    facts = [
        WriteSemanticsFact(
            source_table="s" * 240,
            target_table="t" * 240,
            conclusion="not_assessed",
            mechanism="write_statement_absent",
            matching_rows=1,
            evidence_ids=["e" * 2_000 for _ in range(20)],
        )
        for _ in range(20)
    ]

    payload = write_semantics_payload(facts)

    assert len(json.dumps(payload, ensure_ascii=False)) <= (
        MAX_WRITE_SEMANTICS_PAYLOAD_CHARS
    )
    assert payload["payload_truncated"] is True
