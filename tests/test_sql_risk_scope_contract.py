"""Pure contracts for the opt-in SQL-risk scope/evidence experiment."""

from __future__ import annotations

import pytest

from agents.sql_risk_scope_contract import (
    GetSourceTargetColumnPairRequirement,
    OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV,
    ReadS2TSourceToTargetRequirement,
    build_sql_risk_scope_contract,
    ensure_sql_risk_answer_scope,
    extract_literal_file_id,
    extract_literal_sql_risk_scope,
    missing_sql_risk_requirements,
    render_sql_risk_scope_contract,
    required_sql_risk_tools,
    sql_risk_scope_evidence_architecture,
    sql_risk_scope_evidence_enabled,
)


TABLE_TASK = (
    "Проверь риск для raw_customer_events_v2 → mart_customer_daily_v3."
)
FIELD_TASK = (
    "Для file_id=417 проверь nullable для точной пары колонок "
    "raw_customer_events_v2.customer_key → "
    "mart_customer_daily_v3.customer_key."
)
QUALIFIED_FIELD_TASK = (
    "Для file_id=417 проверь stg.raw_customer_events_v2.customer_key → "
    "dwh.mart_customer_daily_v3.customer_key."
)


def test_experiment_is_off_by_default_and_preserves_answer_byte_for_byte(
    monkeypatch,
):
    monkeypatch.delenv(
        OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV,
        raising=False,
    )
    answer = "  Ответ с Unicode → и хвостовым переводом строки.\n"

    assert sql_risk_scope_evidence_enabled() is False
    assert build_sql_risk_scope_contract(
        TABLE_TASK,
        ["row_filtering"],
    ) is None
    assert required_sql_risk_tools(TABLE_TASK, ["row_filtering"]) == ()
    assert ensure_sql_risk_answer_scope(
        answer,
        TABLE_TASK,
        ["row_filtering"],
    ) == answer


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", "off"),
        ("0", "off"),
        ("false", "off"),
        ("default", "off"),
        ("1", "prompt"),
        ("TRUE", "prompt"),
        (" enabled ", "prompt"),
        ("typed_plan", "typed_plan"),
        (" TYPED_PLAN ", "typed_plan"),
    ],
)
def test_scope_evidence_architecture_preserves_legacy_values(
    value,
    expected,
):
    assert sql_risk_scope_evidence_architecture(value) == expected


def test_typed_plan_mode_enables_the_same_exact_scope_contract(monkeypatch):
    monkeypatch.setenv(
        OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV,
        "typed_plan",
    )

    assert sql_risk_scope_evidence_architecture() == "typed_plan"
    assert sql_risk_scope_evidence_enabled() is True
    contract = build_sql_risk_scope_contract(
        TABLE_TASK,
        ["cardinality"],
    )

    assert contract is not None
    assert contract.scope.label == (
        "raw_customer_events_v2 → mart_customer_daily_v3"
    )


@pytest.mark.parametrize("value", ["1", "TRUE", " yes ", "on", "enabled"])
def test_explicit_enabled_values(value):
    assert sql_risk_scope_evidence_enabled(value) is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "default"])
def test_explicit_disabled_values(value, monkeypatch):
    monkeypatch.delenv(
        OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV,
        raising=False,
    )
    assert sql_risk_scope_evidence_enabled(value) is False


def test_unknown_setting_fails_closed_with_visible_error():
    with pytest.raises(
        ValueError,
        match="Unknown OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT",
    ):
        sql_risk_scope_evidence_enabled("wat")


def test_unknown_setting_does_not_affect_non_sql_risk_operation(monkeypatch):
    monkeypatch.setenv(
        OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT_ENV,
        "wat",
    )
    assert build_sql_risk_scope_contract(TABLE_TASK, []) is None


def test_extracts_one_generic_table_pair_without_fixture_assumptions():
    scope = extract_literal_sql_risk_scope(TABLE_TASK)

    assert scope is not None
    assert scope.source == "raw_customer_events_v2"
    assert scope.target == "mart_customer_daily_v3"
    assert scope.source_table == "raw_customer_events_v2"
    assert scope.target_table == "mart_customer_daily_v3"
    assert scope.source_field is None
    assert scope.target_field is None
    assert scope.file_id is None
    assert scope.label == (
        "raw_customer_events_v2 → mart_customer_daily_v3"
    )


def test_extracts_field_pair_and_only_explicit_positive_file_id():
    scope = extract_literal_sql_risk_scope(FIELD_TASK)

    assert scope is not None
    assert scope.source_table == "raw_customer_events_v2"
    assert scope.source_field == "customer_key"
    assert scope.target_table == "mart_customer_daily_v3"
    assert scope.target_field == "customer_key"
    assert scope.file_id == 417
    assert extract_literal_file_id("version=417, file_id: 23") == 23
    assert extract_literal_file_id("table_v417 → table_v418") is None


def test_repeated_identical_pair_is_not_ambiguous():
    scope = extract_literal_sql_risk_scope(
        "Сначала raw_x → mart_y; затем повторно raw_x → mart_y."
    )

    assert scope is not None
    assert scope.label == "raw_x → mart_y"


@pytest.mark.parametrize(
    "task",
    [
        "Сравни raw_a → mart_b и raw_c → mart_d.",
        "Сравни source → target.",
        "Сравни raw_a.id → mart_b.",
        "Сравни schema.raw_a.id → mart_b.id.",
        "Оцени риск без указанной пары.",
        "Проверь SQL-выражение payload_json -> customer_key.",
    ],
)
def test_ambiguous_or_nontechnical_scope_never_creates_requirements(task):
    assert extract_literal_sql_risk_scope(task) is None
    assert build_sql_risk_scope_contract(
        task,
        ["cardinality"],
        enabled=True,
    ) is None
    assert required_sql_risk_tools(
        task,
        ["cardinality"],
        enabled=True,
    ) == ()


@pytest.mark.parametrize(
    "aspect",
    [
        "row_filtering",
        "cardinality",
        "value_changes",
        "write_semantics",
    ],
)
def test_mapping_aspects_require_only_exact_directed_s2t_reader(aspect):
    task = QUALIFIED_FIELD_TASK if aspect == "value_changes" else TABLE_TASK
    source_table = (
        "stg.raw_customer_events_v2"
        if aspect == "value_changes"
        else "raw_customer_events_v2"
    )
    target_table = (
        "dwh.mart_customer_daily_v3"
        if aspect == "value_changes"
        else "mart_customer_daily_v3"
    )
    contract = build_sql_risk_scope_contract(
        task,
        [aspect],
        enabled=True,
    )

    assert contract is not None
    assert contract.aspects == (aspect,)
    assert contract.tool_names == ("read_s2t_source_to_target",)
    assert contract.requirements == (
        ReadS2TSourceToTargetRequirement(
            source_table=source_table,
            target_table=target_table,
        ),
    )
    assert contract.requirements[0].arguments == {
        "source_table": source_table,
        "target_table": target_table,
    }


def test_constraint_field_pair_with_literal_file_requires_both_exact_readers():
    contract = build_sql_risk_scope_contract(
        FIELD_TASK,
        ["constraint_rejection"],
        enabled=True,
    )

    assert contract is not None
    assert contract.tool_names == (
        "read_s2t_source_to_target",
        "get_source_target_column_pair",
    )
    assert contract.requirements == (
        ReadS2TSourceToTargetRequirement(
            source_table="raw_customer_events_v2",
            target_table="mart_customer_daily_v3",
        ),
        GetSourceTargetColumnPairRequirement(
            file_id=417,
            source_table="raw_customer_events_v2",
            source_column="customer_key",
            target_table="mart_customer_daily_v3",
            target_column="customer_key",
        ),
    )
    assert contract.requirements[1].arguments == {
        "file_id": 417,
        "source_table": "raw_customer_events_v2",
        "source_column": "customer_key",
        "target_table": "mart_customer_daily_v3",
        "target_column": "customer_key",
    }


def test_table_aspect_does_not_guess_schema_qualified_table_shape():
    assert build_sql_risk_scope_contract(
        "Оцени cardinality для stg.orders → dwh.orders.",
        ["cardinality"],
        enabled=True,
    ) is None


def test_field_aspect_splits_schema_qualified_fields_from_the_right():
    contract = build_sql_risk_scope_contract(
        "Для file_id=17 проверь stg.orders.id → dwh.orders.order_id.",
        ["constraint_rejection"],
        enabled=True,
    )

    assert contract is not None
    assert contract.scope.source_table == "stg.orders"
    assert contract.scope.source_field == "id"
    assert contract.scope.target_table == "dwh.orders"
    assert contract.scope.target_field == "order_id"
    assert contract.requirements[0].arguments == {
        "source_table": "stg.orders",
        "target_table": "dwh.orders",
    }


def test_mixed_table_and_field_aspects_do_not_guess_dotted_scope_shape():
    assert build_sql_risk_scope_contract(
        "Проверь stg.orders.id → dwh.orders.id.",
        ["row_filtering", "value_changes"],
        enabled=True,
    ) is None


def test_two_part_value_scope_is_left_to_existing_exact_analysis():
    assert build_sql_risk_scope_contract(
        "Оцени value changes src_alpha.id → tgt_beta.id.",
        ["value_changes"],
        enabled=True,
    ) is None


def test_two_part_constraint_requires_strong_column_grammar_and_file():
    assert build_sql_risk_scope_contract(
        "Для file_id=17 проверь src_alpha.id → tgt_beta.id.",
        ["constraint_rejection"],
        enabled=True,
    ) is None
    assert build_sql_risk_scope_contract(
        "Проверь nullable колонок src_alpha.id → tgt_beta.id.",
        ["constraint_rejection"],
        enabled=True,
    ) is None


@pytest.mark.parametrize(
    "task",
    [
        "Проверь raw_a.id → mart_b.id без выбранного файла.",
        "Для file_id=7 и file_id=8 проверь raw_a.id → mart_b.id.",
        "Для file_id=7 проверь raw_a → mart_b.",
    ],
)
def test_ambiguous_constraint_scope_stays_on_baseline_path(task):
    assert required_sql_risk_tools(
        task,
        ["constraint_rejection"],
        enabled=True,
    ) == ()


def test_multiple_aspects_deduplicate_tools_in_stable_order():
    contract = build_sql_risk_scope_contract(
        QUALIFIED_FIELD_TASK,
        ["value_changes", "constraint_rejection"],
        enabled=True,
    )

    assert contract is not None
    assert contract.aspects == (
        "constraint_rejection",
        "value_changes",
    )
    assert contract.tool_names == (
        "read_s2t_source_to_target",
        "get_source_target_column_pair",
    )


def test_answer_with_exact_directed_scope_is_unchanged():
    answer = (
        "Для `stg.raw_customer_events_v2.customer_key` → "
        "`dwh.mart_customer_daily_v3.customer_key` значение не меняется."
    )

    assert ensure_sql_risk_answer_scope(
        answer,
        QUALIFIED_FIELD_TASK,
        ["value_changes"],
        enabled=True,
    ) == answer


@pytest.mark.parametrize(
    "answer",
    [
        (
            "Для mart_customer_daily_v3 → raw_customer_events_v2 "
            "JOIN может размножить строки."
        ),
        (
            "raw_customer_events_v2 и mart_customer_daily_v3 упомянуты, "
            "но направление не зафиксировано."
        ),
    ],
)
def test_reversed_or_unordered_endpoints_get_correct_scope(answer):
    rendered = ensure_sql_risk_answer_scope(
        answer,
        TABLE_TASK,
        ["cardinality"],
        enabled=True,
    )

    assert rendered == (
        "Scope: raw_customer_events_v2 → mart_customer_daily_v3\n\n"
        + answer
    )


def test_missing_endpoint_gets_compact_exact_scope_line_idempotently():
    answer = "JOIN может размножить строки; уникальность ключа неизвестна."

    rendered = ensure_sql_risk_answer_scope(
        answer,
        TABLE_TASK,
        ["cardinality"],
        enabled=True,
    )

    assert rendered == (
        "Scope: raw_customer_events_v2 → mart_customer_daily_v3\n\n"
        + answer
    )
    assert ensure_sql_risk_answer_scope(
        rendered,
        TABLE_TASK,
        ["cardinality"],
        enabled=True,
    ) == rendered


def test_larger_identifier_substring_does_not_count_as_exact_endpoint():
    answer = (
        "raw_customer_events_v20 and mart_customer_daily_v30 were mentioned."
    )

    rendered = ensure_sql_risk_answer_scope(
        answer,
        TABLE_TASK,
        ["row_filtering"],
        enabled=True,
    )

    assert rendered.startswith(
        "Scope: raw_customer_events_v2 → mart_customer_daily_v3\n\n"
    )


def test_ambiguous_scope_leaves_answer_unchanged_even_when_enabled():
    task = "Сравни raw_a → mart_b и raw_c → mart_d."
    answer = "Не хватает однозначной пары.\n"

    assert ensure_sql_risk_answer_scope(
        answer,
        task,
        ["row_filtering"],
        enabled=True,
    ) == answer


def test_requirement_matcher_checks_exact_name_args_and_truncation():
    contract = build_sql_risk_scope_contract(
        FIELD_TASK,
        ["constraint_rejection"],
        enabled=True,
    )
    assert contract is not None
    calls = [
        {
            "tool_name": "read_s2t_source_to_target",
            "args": {
                "source_table": "raw_customer_events_v2",
                "target_table": "mart_customer_daily_v3",
            },
            "truncated": False,
        },
        {
            "tool_name": "get_source_target_column_pair",
            "args": {
                "file_id": 417,
                "source_table": "raw_customer_events_v2",
                "source_column": "wrong_key",
                "target_table": "mart_customer_daily_v3",
                "target_column": "customer_key",
            },
        },
    ]

    assert missing_sql_risk_requirements(contract, calls) == (
        contract.requirements[1],
    )
    calls[1]["args"]["source_column"] = "customer_key"
    assert missing_sql_risk_requirements(contract, calls) == ()
    calls[0]["truncated"] = True
    assert missing_sql_risk_requirements(contract, calls) == (
        contract.requirements[0],
    )


def test_requirement_matcher_accepts_evidence_artifact_shape():
    class EvidenceLike:
        tool_name = "read_s2t_source_to_target"
        compact_args = {
            "source_table": "raw_customer_events_v2",
            "target_table": "mart_customer_daily_v3",
        }
        truncated = False

    contract = build_sql_risk_scope_contract(
        TABLE_TASK,
        ["row_filtering"],
        enabled=True,
    )

    assert missing_sql_risk_requirements(contract, [EvidenceLike()]) == ()
    assert missing_sql_risk_requirements(None, [EvidenceLike()]) == ()


@pytest.mark.parametrize(
    "stage",
    ["plan", "planner", "observer", "upstream_decision", "upstream"],
)
def test_short_stage_renderer_uses_only_literal_generic_scope(stage):
    contract = build_sql_risk_scope_contract(
        FIELD_TASK,
        ["constraint_rejection"],
        enabled=True,
    )

    rendered = render_sql_risk_scope_contract(contract, stage=stage)

    assert "raw_customer_events_v2.customer_key" in rendered
    assert "mart_customer_daily_v3.customer_key" in rendered
    if stage != "upstream":
        assert "read_s2t_source_to_target" in rendered
        assert "get_source_target_column_pair" in rendered


def test_stage_renderer_is_empty_without_safe_contract_and_rejects_bad_stage():
    assert render_sql_risk_scope_contract(None, stage="plan") == ""
    contract = build_sql_risk_scope_contract(
        TABLE_TASK,
        ["cardinality"],
        enabled=True,
    )
    assert contract is not None
    with pytest.raises(ValueError, match="Unknown SQL-risk scope stage"):
        render_sql_risk_scope_contract(contract, stage="unknown")
