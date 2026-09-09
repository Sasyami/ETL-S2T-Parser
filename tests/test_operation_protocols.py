"""Contracts for reversible prompt-only SQL-risk protocol experiments."""

from __future__ import annotations

import hashlib
from itertools import combinations, product

import pytest

from agents.operation_protocols import (
    OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV,
    SQL_RISK_ASPECTS,
    SQL_RISK_ASPECT_SPECS,
    SQL_RISK_PROTOCOL_CANDIDATES,
    SQL_RISK_PROTOCOL_FAMILIES,
    SQL_RISK_PROTOCOL_STAGES,
    protocol_sha256,
    protocol_variant_sha256,
    render_sql_risk_protocol,
    selected_sql_risk_protocol,
)
from agents.tools.context import (
    OPERATION_SQL_RISK_ASPECTS_EXPERIMENT_ENV,
    _sql_risk_aspect_context,
    load_operation_skills,
)


_FROZEN_CURRENT_CONTEXT_SHA256 = (
    "419f022a8f7785aa0cb17e2a80fb0ef84cacb20173d16cf709b34ffd253e105a"
)


def _candidate(aspect: str, family: str) -> str:
    return f"{aspect}__{family}"


def test_protocol_candidates_are_the_unique_five_by_four_cartesian_product():
    expected = {
        _candidate(aspect, family)
        for aspect, family in product(
            SQL_RISK_ASPECTS,
            SQL_RISK_PROTOCOL_FAMILIES,
        )
    }

    assert len(SQL_RISK_PROTOCOL_CANDIDATES) == 20
    assert len(set(SQL_RISK_PROTOCOL_CANDIDATES)) == 20
    assert set(SQL_RISK_PROTOCOL_CANDIDATES) == expected

    parsed = [
        selected_sql_risk_protocol(candidate)
        for candidate in SQL_RISK_PROTOCOL_CANDIDATES
    ]
    assert all(variant is not None for variant in parsed)
    assert {variant.name for variant in parsed if variant is not None} == expected


@pytest.mark.parametrize("default_value", [None, "", "   ", "default", "current"])
def test_default_protocol_setting_preserves_current_context_byte_for_byte(
    monkeypatch,
    default_value,
):
    monkeypatch.delenv(OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV, raising=False)
    baseline = {
        (aspect, stage): _sql_risk_aspect_context([aspect], stage=stage)
        for aspect in SQL_RISK_ASPECTS
        for stage in SQL_RISK_PROTOCOL_STAGES
    }

    if default_value is None:
        monkeypatch.delenv(
            OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV,
            raising=False,
        )
    else:
        monkeypatch.setenv(
            OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV,
            default_value,
        )

    actual = {
        key: _sql_risk_aspect_context([key[0]], stage=key[1])
        for key in baseline
    }
    assert actual == baseline


def test_current_fallback_contexts_match_frozen_pre_experiment_digest(
    monkeypatch,
):
    monkeypatch.delenv(OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV, raising=False)
    monkeypatch.delenv(OPERATION_SQL_RISK_ASPECTS_EXPERIMENT_ENV, raising=False)
    aspect_subsets = [
        subset
        for length in range(len(SQL_RISK_ASPECTS) + 1)
        for subset in combinations(SQL_RISK_ASPECTS, length)
    ]
    parts = [
        f"[{','.join(aspects)}:{stage}]\n"
        + _sql_risk_aspect_context(aspects, stage=stage)
        for aspects in aspect_subsets
        for stage in SQL_RISK_PROTOCOL_STAGES
    ]

    assert protocol_sha256("\n\n".join(parts)) == (
        _FROZEN_CURRENT_CONTEXT_SHA256
    )


@pytest.mark.parametrize("candidate", SQL_RISK_PROTOCOL_CANDIDATES)
@pytest.mark.parametrize("stage", SQL_RISK_PROTOCOL_STAGES)
def test_every_candidate_renders_every_stage(candidate, stage):
    variant = selected_sql_risk_protocol(candidate)
    assert variant is not None

    context = render_sql_risk_protocol(
        candidate,
        [variant.aspect],
        stage=stage,
    )

    assert context.strip()
    assert f"`{candidate}`" in context
    assert f"`{variant.aspect}`" in context


def test_twenty_candidate_contexts_are_unique():
    contexts = []
    for candidate in SQL_RISK_PROTOCOL_CANDIDATES:
        variant = selected_sql_risk_protocol(candidate)
        assert variant is not None
        contexts.append(
            render_sql_risk_protocol(
                candidate,
                [variant.aspect],
                stage="upstream",
            )
        )

    assert len(set(contexts)) == 20


@pytest.mark.parametrize("invalid", ["unknown", "ROW_FILTERING__decision_table"])
def test_unknown_protocol_setting_fails_closed(invalid):
    with pytest.raises(ValueError, match="Unknown OPERATION_SQL_RISK_PROTOCOL"):
        selected_sql_risk_protocol(invalid)


@pytest.mark.parametrize(
    ("aspects", "message"),
    [
        ([], "exactly one selected typed aspect"),
        (["row_filtering", "cardinality"], "exactly one selected typed aspect"),
        (["unknown_aspect"], "unknown selected aspect"),
    ],
)
def test_candidate_requires_exactly_one_known_selected_aspect(aspects, message):
    with pytest.raises(ValueError, match=message):
        render_sql_risk_protocol(
            "row_filtering__minimal_artifact",
            aspects,
            stage="plan",
        )


def test_candidate_aspect_must_match_selected_aspect(monkeypatch):
    monkeypatch.setenv(
        OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV,
        "row_filtering__minimal_artifact",
    )

    with pytest.raises(ValueError, match="is for aspect.*selected aspect"):
        load_operation_skills(
            ["Анализ SQL-рисков"],
            stage="planner",
            sql_risk_aspects=["cardinality"],
        )


def test_invalid_protocol_env_does_not_affect_non_sql_operation_skills(
    monkeypatch,
):
    monkeypatch.delenv(OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV, raising=False)
    baseline = load_operation_skills(
        ["Совместимость колонок"],
        stage="planner",
    )

    monkeypatch.setenv(
        OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV,
        "not-a-candidate",
    )
    assert (
        load_operation_skills(
            ["Совместимость колонок"],
            stage="planner",
        )
        == baseline
    )


@pytest.mark.parametrize("aspect", SQL_RISK_ASPECTS)
def test_every_protocol_preserves_common_role_scope_and_no_guessing_rules(aspect):
    for family in SQL_RISK_PROTOCOL_FAMILIES:
        candidate = _candidate(aspect, family)
        for stage in SQL_RISK_PROTOCOL_STAGES:
            context = render_sql_risk_protocol(
                candidate,
                [aspect],
                stage=stage,
            )
            assert "точные source/target-роли" in context
            assert "immutable scope" in context
            assert "Не угадывай" in context
            assert "глобальный S2T mapping нельзя сужать по файлу" in context
            assert "только выбранный аспект" in context


@pytest.mark.parametrize("aspect", SQL_RISK_ASPECTS)
def test_terminal_absence_rule_reaches_all_decision_protocol_stages(aspect):
    for family in SQL_RISK_PROTOCOL_FAMILIES:
        candidate = _candidate(aspect, family)
        observer = render_sql_risk_protocol(
            candidate,
            [aspect],
            stage="observer",
        )
        decision = render_sql_risk_protocol(
            candidate,
            [aspect],
            stage="upstream_decision",
        )
        upstream = render_sql_risk_protocol(
            candidate,
            [aspect],
            stage="upstream",
        )
        combined = " ".join((observer, decision, upstream)).casefold()
        assert "terminal" in combined
        if aspect == "write_semantics":
            assert "не оценено" in combined
            assert "не reroute" in combined


@pytest.mark.parametrize("family", SQL_RISK_PROTOCOL_FAMILIES)
def test_every_family_colocates_primary_and_mandatory_support(family):
    plan = render_sql_risk_protocol(
        _candidate("constraint_rejection", family),
        ["constraint_rejection"],
        stage="plan",
    )

    assert "одну самодостаточную worker task" in plan
    assert "не дроби их между workers" in plan


@pytest.mark.parametrize("family", SQL_RISK_PROTOCOL_FAMILIES)
def test_constraint_support_uses_actual_file_scoped_reader_shapes(family):
    candidate = _candidate("constraint_rejection", family)
    combined = " ".join(
        render_sql_risk_protocol(
            candidate,
            ["constraint_rejection"],
            stage=stage,
        )
        for stage in ("plan", "planner")
    )

    assert "при подтверждённом file_id — одно exact-pair чтение" in combined
    assert "со всеми пятью идентификаторами" in combined
    assert "role-preserving full-table batch" in combined
    assert "file_scope=all" in combined
    assert "без суффиксов .field" in combined
    assert "строки разных file_id" in combined
    assert "не смешивай атрибуты версий" in combined


def test_only_cardinality_has_conditional_support_and_keeps_e2_mapping_only():
    assert SQL_RISK_ASPECT_SPECS["cardinality"].conditional_support
    assert all(
        not spec.conditional_support
        for aspect, spec in SQL_RISK_ASPECT_SPECS.items()
        if aspect != "cardinality"
    )

    for family in SQL_RISK_PROTOCOL_FAMILIES:
        candidate = _candidate("cardinality", family)
        plan = render_sql_risk_protocol(
            candidate,
            ["cardinality"],
            stage="plan",
        )
        assert "фактическую уникальность, ключи или точное число" in plan
        assert "Для обычного структурного вопроса" in plan
        assert "E2" not in plan
        assert "этот support не активируется" in plan
        assert "одну самодостаточную worker task" in plan


def test_minimal_artifact_family_has_stage_specific_protocol_characteristics():
    candidate = "cardinality__minimal_artifact"
    contexts = {
        stage: render_sql_risk_protocol(
            candidate,
            ["cardinality"],
            stage=stage,
        )
        for stage in SQL_RISK_PROTOCOL_STAGES
    }

    assert "минимальный data-plan" in contexts["plan"]
    assert "одно точное untruncated чтение" in contexts["planner"]
    assert "не оценивай риск" in contexts["observer"]
    assert "Не делай reroute только" in contexts["upstream_decision"]
    assert "status → mechanism → condition → evidence boundary" in contexts["upstream"]


def test_evidence_ledger_family_has_stage_specific_protocol_characteristics():
    candidate = "constraint_rejection__evidence_ledger"
    contexts = {
        stage: render_sql_risk_protocol(
            candidate,
            ["constraint_rejection"],
            stage=stage,
        )
        for stage in SQL_RISK_PROTOCOL_STAGES
    }

    assert "immutable evidence ledger" in contexts["plan"]
    assert "co-located" in contexts["plan"]
    assert "named ledger-entry" in contexts["planner"]
    assert "`satisfied`, `missing` или `invalid`" in contexts["observer"]
    assert "названной mandatory entry" in contexts["upstream_decision"]
    assert "`S`/`R1`/`R2+`/`C1+`" in contexts["upstream"]


def test_state_machine_family_has_stage_specific_protocol_characteristics():
    candidate = "row_filtering__epistemic_state_machine"
    contexts = {
        stage: render_sql_risk_protocol(
            candidate,
            ["row_filtering"],
            stage=stage,
        )
        for stage in SQL_RISK_PROTOCOL_STAGES
    }

    assert "`UNREAD`" in contexts["plan"]
    assert "`UNREAD → READ_EXACT`" in contexts["planner"]
    assert "`status`, `gap` и `limitations`" in contexts["observer"]
    assert "не выбирай `ANSWERABLE_*`" in contexts["observer"]
    assert "`ANSWERABLE_CONFIRMED`" not in contexts["observer"]
    assert "`ANSWERABLE_CONFIRMED`" in contexts["upstream_decision"]
    assert "Любое `ANSWERABLE_*`" in contexts["upstream_decision"]
    assert "не используй low/medium/high" in contexts["upstream"]


def test_decision_table_family_has_stage_specific_protocol_characteristics():
    candidate = "write_semantics__decision_table"
    contexts = {
        stage: render_sql_risk_protocol(
            candidate,
            ["write_semantics"],
            stage=stage,
        )
        for stage in SQL_RISK_PROTOCOL_STAGES
    }

    assert "только discriminators" in contexts["plan"]
    assert "mandatory/activated discriminator" in contexts["planner"]
    assert "не выбирай conclusion row" in contexts["observer"]
    assert "| Observed inputs | Decision |" in contexts["upstream_decision"]
    assert "Default not_assessed row" in contexts["upstream_decision"]
    assert "Выбери одну decision row" in contexts["upstream"]


def test_protocol_digest_is_stable_utf8_sha256():
    context = render_sql_risk_protocol(
        "value_changes__minimal_artifact",
        ["value_changes"],
        stage="upstream",
    )

    assert protocol_sha256(context) == hashlib.sha256(
        context.encode("utf-8")
    ).hexdigest()
    assert len(protocol_sha256(context)) == 64


def test_aggregate_variant_digest_is_stable_for_name_and_dataclass():
    name = "value_changes__minimal_artifact"
    variant = selected_sql_risk_protocol(name)
    assert variant is not None

    by_name = protocol_variant_sha256(name)
    by_variant = protocol_variant_sha256(variant)

    assert by_name == by_variant
    assert len(by_name) == 64
    assert len(
        {
            protocol_variant_sha256(candidate)
            for candidate in SQL_RISK_PROTOCOL_CANDIDATES
        }
    ) == 20


def test_aggregate_variant_digest_rejects_unknown_candidate():
    with pytest.raises(ValueError, match="Unknown SQL-risk protocol candidate"):
        protocol_variant_sha256("not-a-candidate")
