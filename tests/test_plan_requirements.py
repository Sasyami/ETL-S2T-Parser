import pytest

from agents.contracts import WorkerPlan
from agents.plan_requirements import (
    ReroutePlanRequirementError,
    SqlRiskPlanRequirementError,
    validate_sql_risk_plan_requirements,
    validate_sql_risk_reroute_plan,
)
from agents.tools.context import load_operation_skills


def _plan(*tasks: str) -> WorkerPlan:
    return WorkerPlan(steps=[{"task": task} for task in tasks])


def test_sql_risk_reroute_accepts_mapping_plus_new_metadata():
    validate_sql_risk_reroute_plan(
        _plan(
            "Прочитать полный directed S2T mapping src_np → tgt_np.",
            "Прочитать metadata обеих endpoint-таблиц src_np и tgt_np.",
        ),
        "Оцени риск дубликатов для src_np → tgt_np.",
    )


def test_sql_risk_reroute_rejects_metadata_only_delta():
    with pytest.raises(
        ReroutePlanRequirementError,
        match=r"src_np → tgt_np",
    ):
        validate_sql_risk_reroute_plan(
            _plan(
                "Прочитать metadata обеих endpoint-таблиц src_np и tgt_np."
            ),
            "Оцени риск дубликатов для src_np → tgt_np.",
            sql_risk_aspects=["cardinality"],
        )


def test_sql_risk_reroute_requires_original_direction():
    with pytest.raises(ReroutePlanRequirementError):
        validate_sql_risk_reroute_plan(
            _plan("Прочитать полный S2T-маппинг tgt_np → src_np."),
            "Оцени риск дубликатов для src_np → tgt_np.",
            sql_risk_aspects=["cardinality"],
        )


def test_sql_risk_reroute_accepts_table_mapping_for_field_pair():
    validate_sql_risk_reroute_plan(
        _plan("Прочитать полное S2T-правило src_np → tgt_np."),
        "Может ли src_np.id → tgt_np.id изменить значение?",
    )


def test_sql_risk_reroute_accepts_explicit_transformation_read():
    validate_sql_risk_reroute_plan(
        _plan("Прочитать полную трансформацию src_orders → tgt_orders."),
        "Оцени риск для src_orders → tgt_orders.",
    )


def test_sql_risk_reroute_does_not_treat_transformation_metadata_as_mapping():
    with pytest.raises(ReroutePlanRequirementError):
        validate_sql_risk_reroute_plan(
            _plan(
                "Прочитать metadata трансформации src_orders → tgt_orders."
            ),
            "Оцени риск для src_orders → tgt_orders.",
            sql_risk_aspects=["cardinality"],
        )


def test_sql_risk_reroute_ignores_natural_language_arrow():
    validate_sql_risk_reroute_plan(
        _plan("Прочитать нужные метаданные."),
        "Сначала найди заказ → затем объясни риск.",
        sql_risk_aspects=["cardinality"],
    )


def test_write_semantics_reroute_also_requires_exact_mapping():
    with pytest.raises(ReroutePlanRequirementError):
        validate_sql_risk_reroute_plan(
            _plan("Прочитать явную стратегию записи из Additional object."),
            "Оцени write semantics для src_orders → tgt_orders.",
            sql_risk_aspects=["write_semantics"],
        )


def test_constraint_rejection_requires_atomic_endpoint_metadata_step():
    with pytest.raises(
        SqlRiskPlanRequirementError,
        match="одну самодостаточную worker task",
    ):
        validate_sql_risk_plan_requirements(
            _plan(
                "Прочитать metadata source-колонки src_np.id.",
                "Прочитать metadata target-колонки tgt_np.id.",
                "Прочитать точный S2T mapping src_np.id → tgt_np.id.",
            ),
            "Оцени constraint rejection src_np.id → tgt_np.id.",
            sql_risk_aspects=["constraint_rejection"],
        )


def test_constraint_rejection_accepts_one_exact_pair_metadata_step():
    validate_sql_risk_plan_requirements(
        _plan(
            "Прочитать вместе exact S2T mapping и column metadata обеих "
            "точных колонок src_np.id → tgt_np.id в одном file scope."
        ),
        "Оцени constraint rejection src_np.id → tgt_np.id.",
        sql_risk_aspects=["constraint_rejection"],
    )


def test_constraint_rejection_accepts_roles_without_repeated_arrow():
    validate_sql_risk_plan_requirements(
        _plan(
            "Получить exact S2T mapping и column metadata для source "
            "src_np.id и target tgt_np.id в одном file scope."
        ),
        "Оцени constraint rejection src_np.id → tgt_np.id.",
        sql_risk_aspects=["constraint_rejection"],
    )


def test_constraint_rejection_rejects_metadata_only_combined_step():
    with pytest.raises(SqlRiskPlanRequirementError):
        validate_sql_risk_plan_requirements(
            _plan(
                "Прочитать вместе column metadata обеих точных колонок "
                "src_np.id → tgt_np.id."
            ),
            "Оцени constraint rejection src_np.id → tgt_np.id.",
            sql_risk_aspects=["constraint_rejection"],
        )


def test_other_sql_risk_aspect_does_not_require_column_metadata():
    validate_sql_risk_plan_requirements(
        _plan("Прочитать полный S2T mapping src_np → tgt_np."),
        "Оцени cardinality src_np → tgt_np.",
        sql_risk_aspects=["cardinality"],
    )


def test_conditional_cardinality_rejects_redundant_metadata_step():
    with pytest.raises(
        SqlRiskPlanRequirementError,
        match="ровно одну worker task",
    ):
        validate_sql_risk_plan_requirements(
            _plan(
                "Прочитать полный exact S2T mapping src_np → tgt_np.",
                "Прочитать metadata и ключи src_np и tgt_np.",
            ),
            "Оцени риск появления дубликатов для src_np → tgt_np.",
            sql_risk_aspects=["cardinality"],
        )


def test_conditional_cardinality_rejects_mapping_mixed_with_metadata():
    with pytest.raises(SqlRiskPlanRequirementError):
        validate_sql_risk_plan_requirements(
            _plan(
                "Прочитать полный exact S2T mapping и metadata "
                "src_np → tgt_np."
            ),
            "Оцени риск появления дубликатов для src_np → tgt_np.",
            sql_risk_aspects=["cardinality"],
        )


def test_factual_cardinality_request_may_plan_metadata_after_mapping():
    validate_sql_risk_plan_requirements(
        _plan(
            "Прочитать полный exact S2T mapping src_np → tgt_np.",
            "Проверить фактическую уникальность ключей src_np и tgt_np.",
        ),
        "Проверь фактическую уникальность и точное число дубликатов "
        "для src_np → tgt_np.",
        sql_risk_aspects=["cardinality"],
    )


def test_typed_sql_risk_plan_prompt_explains_clean_reroute():
    prompt = load_operation_skills(
        ["Анализ SQL-рисков"],
        stage="plan",
        sql_risk_aspects=["cardinality"],
    )

    assert "После reroute прошлое evidence удалено" in prompt
    assert "план только из дельты" in prompt

    write_prompt = load_operation_skills(
        ["Анализ SQL-рисков"],
        stage="plan",
        sql_risk_aspects=["write_semantics"],
    )
    assert "план только из дельты" in write_prompt
