"""Focused contracts for field-scoped and negative-evidence SQL risks."""

from agents.tools.context import (
    OPERATION_SQL_RISK_ASPECTS_EXPERIMENT_ENV,
    load_operation_skills,
)


def _typed_context(aspect: str, stage: str) -> str:
    return " ".join(
        load_operation_skills(
            ["Анализ SQL-рисков"],
            stage=stage,
            sql_risk_aspects=[aspect],
        ).split()
    )


def test_value_changes_is_scoped_to_exact_target_projection():
    plan = _typed_context("value_changes", "plan")
    decision = _typed_context("value_changes", "upstream_decision")
    answer = _typed_context("value_changes", "upstream")

    assert "source.field→target.field" in plan
    assert "проекции точного target field" in decision
    assert "другом output alias не доказывают" in answer
    assert "прямая проекция" in answer


def test_write_semantics_accepts_complete_mapping_as_terminal_evidence():
    contexts = {
        stage: _typed_context("write_semantics", stage)
        for stage in (
            "plan",
            "planner",
            "observer",
            "upstream_decision",
            "upstream",
        )
    }

    assert "один полный exact directed mapping" in contexts["plan"]
    assert "один полный mapping" in contexts["planner"]
    assert "terminal negative evidence, а не gap" in contexts["observer"]
    assert "достаточен для pass" in contexts["upstream_decision"]
    assert "не reroute" in contexts["upstream_decision"]
    assert "не оценено" in contexts["upstream"]


def test_legacy_profile_has_the_same_terminal_and_field_scope_rules(monkeypatch):
    monkeypatch.setenv(OPERATION_SQL_RISK_ASPECTS_EXPERIMENT_ENV, "0")

    observer = _typed_context("write_semantics", "observer")
    decision = _typed_context("write_semantics", "upstream_decision")
    answer = _typed_context("value_changes", "upstream")

    assert "terminal negative evidence, а не gap" in observer
    assert "достаточен для `pass`" in decision
    assert "SQL-проекцию именно target.field" in answer
    assert "другом output alias не доказывают" in answer
