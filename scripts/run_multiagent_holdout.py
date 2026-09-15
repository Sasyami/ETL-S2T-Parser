#!/usr/bin/env python
"""Run the preregistered multiagent-only GigaChat-2-Max holdout A/B.

The scenario list, arms, hard warning policy, semantic judge and acceptance
thresholds in this module are deliberately fixed.  The preregistration is
written to disk before either live arm starts so that new results cannot
influence the comparison contract.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from dotenv import load_dotenv

try:
    from scripts.run_live_agent_benchmark import (
        DEFAULT_OUTPUT_DIR,
        SCENARIO_FILE,
        ModeResult,
        _run_mode,
        _selected_scenario_count,
        _scenario_targets,
        _slug,
    )
    from scripts.run_multiagent_experiments import EXPERIMENTS
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from run_live_agent_benchmark import (  # type: ignore[no-redef]
        DEFAULT_OUTPUT_DIR,
        SCENARIO_FILE,
        ModeResult,
        _run_mode,
        _selected_scenario_count,
        _scenario_targets,
        _slug,
    )
    from run_multiagent_experiments import (  # type: ignore[no-redef]
        EXPERIMENTS,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_PROVIDER = "gigachat"
HOLDOUT_MODEL = "GigaChat-2-Max"
DEFAULT_HOLDOUT_DB_PATH = DEFAULT_OUTPUT_DIR / "synthetic_live.db"
DEFAULT_HOLDOUT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "holdout"


@dataclass(frozen=True)
class HoldoutScenario:
    """One frozen scenario and the independent behavior it probes."""

    name: str
    group: str
    hypothesis: str


@dataclass(frozen=True)
class HoldoutArm:
    """One multiagent environment in the A/B comparison."""

    name: str
    hypothesis: str
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HoldoutVerdict:
    """Machine-readable result of the preregistered acceptance contract."""

    status: str
    baseline_successes: int
    candidate_successes: int
    quality_improvement: bool
    efficiency_improvement: bool
    regressions: tuple[str, ...]
    reasons: tuple[str, ...]


HOLDOUT_CASES = (
    HoldoutScenario(
        "test_live_agent_returns_exact_global_sqlite_count",
        "smoke",
        "A literal global SQLite count is read exactly and returned compactly.",
    ),
    HoldoutScenario(
        "test_live_agent_resolves_history_reference_into_task",
        "history",
        "An explicit user-owned history referent is resolved before delegation.",
    ),
    HoldoutScenario(
        "test_live_agent_asks_when_history_reference_is_ambiguous",
        "history",
        "Two unresolved user candidates cause clarification rather than guessing.",
    ),
    HoldoutScenario(
        "test_live_agent_rejects_assistant_only_history_assumption",
        "history",
        "An assistant-only physical identifier is not promoted to user context.",
    ),
    HoldoutScenario(
        "test_live_agent_uses_latest_user_history_rule",
        "history",
        "The latest user definition supersedes an older history rule.",
    ),
    HoldoutScenario(
        "test_live_agent_selects_full_sql_result_for_scrollable_ui",
        "display",
        "The exact full SQL relation is selected for the scrollable UI.",
    ),
    HoldoutScenario(
        "test_live_agent_preserves_exact_s2t_pairs_in_answer_and_full_result",
        "display",
        "Source-target pair cardinality survives answer and display handoff.",
    ),
    HoldoutScenario(
        "test_live_agent_returns_compound_sqlite_summary",
        "display",
        "A compound aggregate remains exact and exposes its full evidence.",
    ),
    HoldoutScenario(
        "test_live_agent_checks_source_and_target_type_compatibility",
        "validation",
        "Type compatibility is grounded in the exact S2T pair and catalogs.",
    ),
    HoldoutScenario(
        "test_live_agent_explains_table_transformation",
        "validation",
        "A saved field transformation is explained from exact S2T evidence.",
    ),
)
HOLDOUT_SCENARIOS = tuple(case.name for case in HOLDOUT_CASES)

_COMMON_ARM_ENVIRONMENT = {
    "WORKER_SPLIT_TOOL_CALL_EXPERIMENT": "0",
    # This preregistered holdout measures the historical agentic arms. Do not
    # let a developer's ambient operation-scope opt-in change that population.
    "OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT": "0",
    "GIGACHAT_JUDGE_MODEL": HOLDOUT_MODEL,
    "GIGACHAT_TEMPERATURE": "0",
    "GIGACHAT_TIMEOUT": "180",
    "LLM_TIMEOUT": "180",
    "LIVE_AGENT_HTTP_TIMEOUT": "600",
    "NEO4J_URI": "",
    "NEO4J_USERNAME": "",
    "NEO4J_USER": "",
    "NEO4J_PASSWORD": "",
    "NEO4J_DATABASE": "",
}
HOLDOUT_ARMS = (
    HoldoutArm(
        "baseline",
        "Legacy broad capability fallback and legacy SQL-risk operation routing.",
        {
            **_COMMON_ARM_ENVIRONMENT,
            "WORKER_CAPABILITY_REROUTE_EXPERIMENT": "0",
            "OPERATION_SQL_RISK_ASPECTS_EXPERIMENT": "0",
        },
    ),
    HoldoutArm(
        "candidate",
        "Current integrated capability reroute and typed SQL-risk routing.",
        {
            **_COMMON_ARM_ENVIRONMENT,
            "WORKER_CAPABILITY_REROUTE_EXPERIMENT": "1",
            "OPERATION_SQL_RISK_ASPECTS_EXPERIMENT": "1",
        },
    ),
)
# Each scenario is evaluated once per arm (20 agent exchanges total).  Odd
# scenarios run A→B and even scenarios B→A to reduce systematic order effects.
COUNTERBALANCED_ORDERS = {
    "AB": ("baseline", "candidate"),
    "BA": ("candidate", "baseline"),
}

# Presentation warnings are correctness failures in this holdout.  Efficiency
# warnings remain observable metrics and are governed by the aggregate limits
# below instead of being promoted ad hoc after results are seen.
HARD_CORRECTNESS_PYTEST_ARGS = (
    "-W",
    "error:live presentation warning:UserWarning",
)

REQUIRED_CANDIDATE_SUCCESSES = len(HOLDOUT_CASES)
EFFICIENCY_TOKEN_RATIO = 0.90
QUALITY_TOKEN_GUARD_RATIO = 1.10
LATENCY_GUARD_RATIO = 1.25


def _defined_live_tests() -> set[str]:
    tree = ast.parse(SCENARIO_FILE.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_live_")
    }


def _validate_holdout_spec() -> None:
    """Fail closed if the checked-in preregistration has drifted."""
    if len(HOLDOUT_SCENARIOS) != 10 or len(set(HOLDOUT_SCENARIOS)) != 10:
        raise ValueError("holdout must contain exactly 10 unique scenarios")
    unknown = sorted(set(HOLDOUT_SCENARIOS) - _defined_live_tests())
    if unknown:
        raise ValueError(f"unknown holdout live scenario: {unknown[0]}")
    tuned_scenarios = {
        scenario
        for experiment in EXPERIMENTS.values()
        for scenario in experiment.scenarios
    }
    overlap = sorted(set(HOLDOUT_SCENARIOS) & tuned_scenarios)
    if overlap:
        raise ValueError(f"holdout overlaps E1-E5: {overlap[0]}")
    if any(case.group == "graph" for case in HOLDOUT_CASES):
        raise ValueError("holdout must not depend on Neo4j")
    if _selected_scenario_count(HOLDOUT_SCENARIOS, []) != 10:
        raise ValueError("each holdout scenario must issue exactly one HTTP exchange")
    if tuple(arm.name for arm in HOLDOUT_ARMS) != ("baseline", "candidate"):
        raise ValueError("holdout requires fixed baseline and candidate arms")
    if COUNTERBALANCED_ORDERS != {
        "AB": ("baseline", "candidate"),
        "BA": ("candidate", "baseline"),
    }:
        raise ValueError("holdout requires fixed odd AB / even BA ordering")
    for arm in HOLDOUT_ARMS:
        if arm.environment.get("GIGACHAT_JUDGE_MODEL") != HOLDOUT_MODEL:
            raise ValueError("both arms must use GigaChat-2-Max as semantic judge")
        if arm.environment.get("WORKER_SPLIT_TOOL_CALL_EXPERIMENT") != "0":
            raise ValueError("split tool-call planning is outside this holdout")


_FIXTURE_PROBES = (
    (
        "at least one stored file",
        "SELECT COUNT(*) >= 1 FROM files",
    ),
    (
        "at least one target catalog row",
        "SELECT COUNT(*) >= 1 FROM target_tables",
    ),
    (
        "at least four complete S2T pairs",
        """
        SELECT COUNT(*) >= 4
        FROM s2t_transformations
        WHERE NULLIF(TRIM(source_table), '') IS NOT NULL
          AND NULLIF(TRIM(source_field), '') IS NOT NULL
          AND NULLIF(TRIM(target_table), '') IS NOT NULL
          AND NULLIF(TRIM(target_field), '') IS NOT NULL
        """,
    ),
    (
        "compound S2T aggregate fixture",
        """
        SELECT EXISTS (
            SELECT 1
            FROM s2t_transformations
            WHERE NULLIF(TRIM(source_table), '') IS NOT NULL
              AND NULLIF(TRIM(target_table), '') IS NOT NULL
        )
        """,
    ),
    (
        "source/target type compatibility fixture",
        """
        SELECT EXISTS (
            SELECT 1
            FROM s2t_transformations AS s2t
            JOIN source_columns AS source_catalog
              ON source_catalog.file_id = s2t.file_id
             AND source_catalog.table_name = s2t.source_table COLLATE NOCASE
             AND source_catalog.column_name = s2t.source_field COLLATE NOCASE
            JOIN target_columns AS target_catalog
              ON target_catalog.file_id = s2t.file_id
             AND target_catalog.table_name = s2t.target_table COLLATE NOCASE
             AND target_catalog.column_name = s2t.target_field COLLATE NOCASE
            WHERE LOWER(s2t.sheet_name) = 's2t'
              AND NULLIF(TRIM(s2t.source_table), '') IS NOT NULL
              AND NULLIF(TRIM(s2t.target_table), '') IS NOT NULL
              AND NULLIF(TRIM(s2t.source_field), '') IS NOT NULL
              AND NULLIF(TRIM(s2t.target_field), '') IS NOT NULL
              AND source_catalog.not_null = 0
              AND target_catalog.not_null = 1
              AND NULLIF(TRIM(source_catalog.data_type), '') IS NOT NULL
              AND NULLIF(TRIM(target_catalog.data_type), '') IS NOT NULL
              AND LOWER(s2t.transformation_rule) LIKE '%join%'
              AND LOWER(s2t.transformation_rule) LIKE '%where%'
        )
        """,
    ),
)


def fixture_preflight_errors(db_path: Path) -> tuple[str, ...]:
    """Check every data-dependent holdout assumption without mutating SQLite."""
    resolved = db_path.expanduser().resolve()
    if not resolved.is_file():
        return (f"SQLite fixture does not exist: {resolved}",)
    errors: list[str] = []
    try:
        connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return (f"cannot open SQLite fixture read-only: {exc}",)
    try:
        required_tables = {
            "files",
            "source_tables",
            "target_tables",
            "source_columns",
            "target_columns",
            "s2t_transformations",
        }
        actual_tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing = sorted(required_tables - actual_tables)
        if missing:
            return ("SQLite fixture misses tables: " + ", ".join(missing),)
        for label, query in _FIXTURE_PROBES:
            try:
                row = connection.execute(query).fetchone()
            except sqlite3.Error as exc:
                errors.append(f"{label}: SQLite error: {exc}")
                continue
            if row is None or not bool(row[0]):
                errors.append(f"{label}: requirement is not satisfied")
    finally:
        connection.close()
    return tuple(errors)


def sqlite_sha256(db_path: Path) -> str:
    """Return the content hash used to prove that the holdout stayed read-only."""
    digest = hashlib.sha256()
    with db_path.expanduser().resolve().open("rb") as database:
        for chunk in iter(lambda: database.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_state(result: ModeResult, scenario: str) -> str:
    return str(result.semantic_statuses.get(scenario) or "").strip().casefold()


def _judge_model_names(result: ModeResult) -> set[str]:
    raw_models = getattr(result, "judge_models", {})
    if isinstance(raw_models, Mapping):
        return {
            str(model)
            for model, count in raw_models.items()
            if int(count or 0) > 0
        }
    if isinstance(raw_models, str):
        return {raw_models} if raw_models else set()
    return {str(model) for model in raw_models or () if str(model)}


def _combined_success(result: ModeResult, scenario: str) -> bool:
    return (
        result.scenario_statuses.get(scenario) == "passed"
        and _semantic_state(result, scenario) == "passed"
    )


def _incomplete_reasons(
    label: str,
    scenario: str,
    result: ModeResult,
) -> list[str]:
    reasons: list[str] = []
    expected = {scenario}
    technical = set(result.scenario_statuses)
    missing_technical = sorted(expected - technical)
    extra_technical = sorted(technical - expected)
    if missing_technical:
        reasons.append(f"{label}: missing technical results: {missing_technical}")
    if extra_technical:
        reasons.append(f"{label}: unexpected technical results: {extra_technical}")
    if result.skipped:
        reasons.append(f"{label}: skipped={result.skipped}")
    if result.errors:
        reasons.append(f"{label}: pytest errors={result.errors}")
    if result.return_code not in {0, 1}:
        reasons.append(f"{label}: pytest return_code={result.return_code}")
    if result.measured_runs != 1:
        reasons.append(
            f"{label}: measured HTTP exchanges={result.measured_runs}, expected=1"
        )
    missing_semantic = sorted(expected - set(result.semantic_statuses))
    if missing_semantic:
        reasons.append(f"{label}: missing semantic results: {missing_semantic}")
    invalid_semantic = {
        semantic_scenario: _semantic_state(result, semantic_scenario) or "missing"
        for semantic_scenario in expected
        if semantic_scenario in result.semantic_statuses
        and _semantic_state(result, semantic_scenario)
        not in {"passed", "failed"}
    }
    if invalid_semantic:
        reasons.append(f"{label}: unevaluable semantic results: {invalid_semantic}")
    if result.llm_calls <= 0 or result.total_tokens <= 0:
        reasons.append(f"{label}: agent token/call metrics are unavailable")
    if result.agent_seconds <= 0:
        reasons.append(f"{label}: agent timing is unavailable")
    judge_attempts = int(getattr(result, "judge_attempts", 0) or 0)
    judge_completed = int(getattr(result, "judge_completed", 0) or 0)
    judge_errors = int(getattr(result, "judge_errors", 0) or 0)
    judge_input_tokens = int(getattr(result, "judge_input_tokens", 0) or 0)
    judge_output_tokens = int(getattr(result, "judge_output_tokens", 0) or 0)
    judge_total_tokens = int(getattr(result, "judge_total_tokens", 0) or 0)
    if judge_attempts <= 0:
        reasons.append(f"{label}: semantic judge attempt telemetry is missing")
    if judge_attempts != judge_completed + judge_errors:
        reasons.append(
            f"{label}: semantic judge attempts do not equal completed+errors"
        )
    if judge_completed <= 0:
        reasons.append(f"{label}: semantic judge completion telemetry is missing")
    if (
        judge_total_tokens <= 0
        or judge_total_tokens != judge_input_tokens + judge_output_tokens
    ):
        reasons.append(
            f"{label}: semantic judge total tokens must equal positive input+output"
        )
    judge_models = dict(getattr(result, "judge_models", {}) or {})
    if judge_models != {HOLDOUT_MODEL: 1}:
        reasons.append(
            f"{label}: semantic judge models={judge_models!r}, "
            f"expected={{{HOLDOUT_MODEL!r}: 1}}"
        )
    return reasons


def evaluate_holdout(
    baseline: Mapping[str, ModeResult],
    candidate: Mapping[str, ModeResult],
) -> HoldoutVerdict:
    """Apply the frozen quality, semantic and efficiency thresholds."""
    expected = set(HOLDOUT_SCENARIOS)
    incomplete: list[str] = []
    for label, results in (("baseline", baseline), ("candidate", candidate)):
        missing = sorted(expected - set(results))
        extra = sorted(set(results) - expected)
        if missing:
            incomplete.append(f"{label}: missing scenario runs: {missing}")
        if extra:
            incomplete.append(f"{label}: unexpected scenario runs: {extra}")
        for scenario in HOLDOUT_SCENARIOS:
            result = results.get(scenario)
            if result is not None:
                incomplete.extend(
                    _incomplete_reasons(f"{label}/{scenario}", scenario, result)
                )
    baseline_successes = {
        scenario
        for scenario, result in baseline.items()
        if scenario in expected and _combined_success(result, scenario)
    }
    candidate_successes = {
        scenario
        for scenario, result in candidate.items()
        if scenario in expected and _combined_success(result, scenario)
    }
    regressions = tuple(sorted(baseline_successes - candidate_successes))
    if incomplete:
        return HoldoutVerdict(
            status="inconclusive",
            baseline_successes=len(baseline_successes),
            candidate_successes=len(candidate_successes),
            quality_improvement=False,
            efficiency_improvement=False,
            regressions=regressions,
            reasons=tuple(incomplete),
        )

    baseline_tokens = sum(result.total_tokens for result in baseline.values())
    candidate_tokens = sum(result.total_tokens for result in candidate.values())
    baseline_calls = sum(result.llm_calls for result in baseline.values())
    candidate_calls = sum(result.llm_calls for result in candidate.values())
    baseline_seconds = sum(result.agent_seconds for result in baseline.values())
    candidate_seconds = sum(result.agent_seconds for result in candidate.values())
    baseline_tool_errors = sum(result.tool_errors for result in baseline.values())
    candidate_tool_errors = sum(result.tool_errors for result in candidate.values())
    baseline_reroutes = sum(result.reroutes for result in baseline.values())
    candidate_reroutes = sum(result.reroutes for result in candidate.values())
    quality_improvement = len(candidate_successes) > len(baseline_successes)
    efficiency_improvement = (
        len(baseline_successes) == len(HOLDOUT_SCENARIOS)
        and len(candidate_successes) == len(HOLDOUT_SCENARIOS)
        and candidate_tokens <= baseline_tokens * EFFICIENCY_TOKEN_RATIO
        and candidate_calls <= baseline_calls
    )
    failures: list[str] = []
    if len(candidate_successes) != REQUIRED_CANDIDATE_SUCCESSES:
        failures.append(
            "candidate must pass technical and semantic checks for all 10 scenarios"
        )
    if regressions:
        failures.append(f"baseline-pass to candidate-fail regressions: {regressions}")
    if len(candidate_successes) < len(baseline_successes):
        failures.append("candidate combined pass count is below baseline")
    if not (quality_improvement or efficiency_improvement):
        failures.append(
            "no preregistered improvement: need a combined hard pass gain, or "
            "at 10/10 in both arms at least 10% fewer agent tokens without "
            "more agent LLM calls"
        )
    candidate_http_500 = sum(result.http_500 for result in candidate.values())
    candidate_presentation_warnings = sum(
        result.presentation_warnings for result in candidate.values()
    )
    if candidate_http_500:
        failures.append(f"candidate HTTP 500 count={candidate_http_500}")
    if candidate_presentation_warnings:
        failures.append(
            "candidate presentation warnings="
            f"{candidate_presentation_warnings}"
        )
    if candidate_tool_errors > baseline_tool_errors:
        failures.append("candidate tool errors exceed baseline")
    if candidate_reroutes > baseline_reroutes:
        failures.append("candidate reroutes exceed baseline")
    if candidate_seconds > baseline_seconds * LATENCY_GUARD_RATIO:
        failures.append("candidate agent time exceeds 125% of baseline")
    if (
        quality_improvement
        and candidate_tokens > baseline_tokens * QUALITY_TOKEN_GUARD_RATIO
    ):
        failures.append("candidate agent tokens exceed 110% of baseline")

    return HoldoutVerdict(
        status="not_improved" if failures else "improved",
        baseline_successes=len(baseline_successes),
        candidate_successes=len(candidate_successes),
        quality_improvement=quality_improvement,
        efficiency_improvement=efficiency_improvement,
        regressions=regressions,
        reasons=tuple(failures or ("all preregistered acceptance gates passed",)),
    )


def _spec_lines(db_path: Path, db_sha256: str) -> list[str]:
    lines = [
        "# Preregistered multiagent holdout A/B",
        "",
        f"- Provider: `{HOLDOUT_PROVIDER}`",
        f"- Agent model: `{HOLDOUT_MODEL}`",
        f"- Semantic judge: `{HOLDOUT_MODEL}` (mandatory for every scenario)",
        "- Runtime mode: `multiagent` in both arms",
        f"- SQLite fixture: `{db_path}` (identical and read-only for both arms)",
        f"- SQLite SHA256 before results: `{db_sha256}`",
        "- Order: odd scenarios A→B, even scenarios B→A; exactly 20 agent "
        "exchanges, all sequential",
        "- LLM judge output is a hard correctness signal, not an optional report",
        "- Agent and judge token/call metrics are reported separately",
        "",
        "## Arms",
        "",
    ]
    for arm in HOLDOUT_ARMS:
        environment = ", ".join(
            f"`{key}={value}`" for key, value in sorted(arm.environment.items())
        )
        lines.extend(
            [
                f"- **{arm.name}** — {arm.hypothesis}",
                f"  Environment: {environment}.",
            ]
        )
    lines.extend(["", "## Frozen scenarios", ""])
    for index, case in enumerate(HOLDOUT_CASES, start=1):
        lines.append(
            f"{index}. `{case.name}` (`{case.group}`): {case.hypothesis}"
        )
    lines.extend(
        [
            "",
            "## Frozen acceptance",
            "",
            "1. Each arm must contain exactly 10 technical results, 10 HTTP "
            "measurements and 10 semantic results. Skips, pytest errors, missing "
            "judge output, `not_evaluated`, `judge_error`, or an unknown judge "
            "status make the comparison **inconclusive**.",
            "   Judge telemetry must contain a completed call, positive token "
            "usage and exactly `GigaChat-2-Max` for every scenario/arm. Recovered "
            "retry errors are reported but do not invalidate a completed verdict.",
            "2. Candidate must be 10/10 on the combined gate: pytest passed **and** "
            "semantic judge passed. A semantic failure counts as a scenario failure.",
            "3. No baseline combined pass may regress in candidate; candidate's "
            "combined pass count cannot be lower.",
            "4. Improvement is either at least one additional combined pass, or — "
            "only when both arms are 10/10 — at least 10% fewer agent tokens with "
            "no increase in agent LLM calls.",
            "5. Candidate must have zero HTTP 500 and zero presentation warnings; "
            "tool errors and reroutes may not exceed baseline.",
            "6. Candidate agent time may not exceed 125% of baseline. When quality "
            "improves, candidate agent tokens may not exceed 110% of baseline.",
            "7. Presentation warnings are promoted by the fixed pytest option "
            "`-W error:live presentation warning:UserWarning`; efficiency warnings "
            "remain metrics governed by rules 4–6.",
            "8. The SQLite file SHA256 is checked before and after every agent "
            "exchange. Any mutation aborts the remaining pairs and makes the "
            "comparison inconclusive.",
        ]
    )
    return lines


def write_preregistered_spec(
    path: Path,
    db_path: Path,
    db_sha256: str,
) -> None:
    path.write_text(
        "\n".join(_spec_lines(db_path, db_sha256)) + "\n",
        encoding="utf-8",
    )


def _status_cell(result: ModeResult, scenario: str) -> str:
    technical = result.scenario_statuses.get(scenario, "missing")
    semantic = _semantic_state(result, scenario) or "missing"
    return f"{technical} / {semantic}"


def write_comparison_report(
    path: Path,
    *,
    baseline: Mapping[str, ModeResult],
    candidate: Mapping[str, ModeResult],
    verdict: HoldoutVerdict,
    preregistration_path: Path,
) -> None:
    """Write results without changing or reinterpreting the frozen spec."""
    lines = [
        "# Multiagent holdout A/B result",
        "",
        f"Preregistration: `{preregistration_path}`",
        "",
        f"Verdict: **{verdict.status}**",
        "",
        "| Arm | Combined passes | Technical pass/fail/error/skip | "
        "Semantic failures | HTTP 500 | Presentation warnings | Reroutes | "
        "Tool errors | Agent LLM calls | Agent tokens | Agent, s |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, results, combined in (
        ("baseline", baseline, verdict.baseline_successes),
        ("candidate", candidate, verdict.candidate_successes),
    ):
        semantic_failures = sum(
            _semantic_state(results[scenario], scenario) == "failed"
            for scenario in HOLDOUT_SCENARIOS
            if scenario in results
        )
        passed = sum(result.passed for result in results.values())
        failed = sum(result.failed for result in results.values())
        errors = sum(result.errors for result in results.values())
        skipped = sum(result.skipped for result in results.values())
        lines.append(
            f"| {label} | {combined}/10 | {passed}/{failed}/{errors}/{skipped} | "
            f"{semantic_failures} | "
            f"{sum(result.http_500 for result in results.values())} | "
            f"{sum(result.presentation_warnings for result in results.values())} | "
            f"{sum(result.reroutes for result in results.values())} | "
            f"{sum(result.tool_errors for result in results.values())} | "
            f"{sum(result.llm_calls for result in results.values())} | "
            f"{sum(result.total_tokens for result in results.values())} | "
            f"{sum(result.agent_seconds for result in results.values()):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Semantic judge telemetry",
            "",
            "Judge metrics are not mixed into the agent efficiency thresholds.",
            "",
            "| Arm | Attempts | Completed | Retry errors | Input tokens | "
            "Output tokens | Total tokens | Cache read | Models |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for label, results in (("baseline", baseline), ("candidate", candidate)):
        models = sorted(
            {
                model
                for result in results.values()
                for model in _judge_model_names(result)
            }
        )
        lines.append(
            f"| {label} | "
            f"{sum(int(getattr(result, 'judge_attempts', 0) or 0) for result in results.values())} | "
            f"{sum(int(getattr(result, 'judge_completed', 0) or 0) for result in results.values())} | "
            f"{sum(int(getattr(result, 'judge_errors', 0) or 0) for result in results.values())} | "
            f"{sum(int(getattr(result, 'judge_input_tokens', 0) or 0) for result in results.values())} | "
            f"{sum(int(getattr(result, 'judge_output_tokens', 0) or 0) for result in results.values())} | "
            f"{sum(int(getattr(result, 'judge_total_tokens', 0) or 0) for result in results.values())} | "
            f"{sum(int(getattr(result, 'judge_cache_read_tokens', 0) or 0) for result in results.values())} | "
            f"{', '.join(models) or 'missing'} |"
        )
    lines.extend(
        [
            "",
            "## Scenario results",
            "",
            "Each cell is `technical / semantic`; only `passed / passed` counts.",
            "",
            "| Scenario | Baseline | Candidate |",
            "|---|---|---|",
        ]
    )
    for scenario in HOLDOUT_SCENARIOS:
        baseline_result = baseline.get(scenario)
        candidate_result = candidate.get(scenario)
        lines.append(
            f"| `{scenario}` | "
            f"{_status_cell(baseline_result, scenario) if baseline_result else 'missing'} | "
            f"{_status_cell(candidate_result, scenario) if candidate_result else 'missing'} |"
        )
    lines.extend(["", "## Acceptance evaluation", ""])
    lines.extend(f"- {reason}" for reason in verdict.reasons)
    lines.extend(["", "## Artifacts", ""])
    for index, scenario in enumerate(HOLDOUT_SCENARIOS, start=1):
        order = "AB" if index % 2 else "BA"
        for arm_name, results in (("baseline", baseline), ("candidate", candidate)):
            result = results.get(scenario)
            if result is None:
                continue
            lines.append(
                f"- `{index:02d}/{order}/{arm_name}/{scenario}`: "
                f"`{result.transcript_path}`; JUnit: `{result.junit_path}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen multiagent-only GigaChat-2-Max 10-case holdout A/B."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help=(
            "Read-only SQLite fixture shared by both arms. Defaults to "
            "LIVE_AGENT_DB_PATH or .test_runs/synthetic_live.db."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_HOLDOUT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the frozen spec without starting HTTP/LLM calls.",
    )
    return parser


def _resolved_db_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    configured = os.getenv("LIVE_AGENT_DB_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_HOLDOUT_DB_PATH.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_holdout_spec()
    except ValueError as exc:
        parser.error(str(exc))
    db_path = _resolved_db_path(args.db_path)
    preflight_errors = fixture_preflight_errors(db_path)
    try:
        initial_db_sha256 = sqlite_sha256(db_path)
    except OSError as exc:
        initial_db_sha256 = "unavailable"
        preflight_errors = (
            *preflight_errors,
            f"cannot hash SQLite fixture: {exc}",
        )
    print("\n".join(_spec_lines(db_path, initial_db_sha256)), flush=True)
    if preflight_errors:
        for error in preflight_errors:
            print(f"fixture preflight failed: {error}", flush=True)
        return 2
    print("fixture preflight: passed", flush=True)
    if args.dry_run:
        print("dry-run: no HTTP or LLM calls started", flush=True)
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.expanduser().resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration_path = output_dir / f"{timestamp}_preregistered.md"
    write_preregistered_spec(
        preregistration_path,
        db_path,
        initial_db_sha256,
    )
    print(f"Preregistration: {preregistration_path}", flush=True)

    arms = {arm.name: arm for arm in HOLDOUT_ARMS}
    results: dict[str, dict[str, ModeResult]] = {
        "baseline": {},
        "candidate": {},
    }
    integrity_error = ""
    for index, case in enumerate(HOLDOUT_CASES, start=1):
        order_name = "AB" if index % 2 else "BA"
        for arm_name in COUNTERBALANCED_ORDERS[order_name]:
            try:
                before_call_hash = sqlite_sha256(db_path)
            except OSError as exc:
                integrity_error = (
                    f"SQLite integrity check failed before {case.name}/{arm_name}: "
                    f"{exc}"
                )
                break
            if before_call_hash != initial_db_sha256:
                integrity_error = (
                    f"SQLite SHA256 changed before {case.name}/{arm_name}: "
                    f"{before_call_hash} != {initial_db_sha256}"
                )
                break
            arm = arms[arm_name]
            label = (
                f"{timestamp}_holdout_{index:02d}_"
                f"{_slug(case.name)}_{order_name.lower()}_{_slug(arm.name)}"
            )
            result = _run_mode(
                mode="multiagent",
                provider=HOLDOUT_PROVIDER,
                model=HOLDOUT_MODEL,
                targets=_scenario_targets((case.name,)),
                pytest_args=HARD_CORRECTNESS_PYTEST_ARGS,
                output_dir=output_dir,
                run_label=label,
                llm_judge=True,
                extra_env={
                    **arm.environment,
                    "LIVE_AGENT_DB_PATH": str(db_path),
                },
            )
            result.mode = f"{order_name}/{arm.name}/{case.name}"
            results[arm_name][case.name] = result
            try:
                after_call_hash = sqlite_sha256(db_path)
            except OSError as exc:
                integrity_error = (
                    f"SQLite integrity check failed after {case.name}/{arm_name}: "
                    f"{exc}"
                )
                break
            if after_call_hash != initial_db_sha256:
                integrity_error = (
                    f"SQLite SHA256 changed after {case.name}/{arm_name}: "
                    f"{after_call_hash} != {initial_db_sha256}"
                )
                break
            run_errors = _incomplete_reasons(
                f"{arm_name}/{case.name}",
                case.name,
                result,
            )
            if run_errors:
                integrity_error = (
                    "invalid holdout run; remaining calls aborted: "
                    + "; ".join(run_errors)
                )
                break
        if integrity_error:
            print(integrity_error, flush=True)
            break

    baseline = results["baseline"]
    candidate = results["candidate"]
    verdict = evaluate_holdout(baseline, candidate)
    if integrity_error:
        verdict = HoldoutVerdict(
            status="inconclusive",
            baseline_successes=verdict.baseline_successes,
            candidate_successes=verdict.candidate_successes,
            quality_improvement=False,
            efficiency_improvement=False,
            regressions=verdict.regressions,
            reasons=(integrity_error, *verdict.reasons),
        )
    report_path = output_dir / f"{timestamp}_comparison.md"
    write_comparison_report(
        report_path,
        baseline=baseline,
        candidate=candidate,
        verdict=verdict,
        preregistration_path=preregistration_path,
    )
    print(f"Holdout report: {report_path}", flush=True)
    if verdict.status == "improved":
        return 0
    if verdict.status == "inconclusive":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
