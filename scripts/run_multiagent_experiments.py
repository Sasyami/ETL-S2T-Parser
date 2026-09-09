#!/usr/bin/env python
"""Run the bounded E1-E5 experiment matrix from the improvement plan.

The experiment runner deliberately reuses the live HTTP scenarios and their
hard execution assertions.  A passing test is therefore the accuracy signal;
the benchmark trace supplies reroutes, tool errors, reader calls, tokens and
latency.  GigaChat Ultra is guarded fail-closed before and after every variant.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from dotenv import load_dotenv

try:
    from scripts.gigachat_budget import (
        DEFAULT_ULTRA_RESERVE_PER_SCENARIO,
        DEFAULT_ULTRA_TOKEN_FLOOR,
        UltraBudgetError,
        guard_ultra_budget,
        ultra_budget_reservations,
    )
    from scripts.run_live_agent_benchmark import (
        DEFAULT_OUTPUT_DIR,
        MODEL_ENV_BY_PROVIDER,
        ModeResult,
        _run_mode,
        _configured_model,
        _selected_scenario_count,
        _scenario_targets,
        _slug,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from gigachat_budget import (  # type: ignore[no-redef]
        DEFAULT_ULTRA_RESERVE_PER_SCENARIO,
        DEFAULT_ULTRA_TOKEN_FLOOR,
        UltraBudgetError,
        guard_ultra_budget,
        ultra_budget_reservations,
    )
    from run_live_agent_benchmark import (  # type: ignore[no-redef]
        DEFAULT_OUTPUT_DIR,
        MODEL_ENV_BY_PROVIDER,
        ModeResult,
        _run_mode,
        _configured_model,
        _selected_scenario_count,
        _scenario_targets,
        _slug,
    )


@dataclass(frozen=True)
class ExperimentVariant:
    """One isolated environment configuration within an experiment."""

    name: str
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    """A bounded scenario set and the configurations compared on it."""

    title: str
    purpose: str
    scenarios: tuple[str, ...]
    variants: tuple[ExperimentVariant, ...]


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "E1": ExperimentSpec(
        title="Capability palettes and split tool-call planning",
        purpose=(
            "Сравнить единый planner-вызов с отдельным выбором tool и "
            "построением аргументов при одинаковой capability palette."
        ),
        scenarios=(
            "test_live_agent_runs_dependent_workers_sequentially",
            "test_live_agent_catalog_13_finds_join_condition",
            "test_live_agent_resolves_table_typo_before_exact_reader",
            "test_live_agent_batches_all_semantic_candidates_into_s2t_search",
        ),
        variants=(
            ExperimentVariant(
                "broad_reroute_baseline",
                {
                    "WORKER_CAPABILITY_REROUTE_EXPERIMENT": "0",
                    "WORKER_SPLIT_TOOL_CALL_EXPERIMENT": "0",
                    "OPERATION_SQL_RISK_ASPECTS_EXPERIMENT": "1",
                },
            ),
            ExperimentVariant(
                "capability_integrated",
                {
                    "WORKER_CAPABILITY_REROUTE_EXPERIMENT": "1",
                    "WORKER_SPLIT_TOOL_CALL_EXPERIMENT": "0",
                    "OPERATION_SQL_RISK_ASPECTS_EXPERIMENT": "1",
                },
            ),
            ExperimentVariant(
                "capability_split_selector_arguments",
                {
                    "WORKER_CAPABILITY_REROUTE_EXPERIMENT": "1",
                    "WORKER_SPLIT_TOOL_CALL_EXPERIMENT": "1",
                    "OPERATION_SQL_RISK_ASPECTS_EXPERIMENT": "1",
                },
            ),
        ),
    ),
    "E2": ExperimentSpec(
        title="SQL-risk aspects",
        purpose=(
            "Проверить выбор только релевантных аспектов row filtering, "
            "cardinality, constraint rejection, value changes и write semantics."
        ),
        scenarios=(
            "test_live_agent_checks_row_loss_risk",
            "test_live_agent_checks_duplicate_risk_in_target",
            "test_live_agent_checks_nulls_in_required_target_fields",
            "test_live_agent_checks_value_change_risk",
            "test_live_agent_checks_write_semantics_risk",
        ),
        variants=(
            ExperimentVariant(
                "legacy_full_sql_risk_profile",
                {
                    "WORKER_CAPABILITY_REROUTE_EXPERIMENT": "1",
                    "WORKER_SPLIT_TOOL_CALL_EXPERIMENT": "0",
                    "OPERATION_SQL_RISK_ASPECTS_EXPERIMENT": "0",
                },
            ),
            ExperimentVariant(
                "typed_aspect_routing",
                {
                    "WORKER_CAPABILITY_REROUTE_EXPERIMENT": "1",
                    "WORKER_SPLIT_TOOL_CALL_EXPERIMENT": "0",
                    "OPERATION_SQL_RISK_ASPECTS_EXPERIMENT": "1",
                },
            ),
        ),
    ),
    "E3": ExperimentSpec(
        title="Extended deterministic validation protocol",
        purpose=(
            "Проверить modes, preflight, reconciliation, expressions, explicit "
            "keys, phases и partial/unavailable states."
        ),
        scenarios=(
            "test_live_validation_protocol_standard_mode",
            "test_live_validation_protocol_exhaustive_mode",
            "test_live_validation_protocol_key_reconciliation",
            "test_live_validation_protocol_field_level_reconciliation",
            "test_live_validation_protocol_preload_constraint_checks",
            "test_live_validation_protocol_expression_projection",
            "test_live_validation_protocol_explicit_key_without_catalog_pk",
            "test_live_validation_protocol_separate_load_scopes",
            "test_live_validation_protocol_without_file",
            "test_live_validation_protocol_without_file_no_catalog_dependency",
        ),
        variants=(ExperimentVariant("deterministic_compiler"),),
    ),
    "E4": ExperimentSpec(
        title="Dependency-based validation readers",
        purpose=(
            "Измерить reader calls, latency и tokens при минимальном наборе "
            "зависимостей каждого check."
        ),
        scenarios=(
            "test_live_validation_protocol_minimal_readers",
            "test_live_validation_protocol_source_catalog_dependency",
        ),
        variants=(ExperimentVariant("dependency_readers"),),
    ),
    "E5": ExperimentSpec(
        title="Shared entity resolution",
        purpose=(
            "Проверить exact bypass, typo, partial, semantic, ambiguity, role "
            "preservation и одинаковую семантику validation/agentic."
        ),
        scenarios=(
            "test_live_validation_protocol_table_typo_resolution",
            "test_live_validation_protocol_ambiguous_typo",
            "test_live_validation_protocol_semantic_file_resolution",
            "test_live_agent_resolves_table_typo_before_exact_reader",
            "test_live_agent_skips_resolution_for_exact_table",
            "test_live_agent_resolves_partial_table_name",
            "test_live_agent_resolves_semantic_table_mention",
            "test_live_agent_batches_all_semantic_candidates_into_s2t_search",
            "test_live_agent_does_not_guess_ambiguous_entity",
            "test_live_entity_resolution_preserves_source_target_role",
            "test_live_validation_and_agentic_use_same_resolution_semantics",
        ),
        variants=(ExperimentVariant("shared_resolver"),),
    ),
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def selected_experiments(names: Sequence[str]) -> list[tuple[str, ExperimentSpec]]:
    """Return requested experiments once, preserving CLI order."""
    selected = list(dict.fromkeys(name.upper() for name in names))
    unknown = [name for name in selected if name not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown experiment: {unknown[0]}")
    return [(name, EXPERIMENTS[name]) for name in selected]


def _accuracy(result: ModeResult) -> float:
    evaluated = result.passed + result.failed + result.errors
    return (100.0 * result.passed / evaluated) if evaluated else 0.0


def _write_report(
    path: Path,
    *,
    provider: str,
    model: str,
    rows: Sequence[tuple[str, ExperimentSpec, ExperimentVariant, ModeResult]],
) -> None:
    lines = [
        f"# Multiagent experiments: {provider} / {model or 'configured default'}",
        "",
        "Accuracy — доля прошедших hard live assertions среди выполненных "
        "(skipped не входят в знаменатель).",
        "Любой skipped либо ноль выполненных сценариев делает матрицу "
        "неполной и приводит к ненулевому exit code.",
        "",
        "| Experiment | Variant | Accuracy | Passed | Failed | Skipped | "
        "Reroutes | Tool errors | Reader calls | LLM calls | Tokens | Agent, s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _spec, variant, result in rows:
        lines.append(
            f"| {name} | {variant.name} | {_accuracy(result):.1f}% | "
            f"{result.passed} | {result.failed + result.errors} | "
            f"{result.skipped} | {getattr(result, 'reroutes', 0)} | "
            f"{getattr(result, 'tool_errors', 0)} | "
            f"{getattr(result, 'reader_calls', result.tool_calls)} | "
            f"{result.llm_calls} | {result.total_tokens} | "
            f"{result.agent_seconds:.3f} |"
        )
    for name, spec in dict((name, spec) for name, spec, _, _ in rows).items():
        lines.extend(
            [
                "",
                f"## {name}. {spec.title}",
                "",
                spec.purpose,
                "",
                "Scenarios: " + ", ".join(f"`{item}`" for item in spec.scenarios),
            ]
        )
    lines.extend(["", "## Artifacts", ""])
    for name, _spec, variant, result in rows:
        lines.append(
            f"- `{name}/{variant.name}`: `{result.transcript_path}`; "
            f"JUnit: `{result.junit_path}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Последовательно выполнить bounded E1-E5 live experiments."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        choices=tuple(EXPERIMENTS),
        default=[],
        help="E1–E5; можно повторять. Без параметра запускается вся матрица.",
    )
    parser.add_argument(
        "--provider",
        choices=tuple(sorted(MODEL_ENV_BY_PROVIDER)),
        default=os.getenv("LLM_PROVIDER", "ollama"),
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--pytest-arg", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "experiments")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--ultra-token-floor",
        type=int,
        default=DEFAULT_ULTRA_TOKEN_FLOOR,
    )
    parser.add_argument(
        "--ultra-reserve-per-scenario",
        type=int,
        default=DEFAULT_ULTRA_RESERVE_PER_SCENARIO,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Keep model selection identical to the benchmark subprocess.  Otherwise
    # an Ultra model configured only in ``.env`` could be discovered too late,
    # after this process had already skipped the fail-closed budget check.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = build_parser()
    args = parser.parse_args(argv)
    requested = args.experiment or list(EXPERIMENTS)
    selected = selected_experiments(requested)
    run_units = [
        (name, spec, variant)
        for name, spec in selected
        for variant in spec.variants
    ]
    if args.dry_run:
        for name, spec, variant in run_units:
            print(
                f"{name}/{variant.name}: {len(spec.scenarios)} scenarios; "
                + ", ".join(
                    f"{key}={value}" for key, value in variant.environment.items()
                )
            )
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _configured_model(args.provider, args.model)
    remaining_calls = sum(
        _selected_scenario_count(spec.scenarios, [])
        for _, spec, _ in run_units
    )
    rows: list[tuple[str, ExperimentSpec, ExperimentVariant, ModeResult]] = []

    for name, spec, variant in run_units:
        if args.provider == "gigachat":
            for reservation in ultra_budget_reservations(
                chat_model=model,
                exchange_count=remaining_calls,
                reserve_per_exchange=args.ultra_reserve_per_scenario,
                judge_enabled=args.llm_judge,
            ):
                try:
                    guard_ultra_budget(
                        model=reservation.model,
                        floor_tokens=args.ultra_token_floor,
                        reserved_tokens=reservation.reserved_tokens,
                    )
                except UltraBudgetError as exc:
                    parser.error(str(exc))
        label = f"{timestamp}_{name.lower()}_{_slug(variant.name)}"
        result = _run_mode(
            mode="multiagent",
            provider=args.provider,
            model=model,
            targets=_scenario_targets(spec.scenarios),
            pytest_args=args.pytest_arg,
            output_dir=output_dir,
            run_label=label,
            llm_judge=args.llm_judge,
            extra_env=variant.environment,
        )
        result.mode = f"{name}/{variant.name}"
        evaluated = result.passed + result.failed + result.errors
        if result.skipped or not evaluated:
            print(
                f"Incomplete experiment {result.mode}: "
                f"evaluated={evaluated}, skipped={result.skipped}",
                flush=True,
            )
            result.return_code = result.return_code or 5
        rows.append((name, spec, variant, result))
        remaining_calls -= _selected_scenario_count(spec.scenarios, [])
        if args.provider == "gigachat":
            for reservation in ultra_budget_reservations(
                chat_model=model,
                exchange_count=remaining_calls,
                reserve_per_exchange=args.ultra_reserve_per_scenario,
                judge_enabled=args.llm_judge,
            ):
                try:
                    guard_ultra_budget(
                        model=reservation.model,
                        floor_tokens=args.ultra_token_floor,
                        reserved_tokens=reservation.reserved_tokens,
                    )
                except UltraBudgetError as exc:
                    parser.error(str(exc))

    report_path = output_dir / f"{timestamp}_experiments.md"
    _write_report(
        report_path,
        provider=args.provider,
        model=model,
        rows=rows,
    )
    print(f"Experiment report: {report_path}")
    return 1 if any(result.return_code for *_, result in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
