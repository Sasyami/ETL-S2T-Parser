from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_multiagent_experiments as experiments
from scripts.run_live_agent_benchmark import ModeResult


def test_experiment_matrix_covers_e1_to_e5_and_required_signals():
    assert set(experiments.EXPERIMENTS) == {"E1", "E2", "E3", "E4", "E5"}
    assert len(experiments.EXPERIMENTS["E1"].variants) == 3
    assert set(experiments.EXPERIMENTS["E1"].scenarios) == {
        "test_live_agent_runs_dependent_workers_sequentially",
        "test_live_agent_catalog_13_finds_join_condition",
        "test_live_agent_resolves_table_typo_before_exact_reader",
        "test_live_agent_batches_all_semantic_candidates_into_s2t_search",
    }
    assert len(experiments.EXPERIMENTS["E2"].variants) == 2
    assert set(experiments.EXPERIMENTS["E2"].scenarios) == {
        "test_live_agent_checks_row_loss_risk",
        "test_live_agent_checks_duplicate_risk_in_target",
        "test_live_agent_checks_nulls_in_required_target_fields",
        "test_live_agent_checks_value_change_risk",
        "test_live_agent_checks_write_semantics_risk",
    }
    assert {
        "test_live_validation_protocol_standard_mode",
        "test_live_validation_protocol_exhaustive_mode",
    } <= set(experiments.EXPERIMENTS["E3"].scenarios)
    assert "test_live_validation_protocol_minimal_readers" in (
        experiments.EXPERIMENTS["E4"].scenarios
    )
    assert "test_live_agent_skips_resolution_for_exact_table" in (
        experiments.EXPERIMENTS["E5"].scenarios
    )
    assert "test_live_agent_batches_all_semantic_candidates_into_s2t_search" in (
        experiments.EXPERIMENTS["E5"].scenarios
    )
    experiment_flags = {
        "WORKER_CAPABILITY_REROUTE_EXPERIMENT",
        "WORKER_SPLIT_TOOL_CALL_EXPERIMENT",
        "OPERATION_SQL_RISK_ASPECTS_EXPERIMENT",
    }
    for experiment_name in ("E1", "E2"):
        for variant in experiments.EXPERIMENTS[experiment_name].variants:
            assert set(variant.environment) == experiment_flags


def test_selected_experiments_deduplicates_in_cli_order():
    selected = experiments.selected_experiments(["e5", "E1", "e5"])
    assert [name for name, _ in selected] == ["E5", "E1"]
    with pytest.raises(ValueError, match="unknown experiment"):
        experiments.selected_experiments(["E6"])


def test_dry_run_never_starts_live_benchmark(monkeypatch, capsys):
    called = False

    def fail_run(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("live benchmark must not run")

    monkeypatch.setattr(experiments, "_run_mode", fail_run)
    assert experiments.main(["--experiment", "E4", "--dry-run"]) == 0
    assert called is False
    assert "E4/dependency_readers: 2 scenarios" in capsys.readouterr().out


def test_experiment_loads_dotenv_before_resolving_model(
    monkeypatch,
    tmp_path,
):
    events = []
    monkeypatch.delenv("GIGACHAT_MODEL", raising=False)
    monkeypatch.delenv("MODEL", raising=False)

    def fake_load_dotenv(path, *, override):
        events.append(("dotenv", path, override))
        monkeypatch.setenv("MODEL", "GigaChat-3-Ultra")

    def fake_run_mode(**kwargs):
        events.append(("run", kwargs["model"]))
        return ModeResult(
            mode=kwargs["mode"],
            return_code=0,
            transcript_path=tmp_path / "trace.md",
            junit_path=tmp_path / "junit.xml",
            passed=1,
        )

    monkeypatch.setattr(experiments, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(experiments, "_run_mode", fake_run_mode)
    monkeypatch.setattr(experiments, "_write_report", lambda *args, **kwargs: None)

    assert experiments.main(
        [
            "--experiment",
            "E4",
            "--provider",
            "gigachat",
            "--output-dir",
            str(tmp_path),
        ]
    ) == 0

    assert events[0] == ("dotenv", experiments.PROJECT_ROOT / ".env", False)
    assert events[1:] == [("run", "GigaChat-3-Ultra")]


def test_skipped_experiment_is_reported_as_incomplete(monkeypatch, tmp_path):
    monkeypatch.setattr(
        experiments,
        "_run_mode",
        lambda **kwargs: ModeResult(
            mode=kwargs["mode"],
            return_code=0,
            transcript_path=tmp_path / "trace.md",
            junit_path=tmp_path / "junit.xml",
            skipped=2,
        ),
    )

    result = experiments.main(
        [
            "--experiment",
            "E4",
            "--provider",
            "ollama",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert result == 1


def test_report_contains_accuracy_and_efficiency_signals(tmp_path: Path):
    transcript = tmp_path / "trace.md"
    junit = tmp_path / "junit.xml"
    result = ModeResult(
        mode="multiagent",
        return_code=0,
        transcript_path=transcript,
        junit_path=junit,
        passed=3,
        failed=1,
        skipped=2,
        agent_seconds=12.5,
        llm_calls=7,
        tool_calls=4,
        total_tokens=900,
    )
    # Newer benchmark metrics are optional for backwards-compatible reports.
    result.__dict__["reroutes"] = 2
    result.__dict__["tool_errors"] = 1
    result.__dict__["reader_calls"] = 4
    spec = experiments.EXPERIMENTS["E4"]
    variant = spec.variants[0]
    path = tmp_path / "report.md"

    experiments._write_report(
        path,
        provider="ollama",
        model="local",
        rows=[("E4", spec, variant, result)],
    )

    text = path.read_text(encoding="utf-8")
    assert "75.0%" in text
    assert "Reroutes" in text
    assert "Tool errors" in text
    assert "Reader calls" in text
    assert "| E4 | dependency_readers |" in text
