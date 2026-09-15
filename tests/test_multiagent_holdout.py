from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

from scripts import run_multiagent_holdout as holdout
from scripts.run_live_agent_benchmark import ModeResult


def _result(
    tmp_path: Path,
    scenario: str,
    *,
    technical: str = "passed",
    semantic: str = "passed",
    agent_tokens: int = 1_000,
    agent_calls: int = 3,
    agent_seconds: float = 2.0,
    judge_model: str = holdout.HOLDOUT_MODEL,
    judge_completed: int = 1,
    judge_tokens: int = 100,
    judge_errors: int = 0,
    reroutes: int = 0,
    tool_errors: int = 0,
) -> ModeResult:
    result = ModeResult(
        mode="multiagent",
        return_code=0 if technical == "passed" else 1,
        transcript_path=tmp_path / f"{scenario}.md",
        junit_path=tmp_path / f"{scenario}.xml",
        passed=int(technical == "passed"),
        failed=int(technical == "failed"),
        errors=int(technical == "error"),
        skipped=int(technical == "skipped"),
        scenario_statuses={scenario: technical},
        measured_runs=1,
        agent_seconds=agent_seconds,
        llm_calls=agent_calls,
        total_tokens=agent_tokens,
        reroutes=reroutes,
        tool_errors=tool_errors,
        semantic_statuses={scenario: semantic},
        judge_attempts=judge_completed + judge_errors,
        judge_completed=judge_completed,
        judge_errors=judge_errors,
        judge_input_tokens=max(judge_tokens - 10, 0),
        judge_output_tokens=min(judge_tokens, 10),
        judge_total_tokens=judge_tokens,
        judge_models={judge_model: judge_completed} if judge_model else {},
    )
    return result


def _arm_results(
    tmp_path: Path,
    *,
    failed_scenario: str | None = None,
    semantic_failure: bool = False,
    agent_tokens: int = 1_000,
    agent_calls: int = 3,
    agent_seconds: float = 2.0,
    judge_errors_on: str | None = None,
) -> dict[str, ModeResult]:
    results = {}
    for scenario in holdout.HOLDOUT_SCENARIOS:
        failed = scenario == failed_scenario
        results[scenario] = _result(
            tmp_path,
            scenario,
            technical="failed" if failed else "passed",
            semantic="failed" if failed and semantic_failure else "passed",
            agent_tokens=agent_tokens,
            agent_calls=agent_calls,
            agent_seconds=agent_seconds,
            judge_errors=int(scenario == judge_errors_on),
        )
    return results


def _write_fixture(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE files (file_id INTEGER);
            CREATE TABLE source_tables (id INTEGER);
            CREATE TABLE target_tables (id INTEGER);
            CREATE TABLE source_columns (
                file_id INTEGER,
                table_name TEXT,
                column_name TEXT,
                not_null INTEGER,
                data_type TEXT
            );
            CREATE TABLE target_columns (
                file_id INTEGER,
                table_name TEXT,
                column_name TEXT,
                not_null INTEGER,
                data_type TEXT
            );
            CREATE TABLE s2t_transformations (
                file_id INTEGER,
                sheet_name TEXT,
                source_table TEXT,
                source_field TEXT,
                target_table TEXT,
                target_field TEXT,
                transformation_rule TEXT
            );
            INSERT INTO files VALUES (1);
            INSERT INTO source_tables VALUES (1);
            INSERT INTO target_tables VALUES (1);
            INSERT INTO source_columns
            VALUES (1, 'src', 'id', 0, 'INTEGER');
            INSERT INTO target_columns
            VALUES (1, 'tgt', 'id', 1, 'BIGINT');
            INSERT INTO s2t_transformations VALUES
                (1, 'S2T', 'src', 'id', 'tgt', 'id',
                 'SELECT s.id FROM src s JOIN aux a ON a.id=s.id WHERE s.ok'),
                (1, 'S2T', 'src', 'a', 'tgt', 'a', 'SELECT a FROM src'),
                (1, 'S2T', 'src', 'b', 'tgt', 'b', 'SELECT b FROM src'),
                (1, 'S2T', 'src', 'c', 'tgt', 'c', 'SELECT c FROM src');
            """
        )
        connection.commit()
    finally:
        connection.close()


def test_holdout_manifest_is_fixed_disjoint_and_multiagent_only():
    holdout._validate_holdout_spec()
    assert len(holdout.HOLDOUT_SCENARIOS) == 10
    assert len(set(holdout.HOLDOUT_SCENARIOS)) == 10
    assert Counter(case.group for case in holdout.HOLDOUT_CASES) == {
        "smoke": 1,
        "history": 4,
        "display": 3,
        "validation": 2,
    }
    assert "test_live_agent_answers_simple_conversation_without_display_results" not in (
        holdout.HOLDOUT_SCENARIOS
    )
    assert "test_live_agent_explains_table_transformation" in (
        holdout.HOLDOUT_SCENARIOS
    )
    tuned = {
        scenario
        for experiment in holdout.EXPERIMENTS.values()
        for scenario in experiment.scenarios
    }
    assert not (set(holdout.HOLDOUT_SCENARIOS) & tuned)
    assert holdout.HOLDOUT_MODEL == "GigaChat-2-Max"
    assert holdout.COUNTERBALANCED_ORDERS == {
        "AB": ("baseline", "candidate"),
        "BA": ("candidate", "baseline"),
    }
    for arm in holdout.HOLDOUT_ARMS:
        assert arm.environment["GIGACHAT_JUDGE_MODEL"] == holdout.HOLDOUT_MODEL
        assert arm.environment["GIGACHAT_TEMPERATURE"] == "0"
        assert arm.environment["GIGACHAT_TIMEOUT"] == "180"
        assert arm.environment["LLM_TIMEOUT"] == "180"
        assert arm.environment["LIVE_AGENT_HTTP_TIMEOUT"] == "600"
        assert all(
            arm.environment[key] == ""
            for key in (
                "NEO4J_URI",
                "NEO4J_USERNAME",
                "NEO4J_USER",
                "NEO4J_PASSWORD",
                "NEO4J_DATABASE",
            )
        )
    assert holdout.HARD_CORRECTNESS_PYTEST_ARGS == (
        "-W",
        "error:live presentation warning:UserWarning",
    )


def test_fixture_preflight_is_read_only_and_covers_dynamic_holdout(tmp_path):
    db_path = tmp_path / "holdout.db"
    _write_fixture(db_path)
    assert holdout.fixture_preflight_errors(db_path) == ()
    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1
        connection.execute("DELETE FROM target_tables")
        connection.commit()
    finally:
        connection.close()
    assert "target catalog" in " ".join(
        holdout.fixture_preflight_errors(db_path)
    )


def test_quality_gain_passes_with_completed_max_judge_telemetry(tmp_path):
    failed = holdout.HOLDOUT_SCENARIOS[0]
    baseline = _arm_results(
        tmp_path,
        failed_scenario=failed,
        judge_errors_on=holdout.HOLDOUT_SCENARIOS[1],
    )
    candidate = _arm_results(tmp_path)

    verdict = holdout.evaluate_holdout(baseline, candidate)

    assert verdict.status == "improved"
    assert verdict.quality_improvement is True
    assert verdict.efficiency_improvement is False
    assert verdict.baseline_successes == 9
    assert verdict.candidate_successes == 10


def test_equal_perfect_quality_requires_ten_percent_agent_efficiency(tmp_path):
    baseline = _arm_results(tmp_path, agent_tokens=1_000, agent_calls=3)
    candidate = _arm_results(tmp_path, agent_tokens=900, agent_calls=3)

    verdict = holdout.evaluate_holdout(baseline, candidate)

    assert verdict.status == "improved"
    assert verdict.quality_improvement is False
    assert verdict.efficiency_improvement is True


def test_semantic_failure_is_a_composite_failure_not_an_improvement(tmp_path):
    failed = holdout.HOLDOUT_SCENARIOS[-1]
    baseline = _arm_results(tmp_path)
    candidate = _arm_results(
        tmp_path,
        failed_scenario=failed,
        semantic_failure=True,
    )

    verdict = holdout.evaluate_holdout(baseline, candidate)

    assert verdict.status == "not_improved"
    assert failed in verdict.regressions
    assert verdict.candidate_successes == 9


def test_missing_or_invalid_judge_telemetry_is_inconclusive(tmp_path):
    baseline = _arm_results(tmp_path)
    candidate = _arm_results(tmp_path)
    scenario = holdout.HOLDOUT_SCENARIOS[0]
    candidate[scenario].semantic_statuses[scenario] = "judge_error"
    candidate[scenario].judge_completed = 0
    candidate[scenario].judge_total_tokens = 0

    verdict = holdout.evaluate_holdout(baseline, candidate)

    assert verdict.status == "inconclusive"
    assert any("unevaluable semantic" in reason for reason in verdict.reasons)
    assert any("judge completion telemetry" in reason for reason in verdict.reasons)


def test_non_max_judge_telemetry_is_inconclusive(tmp_path):
    baseline = _arm_results(tmp_path)
    candidate = _arm_results(tmp_path)
    scenario = holdout.HOLDOUT_SCENARIOS[0]
    candidate[scenario].judge_models = {"GigaChat-2-Pro": 1}

    verdict = holdout.evaluate_holdout(baseline, candidate)

    assert verdict.status == "inconclusive"
    assert any("judge models" in reason for reason in verdict.reasons)


def test_inconsistent_judge_attempt_telemetry_is_inconclusive(tmp_path):
    baseline = _arm_results(tmp_path)
    candidate = _arm_results(tmp_path)
    scenario = holdout.HOLDOUT_SCENARIOS[0]
    candidate[scenario].judge_attempts += 1

    verdict = holdout.evaluate_holdout(baseline, candidate)

    assert verdict.status == "inconclusive"
    assert any("completed+errors" in reason for reason in verdict.reasons)


def test_dry_run_preflights_without_starting_network(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "holdout.db"
    _write_fixture(db_path)

    def fail_run(**_kwargs):
        raise AssertionError("dry-run must not start live scenarios")

    monkeypatch.setattr(holdout, "_run_mode", fail_run)
    assert holdout.main(["--db-path", str(db_path), "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "GigaChat-2-Max" in output
    assert "odd scenarios A→B, even scenarios B→A" in output
    assert "fixture preflight: passed" in output
    assert "no HTTP or LLM calls started" in output


def test_main_runs_exactly_twenty_counterbalanced_max_calls(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "holdout.db"
    _write_fixture(db_path)
    calls: list[dict] = []

    def fake_run_mode(**kwargs):
        preregistrations = list(tmp_path.rglob("*_preregistered.md"))
        assert preregistrations, "spec must exist before the first result"
        calls.append(kwargs)
        scenario = str(kwargs["targets"][0]).rsplit("::", 1)[-1]
        is_baseline = (
            kwargs["extra_env"]["WORKER_CAPABILITY_REROUTE_EXPERIMENT"]
            == "0"
        )
        technical = (
            "failed"
            if is_baseline and scenario == holdout.HOLDOUT_SCENARIOS[0]
            else "passed"
        )
        return _result(tmp_path, scenario, technical=technical)

    monkeypatch.setattr(holdout, "_run_mode", fake_run_mode)

    assert holdout.main(
        [
            "--db-path",
            str(db_path),
            "--output-dir",
            str(tmp_path),
        ]
    ) == 0

    assert len(calls) == 20
    arm_order = [
        (
            "baseline"
            if call["extra_env"]["WORKER_CAPABILITY_REROUTE_EXPERIMENT"]
            == "0"
            else "candidate"
        )
        for call in calls
    ]
    assert arm_order == [
        arm
        for index in range(1, 11)
        for arm in (
            ("baseline", "candidate")
            if index % 2
            else ("candidate", "baseline")
        )
    ]
    assert all(call["mode"] == "multiagent" for call in calls)
    assert all(call["provider"] == "gigachat" for call in calls)
    assert all(call["model"] == holdout.HOLDOUT_MODEL for call in calls)
    assert all(call["llm_judge"] is True for call in calls)
    assert all(len(call["targets"]) == 1 for call in calls)
    assert all(
        call["pytest_args"] == holdout.HARD_CORRECTNESS_PYTEST_ARGS
        for call in calls
    )
    assert all(
        call["extra_env"]["GIGACHAT_JUDGE_MODEL"] == holdout.HOLDOUT_MODEL
        for call in calls
    )
    report = next(tmp_path.rglob("*_comparison.md"))
    text = report.read_text(encoding="utf-8")
    assert "Semantic judge telemetry" in text
    assert "Combined passes" in text
    assert "Verdict: **improved**" in text


def test_main_aborts_remaining_calls_when_sqlite_hash_changes(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "holdout.db"
    _write_fixture(db_path)
    calls = 0

    def mutating_run_mode(**kwargs):
        nonlocal calls
        calls += 1
        scenario = str(kwargs["targets"][0]).rsplit("::", 1)[-1]
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("INSERT INTO files VALUES (2)")
            connection.commit()
        finally:
            connection.close()
        return _result(tmp_path, scenario)

    monkeypatch.setattr(holdout, "_run_mode", mutating_run_mode)

    assert holdout.main(
        [
            "--db-path",
            str(db_path),
            "--output-dir",
            str(tmp_path),
        ]
    ) == 2
    assert calls == 1
    report = next(tmp_path.rglob("*_comparison.md"))
    text = report.read_text(encoding="utf-8")
    assert "Verdict: **inconclusive**" in text
    assert "SQLite SHA256 changed" in text


def test_main_aborts_remaining_calls_on_invalid_judge_telemetry(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "holdout.db"
    _write_fixture(db_path)
    calls = 0

    def invalid_run_mode(**kwargs):
        nonlocal calls
        calls += 1
        scenario = str(kwargs["targets"][0]).rsplit("::", 1)[-1]
        return _result(
            tmp_path,
            scenario,
            judge_completed=0,
            judge_tokens=0,
        )

    monkeypatch.setattr(holdout, "_run_mode", invalid_run_mode)

    assert holdout.main(
        [
            "--db-path",
            str(db_path),
            "--output-dir",
            str(tmp_path),
        ]
    ) == 2
    assert calls == 1
    report = next(tmp_path.rglob("*_comparison.md"))
    text = report.read_text(encoding="utf-8")
    assert "Verdict: **inconclusive**" in text
    assert "remaining calls aborted" in text
