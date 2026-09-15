from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_operation_protocol_experiments as runner
from scripts.run_live_agent_benchmark import ModeResult


def _snapshot() -> runner.RepositorySnapshot:
    return runner.RepositorySnapshot(
        head="a" * 40,
        tracked_status_sha256=runner._sha256_bytes(b""),
        tracked_clean=True,
        sample_count=5,
        samples_sha256="b" * 64,
        env_metadata={
            "exists": True,
            "ignored": True,
            "mode": 0o600,
            "size": 42,
            "mtime_ns": 123,
        },
    )


def _result(
    directory: Path,
    scenario: str,
    *,
    technical: str = "passed",
    semantic: str = "passed",
    judge_completed: int = 1,
    judge_tokens: int = 30,
) -> ModeResult:
    directory.mkdir(parents=True, exist_ok=True)
    transcript = directory / f"{scenario}.md"
    junit = directory / f"{scenario}.xml"
    transcript.write_text("trace\n", encoding="utf-8")
    junit.write_text("<testsuites/>\n", encoding="utf-8")
    return ModeResult(
        mode="multiagent",
        return_code=0 if technical == "passed" else 1,
        transcript_path=transcript,
        junit_path=junit,
        passed=int(technical == "passed"),
        failed=int(technical == "failed"),
        errors=int(technical == "error"),
        skipped=int(technical == "skipped"),
        scenario_statuses={scenario: technical},
        measured_runs=1,
        agent_seconds=1.5,
        llm_calls=4,
        reader_calls=2,
        total_tokens=500,
        semantic_statuses={scenario: semantic},
        judge_attempts=judge_completed,
        judge_completed=judge_completed,
        judge_input_tokens=max(judge_tokens - 5, 0),
        judge_output_tokens=min(judge_tokens, 5),
        judge_total_tokens=judge_tokens,
        judge_models={runner.MODEL: 1} if judge_completed else {},
    )


def _patch_outer_preflight(monkeypatch, db_path: Path) -> runner.RepositorySnapshot:
    snapshot = _snapshot()
    digest = runner.sqlite_sha256(db_path)
    db_file_state = runner._sqlite_file_state(db_path)
    bundle_sha256 = "c" * 64
    plugin_sha256 = runner._file_sha256(runner.PLUGIN_PATH)
    monkeypatch.setattr(
        runner,
        "_preflight",
        lambda _parser, _args: runner.PreflightState(
            db_path=db_path,
            db_sha256=digest,
            db_file_state=db_file_state,
            repository=snapshot,
            base_sha=snapshot.head,
            runtime_bundle_sha256=bundle_sha256,
            plugin_sha256=plugin_sha256,
        ),
    )
    monkeypatch.setattr(
        runner,
        "capture_repository_snapshot",
        lambda _root=runner.PROJECT_ROOT: snapshot,
    )
    monkeypatch.setattr(runner, "load_dotenv", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runner, "_runtime_bundle_sha256", lambda _root: bundle_sha256)
    clones: list[Path] = []

    def fake_prepare(
        _source: Path,
        destination: Path,
        _base_sha: str,
        _bundle_sha256: str,
    ) -> None:
        destination.mkdir(parents=True)
        clones.append(destination)

    monkeypatch.setattr(runner, "_prepare_clone", fake_prepare)
    monkeypatch.setattr(
        runner,
        "_clone_status",
        lambda _clone: (runner._sha256_bytes(b""), True),
    )
    snapshot.__dict__["_test_clones"] = clones
    return snapshot


def test_manifest_is_exact_fixed_five_by_four_e2_matrix():
    runner.validate_manifest()

    assert len(runner.EXPERIMENTS_20) == 20
    assert {case.aspect for case in runner.EXPERIMENTS_20} == set(
        runner.ASPECT_SCENARIOS
    )
    assert {case.family for case in runner.EXPERIMENTS_20} == set(
        runner.PROTOCOL_FAMILIES
    )
    assert {case.scenario for case in runner.EXPERIMENTS_20} == set(
        runner.EXPERIMENTS["E2"].scenarios
    )
    assert len({case.protocol for case in runner.EXPERIMENTS_20}) == 20
    assert [case.order_name for case in runner.EXPERIMENTS_20] == [
        "AB" if index % 2 else "BA" for index in range(1, 21)
    ]
    assert all(
        case.protocol == f"{case.aspect}__{case.family}"
        for case in runner.EXPERIMENTS_20
    )


def test_arm_contract_is_max_multiagent_typed_and_protocol_specific(tmp_path):
    case = runner.EXPERIMENTS_20[0]
    baseline = runner._arm_environment("default", tmp_path / "baseline.db")
    candidate = runner._arm_environment(case.protocol, tmp_path / "candidate.db")

    assert runner.MODEL == "GigaChat-2-Max"
    assert runner.PROVIDER == "gigachat"
    for environment in (baseline, candidate):
        assert environment["CHAT_AGENT_MODE"] == "multiagent"
        assert environment["LIVE_AGENT_MODE"] == "multiagent"
        assert environment["LLM_PROVIDER"] == "gigachat"
        assert environment["GIGACHAT_MODEL"] == runner.MODEL
        assert environment["WORKER_CAPABILITY_REROUTE_EXPERIMENT"] == "1"
        assert environment["WORKER_SPLIT_TOOL_CALL_EXPERIMENT"] == "0"
        assert environment["OPERATION_SQL_RISK_ASPECTS_EXPERIMENT"] == "1"
        assert environment["GIGACHAT_JUDGE_MODEL"] == runner.MODEL
        assert environment["LANGFUSE_ENABLED"] == "0"
    assert baseline[runner.PROTOCOL_ENV] == "default"
    assert candidate[runner.PROTOCOL_ENV] == case.protocol
    assert runner.HARD_CORRECTNESS_PYTEST_ARGS == (
        "-W",
        "error:live presentation warning:UserWarning",
    )


def test_dry_run_creates_no_clone_output_or_network(
    monkeypatch,
    tmp_path,
    capsys,
):
    db_path = tmp_path / "fixture.db"
    db_path.write_bytes(b"fixture")
    _patch_outer_preflight(monkeypatch, db_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run must not clone, load secrets or run an arm")

    monkeypatch.setattr(runner, "_prepare_clone", forbidden)
    monkeypatch.setattr(runner, "_run_arm_subprocess", forbidden)
    monkeypatch.setattr(runner, "load_dotenv", forbidden)
    output = tmp_path / "output"

    assert runner.main(
        ["--db-path", str(db_path), "--output-dir", str(output), "--dry-run"]
    ) == 0

    stdout = capsys.readouterr().out
    assert "20 experiments, 40 Max-only multiagent exchanges" in stdout
    assert "no clone, HTTP, LLM or output artifact" in stdout
    assert not output.exists()


def test_internal_worker_calls_one_semantically_judged_multiagent_exchange(
    monkeypatch,
    tmp_path,
):
    scenario = runner.EXPERIMENTS_20[0].scenario
    captured: dict = {}
    result = _result(tmp_path / "artifacts", scenario)

    def fake_run_mode(**kwargs):
        captured.update(kwargs)
        return result

    benchmark = SimpleNamespace(
        _run_mode=fake_run_mode,
        _scenario_targets=lambda names: [f"clone/tests/live.py::{names[0]}"],
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda _name: benchmark)
    result_json = tmp_path / "result.json"
    args = argparse.Namespace(
        arm_name="candidate",
        arm_protocol=runner.EXPERIMENTS_20[0].protocol,
        arm_scenario=scenario,
        arm_db_path=tmp_path / "candidate.db",
        arm_output_dir=tmp_path / "output",
        arm_run_label="candidate-run",
        arm_result_json=result_json,
    )

    assert runner._run_arm_worker(args) == 0

    assert captured["mode"] == "multiagent"
    assert captured["provider"] == "gigachat"
    assert captured["model"] == runner.MODEL
    assert captured["llm_judge"] is True
    assert captured["pytest_args"] == runner.HARD_CORRECTNESS_PYTEST_ARGS
    assert captured["extra_env"][runner.PROTOCOL_ENV] == args.arm_protocol
    assert captured["extra_env"]["LIVE_AGENT_DB_PATH"] == str(
        args.arm_db_path.resolve()
    )
    assert json.loads(result_json.read_text(encoding="utf-8"))["mode"] == (
        f"candidate/{args.arm_protocol}"
    )


def test_arm_launcher_uses_clone_runner_main_venv_and_explicit_plugin_path(
    monkeypatch,
    tmp_path,
):
    clone = tmp_path / "clone"
    clone.mkdir()
    python = tmp_path / "main-venv-python"
    plugin_dir = tmp_path / "parent-test-runs"
    plugin_dir.mkdir()
    case = runner.EXPERIMENTS_20[0]
    output = tmp_path / "output"
    output.mkdir()
    seen: dict = {}

    class FakeProcess:
        pid = 12345

        def __init__(self, command, *, cwd, env, start_new_session):
            seen.update(
                command=command,
                cwd=cwd,
                env=env,
                start_new_session=start_new_session,
            )
            result_path = Path(command[command.index("--arm-result-json") + 1])
            result = _result(output, case.scenario)
            payload = dict(runner._serialize_result(result))
            payload["_attestation"] = runner._protocol_attestation(case.protocol)
            runner._atomic_write_json(result_path, payload)

        def wait(self, timeout=None):
            seen["timeout"] = timeout
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "must-not-enter-command")

    result = runner._run_arm_subprocess(
        clone=clone,
        python=python,
        plugin_dir=plugin_dir,
        case=case,
        arm="candidate",
        db_path=tmp_path / "candidate.db",
        output_dir=output,
        run_label="run-01",
    )

    assert result.scenario_statuses == {case.scenario: "passed"}
    assert seen["command"][0] == str(python)
    assert seen["command"][1] == str(
        clone / "scripts" / "run_operation_protocol_experiments.py"
    )
    assert "must-not-enter-command" not in " ".join(seen["command"])
    assert seen["cwd"] == clone
    assert seen["env"]["PYTHONPATH"] == f"{clone}{runner.os.pathsep}{plugin_dir}"
    assert seen["env"]["PYTEST_PLUGINS"] == runner.PLUGIN_MODULE
    assert seen["start_new_session"] is True
    assert seen["timeout"] == runner.ARM_SUBPROCESS_TIMEOUT_SECONDS


def test_arm_launcher_rejects_missing_clone_runtime_attestation(
    monkeypatch,
    tmp_path,
):
    case = runner.EXPERIMENTS_20[0]
    clone = tmp_path / "clone"
    output = tmp_path / "output"
    clone.mkdir()
    output.mkdir()

    class FakeProcess:
        pid = 12345

        def __init__(self, command, **_kwargs):
            result_path = Path(command[command.index("--arm-result-json") + 1])
            result = _result(output, case.scenario)
            runner._atomic_write_json(result_path, runner._serialize_result(result))

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)

    with pytest.raises(runner.InfrastructureError, match="attestation"):
        runner._run_arm_subprocess(
            clone=clone,
            python=tmp_path / "python",
            plugin_dir=tmp_path / "plugins",
            case=case,
            arm="candidate",
            db_path=tmp_path / "candidate.db",
            output_dir=output,
            run_label="run-01",
        )


def test_prepare_clone_uses_local_no_hardlinks_and_detached_fixed_sha(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "source"
    clone = tmp_path / "clone"
    source.mkdir()
    calls: list[tuple[tuple[str, ...], Path]] = []

    def fake_git(arguments, *, cwd, allowed_returncodes=(0,)):
        calls.append((tuple(arguments), cwd))
        if arguments[0] == "clone":
            clone.mkdir()
        return subprocess.CompletedProcess(["git", *arguments], 0, "", "")

    monkeypatch.setattr(runner, "_run_git", fake_git)
    monkeypatch.setattr(runner, "_git_output", lambda _args, *, cwd: "c" * 40)
    monkeypatch.setattr(
        runner,
        "_clone_status",
        lambda _clone: (runner._sha256_bytes(b""), True),
    )
    monkeypatch.setattr(runner, "_runtime_bundle_sha256", lambda _root: "d" * 64)

    runner._prepare_clone(source, clone, "c" * 40, "d" * 64)

    clone_args = calls[0][0]
    assert clone_args[:5] == (
        "clone",
        "--local",
        "--no-hardlinks",
        "--quiet",
        "--no-checkout",
    )
    assert calls[1][0] == ("checkout", "--quiet", "--detach", "c" * 40)


def test_semantic_or_hard_failure_is_quality_but_missing_judge_is_infrastructure(
    tmp_path,
):
    scenario = runner.EXPERIMENTS_20[0].scenario
    quality_failure = _result(
        tmp_path / "quality",
        scenario,
        technical="failed",
        semantic="failed",
    )
    invalid_judge = _result(
        tmp_path / "invalid",
        scenario,
        judge_completed=0,
        judge_tokens=0,
    )

    assert runner.judge_infrastructure_errors(quality_failure, scenario) == ()
    assert any(
        "judge" in error
        for error in runner.judge_infrastructure_errors(invalid_judge, scenario)
    )


def test_main_runs_all_forty_calls_counterbalanced_and_rolls_back_each_clone(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "fixture.db"
    db_path.write_bytes(b"immutable fixture")
    snapshot = _patch_outer_preflight(monkeypatch, db_path)
    calls: list[tuple[int, str, str, Path, Path, Path]] = []

    def fake_arm(**kwargs):
        case = kwargs["case"]
        arm = kwargs["arm"]
        calls.append(
            (
                case.index,
                arm,
                runner._arm_protocol(case, arm),
                kwargs["clone"],
                kwargs["db_path"],
                kwargs["plugin_dir"],
            )
        )
        return _result(
            kwargs["output_dir"] / f"fake-{arm}",
            case.scenario,
            technical="failed" if arm == "baseline" else "passed",
            semantic="failed" if arm == "baseline" else "passed",
        )

    monkeypatch.setattr(runner, "_run_arm_subprocess", fake_arm)
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "journal-must-not-contain-this")
    output = tmp_path / "output"

    assert runner.main(
        ["--db-path", str(db_path), "--output-dir", str(output)]
    ) == 0

    assert len(calls) == 40
    assert [(index, arm) for index, arm, *_rest in calls] == [
        (case.index, arm) for case in runner.EXPERIMENTS_20 for arm in case.arm_order
    ]
    assert all(
        protocol == ("default" if arm == "baseline" else runner.EXPERIMENTS_20[index - 1].protocol)
        for index, arm, protocol, _clone, _db, _plugin in calls
    )
    assert all(not clone.exists() for _index, _arm, _protocol, clone, _db, _plugin in calls)
    assert all(not copied_db.exists() for _index, _arm, _protocol, _clone, copied_db, _plugin in calls)
    assert all(
        plugin != runner.PLUGIN_DIR and not plugin.exists()
        for _index, _arm, _protocol, _clone, _db, plugin in calls
    )
    assert all(not clone.exists() for clone in snapshot.__dict__["_test_clones"])

    certificates = list(output.rglob("*_rollback.json"))
    assert len(certificates) == 20
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["rollback_complete"]
        for path in certificates
    )
    journal_path = next(output.rglob("*_journal.json"))
    journal_text = journal_path.read_text(encoding="utf-8")
    journal = json.loads(journal_text)
    assert journal["state"] == "complete"
    assert len(journal["experiments"]) == 20
    assert all(item["verdict"] == "improved" for item in journal["experiments"])
    assert all(
        item["state"] == "completed_rolled_back"
        and item["recovery_required"] is False
        for item in journal["experiments"]
    )
    assert journal["quality_status"] == "improved"
    assert "journal-must-not-contain-this" not in journal_text
    assert next(output.rglob("*_comparison.md")).is_file()


def test_quality_failures_continue_all_twenty_experiments_and_return_one(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "fixture.db"
    db_path.write_bytes(b"immutable fixture")
    _patch_outer_preflight(monkeypatch, db_path)
    calls = 0

    def failed_arm(**kwargs):
        nonlocal calls
        calls += 1
        return _result(
            kwargs["output_dir"] / f"failure-{calls}",
            kwargs["case"].scenario,
            technical="failed",
            semantic="failed",
        )

    monkeypatch.setattr(runner, "_run_arm_subprocess", failed_arm)

    assert runner.main(
        ["--db-path", str(db_path), "--output-dir", str(tmp_path / "out")]
    ) == 1
    assert calls == 40


@pytest.mark.parametrize("failure", ["database", "plugin", "judge"])
def test_integrity_or_judge_infrastructure_failure_aborts_and_rolls_back(
    monkeypatch,
    tmp_path,
    failure,
):
    db_path = tmp_path / "fixture.db"
    db_path.write_bytes(b"immutable fixture")
    _patch_outer_preflight(monkeypatch, db_path)
    calls = 0

    def invalid_arm(**kwargs):
        nonlocal calls
        calls += 1
        result = _result(kwargs["output_dir"] / "invalid", kwargs["case"].scenario)
        if failure == "database":
            kwargs["db_path"].write_bytes(b"mutated")
        elif failure == "plugin":
            (
                kwargs["plugin_dir"] / f"{runner.PLUGIN_MODULE}.py"
            ).write_text("# mutated\n", encoding="utf-8")
        else:
            result.judge_completed = 0
            result.judge_attempts = 0
            result.judge_total_tokens = 0
            result.judge_input_tokens = 0
            result.judge_output_tokens = 0
            result.judge_models = {}
        return result

    monkeypatch.setattr(runner, "_run_arm_subprocess", invalid_arm)
    output = tmp_path / "output"

    assert runner.main(
        ["--db-path", str(db_path), "--output-dir", str(output)]
    ) == 2
    assert calls == 1
    journal = json.loads(next(output.rglob("*_journal.json")).read_text("utf-8"))
    assert journal["state"] == "aborted"
    assert journal["abort_reason"]
    certificate = json.loads(
        next(output.rglob("*_rollback.json")).read_text(encoding="utf-8")
    )
    assert certificate["rollback_complete"] is True
    assert certificate["clone_removed"] is True
    assert certificate["arm_databases_removed"] is True
    assert journal["experiments"][0]["state"] == "aborted_rolled_back"


def test_env_integrity_snapshot_contains_metadata_not_secret(
    monkeypatch,
    tmp_path,
):
    secret = "private-token-that-must-never-be-read-into-report"
    (tmp_path / ".env").write_text(f"TOKEN={secret}\n", encoding="utf-8")
    (tmp_path / ".env").chmod(0o600)
    monkeypatch.setattr(
        runner,
        "_run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )

    metadata = runner._env_metadata(tmp_path)

    serialized = json.dumps(metadata)
    assert metadata["exists"] is True
    assert metadata["ignored"] is True
    assert metadata["mode"] == 0o600
    assert secret not in serialized
    assert set(metadata) == {"exists", "ignored", "mode", "size", "mtime_ns"}


def _summary(
    passed: bool,
    *,
    tokens: int = 100,
    calls: int = 4,
    reroutes: int = 0,
    tool_errors: int = 0,
    http_500: int = 0,
    judge_errors: int = 0,
) -> dict:
    return {
        "combined_pass": passed,
        "agent_tokens": tokens,
        "agent_llm_calls": calls,
        "reroutes": reroutes,
        "tool_errors": tool_errors,
        "http_500": http_500,
        "judge_errors": judge_errors,
        "skipped": 0,
    }


def _family_entries(
    family: str,
    *,
    baseline_passes: set[str],
    candidate_passes: set[str],
    baseline_tokens: int = 100,
    candidate_tokens: int = 100,
) -> list[dict]:
    return [
        {
            "aspect": aspect,
            "family": family,
            "state": "completed_rolled_back",
            "rollback": "complete",
            "error": "",
            "arms": {
                "baseline": _summary(
                    aspect in baseline_passes,
                    tokens=baseline_tokens,
                ),
                "candidate": _summary(
                    aspect in candidate_passes,
                    tokens=candidate_tokens,
                ),
            },
        }
        for aspect in runner.ASPECT_SCENARIOS
    ]


def test_family_accepts_strict_quality_gain_without_regression():
    family = runner.PROTOCOL_FAMILIES[0]
    verdict = runner.evaluate_family(
        _family_entries(
            family,
            baseline_passes={"row_filtering"},
            candidate_passes={"row_filtering", "cardinality"},
        ),
        family,
    )

    assert verdict.status == "improved"
    assert verdict.baseline_successes == 1
    assert verdict.candidate_successes == 2
    assert verdict.regressions == ()


def test_family_accepts_only_preregistered_efficiency_gain_when_both_are_perfect():
    family = runner.PROTOCOL_FAMILIES[0]
    all_aspects = set(runner.ASPECT_SCENARIOS)

    accepted = runner.evaluate_family(
        _family_entries(
            family,
            baseline_passes=all_aspects,
            candidate_passes=all_aspects,
            baseline_tokens=100,
            candidate_tokens=90,
        ),
        family,
    )
    rejected = runner.evaluate_family(
        _family_entries(
            family,
            baseline_passes=all_aspects,
            candidate_passes=all_aspects,
            baseline_tokens=100,
            candidate_tokens=91,
        ),
        family,
    )

    assert accepted.status == "improved"
    assert rejected.status == "not_improved"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda rows: rows[1]["arms"]["candidate"].update(http_500=1), "HTTP 500"),
        (lambda rows: rows[1]["arms"]["candidate"].update(judge_errors=1), "judge"),
        (lambda rows: rows[1]["arms"]["candidate"].update(reroutes=1), "reroutes"),
        (lambda rows: rows[1]["arms"]["candidate"].update(tool_errors=1), "tool errors"),
        (lambda rows: rows[1]["arms"]["candidate"].update(agent_tokens=151), "110%"),
    ],
)
def test_family_rejects_fixed_safety_and_efficiency_gate_violations(
    mutation,
    reason,
):
    family = runner.PROTOCOL_FAMILIES[0]
    rows = _family_entries(
        family,
        baseline_passes={"row_filtering"},
        candidate_passes={"row_filtering", "cardinality"},
    )
    mutation(rows)

    verdict = runner.evaluate_family(rows, family)

    assert verdict.status == "not_improved"
    assert any(reason.casefold() in item.casefold() for item in verdict.reasons)


def test_family_rejects_baseline_pass_to_candidate_fail_even_with_net_gain():
    family = runner.PROTOCOL_FAMILIES[0]
    verdict = runner.evaluate_family(
        _family_entries(
            family,
            baseline_passes={"row_filtering"},
            candidate_passes={"cardinality", "constraint_rejection"},
        ),
        family,
    )

    assert verdict.status == "not_improved"
    assert verdict.regressions == ("row_filtering",)


def test_family_is_inconclusive_when_pair_or_rollback_is_incomplete():
    family = runner.PROTOCOL_FAMILIES[0]
    rows = _family_entries(
        family,
        baseline_passes=set(),
        candidate_passes={"row_filtering"},
    )
    rows[0]["state"] = "aborted_rolled_back"

    verdict = runner.evaluate_family(rows, family)

    assert verdict.status == "inconclusive"


def test_runtime_bundle_fails_closed_when_required_file_is_untracked(
    monkeypatch,
    tmp_path,
):
    tracked = set(runner.COMMITTED_RUNTIME_PATHS) - {
        "scripts/run_operation_protocol_experiments.py"
    }
    monkeypatch.setattr(runner, "_tracked_runtime_paths", lambda _root: tracked)

    with pytest.raises(runner.InfrastructureError, match="not committed"):
        runner._runtime_bundle_sha256(tmp_path)


def test_preflight_rejects_base_sha_that_is_not_clean_head(monkeypatch, tmp_path):
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(base_sha="old", db_path=tmp_path / "unused.db")
    snapshot = _snapshot()
    monkeypatch.setattr(runner, "validate_manifest", lambda: None)
    monkeypatch.setattr(runner, "capture_repository_snapshot", lambda _root: snapshot)
    monkeypatch.setattr(runner, "_resolve_base_sha", lambda _root, _value: "d" * 40)

    with pytest.raises(SystemExit):
        runner._preflight(parser, args)


def test_sqlite_copy_safety_rejects_wal_or_sidecar_fixture(tmp_path):
    database = tmp_path / "wal.db"
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        connection.execute("CREATE TABLE item (id INTEGER)")
        connection.execute("INSERT INTO item VALUES (1)")
        connection.commit()

        errors = runner._sqlite_copy_safety_errors(database)
    finally:
        connection.close()

    assert errors
    assert any("WAL" in error or "sidecar" in error for error in errors)


def test_signal_interruption_is_reported_after_runtime_root_is_removed(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "fixture.db"
    db_path.write_bytes(b"immutable fixture")
    _patch_outer_preflight(monkeypatch, db_path)
    monkeypatch.setattr(
        runner,
        "_run_arm_subprocess",
        lambda **_kwargs: (_ for _ in ()).throw(runner.RunInterrupted(signal.SIGTERM)),
    )
    output = tmp_path / "output"

    assert runner.main(["--db-path", str(db_path), "--output-dir", str(output)]) == (
        128 + signal.SIGTERM
    )

    journal = json.loads(next(output.rglob("*_journal.json")).read_text("utf-8"))
    entry = journal["experiments"][0]
    assert journal["state"] == "aborted"
    assert entry["state"] == "aborted_rolled_back"
    assert entry["recovery_required"] is False
    assert not Path(entry["runtime_temp_root"]).exists()
