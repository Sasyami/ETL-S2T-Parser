#!/usr/bin/env python
"""Run 20 isolated Max-only A/B experiments for SQL-risk protocols.

Every experiment is executed from a fresh local clone checked out at one
preregistered commit.  Runtime databases are disposable copies, so discarding
the clone and copies is the rollback mechanism.  Only immutable benchmark
artifacts, a sanitized journal and rollback certificates survive the run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import signal
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

try:
    from scripts.run_live_agent_benchmark import ModeResult, _selected_scenario_count
    from scripts.run_multiagent_experiments import EXPERIMENTS
    from scripts.run_multiagent_holdout import (
        HARD_CORRECTNESS_PYTEST_ARGS,
        fixture_preflight_errors,
        sqlite_sha256,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from run_live_agent_benchmark import (  # type: ignore[no-redef]
        ModeResult,
        _selected_scenario_count,
    )
    from run_multiagent_experiments import (  # type: ignore[no-redef]
        EXPERIMENTS,
    )
    from run_multiagent_holdout import (  # type: ignore[no-redef]
        HARD_CORRECTNESS_PYTEST_ARGS,
        fixture_preflight_errors,
        sqlite_sha256,
    )

try:
    from agents.operation_protocols import (
        OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV,
        SQL_RISK_ASPECTS,
        SQL_RISK_PROTOCOL_CANDIDATES,
        SQL_RISK_PROTOCOL_FAMILIES,
        protocol_variant_sha256,
    )
except ModuleNotFoundError:  # direct execution without the repository on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agents.operation_protocols import (  # type: ignore[no-redef]
        OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV,
        SQL_RISK_ASPECTS,
        SQL_RISK_PROTOCOL_CANDIDATES,
        SQL_RISK_PROTOCOL_FAMILIES,
        protocol_variant_sha256,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "GigaChat-2-Max"
PROVIDER = "gigachat"
PROTOCOL_ENV = OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV
DEFAULT_DB_PATH = PROJECT_ROOT / ".test_runs" / "synthetic_live.db"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / ".test_runs" / "operation-protocol-experiments"
)
MAIN_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PLUGIN_DIR = PROJECT_ROOT / ".test_runs"
PLUGIN_MODULE = "synthetic_live_support"
PLUGIN_PATH = PLUGIN_DIR / f"{PLUGIN_MODULE}.py"

COMMITTED_RUNTIME_PATHS = (
    "agents/operation_protocols.py",
    "agents/tools/__init__.py",
    "agents/tools/context.py",
    "scripts/run_live_agent_benchmark.py",
    "scripts/run_multiagent_experiments.py",
    "scripts/run_multiagent_holdout.py",
    "scripts/run_operation_protocol_experiments.py",
    "tests/test_live_agent_scenarios.py",
    "tests/test_operation_protocol_experiments.py",
    "tests/test_operation_protocols.py",
)
SQLITE_SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")
QUALITY_TOKEN_GUARD_RATIO = 1.10
EFFICIENCY_TOKEN_RATIO = 0.90
ARM_SUBPROCESS_TIMEOUT_SECONDS = 1200

ASPECT_SCENARIOS: Mapping[str, str] = {
    "row_filtering": "test_live_agent_checks_row_loss_risk",
    "cardinality": "test_live_agent_checks_duplicate_risk_in_target",
    "constraint_rejection": (
        "test_live_agent_checks_nulls_in_required_target_fields"
    ),
    "value_changes": "test_live_agent_checks_value_change_risk",
    "write_semantics": "test_live_agent_checks_write_semantics_risk",
}
PROTOCOL_FAMILIES = tuple(SQL_RISK_PROTOCOL_FAMILIES)


@dataclass(frozen=True)
class ProtocolExperiment:
    """One SQL-risk aspect/family comparison against the default protocol."""

    index: int
    aspect: str
    family: str
    scenario: str

    @property
    def protocol(self) -> str:
        return f"{self.aspect}__{self.family}"

    @property
    def protocol_sha256(self) -> str:
        return protocol_variant_sha256(self.protocol)

    @property
    def order_name(self) -> str:
        return "AB" if self.index % 2 else "BA"

    @property
    def arm_order(self) -> tuple[str, str]:
        return (
            ("baseline", "candidate")
            if self.order_name == "AB"
            else ("candidate", "baseline")
        )


EXPERIMENTS_20 = tuple(
    ProtocolExperiment(index, aspect, family, ASPECT_SCENARIOS[aspect])
    for index, (aspect, family) in enumerate(
        (
            (aspect, family)
            for aspect in ASPECT_SCENARIOS
            for family in PROTOCOL_FAMILIES
        ),
        start=1,
    )
)

COMMON_ARM_ENVIRONMENT: Mapping[str, str] = {
    "CHAT_AGENT_MODE": "multiagent",
    "LIVE_AGENT_MODE": "multiagent",
    "LLM_PROVIDER": PROVIDER,
    "GIGACHAT_MODEL": MODEL,
    "WORKER_CAPABILITY_REROUTE_EXPERIMENT": "1",
    "WORKER_SPLIT_TOOL_CALL_EXPERIMENT": "0",
    "OPERATION_SQL_RISK_ASPECTS_EXPERIMENT": "1",
    "GIGACHAT_JUDGE_MODEL": MODEL,
    "GIGACHAT_TEMPERATURE": "0",
    "GIGACHAT_TIMEOUT": "180",
    "LLM_TIMEOUT": "180",
    "LIVE_AGENT_HTTP_TIMEOUT": "600",
    "NEO4J_URI": "",
    "NEO4J_USERNAME": "",
    "NEO4J_USER": "",
    "NEO4J_PASSWORD": "",
    "NEO4J_DATABASE": "",
    # Experiment rollback must not leave traces in a separate external system.
    "LANGFUSE_ENABLED": "false",
}


@dataclass(frozen=True)
class RepositorySnapshot:
    """Non-secret integrity state of the source checkout."""

    head: str
    tracked_status_sha256: str
    tracked_clean: bool
    sample_count: int
    samples_sha256: str
    env_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class FamilyVerdict:
    """Preregistered aggregate verdict for one five-aspect protocol family."""

    family: str
    status: str
    baseline_successes: int
    candidate_successes: int
    regressions: tuple[str, ...]
    baseline_tokens: int
    candidate_tokens: int
    baseline_calls: int
    candidate_calls: int
    baseline_reroutes: int
    candidate_reroutes: int
    baseline_tool_errors: int
    candidate_tool_errors: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PreflightState:
    """Immutable inputs attested before credentials or network are used."""

    db_path: Path
    db_sha256: str
    db_file_state: Mapping[str, str]
    repository: RepositorySnapshot
    base_sha: str
    runtime_bundle_sha256: str
    plugin_sha256: str


class InfrastructureError(RuntimeError):
    """A failure that invalidates the experiment rather than its hypothesis."""


class RunInterrupted(RuntimeError):
    """A handled process signal that still permits rollback and reporting."""

    def __init__(self, signum: int):
        self.signum = int(signum)
        try:
            label = signal.Signals(self.signum).name
        except ValueError:
            label = str(self.signum)
        super().__init__(f"interrupted by {label}")


_ACTIVE_CHILD_PROCESS: subprocess.Popen[Any] | None = None
_SIGNAL_ALREADY_RAISED = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace one sanitized JSON document atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_error(error: BaseException | str) -> str:
    """Bound diagnostic text without serializing environment or command state."""
    return " ".join(str(error).split())[:1000]


def _run_git(
    arguments: Sequence[str],
    *,
    cwd: Path,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode not in allowed_returncodes:
        diagnostic = (completed.stderr or completed.stdout or "git failed").strip()
        raise InfrastructureError(_safe_error(diagnostic))
    return completed


def _git_output(arguments: Sequence[str], *, cwd: Path) -> str:
    return _run_git(arguments, cwd=cwd).stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_runtime_paths(root: Path) -> set[str]:
    output = _git_output(
        ["-c", "core.quotepath=false", "ls-files", "--", *COMMITTED_RUNTIME_PATHS],
        cwd=root,
    )
    return {line.strip() for line in output.splitlines() if line.strip()}


def _runtime_bundle_sha256(root: Path) -> str:
    """Hash all committed code that defines and executes this experiment."""
    tracked = _tracked_runtime_paths(root)
    missing = sorted(set(COMMITTED_RUNTIME_PATHS) - tracked)
    if missing:
        raise InfrastructureError(
            "experiment runtime path is not committed: " + ", ".join(missing)
        )
    digest = hashlib.sha256()
    for relative in COMMITTED_RUNTIME_PATHS:
        path = root / relative
        if not path.is_file():
            raise InfrastructureError(f"experiment runtime path is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sqlite_sidecars(path: Path) -> tuple[Path, ...]:
    return tuple(Path(f"{path}{suffix}") for suffix in SQLITE_SIDECAR_SUFFIXES)


def _sqlite_file_state(path: Path) -> Mapping[str, str]:
    """Hash the main SQLite file and every rollback/WAL sidecar."""
    resolved = path.expanduser().resolve()
    state: dict[str, str] = {"database": sqlite_sha256(resolved)}
    for sidecar in _sqlite_sidecars(resolved):
        if sidecar.exists():
            state[sidecar.name.removeprefix(resolved.name)] = _file_sha256(sidecar)
    return state


def _sqlite_copy_safety_errors(path: Path) -> tuple[str, ...]:
    """Reject fixtures for which a byte copy/hash would not be authoritative."""
    resolved = path.expanduser().resolve()
    errors: list[str] = []
    present_sidecars = [
        sidecar.name for sidecar in _sqlite_sidecars(resolved) if sidecar.exists()
    ]
    if present_sidecars:
        errors.append("SQLite fixture has sidecar files: " + ", ".join(present_sidecars))
    try:
        connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return tuple((*errors, f"cannot inspect SQLite copy safety: {exc}"))
    try:
        journal_mode = str(
            connection.execute("PRAGMA journal_mode").fetchone()[0]
        ).casefold()
        if journal_mode == "wal":
            errors.append(
                "SQLite fixture uses WAL mode; checkpoint it before the run"
            )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or str(integrity[0]).casefold() != "ok":
            errors.append("SQLite fixture integrity_check is not ok")
    except sqlite3.Error as exc:
        errors.append(f"SQLite copy-safety check failed: {exc}")
    finally:
        connection.close()
    # A read-only connection to a WAL database can materialize sidecars.
    after_sidecars = [
        sidecar.name for sidecar in _sqlite_sidecars(resolved) if sidecar.exists()
    ]
    if after_sidecars and not present_sidecars:
        errors.append(
            "SQLite copy-safety inspection created sidecars: "
            + ", ".join(after_sidecars)
        )
    return tuple(dict.fromkeys(errors))


def _sample_manifest(root: Path) -> tuple[int, str]:
    raw_paths = _git_output(
        ["-c", "core.quotepath=false", "ls-files", "--", "samples"],
        cwd=root,
    )
    paths = [Path(item) for item in raw_paths.splitlines() if item.strip()]
    digest = hashlib.sha256()
    for relative in paths:
        absolute = root / relative
        if not absolute.is_file():
            raise InfrastructureError(f"tracked sample is missing: {relative}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(absolute).encode("ascii"))
        digest.update(b"\0")
    return len(paths), digest.hexdigest()


def _env_metadata(root: Path) -> Mapping[str, Any]:
    """Return metadata only; the .env contents are deliberately never read."""
    path = root / ".env"
    if not path.exists():
        return {"exists": False}
    ignored = _run_git(
        ["check-ignore", "--quiet", "--", ".env"],
        cwd=root,
        allowed_returncodes=(0, 1),
    ).returncode == 0
    if not ignored:
        raise InfrastructureError(".env exists but is not ignored by Git")
    metadata = path.stat()
    return {
        "exists": True,
        "ignored": True,
        "mode": stat.S_IMODE(metadata.st_mode),
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
    }


def capture_repository_snapshot(root: Path = PROJECT_ROOT) -> RepositorySnapshot:
    status = _git_output(
        ["status", "--porcelain=v1", "--untracked-files=no"],
        cwd=root,
    )
    sample_count, samples_sha256 = _sample_manifest(root)
    return RepositorySnapshot(
        head=_git_output(["rev-parse", "HEAD"], cwd=root),
        tracked_status_sha256=_sha256_bytes(status.encode("utf-8")),
        tracked_clean=not bool(status),
        sample_count=sample_count,
        samples_sha256=samples_sha256,
        env_metadata=_env_metadata(root),
    )


def _public_snapshot(snapshot: RepositorySnapshot) -> Mapping[str, Any]:
    return asdict(snapshot)


def _resolve_base_sha(root: Path, requested: str) -> str:
    reference = requested.strip() or "HEAD"
    return _git_output(["rev-parse", "--verify", f"{reference}^{{commit}}"], cwd=root)


def _clone_status(clone: Path) -> tuple[str, bool]:
    status = _git_output(
        ["status", "--porcelain=v1", "--untracked-files=all"],
        cwd=clone,
    )
    return _sha256_bytes(status.encode("utf-8")), not bool(status)


def _prepare_clone(
    source: Path,
    destination: Path,
    base_sha: str,
    runtime_bundle_sha256: str,
) -> None:
    _run_git(
        [
            "clone",
            "--local",
            "--no-hardlinks",
            "--quiet",
            "--no-checkout",
            str(source),
            str(destination),
        ],
        cwd=source,
    )
    _run_git(
        ["checkout", "--quiet", "--detach", base_sha],
        cwd=destination,
    )
    actual_sha = _git_output(["rev-parse", "HEAD"], cwd=destination)
    if actual_sha != base_sha:
        raise InfrastructureError(
            f"clone HEAD mismatch: {actual_sha} != {base_sha}"
        )
    _status_digest, clean = _clone_status(destination)
    if not clean:
        raise InfrastructureError("fresh experiment clone is not clean")
    clone_bundle_sha256 = _runtime_bundle_sha256(destination)
    if clone_bundle_sha256 != runtime_bundle_sha256:
        raise InfrastructureError(
            "clone runtime bundle differs from the preregistered source bundle"
        )


def validate_manifest() -> None:
    if len(EXPERIMENTS_20) != 20:
        raise ValueError("operation protocol matrix must contain exactly 20 experiments")
    names = [case.protocol for case in EXPERIMENTS_20]
    if len(set(names)) != 20:
        raise ValueError("operation protocol names must be unique")
    expected = {
        f"{aspect}__{family}"
        for aspect in ASPECT_SCENARIOS
        for family in PROTOCOL_FAMILIES
    }
    if set(names) != expected:
        raise ValueError("operation protocol matrix is not the fixed 5x4 product")
    if tuple(ASPECT_SCENARIOS) != tuple(SQL_RISK_ASPECTS):
        raise ValueError("runner aspect order differs from the production allowlist")
    if tuple(names) != tuple(SQL_RISK_PROTOCOL_CANDIDATES):
        raise ValueError("runner protocol order differs from the production allowlist")
    e2_scenarios = set(EXPERIMENTS["E2"].scenarios)
    if set(ASPECT_SCENARIOS.values()) != e2_scenarios:
        raise ValueError("operation protocol matrix must reuse exactly the E2 scenarios")
    for case in EXPERIMENTS_20:
        if _selected_scenario_count((case.scenario,), ()) != 1:
            raise ValueError(f"scenario must contain one HTTP exchange: {case.scenario}")
        expected_order = (
            ("baseline", "candidate")
            if case.index % 2
            else ("candidate", "baseline")
        )
        if case.arm_order != expected_order:
            raise ValueError(f"counterbalancing drift at experiment {case.index}")


def _resolved_db_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    configured = os.getenv("LIVE_AGENT_DB_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_DB_PATH.resolve()


def _arm_protocol(case: ProtocolExperiment, arm: str) -> str:
    if arm == "baseline":
        return "default"
    if arm == "candidate":
        return case.protocol
    raise ValueError(f"unknown arm: {arm}")


def _arm_environment(protocol: str, db_path: Path) -> Mapping[str, str]:
    return {
        **COMMON_ARM_ENVIRONMENT,
        PROTOCOL_ENV: protocol,
        "LIVE_AGENT_DB_PATH": str(db_path),
    }


def _protocol_attestation(protocol: str) -> Mapping[str, str]:
    return {
        "runtime_mode": "multiagent",
        "provider": PROVIDER,
        "agent_model": MODEL,
        "judge_model": MODEL,
        "protocol": protocol,
        "protocol_sha256": (
            "default" if protocol == "default" else protocol_variant_sha256(protocol)
        ),
    }


def _serialize_result(result: ModeResult) -> Mapping[str, Any]:
    payload: dict[str, Any] = {}
    for item in fields(ModeResult):
        value = getattr(result, item.name)
        payload[item.name] = str(value) if isinstance(value, Path) else value
    return payload


def _deserialize_result(payload: Mapping[str, Any]) -> ModeResult:
    values = dict(payload)
    values["transcript_path"] = Path(str(values["transcript_path"]))
    values["junit_path"] = Path(str(values["junit_path"]))
    known = {item.name for item in fields(ModeResult)}
    return ModeResult(**{key: value for key, value in values.items() if key in known})


def _run_arm_worker(args: argparse.Namespace) -> int:
    """Internal entry point executed from the isolated clone."""
    benchmark = importlib.import_module("scripts.run_live_agent_benchmark")
    db_path = args.arm_db_path.expanduser().resolve()
    output_dir = args.arm_output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = benchmark._run_mode(
        mode="multiagent",
        provider=PROVIDER,
        model=MODEL,
        targets=benchmark._scenario_targets((args.arm_scenario,)),
        pytest_args=HARD_CORRECTNESS_PYTEST_ARGS,
        output_dir=output_dir,
        run_label=args.arm_run_label,
        llm_judge=True,
        extra_env=_arm_environment(args.arm_protocol, db_path),
    )
    result.mode = f"{args.arm_name}/{args.arm_protocol}"
    payload = dict(_serialize_result(result))
    payload["_attestation"] = _protocol_attestation(args.arm_protocol)
    _atomic_write_json(args.arm_result_json, payload)
    return 0


def _terminate_child(process: subprocess.Popen[Any]) -> None:
    """Bound shutdown of an isolated worker before its clone is removed."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _experiment_signal_handler(signum: int, _frame: Any) -> None:
    """Convert SIGINT/SIGTERM into a cleanup-preserving exception once."""
    global _SIGNAL_ALREADY_RAISED
    process = _ACTIVE_CHILD_PROCESS
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if _SIGNAL_ALREADY_RAISED:
        return
    _SIGNAL_ALREADY_RAISED = True
    raise RunInterrupted(signum)


def _install_experiment_signal_handlers() -> Mapping[int, Any]:
    previous: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _experiment_signal_handler)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    global _SIGNAL_ALREADY_RAISED
    for signum, handler in previous.items():
        signal.signal(signum, handler)
    _SIGNAL_ALREADY_RAISED = False


def _run_arm_subprocess(
    *,
    clone: Path,
    python: Path,
    plugin_dir: Path,
    case: ProtocolExperiment,
    arm: str,
    db_path: Path,
    output_dir: Path,
    run_label: str,
) -> ModeResult:
    result_json = output_dir / f"{run_label}_{arm}_result.json"
    command = [
        str(python),
        str(clone / "scripts" / "run_operation_protocol_experiments.py"),
        "--_arm-worker",
        "--arm-name",
        arm,
        "--arm-protocol",
        _arm_protocol(case, arm),
        "--arm-scenario",
        case.scenario,
        "--arm-db-path",
        str(db_path),
        "--arm-output-dir",
        str(output_dir),
        "--arm-run-label",
        run_label,
        "--arm-result-json",
        str(result_json),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(clone), str(plugin_dir))),
            "PYTEST_PLUGINS": PLUGIN_MODULE,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    global _ACTIVE_CHILD_PROCESS
    process = subprocess.Popen(
        command,
        cwd=clone,
        env=environment,
        start_new_session=True,
    )
    _ACTIVE_CHILD_PROCESS = process
    try:
        try:
            return_code = process.wait(timeout=ARM_SUBPROCESS_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            _terminate_child(process)
            raise InfrastructureError(
                f"isolated arm worker exceeded {ARM_SUBPROCESS_TIMEOUT_SECONDS}s"
            ) from exc
        except RunInterrupted:
            _terminate_child(process)
            raise
    finally:
        _ACTIVE_CHILD_PROCESS = None
    if return_code != 0:
        raise InfrastructureError(
            f"isolated arm worker exited with code {return_code}"
        )
    if not result_json.is_file():
        raise InfrastructureError("isolated arm worker did not write result JSON")
    try:
        payload = json.loads(result_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InfrastructureError(
            f"cannot read isolated arm result: {_safe_error(exc)}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise InfrastructureError("isolated arm result is not a JSON object")
    expected_attestation = _protocol_attestation(_arm_protocol(case, arm))
    if payload.get("_attestation") != expected_attestation:
        raise InfrastructureError(
            "isolated arm runtime attestation differs from the preregistration"
        )
    return _deserialize_result(payload)


def _semantic_status(result: ModeResult, scenario: str) -> str:
    return str(result.semantic_statuses.get(scenario) or "").strip().casefold()


def judge_infrastructure_errors(
    result: ModeResult,
    scenario: str,
) -> tuple[str, ...]:
    """Validate completeness while treating hard/semantic failures as quality."""
    errors: list[str] = []
    expected = {scenario}
    if set(result.scenario_statuses) != expected:
        errors.append(
            "technical result keys differ from the single requested scenario"
        )
    if result.skipped:
        errors.append(f"scenario was skipped ({result.skipped})")
    if result.return_code not in {0, 1}:
        errors.append(f"pytest return code is {result.return_code}, expected 0 or 1")
    if result.measured_runs != 1:
        errors.append(f"measured HTTP exchanges={result.measured_runs}, expected=1")
    semantic = _semantic_status(result, scenario)
    if semantic not in {"passed", "failed"}:
        errors.append(f"semantic status is {semantic or 'missing'}")
    attempts = int(result.judge_attempts or 0)
    completed = int(result.judge_completed or 0)
    judge_errors = int(result.judge_errors or 0)
    if attempts <= 0 or completed <= 0:
        errors.append("semantic judge completion telemetry is missing")
    if attempts != completed + judge_errors:
        errors.append("semantic judge attempts do not equal completed+errors")
    if (
        result.judge_total_tokens <= 0
        or result.judge_total_tokens
        != result.judge_input_tokens + result.judge_output_tokens
    ):
        errors.append("semantic judge token telemetry is missing or inconsistent")
    if dict(result.judge_models or {}) != {MODEL: 1}:
        errors.append(
            f"semantic judge models={dict(result.judge_models or {})!r}, "
            f"expected={{{MODEL!r}: 1}}"
        )
    for artifact_name, artifact in (
        ("transcript", result.transcript_path),
        ("JUnit", result.junit_path),
    ):
        if not artifact.is_file():
            errors.append(f"{artifact_name} artifact is missing")
    return tuple(errors)


def _combined_pass(result: ModeResult, scenario: str) -> bool:
    return (
        result.scenario_statuses.get(scenario) == "passed"
        and _semantic_status(result, scenario) == "passed"
    )


def _result_summary(result: ModeResult, scenario: str) -> Mapping[str, Any]:
    return {
        "technical": result.scenario_statuses.get(scenario, "missing"),
        "semantic": _semantic_status(result, scenario) or "missing",
        "combined_pass": _combined_pass(result, scenario),
        "return_code": result.return_code,
        "measured_runs": result.measured_runs,
        "skipped": result.skipped,
        "http_500": result.http_500,
        "presentation_warnings": result.presentation_warnings,
        "efficiency_warnings": result.efficiency_warnings,
        "reroutes": result.reroutes,
        "tool_errors": result.tool_errors,
        "reader_calls": result.reader_calls,
        "agent_llm_calls": result.llm_calls,
        "agent_tokens": result.total_tokens,
        "agent_seconds": result.agent_seconds,
        "judge_attempts": result.judge_attempts,
        "judge_completed": result.judge_completed,
        "judge_errors": result.judge_errors,
        "judge_tokens": result.judge_total_tokens,
        "judge_models": dict(result.judge_models),
        "transcript": str(result.transcript_path),
        "junit": str(result.junit_path),
    }


def _metric(summary: Mapping[str, Any], name: str) -> int:
    try:
        return int(summary.get(name) or 0)
    except (TypeError, ValueError):
        return 0


def evaluate_family(
    entries: Sequence[Mapping[str, Any]],
    family: str,
) -> FamilyVerdict:
    """Apply the frozen five-aspect quality and efficiency gates."""
    selected = [entry for entry in entries if entry.get("family") == family]
    expected_aspects = set(ASPECT_SCENARIOS)
    actual_aspects = {str(entry.get("aspect") or "") for entry in selected}
    incomplete: list[str] = []
    if len(selected) != len(expected_aspects) or actual_aspects != expected_aspects:
        incomplete.append("family does not contain exactly the five fixed aspects")

    arm_rows: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for entry in selected:
        aspect = str(entry.get("aspect") or "unknown")
        arms = entry.get("arms")
        if (
            entry.get("state") != "completed_rolled_back"
            or entry.get("rollback") != "complete"
            or entry.get("error")
            or not isinstance(arms, Mapping)
            or not isinstance(arms.get("baseline"), Mapping)
            or not isinstance(arms.get("candidate"), Mapping)
        ):
            incomplete.append(f"{aspect}: pair or rollback is incomplete")
            continue
        arm_rows.append((aspect, arms["baseline"], arms["candidate"]))

    if incomplete:
        return FamilyVerdict(
            family=family,
            status="inconclusive",
            baseline_successes=0,
            candidate_successes=0,
            regressions=(),
            baseline_tokens=0,
            candidate_tokens=0,
            baseline_calls=0,
            candidate_calls=0,
            baseline_reroutes=0,
            candidate_reroutes=0,
            baseline_tool_errors=0,
            candidate_tool_errors=0,
            reasons=tuple(dict.fromkeys(incomplete)),
        )

    baseline_successes = {
        aspect for aspect, baseline, _candidate in arm_rows if baseline.get("combined_pass") is True
    }
    candidate_successes = {
        aspect for aspect, _baseline, candidate in arm_rows if candidate.get("combined_pass") is True
    }
    regressions = tuple(sorted(baseline_successes - candidate_successes))

    def total(arm: int, metric: str) -> int:
        return sum(_metric(row[arm], metric) for row in arm_rows)

    baseline_tokens = total(1, "agent_tokens")
    candidate_tokens = total(2, "agent_tokens")
    baseline_calls = total(1, "agent_llm_calls")
    candidate_calls = total(2, "agent_llm_calls")
    baseline_reroutes = total(1, "reroutes")
    candidate_reroutes = total(2, "reroutes")
    baseline_tool_errors = total(1, "tool_errors")
    candidate_tool_errors = total(2, "tool_errors")

    reasons: list[str] = []
    quality_improvement = len(candidate_successes) > len(baseline_successes)
    both_perfect = (
        len(candidate_successes) == len(expected_aspects)
        and len(baseline_successes) == len(expected_aspects)
    )
    efficiency_improvement = (
        both_perfect
        and candidate_tokens <= baseline_tokens * EFFICIENCY_TOKEN_RATIO
        and candidate_calls <= baseline_calls
    )
    if regressions:
        reasons.append(f"baseline-pass to candidate-fail regressions: {regressions}")
    if not (quality_improvement or efficiency_improvement):
        reasons.append(
            "no preregistered improvement: require a strict combined-pass gain, "
            "or at 5/5 in both arms at least 10% fewer agent tokens without more calls"
        )
    if any(_metric(row[arm], "skipped") for row in arm_rows for arm in (1, 2)):
        reasons.append("at least one arm was skipped")
    if any(_metric(row[arm], "judge_errors") for row in arm_rows for arm in (1, 2)):
        reasons.append("at least one semantic judge attempt errored")
    if any(_metric(row[arm], "http_500") for row in arm_rows for arm in (1, 2)):
        reasons.append("at least one /chat exchange returned HTTP 500")
    if candidate_reroutes > baseline_reroutes:
        reasons.append("candidate reroutes exceed baseline")
    if candidate_tool_errors > baseline_tool_errors:
        reasons.append("candidate tool errors exceed baseline")
    if (
        quality_improvement
        and candidate_tokens > baseline_tokens * QUALITY_TOKEN_GUARD_RATIO
    ):
        reasons.append("candidate agent tokens exceed 110% of baseline")
    if both_perfect and not efficiency_improvement:
        reasons.append(
            "both arms are 5/5 but candidate lacks the required 10% token "
            "reduction without extra agent calls"
        )

    return FamilyVerdict(
        family=family,
        status="not_improved" if reasons else "improved",
        baseline_successes=len(baseline_successes),
        candidate_successes=len(candidate_successes),
        regressions=regressions,
        baseline_tokens=baseline_tokens,
        candidate_tokens=candidate_tokens,
        baseline_calls=baseline_calls,
        candidate_calls=candidate_calls,
        baseline_reroutes=baseline_reroutes,
        candidate_reroutes=candidate_reroutes,
        baseline_tool_errors=baseline_tool_errors,
        candidate_tool_errors=candidate_tool_errors,
        reasons=tuple(reasons or ("all preregistered family gates passed",)),
    )


def evaluate_families(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[FamilyVerdict, ...]:
    return tuple(evaluate_family(entries, family) for family in PROTOCOL_FAMILIES)


def _pair_verdict(
    baseline: ModeResult | None,
    candidate: ModeResult | None,
    scenario: str,
) -> str:
    if baseline is None or candidate is None:
        return "inconclusive"
    baseline_pass = _combined_pass(baseline, scenario)
    candidate_pass = _combined_pass(candidate, scenario)
    if candidate_pass and not baseline_pass:
        return "improved"
    if baseline_pass and not candidate_pass:
        return "regressed"
    return "tie_pass" if candidate_pass else "tie_fail"


def _write_pair_report(
    path: Path,
    *,
    case: ProtocolExperiment,
    base_sha: str,
    results: Mapping[str, ModeResult],
    infrastructure_error: str,
) -> None:
    lines = [
        f"# Operation protocol experiment {case.index:02d}",
        "",
        f"- Aspect: `{case.aspect}`",
        f"- Family: `{case.family}`",
        f"- Candidate protocol: `{case.protocol}`",
        f"- Candidate protocol SHA256: `{case.protocol_sha256}`",
        "- Baseline protocol: `default`",
        f"- Scenario: `{case.scenario}`",
        f"- Order: `{case.order_name}` ({' → '.join(case.arm_order)})",
        f"- Fixed base SHA: `{base_sha}`",
        f"- Agent and semantic judge: `{MODEL}`",
        "- Runtime: `multiagent`; semantic judge is mandatory",
        "- Automatic promotion: disabled",
        "",
        f"Verdict: **{_pair_verdict(results.get('baseline'), results.get('candidate'), case.scenario)}**",
        "",
        "| Arm | Protocol | Technical | Semantic | Combined | Agent calls | "
        "Agent tokens | Judge calls | Judge tokens | Reader calls | Reroutes | Seconds |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("baseline", "candidate"):
        result = results.get(arm)
        if result is None:
            lines.append(
                f"| {arm} | `{_arm_protocol(case, arm)}` | missing | missing | "
                "0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000 |"
            )
            continue
        lines.append(
            f"| {arm} | `{_arm_protocol(case, arm)}` | "
            f"{result.scenario_statuses.get(case.scenario, 'missing')} | "
            f"{_semantic_status(result, case.scenario) or 'missing'} | "
            f"{int(_combined_pass(result, case.scenario))} | "
            f"{result.llm_calls} | {result.total_tokens} | "
            f"{result.judge_completed} | {result.judge_total_tokens} | "
            f"{result.reader_calls} | {result.reroutes} | "
            f"{result.agent_seconds:.3f} |"
        )
    if infrastructure_error:
        lines.extend(["", "## Infrastructure failure", "", f"- {infrastructure_error}"])
    lines.extend(["", "## Artifacts", ""])
    for arm in ("baseline", "candidate"):
        result = results.get(arm)
        if result is not None:
            lines.append(
                f"- `{arm}`: `{result.transcript_path}`; JUnit: `{result.junit_path}`"
            )
    _write_text(path, lines)


def _write_preregistration(
    path: Path,
    *,
    base_sha: str,
    db_path: Path,
    db_sha256: str,
    snapshot: RepositorySnapshot,
    runtime_bundle_sha256: str,
    plugin_sha256: str,
) -> None:
    lines = [
        "# Preregistered operation-protocol experiments",
        "",
        f"- Fixed base SHA for all fresh clones: `{base_sha}`",
        f"- Runtime bundle SHA256: `{runtime_bundle_sha256}`",
        f"- Frozen synthetic plugin SHA256: `{plugin_sha256}`",
        f"- Source SQLite: `{db_path}`",
        f"- Source SQLite SHA256: `{db_sha256}`",
        f"- Tracked source checkout clean: `{snapshot.tracked_clean}`",
        f"- Tracked samples: `{snapshot.sample_count}`; manifest SHA256: `{snapshot.samples_sha256}`",
        f"- Agent model: `{MODEL}`",
        f"- Mandatory semantic judge: `{MODEL}`",
        "- Runtime: only `multiagent`",
        "- Common flags: capability=1, split=0, typed SQL-risk aspects=1",
        "- Baseline protocol: `default`",
        "- Candidate protocols: fixed `<aspect>__<family>` allowlist",
        "- Order: odd experiments A→B, even experiments B→A",
        "- Presentation warnings are hard pytest failures",
        "- Quality failures do not stop the matrix; integrity or judge-infrastructure failures do",
        "- Per family (five aspects), acceptance requires a strict combined hard+semantic pass gain with no regression, or when both arms are 5/5 at least 10% fewer agent tokens without more calls",
        "- Every accepted family also requires no skips, judge errors or HTTP 500; non-increasing reroutes/tool errors; and at most 110% of baseline tokens when quality improves",
        "- Overall exploratory status is improved when at least one family passes every fixed gate; other families remain separately reported",
        "- No candidate is promoted automatically",
        "- Every experiment is rolled back by deleting its fresh no-hardlinks clone and disposable DB copies",
        "- All 20 cells exercise the full model-owned upstream decision and answer path",
        "- Deterministic SQL-risk facts only supplement evidence; they do not render the final answer or suppress reroute",
        "",
        "## Fixed 20-experiment matrix",
        "",
    ]
    for case in EXPERIMENTS_20:
        lines.append(
            f"{case.index}. `{case.protocol}` (SHA256 `{case.protocol_sha256}`) "
            f"→ `{case.scenario}`; order `{case.order_name}`"
        )
    _write_text(path, lines)


def _write_aggregate_report(
    path: Path,
    *,
    base_sha: str,
    preregistration: Path,
    entries: Sequence[Mapping[str, Any]],
    family_verdicts: Sequence[FamilyVerdict],
    abort_reason: str,
) -> None:
    completed = [
        entry for entry in entries if entry.get("state") == "completed_rolled_back"
    ]
    counts: dict[str, int] = {}
    for entry in completed:
        verdict = str(entry.get("verdict") or "inconclusive")
        counts[verdict] = counts.get(verdict, 0) + 1
    quality_status = (
        "inconclusive"
        if abort_reason
        else (
            "improved"
            if any(verdict.status == "improved" for verdict in family_verdicts)
            else "not_improved"
        )
    )
    lines = [
        "# Operation protocol experiment result",
        "",
        f"Preregistration: `{preregistration}`",
        "",
        f"- Fixed base SHA: `{base_sha}`",
        f"- Completed and rolled back: `{len(completed)}/20`",
        f"- Infrastructure status: `{'aborted' if abort_reason else 'complete'}`",
        f"- Exploratory quality status: `{quality_status}`",
        "- All five aspects exercise a model-generated upstream decision and final answer; deterministic facts only supplement evidence",
        "- Automatic promotion: `disabled`",
        "",
        "| # | Aspect | Family | Order | Baseline | Candidate | Verdict | Rollback |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for entry in entries:
        arms = entry.get("arms") if isinstance(entry.get("arms"), Mapping) else {}
        baseline = arms.get("baseline") if isinstance(arms, Mapping) else None
        candidate = arms.get("candidate") if isinstance(arms, Mapping) else None
        baseline_cell = (
            f"{baseline.get('technical', 'missing')} / {baseline.get('semantic', 'missing')}"
            if isinstance(baseline, Mapping)
            else "missing"
        )
        candidate_cell = (
            f"{candidate.get('technical', 'missing')} / {candidate.get('semantic', 'missing')}"
            if isinstance(candidate, Mapping)
            else "missing"
        )
        lines.append(
            f"| {entry.get('index')} | `{entry.get('aspect')}` | "
            f"`{entry.get('family')}` | {entry.get('order')} | "
            f"{baseline_cell} | {candidate_cell} | "
            f"{entry.get('verdict', 'inconclusive')} | "
            f"{entry.get('rollback', 'missing')} |"
        )
    lines.extend(["", "## Verdict counts", ""])
    if counts:
        lines.extend(f"- `{name}`: {count}" for name, count in sorted(counts.items()))
    else:
        lines.append("- No complete experiment pairs.")
    lines.extend(
        [
            "",
            "## Preregistered family acceptance",
            "",
            "| Family | Baseline passes | Candidate passes | Tokens B/C | Calls B/C | Reroutes B/C | Tool errors B/C | Status | Reasons |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for verdict in family_verdicts:
        reasons = "; ".join(verdict.reasons).replace("|", "\\|")
        lines.append(
            f"| `{verdict.family}` | {verdict.baseline_successes}/5 | "
            f"{verdict.candidate_successes}/5 | {verdict.baseline_tokens}/"
            f"{verdict.candidate_tokens} | {verdict.baseline_calls}/"
            f"{verdict.candidate_calls} | {verdict.baseline_reroutes}/"
            f"{verdict.candidate_reroutes} | {verdict.baseline_tool_errors}/"
            f"{verdict.candidate_tool_errors} | {verdict.status} | {reasons} |"
        )
    if abort_reason:
        lines.extend(["", "## Abort reason", "", f"- {abort_reason}"])
    lines.extend(["", "## Per-experiment reports", ""])
    for entry in entries:
        if entry.get("report"):
            lines.append(f"- `{entry['index']}`: `{entry['report']}`")
    _write_text(path, lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly 20 isolated multiagent SQL-risk protocol A/B experiments."
        )
    )
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--base-sha",
        default="",
        help="Committed base for every fresh clone; defaults to current HEAD.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the fixed design without clones, HTTP or LLM calls.",
    )
    parser.add_argument("--_arm-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--arm-protocol", default="", help=argparse.SUPPRESS)
    parser.add_argument("--arm-scenario", default="", help=argparse.SUPPRESS)
    parser.add_argument("--arm-db-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--arm-output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--arm-run-label", default="", help=argparse.SUPPRESS)
    parser.add_argument("--arm-result-json", type=Path, help=argparse.SUPPRESS)
    return parser


def _validate_worker_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    required = {
        "--arm-name": args.arm_name,
        "--arm-protocol": args.arm_protocol,
        "--arm-scenario": args.arm_scenario,
        "--arm-db-path": args.arm_db_path,
        "--arm-output-dir": args.arm_output_dir,
        "--arm-run-label": args.arm_run_label,
        "--arm-result-json": args.arm_result_json,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("internal arm worker misses " + ", ".join(missing))
    if args.arm_name not in {"baseline", "candidate"}:
        parser.error("internal arm name must be baseline or candidate")
    if args.arm_scenario not in set(ASPECT_SCENARIOS.values()):
        parser.error("internal arm scenario is outside the fixed matrix")
    if args.arm_name == "baseline" and args.arm_protocol != "default":
        parser.error("internal baseline arm must use the default protocol")
    if (
        args.arm_name == "candidate"
        and args.arm_protocol not in SQL_RISK_PROTOCOL_CANDIDATES
    ):
        parser.error("internal candidate arm must use a fixed protocol candidate")
    if args.arm_name == "candidate":
        aspect = args.arm_protocol.partition("__")[0]
        if ASPECT_SCENARIOS.get(aspect) != args.arm_scenario:
            parser.error("internal candidate protocol/scenario aspect mismatch")


def _preflight(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> PreflightState:
    try:
        validate_manifest()
        snapshot = capture_repository_snapshot(PROJECT_ROOT)
        if not snapshot.tracked_clean:
            raise InfrastructureError(
                "source checkout has tracked changes; commit or restore them first"
            )
        base_sha = _resolve_base_sha(PROJECT_ROOT, args.base_sha)
        if base_sha != snapshot.head:
            raise InfrastructureError(
                "base SHA must equal the clean committed HEAD; check out the desired "
                "commit before running experiments"
            )
        runtime_bundle_sha256 = _runtime_bundle_sha256(PROJECT_ROOT)
        db_path = _resolved_db_path(args.db_path)
        errors = fixture_preflight_errors(db_path)
        errors = (*errors, *_sqlite_copy_safety_errors(db_path))
        if errors:
            raise InfrastructureError("; ".join(errors))
        db_sha256 = sqlite_sha256(db_path)
        db_file_state = _sqlite_file_state(db_path)
        if not MAIN_VENV_PYTHON.is_file():
            raise InfrastructureError(f"main venv Python is missing: {MAIN_VENV_PYTHON}")
        if not PLUGIN_PATH.is_file():
            raise InfrastructureError(
                f"live fixture plugin is missing: {PLUGIN_MODULE}"
            )
        plugin_sha256 = _file_sha256(PLUGIN_PATH)
    except (InfrastructureError, OSError, ValueError) as exc:
        parser.error(_safe_error(exc))
    return PreflightState(
        db_path=db_path,
        db_sha256=db_sha256,
        db_file_state=db_file_state,
        repository=snapshot,
        base_sha=base_sha,
        runtime_bundle_sha256=runtime_bundle_sha256,
        plugin_sha256=plugin_sha256,
    )


def _new_journal(
    *,
    base_sha: str,
    db_path: Path,
    db_sha256: str,
    db_file_state: Mapping[str, str],
    snapshot: RepositorySnapshot,
    preregistration: Path,
    runtime_bundle_sha256: str,
    plugin_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runner": "operation_protocol_experiments",
        "started_at": _utc_now(),
        "finished_at": None,
        "state": "running",
        "model": MODEL,
        "judge_model": MODEL,
        "runtime_mode": "multiagent",
        "base_sha": base_sha,
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "plugin_sha256": plugin_sha256,
        "source_db": str(db_path),
        "source_db_sha256": db_sha256,
        "source_db_file_state": dict(db_file_state),
        "repository_before": _public_snapshot(snapshot),
        "preregistration": str(preregistration),
        "experiments": [],
        "family_verdicts": [],
        "quality_status": "pending",
        "abort_reason": "",
    }


def _entry_for(case: ProtocolExperiment) -> dict[str, Any]:
    return {
        "index": case.index,
        "aspect": case.aspect,
        "family": case.family,
        "protocol": case.protocol,
        "protocol_sha256": case.protocol_sha256,
        "scenario": case.scenario,
        "order": case.order_name,
        "state": "preparing",
        "arms": {},
        "verdict": "inconclusive",
        "rollback": "pending",
        "report": "",
        "certificate": "",
        "runtime_temp_root": "",
        "recovery_required": False,
        "error": "",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args._arm_worker:
        _validate_worker_args(parser, args)
        return _run_arm_worker(args)

    preflight = _preflight(parser, args)
    if args.dry_run:
        print(
            "operation protocol dry-run: 20 experiments, 40 Max-only "
            "multiagent exchanges; odd AB/even BA",
            flush=True,
        )
        for case in EXPERIMENTS_20:
            print(
                f"{case.index:02d} {case.protocol}: {case.scenario} "
                f"order={case.order_name}",
                flush=True,
            )
        print(
            "dry-run: no clone, HTTP, LLM or output artifact was created",
            flush=True,
        )
        return 0

    # Read credentials into process memory only after the no-network dry-run
    # branch.  Neither the .env file nor its values are copied or serialized.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    plugin_bytes = PLUGIN_PATH.read_bytes()
    if _sha256_bytes(plugin_bytes) != preflight.plugin_sha256:
        parser.error("synthetic fixture plugin changed after preflight")

    db_path = preflight.db_path
    initial_db_sha256 = preflight.db_sha256
    initial_db_file_state = preflight.db_file_state
    initial_snapshot = preflight.repository
    base_sha = preflight.base_sha

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.expanduser().resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    preregistration = output_dir / f"{timestamp}_preregistered.md"
    journal_path = output_dir / f"{timestamp}_journal.json"
    aggregate_report = output_dir / f"{timestamp}_comparison.md"
    _write_preregistration(
        preregistration,
        base_sha=base_sha,
        db_path=db_path,
        db_sha256=initial_db_sha256,
        snapshot=initial_snapshot,
        runtime_bundle_sha256=preflight.runtime_bundle_sha256,
        plugin_sha256=preflight.plugin_sha256,
    )
    journal = _new_journal(
        base_sha=base_sha,
        db_path=db_path,
        db_sha256=initial_db_sha256,
        db_file_state=initial_db_file_state,
        snapshot=initial_snapshot,
        preregistration=preregistration,
        runtime_bundle_sha256=preflight.runtime_bundle_sha256,
        plugin_sha256=preflight.plugin_sha256,
    )
    _atomic_write_json(journal_path, journal)
    print(f"Preregistration: {preregistration}", flush=True)

    abort_reason = ""
    interrupted_signal = 0
    for case in EXPERIMENTS_20:
        entry = _entry_for(case)
        journal["experiments"].append(entry)
        _atomic_write_json(journal_path, journal)
        experiment_dir = output_dir / f"{case.index:02d}_{case.protocol}"
        experiment_dir.mkdir(parents=True, exist_ok=False)
        report_path = experiment_dir / f"{case.index:02d}_{case.protocol}_comparison.md"
        certificate_path = experiment_dir / f"{case.index:02d}_{case.protocol}_rollback.json"
        entry["report"] = str(report_path)
        entry["certificate"] = str(certificate_path)
        results: dict[str, ModeResult] = {}
        temporary_root = Path(
            tempfile.mkdtemp(
                prefix=f".runtime-{case.index:02d}-",
                dir=experiment_dir,
            )
        )
        entry["runtime_temp_root"] = str(temporary_root)
        entry["recovery_required"] = True
        _atomic_write_json(journal_path, journal)
        clone = temporary_root / "repo"
        frozen_plugin_dir = temporary_root / "plugins"
        frozen_plugin_path = frozen_plugin_dir / f"{PLUGIN_MODULE}.py"
        arm_databases = {
            "baseline": temporary_root / "baseline.db",
            "candidate": temporary_root / "candidate.db",
        }
        certificate: dict[str, Any] = {
            "schema_version": 1,
            "experiment": case.index,
            "protocol": case.protocol,
            "protocol_sha256": case.protocol_sha256,
            "base_sha": base_sha,
            "runtime_bundle_sha256": preflight.runtime_bundle_sha256,
            "plugin_sha256": preflight.plugin_sha256,
            "started_at": _utc_now(),
            "clone_clean_before": False,
            "clone_clean_after": False,
            "source_db_sha256_before": initial_db_sha256,
            "source_db_sha256_after": "unavailable",
            "source_db_file_state_before": dict(initial_db_file_state),
            "source_db_file_state_after": {},
            "arm_db_sha256_before": {},
            "arm_db_sha256_after": {},
            "clone_removed": False,
            "arm_databases_removed": False,
            "rollback_complete": False,
            "error": "",
        }
        experiment_error = ""
        previous_signal_handlers = _install_experiment_signal_handlers()
        try:
            if _sqlite_file_state(db_path) != initial_db_file_state:
                raise InfrastructureError("source SQLite changed before experiment")
            if _file_sha256(PLUGIN_PATH) != preflight.plugin_sha256:
                raise InfrastructureError("synthetic fixture plugin changed")
            if _runtime_bundle_sha256(PROJECT_ROOT) != preflight.runtime_bundle_sha256:
                raise InfrastructureError("main runtime bundle changed")
            current_snapshot = capture_repository_snapshot(PROJECT_ROOT)
            if current_snapshot != initial_snapshot:
                raise InfrastructureError("main repository integrity changed")
            _prepare_clone(
                PROJECT_ROOT,
                clone,
                base_sha,
                preflight.runtime_bundle_sha256,
            )
            certificate["clone_clean_before"] = _clone_status(clone)[1]
            frozen_plugin_dir.mkdir()
            frozen_plugin_path.write_bytes(plugin_bytes)
            if _file_sha256(frozen_plugin_path) != preflight.plugin_sha256:
                raise InfrastructureError("frozen plugin copy hash mismatch")
            for arm_db in arm_databases.values():
                shutil.copy2(db_path, arm_db)
            certificate["arm_db_sha256_before"] = {
                arm: sqlite_sha256(path) for arm, path in arm_databases.items()
            }
            certificate["arm_db_file_state_before"] = {
                arm: dict(_sqlite_file_state(path))
                for arm, path in arm_databases.items()
            }
            if any(
                state != initial_db_file_state
                for state in certificate["arm_db_file_state_before"].values()
            ):
                raise InfrastructureError("disposable SQLite copy hash mismatch")

            entry["state"] = "running"
            _atomic_write_json(journal_path, journal)
            for arm in case.arm_order:
                if _file_sha256(frozen_plugin_path) != preflight.plugin_sha256:
                    raise InfrastructureError(
                        "frozen synthetic fixture plugin changed before an arm exchange"
                    )
                before_state = _sqlite_file_state(arm_databases[arm])
                if before_state != initial_db_file_state:
                    raise InfrastructureError(
                        f"{arm} SQLite changed before its exchange"
                    )
                if _sqlite_file_state(db_path) != initial_db_file_state:
                    raise InfrastructureError(
                        "source SQLite changed before an arm exchange"
                    )
                run_label = (
                    f"{timestamp}_operation_{case.index:02d}_"
                    f"{case.protocol}_{case.order_name.lower()}_{arm}"
                )
                result = _run_arm_subprocess(
                    clone=clone,
                    python=MAIN_VENV_PYTHON,
                    plugin_dir=frozen_plugin_dir,
                    case=case,
                    arm=arm,
                    db_path=arm_databases[arm],
                    output_dir=experiment_dir,
                    run_label=run_label,
                )
                results[arm] = result
                entry["arms"][arm] = _result_summary(result, case.scenario)
                _atomic_write_json(journal_path, journal)
                after_state = _sqlite_file_state(arm_databases[arm])
                if after_state != initial_db_file_state:
                    raise InfrastructureError(
                        f"{arm} SQLite changed during its exchange"
                    )
                if _sqlite_file_state(db_path) != initial_db_file_state:
                    raise InfrastructureError(
                        "source SQLite changed during an arm exchange"
                    )
                if _file_sha256(frozen_plugin_path) != preflight.plugin_sha256:
                    raise InfrastructureError(
                        "frozen synthetic fixture plugin changed during an arm exchange"
                    )
                judge_errors = judge_infrastructure_errors(result, case.scenario)
                if judge_errors:
                    raise InfrastructureError(
                        f"{arm} judge/run infrastructure invalid: "
                        + "; ".join(judge_errors)
                    )

            certificate["arm_db_sha256_after"] = {
                arm: sqlite_sha256(path) for arm, path in arm_databases.items()
            }
            certificate["source_db_sha256_after"] = sqlite_sha256(db_path)
            certificate["source_db_file_state_after"] = dict(
                _sqlite_file_state(db_path)
            )
            certificate["arm_db_file_state_after"] = {
                arm: dict(_sqlite_file_state(path))
                for arm, path in arm_databases.items()
            }
            clone_digest, clone_clean = _clone_status(clone)
            certificate["clone_status_sha256_after"] = clone_digest
            certificate["clone_clean_after"] = clone_clean
            if not clone_clean:
                raise InfrastructureError("experiment clone is not clean after arms")
            if certificate["source_db_file_state_after"] != initial_db_file_state:
                raise InfrastructureError("source SQLite changed after experiment")
            if any(
                state != initial_db_file_state
                for state in certificate["arm_db_file_state_after"].values()
            ):
                raise InfrastructureError("disposable SQLite changed after experiment")
            if capture_repository_snapshot(PROJECT_ROOT) != initial_snapshot:
                raise InfrastructureError("main repository integrity changed")
            entry["verdict"] = _pair_verdict(
                results.get("baseline"), results.get("candidate"), case.scenario
            )
        except RunInterrupted as exc:
            interrupted_signal = exc.signum
            experiment_error = _safe_error(exc)
            entry["error"] = experiment_error
            entry["verdict"] = "inconclusive"
            abort_reason = (
                f"experiment {case.index:02d}/{case.protocol}: {experiment_error}"
            )
        except Exception as exc:
            experiment_error = _safe_error(exc)
            entry["error"] = experiment_error
            entry["verdict"] = "inconclusive"
            abort_reason = (
                f"experiment {case.index:02d}/{case.protocol}: {experiment_error}"
            )
        finally:
            if clone.is_dir() and not certificate["clone_clean_after"]:
                try:
                    clone_digest, clone_clean = _clone_status(clone)
                    certificate["clone_status_sha256_after"] = clone_digest
                    certificate["clone_clean_after"] = clone_clean
                except (InfrastructureError, OSError) as exc:
                    certificate["error"] = _safe_error(exc)
            try:
                shutil.rmtree(temporary_root)
            except OSError as exc:
                cleanup_error = f"rollback cleanup failed: {_safe_error(exc)}"
                experiment_error = experiment_error or cleanup_error
                abort_reason = abort_reason or (
                    f"experiment {case.index:02d}/{case.protocol}: {cleanup_error}"
                )
            certificate["clone_removed"] = not clone.exists()
            certificate["arm_databases_removed"] = all(
                not path.exists() for path in arm_databases.values()
            )
            certificate["rollback_complete"] = bool(
                certificate["clone_removed"]
                and certificate["arm_databases_removed"]
            )
            certificate["finished_at"] = _utc_now()
            if experiment_error and not certificate["error"]:
                certificate["error"] = experiment_error
            entry["rollback"] = (
                "complete" if certificate["rollback_complete"] else "failed"
            )
            entry["recovery_required"] = not certificate["rollback_complete"]
            if certificate["rollback_complete"]:
                entry["state"] = (
                    "aborted_rolled_back"
                    if experiment_error
                    else "completed_rolled_back"
                )
            else:
                entry["state"] = "rollback_failed"
            _restore_signal_handlers(previous_signal_handlers)
            _atomic_write_json(certificate_path, certificate)
            _write_pair_report(
                report_path,
                case=case,
                base_sha=base_sha,
                results=results,
                infrastructure_error=experiment_error,
            )
            _atomic_write_json(journal_path, journal)
        if abort_reason:
            print(f"Aborted: {abort_reason}", flush=True)
            break
        print(
            f"{case.index:02d}/20 {case.protocol}: {entry['verdict']}; "
            "rollback complete",
            flush=True,
        )

    try:
        final_db_sha256 = sqlite_sha256(db_path)
        final_db_file_state = _sqlite_file_state(db_path)
        final_snapshot = capture_repository_snapshot(PROJECT_ROOT)
        if final_db_file_state != initial_db_file_state:
            abort_reason = abort_reason or "source SQLite changed by the matrix"
        if final_snapshot != initial_snapshot:
            abort_reason = abort_reason or "main repository integrity changed by the matrix"
        if _runtime_bundle_sha256(PROJECT_ROOT) != preflight.runtime_bundle_sha256:
            abort_reason = abort_reason or "main runtime bundle changed by the matrix"
        if _file_sha256(PLUGIN_PATH) != preflight.plugin_sha256:
            abort_reason = abort_reason or "synthetic fixture plugin changed by the matrix"
    except (InfrastructureError, OSError) as exc:
        final_db_sha256 = "unavailable"
        final_db_file_state = {}
        final_snapshot = initial_snapshot
        abort_reason = abort_reason or _safe_error(exc)

    family_verdicts = evaluate_families(journal["experiments"])
    if abort_reason:
        quality_status = "inconclusive"
    elif any(verdict.status == "improved" for verdict in family_verdicts):
        quality_status = "improved"
    else:
        quality_status = "not_improved"
    journal["source_db_sha256_after"] = final_db_sha256
    journal["source_db_file_state_after"] = dict(final_db_file_state)
    journal["repository_after"] = _public_snapshot(final_snapshot)
    journal["family_verdicts"] = [asdict(verdict) for verdict in family_verdicts]
    journal["quality_status"] = quality_status
    journal["abort_reason"] = abort_reason
    journal["state"] = "aborted" if abort_reason else "complete"
    journal["finished_at"] = _utc_now()
    _atomic_write_json(journal_path, journal)
    _write_aggregate_report(
        aggregate_report,
        base_sha=base_sha,
        preregistration=preregistration,
        entries=journal["experiments"],
        family_verdicts=family_verdicts,
        abort_reason=abort_reason,
    )
    print(f"Experiment report: {aggregate_report}", flush=True)
    print(f"Sanitized journal: {journal_path}", flush=True)
    if abort_reason:
        return 128 + interrupted_signal if interrupted_signal else 2
    return 0 if quality_status == "improved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
