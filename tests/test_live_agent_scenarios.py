"""Opt-in end-to-end scenarios for the configured real chat model.

These tests intentionally execute one user request per test without mocks,
parameterization, batching, or parallel calls. Enable them explicitly with
``RUN_LIVE_AGENT_SCENARIOS=1``. Set ``LIVE_AGENT_MODE=single_agent`` to run
the same acceptance scenarios through the non-multiagent baseline; the default
is ``multiagent``.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
import warnings
from csv import DictReader
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Iterator
from uuid import uuid4

import pytest
import sqlglot

import storage.database as db_storage
from agents.run_metrics import AgentRunMetrics, consume_agent_run_metrics
from services.sql_dialects import GREENPLUM_DIALECT  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_DB_PATH = PROJECT_ROOT / "excel_data.db"
LIVE_TRANSCRIPT_PATH = os.getenv("LIVE_AGENT_TRANSCRIPT_PATH", "").strip()
LIVE_AGENT_MODE = (
    os.getenv("LIVE_AGENT_MODE", "multiagent").strip().lower()
    or "multiagent"
)
if LIVE_AGENT_MODE not in {"multiagent", "single_agent"}:
    raise ValueError(
        "LIVE_AGENT_MODE must be 'multiagent' or 'single_agent'"
    )
LIVE_AGENT_ENABLED = os.getenv("RUN_LIVE_AGENT_SCENARIOS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LIVE_AGENT_LLM_JUDGE = os.getenv(
    "LIVE_AGENT_LLM_JUDGE", ""
).strip().lower() in {"1", "true", "yes", "on"}
STRICT_RETRIEVAL_ENABLED = os.getenv(
    "S2T_NARROW_TOOLS_EXPERIMENT", ""
).strip().lower() in {"1", "true", "yes", "on"}
_LIVE_TRANSCRIPT_LOCK = threading.Lock()
_LIVE_TRANSCRIPT_INDEX = 0
_LIVE_SEMANTIC_RESULTS: list[dict[str, str]] = []

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live_agent,
    pytest.mark.skipif(
        not LIVE_AGENT_ENABLED,
        reason="set RUN_LIVE_AGENT_SCENARIOS=1 to call the configured real LLM",
    ),
]


def _assert_public_answer(answer: str) -> None:
    assert answer.strip()
    lowered = answer.lower()
    for internal_term in (
        "supervisor",
        "coordinator",
        "finish_worker",
        "result_key",
        "evidence_id",
        "display_ref",
        "dataset_ref",
        "result_id",
    ):
        _warn_unless(
            internal_term not in lowered,
            "presentation",
            f"public answer exposes internal term {internal_term!r}",
        )


def _display_payloads(result) -> list[dict]:
    payloads: list[dict] = []
    for item in result.display_items:
        try:
            payload = json.loads(item.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _payload_contains_value(payload, expected) -> bool:
    if isinstance(payload, dict):
        return any(
            _payload_contains_value(value, expected)
            for value in payload.values()
        )
    if isinstance(payload, list):
        return any(
            _payload_contains_value(value, expected)
            for value in payload
        )
    return payload == expected or str(payload) == str(expected)


def _payload_table_paths(payload: dict) -> list[list[str]]:
    paths: list[list[str]] = []
    for collection_name in ("paths", "chains", "rows"):
        collection = payload.get(collection_name)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            table_path = item.get("table_path")
            if isinstance(table_path, list):
                paths.append([str(value) for value in table_path])
    return paths


@dataclass(frozen=True)
class _LiveExchange:
    query: str
    result: object
    metrics: AgentRunMetrics
    http_elapsed_seconds: float


def _chat(
    client,
    query: str,
    *,
    history: list[dict] | None = None,
):
    from agents.chat_graph import WorkerRunResult

    session_id = f"live-agent-{uuid4()}"
    started_at = perf_counter()
    response = client.post(
        "/chat",
        json={
            "query": query,
            "history": list(history or []),
            "session_id": session_id,
        },
    )
    http_elapsed_seconds = perf_counter() - started_at
    payload = response.get_json()
    metrics = consume_agent_run_metrics(session_id)
    _record_live_exchange(
        query,
        response.status_code,
        payload,
        metrics=metrics,
        http_elapsed_seconds=http_elapsed_seconds,
    )
    assert response.status_code == 200, payload
    assert metrics is not None, "live run did not publish agent metrics"
    return _LiveExchange(
        query=query,
        result=WorkerRunResult.model_validate(payload),
        metrics=metrics,
        http_elapsed_seconds=http_elapsed_seconds,
    )


def _record_live_exchange(
    query: str,
    status_code: int,
    payload,
    *,
    metrics: AgentRunMetrics | None,
    http_elapsed_seconds: float,
) -> None:
    """Append a human-readable request/response pair for an opt-in live run."""
    global _LIVE_TRANSCRIPT_INDEX

    if isinstance(payload, dict):
        answer = payload.get("answer") or payload.get("error") or payload
        display_names = [
            str(item.get("name", ""))
            for item in payload.get("display_items", [])
            if isinstance(item, dict) and item.get("name")
        ]
    else:
        answer = payload
        display_names = []

    semantic_status = "not_evaluated"
    semantic_reason = (
        "Не выполнялась; ответ сохранён для ручного разбора и будущего "
        "LLM-as-judge."
    )
    if LIVE_AGENT_LLM_JUDGE:
        if status_code != 200:
            semantic_status = "failed"
            semantic_reason = f"Технический HTTP {status_code} не решил задачу."
        else:
            try:
                from agents.semantic_judge import judge_agent_response

                verdict = judge_agent_response(
                    query=query,
                    answer=answer,
                    display_items=(
                        payload.get("display_items", [])
                        if isinstance(payload, dict)
                        else []
                    ),
                )
                semantic_status = verdict.status
                semantic_reason = verdict.reason
            except Exception as exc:
                semantic_status = "judge_error"
                semantic_reason = (
                    "LLM-as-judge завершился технической ошибкой: "
                    f"{type(exc).__name__}."
                )
        if status_code == 200:
            _LIVE_SEMANTIC_RESULTS.append(
                {
                    "status": semantic_status,
                    "reason": semantic_reason,
                }
            )

    if not LIVE_TRANSCRIPT_PATH:
        return
    transcript_path = Path(LIVE_TRANSCRIPT_PATH)
    if not transcript_path.is_absolute():
        transcript_path = PROJECT_ROOT / transcript_path

    metrics_block = "Метрики недоступны"
    trace_block = "Трасса недоступна"
    if metrics is not None:
        stage_lines = "\n".join(
            (
                f"stage_tokens[{item.stage}]: calls={item.calls}, "
                f"errors={item.error_calls}, input={item.input_tokens}, "
                f"output={item.output_tokens}, total={item.total_tokens}, "
                f"cache_read={item.cache_read_tokens}, "
                f"seconds={item.elapsed_seconds:.3f}"
            )
            for item in metrics.llm_stages
        )
        metrics_block = (
            f"agent_seconds: {metrics.elapsed_seconds:.3f}\n"
            f"http_seconds: {http_elapsed_seconds:.3f}\n"
            f"llm_calls: {len(metrics.llm_calls)}\n"
            f"tokens: input={metrics.input_tokens}, "
            f"output={metrics.output_tokens}, total={metrics.total_tokens}, "
            f"cache_read={metrics.cache_read_tokens}\n"
            f"{stage_lines}\n"
            f"workers: {len(metrics.worker_tasks)}\n"
            "tools: "
            + (", ".join(item.name for item in metrics.tool_calls) or "Нет")
            + "\n"
            "displays: "
            + (", ".join(metrics.display_tools) or "Нет")
        )
        trace_block = "```json\n" + json.dumps(
            {
                "llm_calls": [
                    item.model_dump(mode="json")
                    for item in metrics.llm_calls
                ],
                "llm_stages": [
                    item.model_dump(mode="json")
                    for item in metrics.llm_stages
                ],
                "tool_calls": [
                    {
                        key: value
                        for key, value in item.model_dump(mode="json").items()
                        if key != "input_preview"
                    }
                    for item in metrics.tool_calls
                ],
                "coordinator_plan": metrics.coordinator_plan,
                "worker_tasks": metrics.worker_tasks,
                "worker_routes": [
                    item.model_dump(mode="json")
                    for item in metrics.worker_routes
                ],
                "observations": [
                    item.model_dump(mode="json")
                    for item in metrics.observations
                ],
                "upstream_output": metrics.upstream_output,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n```"

    with _LIVE_TRANSCRIPT_LOCK:
        _LIVE_TRANSCRIPT_INDEX += 1
        block = (
            f"## {_LIVE_TRANSCRIPT_INDEX}. Запрос\n\n"
            f"agent_mode: {LIVE_AGENT_MODE}\n\n"
            f"{query}\n\n"
            f"### Ответ — HTTP {status_code}\n\n"
            f"{answer}\n\n"
            "### Display-results\n\n"
            f"{', '.join(display_names) if display_names else 'Нет'}\n\n"
            "### Execution metrics\n\n"
            f"{metrics_block}\n\n"
            "### Agent trace\n\n"
            f"{trace_block}\n\n"
            "### Semantic evaluation\n\n"
            f"{semantic_reason}\n\n"
            "<!-- LIVE_SEMANTIC "
            + json.dumps(
                {
                    "scenario": _current_live_scenario(),
                    "status": semantic_status,
                    "reason": semantic_reason,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + " -->\n\n"
        )
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with transcript_path.open("a", encoding="utf-8", newline="\n") as transcript:
            transcript.write(block)


def _assert_minimal_name_sequence(
    actual: list[str],
    expected: list[str | set[str]],
) -> None:
    cursor = 0
    for index, expected_name in enumerate(expected, start=1):
        accepted_names = (
            expected_name
            if isinstance(expected_name, set)
            else {expected_name}
        )
        while cursor < len(actual) and actual[cursor] not in accepted_names:
            cursor += 1
        assert cursor < len(actual), {
            "position": index,
            "actual": actual,
            "accepted": sorted(accepted_names),
        }
        cursor += 1


def _current_live_scenario() -> str:
    current_test = os.getenv("PYTEST_CURRENT_TEST", "").split(" ", 1)[0]
    return current_test.rsplit("::", 1)[-1] or "unknown"


def _record_acceptance_warning(category: str, message: str) -> None:
    clean_category = str(category or "warning").strip().lower()
    clean_message = str(message or "").strip()
    payload = {
        "category": clean_category,
        "scenario": _current_live_scenario(),
        "message": clean_message,
    }
    warnings.warn(
        f"live {clean_category} warning: {clean_message}",
        UserWarning,
        stacklevel=2,
    )
    if not LIVE_TRANSCRIPT_PATH:
        return
    transcript_path = Path(LIVE_TRANSCRIPT_PATH)
    if not transcript_path.is_absolute():
        transcript_path = PROJECT_ROOT / transcript_path
    marker = (
        "<!-- LIVE_WARNING "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + " -->\n"
    )
    with _LIVE_TRANSCRIPT_LOCK:
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with transcript_path.open("a", encoding="utf-8", newline="\n") as transcript:
            transcript.write(marker)


def _warn_unless(condition: bool, category: str, message: str) -> bool:
    if condition:
        return True
    _record_acceptance_warning(category, message)
    return False


def _sequence_matches_exactly(
    actual: list[str],
    expected: list[str | set[str]],
) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        actual_name in (
            expected_name if isinstance(expected_name, set) else {expected_name}
        )
        for actual_name, expected_name in zip(actual, expected)
    )


def _assert_execution(
    exchange: _LiveExchange,
    *,
    expected_tools: list[str | set[str]],
    expected_displays: list[str | set[str]],
    forbidden_tools: set[str] | None = None,
    max_seconds: float | None,
    max_llm_calls: int,
    max_total_tokens: int,
) -> None:
    metrics = exchange.metrics
    assert metrics.error is None, metrics.error
    assert metrics.elapsed_seconds > 0, metrics
    assert exchange.http_elapsed_seconds > 0, metrics
    if max_seconds is not None:
        _warn_unless(
            metrics.elapsed_seconds <= max_seconds
            and exchange.http_elapsed_seconds <= max_seconds + 5,
            "efficiency",
            f"elapsed={metrics.elapsed_seconds:.3f}s exceeds budget={max_seconds}s",
        )
    actual_tools = [item.name for item in metrics.tool_calls]
    tools_match = True
    try:
        _assert_minimal_name_sequence(actual_tools, expected_tools)
    except AssertionError:
        tools_match = False
    _warn_unless(
        tools_match,
        "efficiency",
        f"tool route differs: actual={actual_tools}, expected={expected_tools}",
    )
    forbidden_used = sorted(set(actual_tools) & set(forbidden_tools or ()))
    _warn_unless(
        not forbidden_used,
        "efficiency",
        f"explicitly excluded tools were used: {forbidden_used}",
    )
    _warn_unless(
        _sequence_matches_exactly(actual_tools, expected_tools),
        "efficiency",
        f"tool calls include extras: actual={actual_tools}, expected={expected_tools}",
    )
    if LIVE_AGENT_MODE == "multiagent":
        actual_worker_count = len(metrics.worker_tasks)
        _warn_unless(
            len(metrics.coordinator_plan) == actual_worker_count,
            "efficiency",
            "coordinator plan and executed worker counts differ: "
            f"plan={len(metrics.coordinator_plan)}, workers={actual_worker_count}",
        )
    else:
        _warn_unless(
            metrics.worker_tasks == [] and metrics.coordinator_plan == [],
            "efficiency",
            "single-agent run unexpectedly contains coordinator activity",
        )
    displays_match = True
    try:
        _assert_minimal_name_sequence(metrics.display_tools, expected_displays)
    except AssertionError:
        displays_match = False
    _warn_unless(
        displays_match
        and _sequence_matches_exactly(metrics.display_tools, expected_displays),
        "presentation",
        "display results differ: "
        f"actual={metrics.display_tools}, expected={expected_displays}",
    )
    assert len(metrics.llm_calls) > 0, metrics.llm_calls
    _warn_unless(
        len(metrics.llm_calls) <= max_llm_calls,
        "efficiency",
        f"llm_calls={len(metrics.llm_calls)} exceeds budget={max_llm_calls}",
    )
    assert all(item.total_tokens > 0 for item in metrics.llm_calls), metrics.llm_calls
    assert metrics.input_tokens > 0
    assert metrics.output_tokens > 0
    assert metrics.total_tokens == metrics.input_tokens + metrics.output_tokens
    _warn_unless(
        metrics.total_tokens <= max_total_tokens,
        "efficiency",
        f"total_tokens={metrics.total_tokens} exceeds budget={max_total_tokens}",
    )


class _LiveHttpResponse:
    def __init__(self, status_code: int, data: bytes, headers) -> None:
        self.status_code = status_code
        self.data = data
        self.headers = headers

    def get_json(self):
        return json.loads(self.data.decode("utf-8"))


class _LiveHttpClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, payload: dict | None = None):
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return _LiveHttpResponse(
                    response.status,
                    response.read(),
                    response.headers,
                )
        except urllib.error.HTTPError as exc:
            return _LiveHttpResponse(exc.code, exc.read(), exc.headers)

    def post(self, path: str, *, json: dict):
        return self._request("POST", path, json)

    def get(self, path: str):
        return self._request("GET", path)


@pytest.fixture(autouse=True)
def validate_llm_judge_verdict() -> Iterator[None]:
    """Fail the scenario after its normal checks when semantic judge rejects it."""
    _LIVE_SEMANTIC_RESULTS.clear()
    yield
    if not LIVE_AGENT_LLM_JUDGE:
        return
    failures = [
        item
        for item in _LIVE_SEMANTIC_RESULTS
        if item.get("status") != "passed"
    ]
    if failures:
        details = "; ".join(
            f"{item.get('status')}: {item.get('reason')}" for item in failures
        )
        pytest.fail("LLM-as-judge отклонил пользовательский результат: " + details)


@pytest.fixture
def live_workspace_db(monkeypatch) -> Iterator[Path]:
    if not LIVE_DB_PATH.is_file():
        pytest.skip("workspace excel_data.db is absent")
    monkeypatch.setattr(db_storage, "DB_PATH", str(LIVE_DB_PATH))
    yield LIVE_DB_PATH


@pytest.fixture
def live_chat_client(live_workspace_db):
    from app import app as flask_app
    from werkzeug.serving import make_server

    previous_testing = flask_app.config.get("TESTING", False)
    previous_agent_mode = flask_app.config.get("CHAT_AGENT_MODE", "multiagent")
    flask_app.config["TESTING"] = False
    flask_app.config["CHAT_AGENT_MODE"] = LIVE_AGENT_MODE
    server = make_server("127.0.0.1", 0, flask_app, threaded=False)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        yield _LiveHttpClient(f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=10)
        flask_app.config["TESTING"] = previous_testing
        flask_app.config["CHAT_AGENT_MODE"] = previous_agent_mode


@pytest.fixture(autouse=True)
def generated_sql_exports() -> Iterator[set[Path]]:
    from agents.tools.sql import SQL_EXPORT_DIR

    before = set(SQL_EXPORT_DIR.glob("sql_result_*.csv"))
    registered: set[Path] = set()
    yield registered
    export_root = SQL_EXPORT_DIR.resolve()
    created = set(SQL_EXPORT_DIR.glob("sql_result_*.csv")) - before
    for path in registered | created:
        resolved = path.resolve()
        if resolved.parent == export_root and resolved.name.startswith("sql_result_"):
            resolved.unlink(missing_ok=True)


def _download_sql_export(client, payload: dict, generated: set[Path]) -> list[dict]:
    from agents.tools.sql import SQL_EXPORT_DIR

    path = Path(payload["csv_path"]).resolve()
    assert path.parent == SQL_EXPORT_DIR.resolve(), path
    assert path.is_file(), path
    generated.add(path)
    response = client.get(payload["csv_url"])
    assert response.status_code == 200
    assert "attachment" in response.headers.get("Content-Disposition", "")
    text = response.data.decode("utf-8-sig")
    return list(DictReader(StringIO(text)))


def _fetch_one(query: str, parameters: tuple[object, ...] = ()) -> tuple:
    conn = db_storage.get_db_connection()
    try:
        row = conn.execute(query, parameters).fetchone()
    finally:
        conn.close()
    if row is None:
        pytest.skip("workspace database has no row required by the scenario")
    return tuple(row)


def _s2t_work_case_fixture() -> tuple[int, str, str, str, str]:
    row = _fetch_one(
        """
        SELECT s2t.file_id, s2t.target_table, s2t.source_table,
               s2t.target_field, s2t.source_field
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
          AND s2t.target_table IS NOT NULL AND TRIM(s2t.target_table) <> ''
          AND s2t.source_table IS NOT NULL AND TRIM(s2t.source_table) <> ''
          AND s2t.target_field IS NOT NULL AND TRIM(s2t.target_field) <> ''
          AND s2t.source_field IS NOT NULL AND TRIM(s2t.source_field) <> ''
          AND s2t.transformation_rule IS NOT NULL
          AND source_catalog.not_null = 0
          AND target_catalog.not_null = 1
          AND source_catalog.data_type IS NOT NULL
          AND TRIM(source_catalog.data_type) <> ''
          AND target_catalog.data_type IS NOT NULL
          AND TRIM(target_catalog.data_type) <> ''
          AND LOWER(s2t.transformation_rule) LIKE '%join%'
          AND LOWER(s2t.transformation_rule) LIKE '%where%'
        ORDER BY LENGTH(s2t.transformation_rule), s2t.id
        LIMIT 1
        """
    )
    return (
        int(row[0]),
        str(row[1]),
        str(row[2]),
        str(row[3]),
        str(row[4]),
    )


def _multi_source_validation_case_fixture() -> tuple[int, str, str, str]:
    row = _fetch_one(
        """
        SELECT 2, 't_agr_dep',
               'b3050000420007_product',
               'b3050000420004_nsoadditionalinfo'
        WHERE (
            SELECT COUNT(DISTINCT LOWER(TRIM(source_table)))
            FROM s2t_transformations
            WHERE LOWER(TRIM(target_table)) = 't_agr_dep'
              AND LOWER(TRIM(source_table)) IN (
                  'b3050000420007_product',
                  'b3050000420004_nsoadditionalinfo'
              )
              AND COALESCE(TRIM(transformation_rule), '') <> ''
        ) = 2
          AND EXISTS (
              SELECT 1
              FROM target_columns
              WHERE file_id = 2
                AND LOWER(TRIM(table_name)) = 't_agr_dep'
                AND primary_key = 1
                AND not_null = 1
          )
        """
    )
    return int(row[0]), str(row[1]), str(row[2]), str(row[3])


def _multi_target_validation_case_fixture() -> tuple[int, str, str, str]:
    row = _fetch_one(
        """
        SELECT 3, 'b3050000420005_paymentdetails', 't_optn', 't_optn_type'
        WHERE (
            SELECT COUNT(DISTINCT LOWER(TRIM(target_table)))
            FROM s2t_transformations
            WHERE LOWER(TRIM(source_table)) =
                  'b3050000420005_paymentdetails'
              AND LOWER(TRIM(target_table)) IN ('t_optn', 't_optn_type')
              AND COALESCE(TRIM(transformation_rule), '') <> ''
        ) = 2
          AND (
              SELECT COUNT(DISTINCT LOWER(TRIM(table_name)))
              FROM target_columns
              WHERE file_id = 3
                AND LOWER(TRIM(table_name)) IN ('t_optn', 't_optn_type')
          ) = 2
        """
    )
    return int(row[0]), str(row[1]), str(row[2]), str(row[3])


def _independent_protocol_case_fixture() -> tuple[int, str, str]:
    row = _fetch_one(
        """
        SELECT 3, 'b3050000420007_product', 't_crncy'
        WHERE (
            SELECT COUNT(DISTINCT file_id)
            FROM s2t_transformations
            WHERE LOWER(TRIM(source_table)) = 'b3050000420007_product'
              AND LOWER(TRIM(target_table)) = 't_crncy'
        ) = 1
          AND (
            SELECT MIN(file_id)
            FROM s2t_transformations
            WHERE LOWER(TRIM(source_table)) = 'b3050000420007_product'
              AND LOWER(TRIM(target_table)) = 't_crncy'
        ) = 3
          AND (
            SELECT COUNT(DISTINCT TRIM(transformation_rule))
            FROM s2t_transformations
            WHERE LOWER(TRIM(source_table)) = 'b3050000420007_product'
              AND LOWER(TRIM(target_table)) = 't_crncy'
              AND COALESCE(TRIM(transformation_rule), '') <> ''
              AND (
                  LOWER(LTRIM(transformation_rule)) LIKE 'select%'
                  OR LOWER(LTRIM(transformation_rule)) LIKE 'with%'
              )
        ) = 1
          AND (
            SELECT COUNT(DISTINCT file_id)
            FROM s2t_transformations
            WHERE LOWER(TRIM(target_table)) = 't_crncy'
        ) = 1
          AND EXISTS (
              SELECT 1
              FROM target_columns
              WHERE file_id = 3
                AND LOWER(TRIM(table_name)) = 't_crncy'
                AND primary_key = 1
                AND not_null = 1
          )
        """
    )
    return int(row[0]), str(row[1]), str(row[2])


def _assert_s2t_work_case_execution(
    exchange: _LiveExchange,
    *,
    required_tools: set[str] | None = None,
    require_analysis: bool = False,
    max_seconds: float = 240,
    max_llm_calls: int = 100,
    max_total_tokens: int = 180_000,
) -> None:
    metrics = exchange.metrics
    allowed_tools = {
        "get_excel_row",
        "get_source_target_column_pair",
        "list_additional_objects",
        "list_column_catalog",
        "list_column_metadata",
        "list_columns",
        "list_file_sheet_headers",
        "list_source_column_catalog",
        "list_target_column_catalog",
        "list_s2t_field_mapping",
        "list_s2t_source_field",
        "list_s2t_source_table",
        "list_s2t_table_mapping",
        "list_s2t_occurrences",
        "list_s2t_target_field",
        "list_s2t_target_table",
        "list_s2t_transformations",
        "parse_sql_column_lineage",
        "parse_sql_table_lineage",
        "read_s2t_by_source_table",
        "read_s2t_by_target_table",
        "read_s2t_mapping",
        "read_s2t_source_to_target",
        "query_saved_result",
        "read_previous_result",
        "run_sql",
        "search_column_catalog",
        "search_additional_objects",
        "search_excel_values",
        "search_s2t_transformations",
        "show_plan",
        "trace_transformation_path",
    }
    tool_names = [item.name for item in metrics.tool_calls]

    assert metrics.error is None, metrics.error
    assert metrics.elapsed_seconds > 0, metrics
    assert exchange.http_elapsed_seconds > 0, metrics
    _warn_unless(
        metrics.elapsed_seconds <= max_seconds
        and exchange.http_elapsed_seconds <= max_seconds + 5,
        "efficiency",
        f"elapsed={metrics.elapsed_seconds:.3f}s exceeds budget={max_seconds}s",
    )
    _warn_unless(
        bool(tool_names),
        "efficiency",
        "scenario returned without inspecting stored data",
    )
    _warn_unless(
        set(tool_names) <= allowed_tools,
        "efficiency",
        "scenario used additional read-only tools: "
        f"{sorted(set(tool_names) - allowed_tools)}",
    )
    _warn_unless(
        set(required_tools or ()) <= set(tool_names),
        "efficiency",
        "preferred data tools were not used: "
        f"actual={tool_names}, expected={sorted(required_tools or ())}",
    )
    if require_analysis and LIVE_AGENT_MODE == "multiagent":
        _warn_unless(
            metrics.upstream_output is not None
            and str(metrics.upstream_output.get("answer") or "").strip(),
            "efficiency",
            "scenario completed without a recorded upstream analysis result",
        )
    _warn_unless(
        len(tool_names) <= 12,
        "efficiency",
        f"tool_calls={len(tool_names)} exceeds budget=12: {tool_names}",
    )
    if LIVE_AGENT_MODE == "multiagent":
        direct_pipeline = any(
            str(step.get("pipeline") or "") == "validation_protocol"
            for step in metrics.coordinator_plan
        )
        if direct_pipeline:
            _warn_unless(
                bool(metrics.coordinator_plan) and not metrics.worker_tasks,
                "efficiency",
                "direct coordinator pipeline trace is inconsistent",
            )
        else:
            _warn_unless(
                bool(metrics.coordinator_plan)
                and 0 < len(metrics.worker_tasks) <= len(metrics.coordinator_plan),
                "efficiency",
                "coordinator/worker trace is incomplete",
            )
    else:
        _warn_unless(
            metrics.worker_tasks == [] and metrics.coordinator_plan == [],
            "efficiency",
            "single-agent run unexpectedly contains coordinator activity",
        )
    assert len(metrics.llm_calls) > 0, metrics.llm_calls
    _warn_unless(
        len(metrics.llm_calls) <= max_llm_calls,
        "efficiency",
        f"llm_calls={len(metrics.llm_calls)} exceeds budget={max_llm_calls}",
    )
    assert all(item.total_tokens > 0 for item in metrics.llm_calls), metrics.llm_calls
    assert metrics.input_tokens > 0
    assert metrics.output_tokens > 0
    assert metrics.total_tokens == metrics.input_tokens + metrics.output_tokens
    _warn_unless(
        metrics.total_tokens <= max_total_tokens,
        "efficiency",
        f"total_tokens={metrics.total_tokens} exceeds budget={max_total_tokens}",
    )


def _assert_compiled_test_protocol(
    exchange: _LiveExchange,
    *target_tables: str,
    source_tables: tuple[str, ...] = (),
    expected_pair_reads: int,
) -> None:
    answer = exchange.result.answer
    folded = answer.casefold()
    check_titles = (
        "проверка количества строк",
        "проверка уникальности ключа",
        "проверка null-rate обязательных полей",
        "проверка корректности трансформаций",
    )
    for required in (
        *check_titles,
        "цель:",
        "sql-шаблон:",
        "критерий прохождения:",
        "```sql",
        "фактические метрики не вычислялись",
    ):
        assert required in folded, answer
    expected_protocol_count = len(target_tables)
    for title in check_titles:
        assert folded.count(title) == expected_protocol_count, answer
    assert folded.count("sql-шаблон:") == 4 * expected_protocol_count, answer
    assert folded.count("```sql") == 4 * expected_protocol_count, answer
    assert "sql-шаблон не сформирован" not in folded, answer
    sql_blocks = re.findall(r"```sql\n(.*?)\n```", answer, flags=re.DOTALL)
    assert len(sql_blocks) == 4 * expected_protocol_count, answer
    for sql_template in sql_blocks:
        parseable = sql_template.replace("{{LOAD_SCOPE_PREDICATE}}", "TRUE")
        parseable = re.sub(
            r"(?<![\w$])\$\$([A-Za-z0-9_]+)(?=\.)",
            lambda match: f'"$${match.group(1)}"',
            parseable,
        )
        statements = sqlglot.parse(parseable, read=GREENPLUM_DIALECT)
        assert len(statements) == 1, sql_template
    for source_table in source_tables:
        assert source_table.casefold() in folded, answer
    for target_table in target_tables:
        assert target_table.casefold() in folded, answer
    metrics = exchange.metrics
    assert any(
        str(step.get("pipeline") or "") == "validation_protocol"
        for step in metrics.coordinator_plan
    ), metrics.coordinator_plan
    assert not metrics.worker_tasks, metrics.worker_tasks
    tool_names = [item.name for item in metrics.tool_calls]
    assert tool_names.count("read_s2t_source_to_target") == expected_pair_reads
    assert tool_names.count("read_s2t_by_target_table") == len(target_tables)
    assert tool_names.count("list_target_column_catalog") == len(target_tables)
    display_names = [item.name for item in exchange.result.display_items]
    assert display_names.count("read_s2t_source_to_target") == len(target_tables)
    assert display_names.count("list_target_column_catalog") == len(target_tables)
    assert len(_display_payloads(exchange.result)) == 2 * len(target_tables)


def test_live_agent_answers_simple_conversation_without_display_results(
    live_chat_client,
):
    exchange = _chat(live_chat_client, "Ответь одним словом: привет")
    result = exchange.result

    _assert_public_answer(result.answer)
    _warn_unless(
        result.display_items == [],
        "presentation",
        "simple conversation returned unexpected display items",
    )
    _assert_execution(
        exchange,
        expected_tools=[],
        expected_displays=[],
        max_seconds=None,
        max_llm_calls=3,
        max_total_tokens=15_000,
    )


def test_live_agent_returns_exact_global_sqlite_count(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Через SQLite посчитай точное число строк в s2t_transformations. "
        "Нужен только итоговый count."
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    _warn_unless(
        len(result.answer) <= 220,
        "presentation",
        f"count-only answer is too verbose: {len(result.answer)} chars",
    )
    _warn_unless(
        result.display_items == [],
        "presentation",
        "count-only response returned unexpected display items",
    )
    _assert_execution(
        exchange,
        expected_tools=[{"run_sql", "list_s2t_transformations"}],
        expected_displays=[],
        max_seconds=90,
        max_llm_calls=12,
        max_total_tokens=60_000,
    )


def test_live_agent_resolves_history_reference_into_task(
    live_chat_client,
):
    history = [
        {
            "role": "user",
            "content": "Речь о физической SQLite-таблице s2t_transformations.",
        },
        {
            "role": "assistant",
            "content": "Таблица s2t_transformations зафиксирована.",
        },
    ]
    exchange = _chat(
        live_chat_client,
        "Через SQLite посчитай в ней точное количество строк. Только число.",
        history=history,
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    _warn_unless(
        result.display_items == [],
        "presentation",
        "history-resolution response returned unexpected display items",
    )
    _assert_execution(
        exchange,
        expected_tools=[{"run_sql", "list_s2t_transformations"}],
        expected_displays=[],
        max_seconds=90,
        max_llm_calls=12,
        max_total_tokens=60_000,
    )


def test_live_agent_selects_full_sql_result_for_scrollable_ui(
    live_chat_client,
    generated_sql_exports,
):
    conn = db_storage.get_db_connection()
    try:
        expected_rows = [
            {"file_id": row[0], "filename": row[1]}
            for row in conn.execute(
                "SELECT file_id, filename FROM files ORDER BY file_id"
            ).fetchall()
        ]
    finally:
        conn.close()
    if not expected_rows:
        pytest.skip("workspace files table is empty")

    exchange = _chat(
        live_chat_client,
        "Через SQLite выполни SELECT file_id, filename FROM files "
        "ORDER BY file_id и покажи полный результат отдельно в scrollable UI."
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    _warn_unless(
        len(result.answer) <= 500,
        "presentation",
        f"SQL summary is too verbose: {len(result.answer)} chars",
    )
    _warn_unless(
        bool(result.display_items),
        "presentation",
        "full SQL result was not selected for display",
    )
    matching_payloads = [
        payload
        for payload in _display_payloads(result)
        if payload.get("preview_rows") == expected_rows
        or payload.get("rows") == expected_rows
    ]
    _warn_unless(
        bool(matching_payloads),
        "presentation",
        "display payload does not contain the complete SQL rows",
    )
    payload = matching_payloads[0] if matching_payloads else None
    if payload and payload.get("csv_url"):
        downloaded_rows = _download_sql_export(
            live_chat_client,
            payload,
            generated_sql_exports,
        )
        _warn_unless(
            downloaded_rows
            == [
                {key: str(value) for key, value in row.items()}
                for row in expected_rows
            ],
            "presentation",
            "downloaded display export differs from the complete SQL rows",
        )
    _assert_execution(
        exchange,
        expected_tools=["run_sql"],
        expected_displays=["run_sql"],
        max_seconds=90,
        max_llm_calls=12,
        max_total_tokens=60_000,
    )


def test_live_agent_runs_dependent_workers_sequentially(
    live_chat_client,
):
    target_table = _fetch_one(
        """
        SELECT target_table
        FROM s2t_transformations
        WHERE target_table IS NOT NULL AND TRIM(target_table) <> ''
        GROUP BY target_table
        ORDER BY COUNT(*) DESC, target_table
        LIMIT 1
        """
    )[0]
    source_count = int(
        _fetch_one(
            """
            SELECT COUNT(DISTINCT source_table)
            FROM s2t_transformations
            WHERE target_table = ?
              AND source_table IS NOT NULL
              AND TRIM(source_table) <> ''
            """,
            (target_table,),
        )[0]
    )

    exchange = _chat(
        live_chat_client,
        "Через SQLite сначала найди target_table с максимальным числом строк "
        "в s2t_transformations. Затем отдельным зависимым шагом для найденной "
        "target_table посчитай точное число различных непустых source_table. "
        "Верни имя target_table, число её строк и число source_table. "
        "Полный результат второго шага покажи отдельно.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    payloads = _display_payloads(result)
    _warn_unless(
        any(
            _payload_contains_value(payload, source_count)
            for payload in payloads
        ),
        "presentation",
        "dependent-step display does not contain the source_table count",
    )
    _assert_execution(
        exchange,
        expected_tools=["run_sql", "run_sql"],
        expected_displays=["run_sql"],
        max_seconds=150,
        max_llm_calls=20,
        max_total_tokens=120_000,
    )


def _two_hop_neo4j_path() -> list[str]:
    from graph_storage import execute_neo4j_read

    try:
        rows = execute_neo4j_read(
            """
            MATCH path=(source:ETLProjection:ETLTable)
                  -[:TABLE_TRANSFORMS_TO*2]->
                  (target:ETLProjection:ETLTable)
            WITH [node IN nodes(path) | node.name] AS names
            WHERE all(name IN names WHERE name IS NOT NULL AND trim(name) <> '')
            RETURN names
            LIMIT 1
            """,
            {},
        )
    except Exception as exc:
        pytest.skip(f"Neo4j is unavailable: {type(exc).__name__}")
    if not rows:
        pytest.skip("Neo4j has no two-hop ETLTable path")
    names = [str(value) for value in rows[0].get("names") or []]
    if len(names) != 3:
        pytest.skip("Neo4j path fixture did not return exactly three nodes")
    return names


def _neo4j_path(edge_count: int) -> list[str]:
    from graph_storage import execute_neo4j_read

    try:
        rows = execute_neo4j_read(
            f"""
            MATCH path=(source:ETLProjection:ETLTable)
                  -[:TABLE_TRANSFORMS_TO*{edge_count}]->
                  (target:ETLProjection:ETLTable)
            WITH DISTINCT [node IN nodes(path) | node.name] AS names
            WHERE all(name IN names WHERE name IS NOT NULL AND trim(name) <> '')
            RETURN names
            LIMIT 1
            """,
            {},
        )
    except Exception as exc:
        pytest.skip(f"Neo4j is unavailable: {type(exc).__name__}")
    if not rows:
        pytest.skip(f"Neo4j has no {edge_count}-edge ETLTable path")
    names = [str(value) for value in rows[0].get("names") or []]
    if len(names) != edge_count + 1:
        pytest.skip("Neo4j path fixture returned an unexpected node count")
    return names


def _neo4j_paths_between(
    source: str,
    target: str,
    edge_count: int,
) -> list[list[str]]:
    from graph_storage import execute_neo4j_read

    rows = execute_neo4j_read(
        f"""
        MATCH path=(source:ETLProjection:ETLTable {{name: $source}})
              -[:TABLE_TRANSFORMS_TO*{edge_count}]->
              (target:ETLProjection:ETLTable {{name: $target}})
        WITH DISTINCT [node IN nodes(path) | node.name] AS names
        WHERE all(name IN names WHERE name IS NOT NULL AND trim(name) <> '')
        RETURN names
        ORDER BY names
        LIMIT 20
        """,
        {"source": source, "target": target},
    )
    paths = [
        [str(value) for value in row.get("names") or []]
        for row in rows
    ]
    return [path for path in paths if len(path) == edge_count + 1]


def test_live_agent_returns_exact_neo4j_path_and_full_result(live_chat_client):
    source, middle, target = _two_hop_neo4j_path()
    exchange = _chat(
        live_chat_client,
        f"Через Neo4j найди точный путь длины 2 от таблицы {source} "
        f"до таблицы {target}. Не используй SQLite. Покажи только все узлы "
        "по порядку и глубину, а полный результат инструмента — отдельно."
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    _warn_unless(
        "sqlite" not in result.answer.lower(),
        "presentation",
        "Neo4j-only answer mentions SQLite",
    )
    display_contents = [item.content for item in result.display_items]
    _warn_unless(
        any(
            all(table_name in content for table_name in (source, middle, target))
            for content in display_contents
        ),
        "presentation",
        "Neo4j display does not contain the complete two-edge path",
    )
    _assert_execution(
        exchange,
        expected_tools=[{"run_cypher", "trace_neo4j_table_path"}],
        expected_displays=[{"run_cypher", "trace_neo4j_table_path"}],
        forbidden_tools={"run_sql"},
        max_seconds=150,
        max_llm_calls=12,
        max_total_tokens=80_000,
    )


def test_live_agent_returns_complete_three_edge_neo4j_path(live_chat_client):
    fixture_path = _neo4j_path(3)
    source, target = fixture_path[0], fixture_path[-1]
    expected_paths = _neo4j_paths_between(source, target, 3)
    if not expected_paths:
        pytest.skip("Neo4j has no stable three-edge path for the selected endpoints")
    exchange = _chat(
        live_chat_client,
        f"Через Neo4j найди полный точный направленный путь длины 3 от таблицы "
        f"{source} до таблицы {target}. Не используй SQLite. В ответе покажи "
        "только все четыре узла по порядку и глубину. Полный результат со всеми "
        "шагами пути покажи отдельно в scrollable UI."
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    lowered = result.answer.lower()
    _warn_unless(
        "sqlite" not in lowered,
        "presentation",
        "Neo4j-only answer mentions SQLite",
    )
    _warn_unless(
        "трансформац" not in lowered and "mapping" not in lowered,
        "presentation",
        "path-only answer includes transformation commentary",
    )
    matching_paths = expected_paths

    path_payloads = _display_payloads(result)
    _warn_unless(
        any(
            any(
                expected_path in _payload_table_paths(payload)
                for expected_path in matching_paths
            )
            and _payload_contains_value(payload, 3)
            for payload in path_payloads
        ),
        "presentation",
        "Neo4j display does not contain a complete three-edge path",
    )
    _assert_execution(
        exchange,
        expected_tools=[{"run_cypher", "trace_neo4j_table_path"}],
        expected_displays=[{"run_cypher", "trace_neo4j_table_path"}],
        forbidden_tools={"run_sql"},
        max_seconds=180,
        max_llm_calls=12,
        max_total_tokens=100_000,
    )


def test_live_agent_preserves_exact_s2t_pairs_in_answer_and_full_result(
    live_chat_client,
    generated_sql_exports,
):
    query = (
        "SELECT source_table, source_field, target_table, target_field "
        "FROM s2t_transformations "
        "WHERE source_table IS NOT NULL AND TRIM(source_table) <> '' "
        "AND source_field IS NOT NULL AND TRIM(source_field) <> '' "
        "AND target_table IS NOT NULL AND TRIM(target_table) <> '' "
        "AND target_field IS NOT NULL AND TRIM(target_field) <> '' "
        "ORDER BY id LIMIT 4"
    )
    conn = db_storage.get_db_connection()
    try:
        expected_rows = [dict(row) for row in conn.execute(query).fetchall()]
    finally:
        conn.close()
    if len(expected_rows) != 4:
        pytest.skip("workspace database has fewer than four complete S2T pairs")

    exchange = _chat(
        live_chat_client,
        f"Через SQLite выполни ровно этот read-only запрос: {query}. "
        "Перечисли все 4 точные пары source_table.source_field -> "
        "target_table.target_field, не разделяя связанные стороны на отдельные "
        "списки. Полный табличный результат покажи отдельно в scrollable UI."
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    matching_payloads = [
        payload
        for payload in _display_payloads(result)
        if payload.get("preview_rows") == expected_rows
        or payload.get("rows") == expected_rows
    ]
    _warn_unless(
        bool(matching_payloads),
        "presentation",
        "display payload does not preserve all four S2T pairs",
    )
    payload = matching_payloads[0] if matching_payloads else None
    if payload and payload.get("csv_url"):
        _warn_unless(
            payload.get("preview_rows") == expected_rows,
            "presentation",
            "CSV display preview differs from the requested S2T pairs",
        )
        downloaded_rows = _download_sql_export(
            live_chat_client,
            payload,
            generated_sql_exports,
        )
        _warn_unless(
            downloaded_rows
            == [
                {key: str(value) for key, value in row.items()}
                for row in expected_rows
            ],
            "presentation",
            "downloaded CSV differs from the requested S2T pairs",
        )
    elif payload:
        _warn_unless(
            payload.get("rows") == expected_rows
            and payload.get("returned_rows") == len(expected_rows)
            and payload.get("truncated") is False,
            "presentation",
            "inline display is incomplete or differs from the requested S2T pairs",
        )
    _assert_execution(
        exchange,
        expected_tools=["run_sql"],
        expected_displays=["run_sql"],
        max_seconds=150,
        max_llm_calls=20,
        max_total_tokens=100_000,
    )


def test_live_agent_runs_three_dependent_sqlite_workers(
    live_chat_client,
):
    target_table = _fetch_one(
        """
        SELECT target_table
        FROM s2t_transformations
        WHERE target_table IS NOT NULL AND TRIM(target_table) <> ''
        GROUP BY target_table
        ORDER BY COUNT(*) DESC, target_table
        LIMIT 1
        """
    )[0]
    top_source, top_source_count = _fetch_one(
        """
        SELECT source_table, COUNT(*) AS row_count
        FROM s2t_transformations
        WHERE target_table = ?
          AND source_table IS NOT NULL
          AND TRIM(source_table) <> ''
        GROUP BY source_table
        ORDER BY row_count DESC, source_table
        LIMIT 1
        """,
        (target_table,),
    )

    exchange = _chat(
        live_chat_client,
        "Через SQLite составь сводку для target_table с наибольшим числом строк "
        "в s2t_transformations: имя и число строк target_table, число различных "
        "непустых source_table, а также самый частый source_table и число его "
        "строк. При равенстве выбери лексикографически первый source_table. "
        "Полную сводку покажи отдельно.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    payloads = _display_payloads(result)
    _warn_unless(
        any(
            _payload_contains_value(payload, top_source)
            and _payload_contains_value(payload, int(top_source_count))
            for payload in payloads
        ),
        "presentation",
        "third dependent-step display does not contain the leading source",
    )
    _assert_execution(
        exchange,
        expected_tools=["run_sql"],
        expected_displays=["run_sql"],
        max_seconds=180,
        max_llm_calls=28,
        max_total_tokens=160_000,
    )


def test_live_agent_passes_sqlite_result_into_full_neo4j_path(
    live_chat_client,
):
    expected_path = _neo4j_path(3)
    root_source = expected_path[0]
    expected_target = expected_path[-1]

    exchange = _chat(
        live_chat_client,
        f"Через Neo4j покажи полный направленный путь от {root_source} до "
        f"{expected_target}: все узлы по порядку и глубину. Полный результат "
        "пути покажи отдельно.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    payloads = _display_payloads(result)
    _warn_unless(
        any(
            expected_path in _payload_table_paths(payload)
            for payload in payloads
        ),
        "presentation",
        "Neo4j display does not contain the dependent full path",
    )
    _assert_execution(
        exchange,
        expected_tools=[{"run_cypher", "trace_neo4j_table_path"}],
        expected_displays=[{"run_cypher", "trace_neo4j_table_path"}],
        max_seconds=180,
        max_llm_calls=20,
        max_total_tokens=140_000,
    )


def test_live_agent_checks_nulls_in_required_target_fields(live_chat_client):
    file_id, target_table, source_table, target_field, source_field = (
        _s2t_work_case_fixture()
    )
    exchange = _chat(
        live_chat_client,
        f"Для file_id={file_id} оцени совместимость nullable-ограничений "
        f"{source_table}.{source_field} → {target_table}.{target_field}. Верни "
        "source_not_null=<0|1>, target_not_null=<0|1> и вывод.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    _assert_s2t_work_case_execution(
        exchange,
        required_tools={
            "get_source_target_column_pair"
            if STRICT_RETRIEVAL_ENABLED
            else "list_column_catalog"
        },
        require_analysis=True,
    )


def test_live_agent_checks_source_and_target_type_compatibility(live_chat_client):
    file_id, target_table, source_table, target_field, source_field = (
        _s2t_work_case_fixture()
    )
    exchange = _chat(
        live_chat_client,
        f"Для file_id={file_id} оцени совместимость типов "
        f"{source_table}.{source_field} → {target_table}.{target_field}. Верни "
        "source_data_type=<тип>, target_data_type=<тип> и вывод.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    _assert_s2t_work_case_execution(
        exchange,
        required_tools={
            "get_source_target_column_pair"
            if STRICT_RETRIEVAL_ENABLED
            else "list_column_catalog"
        },
        require_analysis=True,
    )


def test_live_agent_checks_duplicate_risk_in_target(live_chat_client):
    _, target_table, source_table, _, _ = _s2t_work_case_fixture()
    exchange = _chat(
        live_chat_client,
        f"Оцени риск появления дубликатов при сохранённой S2T-трансформации "
        f"{source_table} → {target_table}.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    _assert_s2t_work_case_execution(
        exchange,
        required_tools=(
            {"read_s2t_source_to_target"}
            if STRICT_RETRIEVAL_ENABLED
            else None
        ),
        require_analysis=True,
    )


def test_live_agent_checks_unmapped_required_target_fields(live_chat_client):
    file_id, target_table, _, _, _ = _s2t_work_case_fixture()
    exchange = _chat(
        live_chat_client,
        f"Для file_id={file_id} найди обязательные поля {target_table} без "
        "сохранённого S2T-маппинга. Верни "
        "mandatory_fields_count=<число>, "
        "mandatory_fields_without_mapping_count=<число> и имена полей без "
        "маппинга.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    _assert_s2t_work_case_execution(
        exchange,
        required_tools=(
            {"list_target_column_catalog", "read_s2t_by_target_table"}
            if STRICT_RETRIEVAL_ENABLED
            else {"list_column_catalog"}
        ),
        require_analysis=True,
    )


def test_live_agent_checks_row_loss_risk(live_chat_client):
    _, target_table, source_table, _, _ = _s2t_work_case_fixture()
    exchange = _chat(
        live_chat_client,
        f"Оцени риск потери строк в сохранённой S2T-трансформации "
        f"{source_table} → {target_table}.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    _assert_s2t_work_case_execution(
        exchange,
        required_tools=(
            {"read_s2t_source_to_target"}
            if STRICT_RETRIEVAL_ENABLED
            else None
        ),
        require_analysis=True,
    )


def test_live_agent_explains_table_transformation(live_chat_client):
    _, target_table, source_table, target_field, source_field = (
        _s2t_work_case_fixture()
    )
    exchange = _chat(
        live_chat_client,
        f"Объясни сохранённую S2T-трансформацию "
        f"{source_table}.{source_field} → {target_table}.{target_field}.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    _assert_s2t_work_case_execution(
        exchange,
        required_tools=(
            {"read_s2t_source_to_target"}
            if STRICT_RETRIEVAL_ENABLED
            else None
        ),
        require_analysis=True,
    )


def test_live_agent_writes_s2t_test_protocol(live_chat_client):
    file_id, target_table, source_table, _, _ = _s2t_work_case_fixture()
    exchange = _chat(
        live_chat_client,
        f"Для file_id={file_id} по сохранённой S2T-спецификации "
        f"{source_table} → {target_table} составь тест-протокол для проверки "
        "ETL-загрузки во внешней СУБД. Включи проверки количества строк, "
        "уникальности ключа, null-rate обязательных полей и корректности "
        "трансформаций. Для каждой проверки дай цель, SQL-шаблон и критерий "
        "прохождения. Используй подтверждённые таблицы, колонки и правила; "
        "фактические метрики не вычисляй.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    _assert_compiled_test_protocol(
        exchange,
        target_table,
        source_tables=(source_table,),
        expected_pair_reads=1,
    )
    _assert_s2t_work_case_execution(
        exchange,
        required_tools=(
            {
                "read_s2t_by_target_table",
                "list_target_column_catalog",
            }
            if STRICT_RETRIEVAL_ENABLED
            else None
        ),
        require_analysis=True,
    )


def test_live_agent_writes_independent_s2t_test_protocol(live_chat_client):
    file_id, source_table, target_table = _independent_protocol_case_fixture()
    exchange = _chat(
        live_chat_client,
        f"Для file_id={file_id} подготовь приёмочный протокол из Greenplum SQL "
        f"для сохранённой загрузки {source_table} → {target_table}. Ничего не "
        "запускай. Нужны четыре контроля: совпадает ли рассчитанный по правилу "
        "набор с target; отсутствуют ли NULL в обязательных колонках; не "
        "повторяется ли подтверждённый ключ; одинаково ли число ожидаемых и "
        "загруженных строк. Для каждого укажи назначение, запрос и однозначное "
        "условие успешной приёмки.",
    )

    _assert_public_answer(exchange.result.answer)
    _assert_compiled_test_protocol(
        exchange,
        target_table,
        source_tables=(source_table,),
        expected_pair_reads=1,
    )
    sql_blocks = re.findall(
        r"```sql\n(.*?)\n```",
        exchange.result.answer,
        flags=re.DOTALL,
    )
    folded_blocks = [block.casefold() for block in sql_blocks]
    assert all(target_table.casefold() in block for block in folded_blocks)
    assert all("{{load_scope_predicate}}" in block for block in folded_blocks)
    combined_sql = "\n".join(folded_blocks)
    assert "group by" in combined_sql
    assert "having count(*) > 1" in combined_sql
    assert "is null" in combined_sql
    assert combined_sql.count("except all") == 2
    _assert_s2t_work_case_execution(
        exchange,
        required_tools=(
            {
                "read_s2t_source_to_target",
                "read_s2t_by_target_table",
                "list_target_column_catalog",
            }
            if STRICT_RETRIEVAL_ENABLED
            else None
        ),
        require_analysis=True,
    )


def test_live_agent_analyzes_s2t_validation_risks(live_chat_client):
    file_id, target_table, source_table, _, _ = _s2t_work_case_fixture()
    exchange = _chat(
        live_chat_client,
        f"Для file_id={file_id} по сохранённой S2T-спецификации "
        f"{source_table} → {target_table} оцени риск потери строк, риск "
        "дубликатов, обязательные target-поля без S2T-маппинга и "
        "согласованность трансформации. Используй только S2T и каталог "
        "колонок; не обращайся к физическим данным ETL-таблиц.",
    )

    _assert_public_answer(exchange.result.answer)
    _assert_s2t_work_case_execution(
        exchange,
        required_tools=(
            {"read_s2t_by_target_table", "list_target_column_catalog"}
            if STRICT_RETRIEVAL_ENABLED
            else None
        ),
        require_analysis=True,
    )


def test_live_agent_writes_multi_source_s2t_validation_protocol(
    live_chat_client,
):
    file_id, target_table, first_source, second_source = (
        _multi_source_validation_case_fixture()
    )
    exchange = _chat(
        live_chat_client,
        f"Для file_id={file_id} по сохранённой S2T-загрузке из "
        f"{first_source} и {second_source} в {target_table} составь единый "
        "тест-протокол для внешней СУБД. Включи проверки количества строк, "
        "уникальности ключа, null-rate обязательных полей и корректности "
        "трансформаций. Для каждой проверки дай цель, SQL-шаблон и критерий "
        "прохождения; фактические метрики не вычисляй.",
    )

    _assert_public_answer(exchange.result.answer)
    _assert_compiled_test_protocol(
        exchange,
        target_table,
        source_tables=(first_source, second_source),
        expected_pair_reads=2,
    )
    _assert_s2t_work_case_execution(
        exchange,
        required_tools=(
            {"read_s2t_by_target_table", "list_target_column_catalog"}
            if STRICT_RETRIEVAL_ENABLED
            else None
        ),
        require_analysis=True,
    )


def test_live_agent_writes_multi_target_s2t_validation_protocol(
    live_chat_client,
):
    file_id, source_table, first_target, second_target = (
        _multi_target_validation_case_fixture()
    )
    exchange = _chat(
        live_chat_client,
        f"Для file_id={file_id} по сохранённым S2T-загрузкам из "
        f"{source_table} в {first_target} и {second_target} составь отдельный "
        "тест-протокол для каждой target-таблицы во внешней СУБД. В каждый "
        "включи проверки количества строк, уникальности ключа, null-rate "
        "обязательных полей и корректности трансформаций. Для каждой проверки "
        "дай цель, SQL-шаблон и критерий прохождения; фактические метрики не "
        "вычисляй.",
    )

    _assert_public_answer(exchange.result.answer)
    _assert_compiled_test_protocol(
        exchange,
        first_target,
        second_target,
        source_tables=(source_table,),
        expected_pair_reads=2,
    )
    _assert_s2t_work_case_execution(
        exchange,
        required_tools=(
            {
                "read_s2t_by_target_table",
                "list_target_column_catalog",
            }
            if STRICT_RETRIEVAL_ENABLED
            else None
        ),
        require_analysis=True,
    )


def _assert_s2t_catalog_scenario(
    exchange: _LiveExchange,
) -> None:
    answer = exchange.result.answer
    _assert_public_answer(answer)

    metrics = exchange.metrics
    assert metrics.error is None, metrics.error
    assert metrics.elapsed_seconds > 0, metrics
    assert exchange.http_elapsed_seconds > 0, metrics
    _warn_unless(
        metrics.elapsed_seconds <= 300
        and exchange.http_elapsed_seconds <= 305,
        "efficiency",
        f"elapsed={metrics.elapsed_seconds:.3f}s exceeds catalog budget=300s",
    )
    _warn_unless(
        bool(metrics.tool_calls),
        "efficiency",
        "scenario returned without inspecting stored data",
    )
    if LIVE_AGENT_MODE == "multiagent":
        _warn_unless(
            bool(metrics.coordinator_plan)
            and 0 < len(metrics.worker_tasks) <= len(metrics.coordinator_plan),
            "efficiency",
            "coordinator/worker trace is incomplete",
        )
    else:
        _warn_unless(
            metrics.worker_tasks == [] and metrics.coordinator_plan == [],
            "efficiency",
            "single-agent run unexpectedly contains coordinator activity",
        )
    assert len(metrics.llm_calls) > 0, metrics.llm_calls
    _warn_unless(
        len(metrics.llm_calls) <= 80,
        "efficiency",
        f"llm_calls={len(metrics.llm_calls)} exceeds catalog budget=80",
    )
    assert metrics.total_tokens == metrics.input_tokens + metrics.output_tokens
    assert metrics.total_tokens > 0, metrics
    _warn_unless(
        metrics.total_tokens <= 320_000,
        "efficiency",
        f"total_tokens={metrics.total_tokens} exceeds catalog budget=320000",
    )


def test_live_agent_catalog_01_finds_target_field_source(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Откуда заполняется optn_id в t_optn? Найди source table, source field "
        "и покажи transformation rule. Используй глобальную s2t_transformations.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_02_finds_source_field_targets(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "В какие целевые таблицы передаётся c_closedate из "
        "s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract? Найди все "
        "downstream S2T, не останавливайся на первом совпадении.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_03_lists_table_mapping(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Покажи полный маппинг b3050000420005_paymentdetails -> t_optn: "
        "перечисли source column -> target column и transformation rules.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_04_explains_calculated_field(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Как рассчитывается agr_cred_sum_crncy_amt в "
        "b7000000250004_loansagreement? Покажи expression и все исходные поля.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_05_finds_business_metric_source(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Из какого поля берётся сумма задолженности или кредитного лимита "
        "клиента? Ищи по бизнес-смыслу и описаниям, верни наиболее вероятные "
        "S2T и объясни выбор техническими полями.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_06_semantic_close_date_search(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Где у нас хранится дата закрытия договора? Найди технические поля без "
        "требования точного совпадения русского текста и покажи S2T.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_07_finds_business_filter_rule(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Как определяется, что клиент связан с депозитным договором в "
        "t_agr_dep_cust? Покажи условия отбора и поля клиента.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_08_searches_client_id_synonyms(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Найди идентификатор клиента в S2T, учитывая варианты client_id, "
        "cust_id, client_entityid_uid и baseclientid. Верни таблицы и поля.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_09_maps_russian_term_to_technical_field(
    live_chat_client,
):
    exchange = _chat(
        live_chat_client,
        "Найди техническое поле для даты удаления записи и соответствующее "
        "S2T-правило. Ищи по русскому бизнес-термину, а не по заданному имени.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_10_builds_full_lineage(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Покажи всю цепочку происхождения b700000025_agr_cred.c_closedate до "
        "первичных source-таблиц. Включи subquery и branch по порядку.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_11_lists_intermediate_tables(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Через какие промежуточные таблицы проходит c_closedate от "
        "s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract до "
        "b700000025_agr_cred? Перечисли маршрут по порядку.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_12_compares_two_field_origins(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "fk_status_id в b700000025_agr_cred и fk_status_id в "
        "b700000025_agr_grntee берутся из одного источника? Построй lineage для "
        "обоих и дай явный итог с общими и различающимися источниками.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_13_finds_join_condition(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "По каким полям соединяются l_000025_t_loansagreement_stg и "
        "l_000025_t_loanscontract_stg в сохранённых Additional objects? "
        "Покажи JOIN condition и роли алиасов.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_14_finds_filtering(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Какие записи из b3050000420007_product не попадут в t_agr_dep? "
        "Найди WHERE/FILTER условия и объясни исключение записей.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_15_finds_constant_or_default(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Где при загрузке fk_productkind_id в "
        "b700000025_agr_cred::subquery::v_agr_cred2 устанавливается константа "
        "или default? Покажи literal и правило.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_16_finds_case_transformation(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Где используется CASE при расчёте del_dt в "
        "b700000025_agr_cred::subquery::v_agr_cred1? Покажи условия и "
        "результирующие значения.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_17_finds_aggregation(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Откуда берётся agr_dep_purpose_type_cd в t_agr_dep_purpose_type и "
        "как данные агрегируются? Покажи агрегат и уровень GROUP BY.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_18_investigates_wrong_value(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "В b700000025_agr_cred.c_closedate неправильная дата. Из каких "
        "источников и преобразований она могла прийти? Восстанови lineage назад "
        "и выдели места возможного изменения.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_19_investigates_null(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "del_dt в b700000025_agr_cred пустое. Посмотри, откуда оно загружается "
        "и какие CASE/JOIN/FILTER могут привести к NULL.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_20_finds_data_loss_points(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "В a_000025_t_loanscontract запись есть, а в b700000025_agr_cred её "
        "нет. Какие S2T, промежуточные таблицы, JOIN и FILTER надо проверить? "
        "Выдели возможные места потери записи.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_21_traces_value_change(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "В источнике ctl_action='D', а в b700000025_agr_cred рассчитано del_dt. "
        "Найди все преобразования по пути и укажи, где меняется представление "
        "значения.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_22_finds_multiple_sources(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Из каких источников может заполняться agr_cred_sum_crncy_amt в "
        "b7000000250004_loansagreement? Учти CASE, COALESCE и альтернативные "
        "source fields.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_23_performs_impact_analysis(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Что затронет изменение "
        "s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract.c_closedate? "
        "Выполни reverse lineage и перечисли downstream-поля, таблицы и "
        "зависимые transformations.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_24_compares_two_mart_rules(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Сравни расчёт del_dt в b700000025_agr_cred и "
        "b700000025_agr_grntee. Найди оба lineage и rules, явно скажи, "
        "совпадает логика или различается и чем.",
    )
    _assert_s2t_catalog_scenario(exchange)


def test_live_agent_catalog_25_finds_conflicting_s2t(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Есть ли несколько S2T, которые описывают загрузку "
        "b700000025_agr_cred::subquery::v_agr_cred1.del_dt по-разному? Найди "
        "все mappings, сравни source fields и transformation, выдели конфликт "
        "или объясни, почему mappings дополняют друг друга.",
    )
    _assert_s2t_catalog_scenario(exchange)
