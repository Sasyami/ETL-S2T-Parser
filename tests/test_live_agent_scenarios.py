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
from csv import DictReader
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Iterator
from uuid import uuid4

import pytest

import storage.database as db_storage
from agents.run_metrics import AgentRunMetrics, consume_agent_run_metrics


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
_LIVE_TRANSCRIPT_LOCK = threading.Lock()
_LIVE_TRANSCRIPT_INDEX = 0

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live_agent,
    pytest.mark.skipif(
        not LIVE_AGENT_ENABLED,
        reason="set RUN_LIVE_AGENT_SCENARIOS=1 to call the configured real LLM",
    ),
]


def _integer_values(text: str) -> set[int]:
    values: set[int] = set()
    for match in re.findall(
        r"(?<![\w.,])\d(?:[\d _\u00a0]*\d)?(?![\w]|[.,]\d)",
        str(text or ""),
    ):
        normalized = re.sub(r"[ _\u00a0]", "", match)
        values.add(int(normalized))
    return values


def _assert_public_answer(answer: str) -> None:
    assert answer.strip()
    lowered = answer.lower()
    for internal_term in (
        "supervisor",
        "coordinator",
        "finish_worker",
        "result_key",
    ):
        assert internal_term not in lowered, answer


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
    result: object
    metrics: AgentRunMetrics
    http_elapsed_seconds: float


def _chat(client, query: str, *, history: list[dict] | None = None):
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
    if not LIVE_TRANSCRIPT_PATH:
        return

    global _LIVE_TRANSCRIPT_INDEX
    transcript_path = Path(LIVE_TRANSCRIPT_PATH)
    if not transcript_path.is_absolute():
        transcript_path = PROJECT_ROOT / transcript_path

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

    metrics_block = "Метрики недоступны"
    trace_block = "Трасса недоступна"
    if metrics is not None:
        metrics_block = (
            f"agent_seconds: {metrics.elapsed_seconds:.3f}\n"
            f"http_seconds: {http_elapsed_seconds:.3f}\n"
            f"llm_calls: {len(metrics.llm_calls)}\n"
            f"tokens: input={metrics.input_tokens}, "
            f"output={metrics.output_tokens}, total={metrics.total_tokens}, "
            f"cache_read={metrics.cache_read_tokens}\n"
            f"workers: {len(metrics.worker_tasks)}\n"
            "tools: "
            + (", ".join(item.name for item in metrics.tool_calls) or "Нет")
            + "\n"
            "displays: "
            + (", ".join(metrics.display_tools) or "Нет")
        )
        trace_block = "```json\n" + json.dumps(
            {
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
                "coordinate_result": metrics.coordinate_result,
                "aggregate_result": metrics.aggregate_result,
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
        )
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with transcript_path.open("a", encoding="utf-8", newline="\n") as transcript:
            transcript.write(block)


def _assert_minimal_name_sequence(
    actual: list[str],
    expected: list[str | set[str]],
) -> None:
    assert len(actual) == len(expected), {
        "actual": actual,
        "expected": expected,
    }
    for index, (actual_name, expected_name) in enumerate(
        zip(actual, expected),
        start=1,
    ):
        accepted_names = (
            expected_name
            if isinstance(expected_name, set)
            else {expected_name}
        )
        assert actual_name in accepted_names, {
            "position": index,
            "actual": actual_name,
            "accepted": sorted(accepted_names),
        }


def _assert_execution(
    exchange: _LiveExchange,
    *,
    expected_tools: list[str | set[str]],
    expected_displays: list[str | set[str]],
    max_seconds: float | None,
    max_llm_calls: int,
    max_total_tokens: int,
) -> None:
    metrics = exchange.metrics
    assert metrics.error is None, metrics.error
    assert metrics.elapsed_seconds > 0, metrics
    assert exchange.http_elapsed_seconds > 0, metrics
    if max_seconds is not None:
        assert metrics.elapsed_seconds <= max_seconds, metrics
        assert exchange.http_elapsed_seconds <= max_seconds + 5, metrics
    _assert_minimal_name_sequence(
        [item.name for item in metrics.tool_calls],
        expected_tools,
    )
    if LIVE_AGENT_MODE == "multiagent":
        actual_worker_count = len(metrics.worker_tasks)
        assert len(metrics.coordinator_plan) == actual_worker_count, (
            metrics.coordinator_plan
        )
    else:
        assert metrics.worker_tasks == [], metrics.worker_tasks
        assert metrics.coordinator_plan == [], metrics.coordinator_plan
    _assert_minimal_name_sequence(metrics.display_tools, expected_displays)
    assert 0 < len(metrics.llm_calls) <= max_llm_calls, metrics.llm_calls
    assert all(item.total_tokens > 0 for item in metrics.llm_calls), metrics.llm_calls
    assert metrics.input_tokens > 0
    assert metrics.output_tokens > 0
    assert metrics.total_tokens == metrics.input_tokens + metrics.output_tokens
    assert metrics.total_tokens <= max_total_tokens, metrics


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


def _work_case_column_metadata() -> tuple[dict, dict]:
    file_id, target_table, source_table, target_field, source_field = (
        _s2t_work_case_fixture()
    )
    conn = db_storage.get_db_connection()
    try:
        source_row = conn.execute(
            """
            SELECT table_name, column_name, data_type, primary_key, not_null,
                   description
            FROM source_columns
            WHERE file_id = ?
              AND table_name = ? COLLATE NOCASE
              AND column_name = ? COLLATE NOCASE
            ORDER BY id
            LIMIT 1
            """,
            (file_id, source_table, source_field),
        ).fetchone()
        target_row = conn.execute(
            """
            SELECT table_name, column_name, data_type, primary_key, not_null,
                   description
            FROM target_columns
            WHERE file_id = ?
              AND table_name = ? COLLATE NOCASE
              AND column_name = ? COLLATE NOCASE
            ORDER BY id
            LIMIT 1
            """,
            (file_id, target_table, target_field),
        ).fetchone()
    finally:
        conn.close()
    if source_row is None or target_row is None:
        pytest.skip("workspace column catalogs lack the selected S2T mapping")
    return dict(source_row), dict(target_row)


def _work_case_data_types() -> tuple[str, str]:
    source_metadata, target_metadata = _work_case_column_metadata()
    source_type = str(source_metadata.get("data_type") or "").strip()
    target_type = str(target_metadata.get("data_type") or "").strip()
    if not source_type or not target_type:
        pytest.skip("workspace column catalogs lack data types for the mapping")
    return source_type, target_type


def _work_case_required_target_fields() -> tuple[list[str], list[str]]:
    file_id, target_table, _, _, _ = _s2t_work_case_fixture()
    conn = db_storage.get_db_connection()
    try:
        mandatory_rows = conn.execute(
            """
            SELECT DISTINCT TRIM(column_name) AS column_name
            FROM target_columns
            WHERE file_id = ?
              AND table_name = ? COLLATE NOCASE
              AND not_null = 1
              AND column_name IS NOT NULL
              AND TRIM(column_name) <> ''
            ORDER BY column_name COLLATE NOCASE
            """,
            (file_id, target_table),
        ).fetchall()
        mapped_rows = conn.execute(
            """
            SELECT DISTINCT TRIM(target_field) AS target_field
            FROM s2t_transformations
            WHERE target_table = ? COLLATE NOCASE
              AND target_field IS NOT NULL
              AND TRIM(target_field) <> ''
            """,
            (target_table,),
        ).fetchall()
    finally:
        conn.close()
    mandatory = [str(row[0]) for row in mandatory_rows]
    mapped = {str(row[0]).casefold() for row in mapped_rows}
    unmapped = [name for name in mandatory if name.casefold() not in mapped]
    return mandatory, unmapped


def _assert_s2t_work_case_execution(
    exchange: _LiveExchange,
    *,
    required_tools: set[str] | None = None,
    max_seconds: float = 240,
    max_llm_calls: int = 32,
    max_total_tokens: int = 180_000,
) -> None:
    metrics = exchange.metrics
    allowed_tools = {
        "get_excel_row",
        "list_column_catalog",
        "list_columns",
        "list_file_sheet_headers",
        "list_s2t_transformations",
        "parse_sql_column_lineage",
        "parse_sql_table_lineage",
        "query_saved_result",
        "run_sql",
        "search_column_catalog",
        "search_excel_values",
        "search_s2t_transformations",
        "show_plan",
        "trace_transformation_path",
    }
    tool_names = [item.name for item in metrics.tool_calls]

    assert metrics.error is None, metrics.error
    assert 0 < metrics.elapsed_seconds <= max_seconds, metrics
    assert 0 < exchange.http_elapsed_seconds <= max_seconds + 5, metrics
    assert tool_names, "scenario returned without inspecting stored data"
    assert set(tool_names) <= allowed_tools, tool_names
    assert set(required_tools or ()) <= set(tool_names), tool_names
    assert len(tool_names) <= 12, tool_names
    if LIVE_AGENT_MODE == "multiagent":
        assert metrics.coordinator_plan, metrics.coordinator_plan
        assert 0 < len(metrics.worker_tasks) <= len(metrics.coordinator_plan), (
            metrics.coordinator_plan,
            metrics.worker_tasks,
        )
    else:
        assert metrics.worker_tasks == [], metrics.worker_tasks
        assert metrics.coordinator_plan == [], metrics.coordinator_plan
    assert 0 < len(metrics.llm_calls) <= max_llm_calls, metrics.llm_calls
    assert all(item.total_tokens > 0 for item in metrics.llm_calls), metrics.llm_calls
    assert metrics.input_tokens > 0
    assert metrics.output_tokens > 0
    assert metrics.total_tokens == metrics.input_tokens + metrics.output_tokens
    assert metrics.total_tokens <= max_total_tokens, metrics


def _assert_answer_contains_any(answer: str, values: tuple[str, ...]) -> None:
    lowered = answer.casefold()
    assert any(value.casefold() in lowered for value in values), answer


def test_live_agent_answers_simple_conversation_without_display_results(
    live_chat_client,
):
    exchange = _chat(live_chat_client, "Ответь одним словом: привет")
    result = exchange.result

    _assert_public_answer(result.answer)
    assert result.answer.strip().casefold() == "привет", result.answer
    assert result.display_items == []
    _assert_execution(
        exchange,
        expected_tools=[],
        expected_displays=[],
        max_seconds=None,
        max_llm_calls=3,
        max_total_tokens=15_000,
    )


def test_live_agent_returns_exact_global_sqlite_count(live_chat_client):
    expected_count = int(
        _fetch_one("SELECT COUNT(*) FROM s2t_transformations")[0]
    )
    exchange = _chat(
        live_chat_client,
        "Через SQLite посчитай точное число строк в s2t_transformations. "
        "Нужен только итоговый count."
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    assert expected_count in _integer_values(result.answer), result.answer
    assert len(result.answer) <= 220
    assert result.display_items == []
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
    expected_count = int(
        _fetch_one("SELECT COUNT(*) FROM s2t_transformations")[0]
    )
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
    assert expected_count in _integer_values(result.answer), result.answer
    assert result.display_items == []
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
    assert len(result.answer) <= 500
    assert result.display_items, result.answer
    matching_payloads = [
        payload
        for payload in _display_payloads(result)
        if payload.get("preview_rows") == expected_rows
        or payload.get("rows") == expected_rows
    ]
    assert matching_payloads, [item.content for item in result.display_items]
    payload = matching_payloads[0]
    if payload.get("csv_url"):
        downloaded_rows = _download_sql_export(
            live_chat_client,
            payload,
            generated_sql_exports,
        )
        assert downloaded_rows == [
            {key: str(value) for key, value in row.items()}
            for row in expected_rows
        ]
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
    target_table, target_row_count = _fetch_one(
        """
        SELECT target_table, COUNT(*) AS row_count
        FROM s2t_transformations
        WHERE target_table IS NOT NULL AND TRIM(target_table) <> ''
        GROUP BY target_table
        ORDER BY row_count DESC, target_table
        LIMIT 1
        """
    )
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
    assert str(target_table) in result.answer, result.answer
    answer_numbers = _integer_values(result.answer)
    assert int(target_row_count) in answer_numbers, result.answer
    assert source_count in answer_numbers, result.answer
    payloads = _display_payloads(result)
    assert any(
        _payload_contains_value(payload, source_count)
        for payload in payloads
    ), payloads
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
    assert "sqlite" not in result.answer.lower(), result.answer
    for table_name in (source, middle, target):
        assert table_name in result.answer, result.answer
    assert 2 in _integer_values(result.answer), result.answer
    display_contents = [item.content for item in result.display_items]
    assert any(
        all(table_name in content for table_name in (source, middle, target))
        for content in display_contents
    ), display_contents
    _assert_execution(
        exchange,
        expected_tools=[{"run_cypher", "trace_neo4j_table_path"}],
        expected_displays=[{"run_cypher", "trace_neo4j_table_path"}],
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
    assert "sqlite" not in lowered, result.answer
    assert "трансформац" not in lowered, result.answer
    assert "mapping" not in lowered, result.answer
    matching_paths = [
        path
        for path in expected_paths
        if all(table_name in result.answer for table_name in path)
    ]
    assert matching_paths, result.answer
    assert 3 in _integer_values(result.answer), result.answer

    path_payloads = _display_payloads(result)
    assert any(
        any(
            expected_path in _payload_table_paths(payload)
            for expected_path in matching_paths
        )
        and _payload_contains_value(payload, 3)
        for payload in path_payloads
    ), path_payloads
    _assert_execution(
        exchange,
        expected_tools=[{"run_cypher", "trace_neo4j_table_path"}],
        expected_displays=[{"run_cypher", "trace_neo4j_table_path"}],
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
    for row in expected_rows:
        for field_name in (
            "source_table",
            "source_field",
            "target_table",
            "target_field",
        ):
            assert str(row[field_name]) in result.answer, result.answer
    matching_payloads = [
        payload
        for payload in _display_payloads(result)
        if payload.get("preview_rows") == expected_rows
        or payload.get("rows") == expected_rows
    ]
    assert matching_payloads, [item.content for item in result.display_items]
    payload = matching_payloads[0]
    if payload.get("csv_url"):
        assert payload.get("preview_rows") == expected_rows
        downloaded_rows = _download_sql_export(
            live_chat_client,
            payload,
            generated_sql_exports,
        )
        assert downloaded_rows == [
            {key: str(value) for key, value in row.items()}
            for row in expected_rows
        ]
    else:
        assert payload.get("rows") == expected_rows
        assert payload.get("returned_rows") == len(expected_rows)
        assert payload.get("truncated") is False
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
    target_table, target_row_count = _fetch_one(
        """
        SELECT target_table, COUNT(*) AS row_count
        FROM s2t_transformations
        WHERE target_table IS NOT NULL AND TRIM(target_table) <> ''
        GROUP BY target_table
        ORDER BY row_count DESC, target_table
        LIMIT 1
        """
    )
    distinct_source_count = int(
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
        "Через SQLite выполни три отдельных зависимых шага. Шаг 1: в "
        "s2t_transformations найди target_table с максимальным числом строк. "
        "Шаг 2: для найденной target_table посчитай различные непустые "
        "source_table. Шаг 3: для той же target_table среди непустых "
        "source_table найди source_table с максимальным числом строк; при "
        "равенстве выбери лексикографически первый source_table. Верни "
        "target_table и число строк, число "
        "различных source_table, лидирующую source_table и число её строк. "
        "Полный результат третьего шага покажи отдельно.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    assert str(target_table) in result.answer, result.answer
    assert str(top_source) in result.answer, result.answer
    numbers = _integer_values(result.answer)
    assert int(target_row_count) in numbers, result.answer
    assert distinct_source_count in numbers, result.answer
    assert int(top_source_count) in numbers, result.answer
    payloads = _display_payloads(result)
    assert any(
        _payload_contains_value(payload, top_source)
        and _payload_contains_value(payload, int(top_source_count))
        for payload in payloads
    ), payloads
    _assert_execution(
        exchange,
        expected_tools=["run_sql", "run_sql", "run_sql"],
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
    immediate_source = expected_path[-2]
    expected_target, expected_count = _fetch_one(
        """
        SELECT target_table, COUNT(*) AS row_count
        FROM s2t_transformations
        WHERE source_table = ?
          AND target_table IS NOT NULL
          AND TRIM(target_table) <> ''
        GROUP BY target_table
        ORDER BY row_count DESC, target_table
        LIMIT 1
        """,
        (immediate_source,),
    )
    if str(expected_target) != expected_path[-1]:
        pytest.skip("SQLite and Neo4j fixtures do not describe the same path end")

    exchange = _chat(
        live_chat_client,
        f"Выполни два отдельных зависимых шага. Сначала через SQLite в "
        f"s2t_transformations для source_table = {immediate_source} найди "
        "target_table с максимальным числом строк и верни её имя и count. Затем "
        f"через Neo4j, используя найденное имя target_table, построй полный "
        f"направленный путь от {root_source} до неё. В итоговом ответе дай count, "
        "все узлы пути по порядку и глубину. Полные результаты обоих шагов "
        "покажи отдельно в scrollable UI.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    assert int(expected_count) in _integer_values(result.answer), result.answer
    for table_name in expected_path:
        assert table_name in result.answer, result.answer
    assert 3 in _integer_values(result.answer), result.answer
    payloads = _display_payloads(result)
    assert any(
        _payload_contains_value(payload, expected_target)
        and _payload_contains_value(payload, int(expected_count))
        for payload in payloads
    ), payloads
    assert any(
        expected_path in _payload_table_paths(payload)
        for payload in payloads
    ), payloads
    _assert_execution(
        exchange,
        expected_tools=[
            "run_sql",
            {"run_cypher", "trace_neo4j_table_path"},
        ],
        expected_displays=[
            "run_sql",
            {"run_cypher", "trace_neo4j_table_path"},
        ],
        max_seconds=180,
        max_llm_calls=20,
        max_total_tokens=140_000,
    )


def test_live_agent_checks_nulls_in_required_target_fields(live_chat_client):
    file_id, target_table, source_table, target_field, source_field = (
        _s2t_work_case_fixture()
    )
    source_metadata, target_metadata = _work_case_column_metadata()
    exchange = _chat(
        live_chat_client,
        f"Для загруженного файла file_id={file_id} проверь кейс: в обязательном "
        f"целевом поле {target_table}.{target_field} могут появляться NULL. "
        "Получи точные метаданные через публичные каталоги source_columns и "
        "target_columns, не восстанавливай их из сырых строк data. Для "
        f"источника {source_table}.{source_field} укажи обязательность target, "
        "может ли source быть NULL и есть ли в глобальной s2t_transformations "
        "защита COALESCE или WHERE. В ответе явно запиши "
        "source_not_null=<0|1> и target_not_null=<0|1>, затем дай вывод с "
        "конкретными доказательствами; при нехватке данных укажи это.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    assert target_table in result.answer, result.answer
    assert source_table in result.answer, result.answer
    assert target_field in result.answer, result.answer
    assert source_field in result.answer, result.answer
    normalized_answer = result.answer.casefold().replace(" ", "")
    assert (
        f"source_not_null={int(source_metadata['not_null'])}"
        in normalized_answer
    ), result.answer
    assert (
        f"target_not_null={int(target_metadata['not_null'])}"
        in normalized_answer
    ), result.answer
    _assert_answer_contains_any(result.answer, ("null", "coalesce", "where"))
    _assert_s2t_work_case_execution(
        exchange,
        required_tools={"list_column_catalog"},
    )


def test_live_agent_checks_source_and_target_type_compatibility(live_chat_client):
    file_id, target_table, source_table, target_field, source_field = (
        _s2t_work_case_fixture()
    )
    source_type, target_type = _work_case_data_types()
    exchange = _chat(
        live_chat_client,
        f"Через SQLite для файла file_id={file_id} проверь совместимость типов "
        "в маппинге "
        f"{source_table}.{source_field} -> {target_table}.{target_field}. "
        "Получи обе строки через публичные source_columns и target_columns по "
        "точным file_id/table_name/column_name с помощью каталога колонок; не "
        "читай для этого сырые строки data. Отдельно проверь CAST в глобальной "
        "s2t_transformations по точным source_table, source_field, target_table "
        "и target_field без фильтра file_id. В ответе явно запиши "
        "source_data_type=<тип> и target_data_type=<тип>, приведи доказательства "
        "и однозначный вывод: совместимо, несовместимо или данных недостаточно.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    assert target_table in result.answer, result.answer
    normalized_answer = result.answer.casefold().replace(" ", "")
    assert f"source_data_type={source_type.casefold()}" in normalized_answer, (
        result.answer
    )
    assert f"target_data_type={target_type.casefold()}" in normalized_answer, (
        result.answer
    )
    _assert_answer_contains_any(result.answer, ("тип", "cast"))
    _assert_answer_contains_any(
        result.answer,
        ("совместим", "несовместим", "недостаточно"),
    )
    _assert_s2t_work_case_execution(
        exchange,
        required_tools={"list_column_catalog"},
    )


def test_live_agent_checks_duplicate_risk_in_target(live_chat_client):
    file_id, target_table, source_table, _, _ = _s2t_work_case_fixture()
    exchange = _chat(
        live_chat_client,
        f"Для файла file_id={file_id} оцени риск дубликатов в target_table "
        f"{target_table} из источника {source_table}. Проверь S2T и, если "
        "источник порождён представлением, Additional objects. Покажи, есть ли "
        "GROUP BY, DISTINCT, оконная дедупликация и какие JOIN могут размножать "
        "строки. Нужен вывод с конкретными SQL-признаками.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    assert target_table in result.answer, result.answer
    _assert_answer_contains_any(
        result.answer,
        ("group by", "distinct", "join", "row_number"),
    )
    _assert_s2t_work_case_execution(exchange)


def test_live_agent_checks_unmapped_required_target_fields(live_chat_client):
    file_id, target_table, _, _, _ = _s2t_work_case_fixture()
    mandatory_fields, unmapped_fields = _work_case_required_target_fields()
    exchange = _chat(
        live_chat_client,
        f"Для файла file_id={file_id} проверь, все ли обязательные поля "
        f"target_table {target_table} заполняются. Получи обязательные поля "
        "точным запросом к публичному target_columns с not_null=true, не читай "
        "их из сырых строк data. Сопоставь их с target_field глобальной "
        "s2t_transformations без фильтра file_id. Верни "
        "mandatory_fields_count=<число>, "
        "mandatory_fields_without_mapping_count=<число> и имена полей без "
        "маппинга. Не считай необязательные поля дефектами.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    assert target_table in result.answer, result.answer
    normalized_answer = result.answer.casefold().replace(" ", "")
    assert (
        f"mandatory_fields_count={len(mandatory_fields)}"
        in normalized_answer
    ), result.answer
    assert (
        "mandatory_fields_without_mapping_count="
        f"{len(unmapped_fields)}" in normalized_answer
    ), result.answer
    for field_name in unmapped_fields:
        assert field_name.casefold() in result.answer.casefold(), result.answer
    _assert_s2t_work_case_execution(
        exchange,
        required_tools={"list_column_catalog"},
    )


def test_live_agent_checks_row_loss_risk(live_chat_client):
    file_id, target_table, source_table, _, _ = _s2t_work_case_fixture()
    exchange = _chat(
        live_chat_client,
        f"Через SQLite для файла file_id={file_id} оцени риск потери строк при "
        "загрузке "
        f"из {source_table} в {target_table}. Проверь S2T и связанный SQL из "
        "Additional objects, если он есть. s2t_transformations глобальна: не "
        "ограничивай её по file_id; file_id используй только для файловых "
        "листов и Additional objects. ETL-имена здесь являются значениями, а "
        "не именами Excel-листов: не ищи лист с именем source_table. Сначала "
        "получи transformation_rule по точным source_table и target_table. "
        "Перечисли конкретные WHERE и INNER JOIN, которые могут уменьшить число "
        "строк, и отдели доказанные факты от предположений.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    assert target_table in result.answer, result.answer
    _assert_answer_contains_any(result.answer, ("where", "inner join", "join"))
    _assert_s2t_work_case_execution(exchange)


def test_live_agent_explains_table_transformation(live_chat_client):
    file_id, target_table, source_table, target_field, source_field = (
        _s2t_work_case_fixture()
    )
    exchange = _chat(
        live_chat_client,
        f"Для файла file_id={file_id} объясни простыми шагами табличную "
        f"трансформацию для маппинга {source_table}.{source_field} -> "
        f"{target_table}.{target_field}. Используй сохранённое правило S2T: "
        "какие таблицы соединяются, какие фильтры применяются, как устраняются "
        "дубликаты и какое значение попадает в target. Не выдумывай отсутствующие "
        "детали.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    assert target_table in result.answer, result.answer
    assert source_table in result.answer, result.answer
    _assert_answer_contains_any(result.answer, ("join", "where", "row_number"))
    _assert_s2t_work_case_execution(exchange)


def test_live_agent_writes_s2t_test_protocol(live_chat_client):
    file_id, target_table, source_table, _, _ = _s2t_work_case_fixture()
    exchange = _chat(
        live_chat_client,
        f"Через SQLite получи из глобальной s2t_transformations фактические "
        f"маппинги загрузки {source_table} -> {target_table}, а затем для файла "
        f"file_id={file_id} составь исполнимый тест-протокол. "
        "Включи проверки count(source) vs count(target), уникальности ключа, null-rate "
        "обязательных полей и контроль трансформаций. Для каждой проверки дай "
        "цель, SQL-шаблон и критерий прохождения. Source/target — логические "
        "ETL-таблицы, а не имена Excel-листов: не ищи листы с такими именами. "
        "Неизвестные физические имена не придумывай, обозначь их явными "
        "плейсхолдерами в SQL-шаблонах.",
    )
    result = exchange.result

    _assert_public_answer(result.answer)
    assert target_table in result.answer, result.answer
    _assert_answer_contains_any(result.answer, ("count", "количеств"))
    _assert_answer_contains_any(result.answer, ("null", "null-rate"))
    _assert_answer_contains_any(result.answer, ("distinct", "уникальн", "ключ"))
    _assert_s2t_work_case_execution(exchange)


def _assert_s2t_catalog_scenario(
    exchange: _LiveExchange,
    *,
    required: tuple[str, ...] = (),
    any_of: tuple[str, ...] = (),
) -> None:
    answer = exchange.result.answer
    _assert_public_answer(answer)
    lowered = answer.casefold()
    for value in required:
        assert value.casefold() in lowered, answer
    if any_of:
        _assert_answer_contains_any(answer, any_of)

    metrics = exchange.metrics
    assert metrics.error is None, metrics.error
    assert 0 < metrics.elapsed_seconds <= 300, metrics
    assert 0 < exchange.http_elapsed_seconds <= 305, metrics
    assert metrics.tool_calls, "scenario returned without inspecting stored data"
    if LIVE_AGENT_MODE == "multiagent":
        assert metrics.coordinator_plan, metrics.coordinator_plan
        assert 0 < len(metrics.worker_tasks) <= len(metrics.coordinator_plan), (
            metrics.coordinator_plan,
            metrics.worker_tasks,
        )
    else:
        assert metrics.worker_tasks == [], metrics.worker_tasks
        assert metrics.coordinator_plan == [], metrics.coordinator_plan
    assert 0 < len(metrics.llm_calls) <= 80, metrics.llm_calls
    assert metrics.total_tokens == metrics.input_tokens + metrics.output_tokens
    assert 0 < metrics.total_tokens <= 320_000, metrics


def test_live_agent_catalog_01_finds_target_field_source(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Откуда заполняется optn_id в t_optn? Найди source table, source field "
        "и покажи transformation rule. Используй глобальную s2t_transformations.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("t_optn", "optn_id", "b3050000420005_paymentdetails"),
        any_of=("object_id_uid", "product_entityid_uid"),
    )


def test_live_agent_catalog_02_finds_source_field_targets(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "В какие целевые таблицы передаётся c_closedate из "
        "s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract? Найди все "
        "downstream S2T, не останавливайся на первом совпадении.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("c_closedate",),
        any_of=("b700000025_agr_cred", "b700000025_agr_grntee"),
    )


def test_live_agent_catalog_03_lists_table_mapping(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Покажи полный маппинг b3050000420005_paymentdetails -> t_optn: "
        "перечисли source column -> target column и transformation rules.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("b3050000420005_paymentdetails", "t_optn"),
        any_of=("object_id_uid", "product_entityid_uid"),
    )


def test_live_agent_catalog_04_explains_calculated_field(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Как рассчитывается agr_cred_sum_crncy_amt в "
        "b7000000250004_loansagreement? Покажи expression и все исходные поля.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("agr_cred_sum_crncy_amt",),
        any_of=("c_debtlimit", "c_expenseslimit", "coalesce"),
    )


def test_live_agent_catalog_05_finds_business_metric_source(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Из какого поля берётся сумма задолженности или кредитного лимита "
        "клиента? Ищи по бизнес-смыслу и описаниям, верни наиболее вероятные "
        "S2T и объясни выбор техническими полями.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        any_of=("agr_cred_sum_crncy_amt", "c_debtlimit", "c_expenseslimit"),
    )


def test_live_agent_catalog_06_semantic_close_date_search(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Где у нас хранится дата закрытия договора? Найди технические поля без "
        "требования точного совпадения русского текста и покажи S2T.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        any_of=("close_dt", "c_closedate", "closedate", "dateclose"),
    )


def test_live_agent_catalog_07_finds_business_filter_rule(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Как определяется, что клиент связан с депозитным договором в "
        "t_agr_dep_cust? Покажи условия отбора и поля клиента.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("t_agr_dep_cust",),
        any_of=("client_entityid_uid", "cust_id", "is not null"),
    )


def test_live_agent_catalog_08_searches_client_id_synonyms(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Найди идентификатор клиента в S2T, учитывая варианты client_id, "
        "cust_id, client_entityid_uid и baseclientid. Верни таблицы и поля.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        any_of=("cust_id", "client_entityid_uid", "c_baseclientid"),
    )


def test_live_agent_catalog_09_maps_russian_term_to_technical_field(
    live_chat_client,
):
    exchange = _chat(
        live_chat_client,
        "Найди техническое поле для даты удаления записи и соответствующее "
        "S2T-правило. Ищи по русскому бизнес-термину, а не по заданному имени.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("del_dt",),
        any_of=("ctl_action", "ctl_validfrom"),
    )


def test_live_agent_catalog_10_builds_full_lineage(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Покажи всю цепочку происхождения b700000025_agr_cred.c_closedate до "
        "первичных source-таблиц. Включи subquery и branch по порядку.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("b700000025_agr_cred", "c_closedate"),
        any_of=("branch", "subquery", "a_000025_t_loanscontract"),
    )


def test_live_agent_catalog_11_lists_intermediate_tables(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Через какие промежуточные таблицы проходит c_closedate от "
        "s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract до "
        "b700000025_agr_cred? Перечисли маршрут по порядку.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("c_closedate", "b700000025_agr_cred"),
        any_of=("branch", "v_agr_cred", "subquery"),
    )


def test_live_agent_catalog_12_compares_two_field_origins(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "fk_status_id в b700000025_agr_cred и fk_status_id в "
        "b700000025_agr_grntee берутся из одного источника? Построй lineage для "
        "обоих и дай явный итог с общими и различающимися источниками.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("b700000025_agr_cred", "b700000025_agr_grntee", "fk_status_id"),
        any_of=("loanscontract", "loans_productparty"),
    )


def test_live_agent_catalog_13_finds_join_condition(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "По каким полям соединяются l_000025_t_loansagreement_stg и "
        "l_000025_t_loanscontract_stg в сохранённых Additional objects? "
        "Покажи JOIN condition и роли алиасов.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("fk_contract_id", "c_id"),
        any_of=("join", "соедин"),
    )


def test_live_agent_catalog_14_finds_filtering(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Какие записи из b3050000420007_product не попадут в t_agr_dep? "
        "Найди WHERE/FILTER условия и объясни исключение записей.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("b3050000420007_product", "t_agr_dep"),
        any_of=("incr_flag", "where", "фильтр"),
    )


def test_live_agent_catalog_15_finds_constant_or_default(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Где при загрузке fk_productkind_id в "
        "b700000025_agr_cred::subquery::v_agr_cred2 устанавливается константа "
        "или default? Покажи literal и правило.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("fk_productkind_id", "-1010"),
        any_of=("cast", "констант", "default"),
    )


def test_live_agent_catalog_16_finds_case_transformation(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Где используется CASE при расчёте del_dt в "
        "b700000025_agr_cred::subquery::v_agr_cred1? Покажи условия и "
        "результирующие значения.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("del_dt", "case"),
        any_of=("ctl_action", "ctl_validfrom", "null"),
    )


def test_live_agent_catalog_17_finds_aggregation(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Откуда берётся agr_dep_purpose_type_cd в t_agr_dep_purpose_type и "
        "как данные агрегируются? Покажи агрегат и уровень GROUP BY.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("agr_dep_purpose_type_cd", "t_agr_dep_purpose_type"),
        any_of=("max", "group by", "specialcode"),
    )


def test_live_agent_catalog_18_investigates_wrong_value(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "В b700000025_agr_cred.c_closedate неправильная дата. Из каких "
        "источников и преобразований она могла прийти? Восстанови lineage назад "
        "и выдели места возможного изменения.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("b700000025_agr_cred", "c_closedate"),
        any_of=("union all", "branch", "loanscontract", "loans_productparty"),
    )


def test_live_agent_catalog_19_investigates_null(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "del_dt в b700000025_agr_cred пустое. Посмотри, откуда оно загружается "
        "и какие CASE/JOIN/FILTER могут привести к NULL.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("b700000025_agr_cred", "del_dt"),
        any_of=("null", "case", "ctl_action"),
    )


def test_live_agent_catalog_20_finds_data_loss_points(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "В a_000025_t_loanscontract запись есть, а в b700000025_agr_cred её "
        "нет. Какие S2T, промежуточные таблицы, JOIN и FILTER надо проверить? "
        "Выдели возможные места потери записи.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("a_000025_t_loanscontract", "b700000025_agr_cred"),
        any_of=("join", "where", "branch", "union"),
    )
    _assert_answer_contains_any(
        exchange.result.answer,
        ("::subquery::", "::branch::", "v_agr_cred"),
    )


def test_live_agent_catalog_21_traces_value_change(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "В источнике ctl_action='D', а в b700000025_agr_cred рассчитано del_dt. "
        "Найди все преобразования по пути и укажи, где меняется представление "
        "значения.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("ctl_action", "del_dt", "case", "ctl_validfrom"),
        any_of=("null", "union all", "branch"),
    )


def test_live_agent_catalog_22_finds_multiple_sources(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Из каких источников может заполняться agr_cred_sum_crncy_amt в "
        "b7000000250004_loansagreement? Учти CASE, COALESCE и альтернативные "
        "source fields.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("agr_cred_sum_crncy_amt", "c_debtlimit", "c_expenseslimit"),
        any_of=("case", "coalesce"),
    )


def test_live_agent_catalog_23_performs_impact_analysis(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Что затронет изменение "
        "s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract.c_closedate? "
        "Выполни reverse lineage и перечисли downstream-поля, таблицы и "
        "зависимые transformations.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("c_closedate", "branch", "union all"),
        any_of=("b700000025_agr_cred", "b700000025_agr_grntee"),
    )


def test_live_agent_catalog_24_compares_two_mart_rules(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Сравни расчёт del_dt в b700000025_agr_cred и "
        "b700000025_agr_grntee. Найди оба lineage и rules, явно скажи, "
        "совпадает логика или различается и чем.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("b700000025_agr_cred", "b700000025_agr_grntee", "del_dt"),
        any_of=("case", "ctl_action", "ctl_validfrom"),
    )


def test_live_agent_catalog_25_finds_conflicting_s2t(live_chat_client):
    exchange = _chat(
        live_chat_client,
        "Есть ли несколько S2T, которые описывают загрузку "
        "b700000025_agr_cred::subquery::v_agr_cred1.del_dt по-разному? Найди "
        "все mappings, сравни source fields и transformation, выдели конфликт "
        "или объясни, почему mappings дополняют друг друга.",
    )
    _assert_s2t_catalog_scenario(
        exchange,
        required=("del_dt", "ctl_action", "ctl_validfrom"),
        any_of=("case", "конфликт", "дополня"),
    )
