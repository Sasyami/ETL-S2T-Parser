"""Opt-in end-to-end scenarios for the configured real chat model.

These tests intentionally execute one user request per test without mocks,
parameterization, batching, or parallel calls. Enable them explicitly with
``RUN_LIVE_AGENT_SCENARIOS=1``.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from csv import DictReader
from io import StringIO
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest

import storage.database as db_storage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_DB_PATH = PROJECT_ROOT / "excel_data.db"
LIVE_TRANSCRIPT_PATH = os.getenv("LIVE_AGENT_TRANSCRIPT_PATH", "").strip()
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


def _chat(client, query: str, *, history: list[dict] | None = None):
    from agents.chat_graph import WorkerRunResult

    response = client.post(
        "/chat",
        json={
            "query": query,
            "history": list(history or []),
            "session_id": f"live-agent-{uuid4()}",
        },
    )
    payload = response.get_json()
    _record_live_exchange(query, response.status_code, payload)
    assert response.status_code == 200, payload
    return WorkerRunResult.model_validate(payload)


def _record_live_exchange(query: str, status_code: int, payload) -> None:
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

    with _LIVE_TRANSCRIPT_LOCK:
        _LIVE_TRANSCRIPT_INDEX += 1
        block = (
            f"## {_LIVE_TRANSCRIPT_INDEX}. Запрос\n\n"
            f"{query}\n\n"
            f"### Ответ — HTTP {status_code}\n\n"
            f"{answer}\n\n"
            "### Display-results\n\n"
            f"{', '.join(display_names) if display_names else 'Нет'}\n\n"
        )
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with transcript_path.open("a", encoding="utf-8", newline="\n") as transcript:
            transcript.write(block)


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
    flask_app.config["TESTING"] = False
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


def test_live_agent_answers_simple_conversation_without_display_results(
    live_chat_client,
):
    result = _chat(live_chat_client, "Ответь одним словом: привет")

    _assert_public_answer(result.answer)
    assert result.display_items == []


def test_live_agent_returns_exact_global_sqlite_count(live_chat_client):
    expected_count = int(
        _fetch_one("SELECT COUNT(*) FROM s2t_transformations")[0]
    )
    result = _chat(
        live_chat_client,
        "Через SQLite посчитай точное число строк в s2t_transformations. "
        "Нужен только итоговый count."
    )

    _assert_public_answer(result.answer)
    assert expected_count in _integer_values(result.answer), result.answer
    assert len(result.answer) <= 220
    assert result.display_items == []


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
    result = _chat(
        live_chat_client,
        "Через SQLite посчитай в ней точное количество строк. Только число.",
        history=history,
    )

    _assert_public_answer(result.answer)
    assert expected_count in _integer_values(result.answer), result.answer
    assert result.display_items == []


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

    result = _chat(
        live_chat_client,
        "Через SQLite выполни SELECT file_id, filename FROM files "
        "ORDER BY file_id и покажи полный результат отдельно в scrollable UI."
    )

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

    result = _chat(
        live_chat_client,
        "Через SQLite сначала найди target_table с максимальным числом строк "
        "в s2t_transformations. Затем отдельным зависимым шагом для найденной "
        "target_table посчитай точное число различных непустых source_table. "
        "Верни имя target_table, число её строк и число source_table. "
        "Полный результат второго шага покажи отдельно.",
    )

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
    result = _chat(
        live_chat_client,
        f"Через Neo4j найди точный путь длины 2 от таблицы {source} "
        f"до таблицы {target}. Не используй SQLite. Покажи только все узлы "
        "по порядку и глубину, а полный результат инструмента — отдельно."
    )

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


def test_live_agent_returns_complete_three_edge_neo4j_path(live_chat_client):
    fixture_path = _neo4j_path(3)
    source, target = fixture_path[0], fixture_path[-1]
    expected_paths = _neo4j_paths_between(source, target, 3)
    if not expected_paths:
        pytest.skip("Neo4j has no stable three-edge path for the selected endpoints")
    result = _chat(
        live_chat_client,
        f"Через Neo4j найди полный точный направленный путь длины 3 от таблицы "
        f"{source} до таблицы {target}. Не используй SQLite. В ответе покажи "
        "только все четыре узла по порядку и глубину. Полный результат со всеми "
        "шагами пути покажи отдельно в scrollable UI."
    )

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

    result = _chat(
        live_chat_client,
        f"Через SQLite выполни ровно этот read-only запрос: {query}. "
        "Перечисли все 4 точные пары source_table.source_field -> "
        "target_table.target_field, не разделяя связанные стороны на отдельные "
        "списки. Полный табличный результат покажи отдельно в scrollable UI."
    )

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

    result = _chat(
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

    result = _chat(
        live_chat_client,
        f"Выполни два отдельных зависимых шага. Сначала через SQLite в "
        f"s2t_transformations для source_table = {immediate_source} найди "
        "target_table с максимальным числом строк и верни её имя и count. Затем "
        f"через Neo4j, используя найденное имя target_table, построй полный "
        f"направленный путь от {root_source} до неё. В итоговом ответе дай count, "
        "все узлы пути по порядку и глубину. Полные результаты обоих шагов "
        "покажи отдельно в scrollable UI.",
    )

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
