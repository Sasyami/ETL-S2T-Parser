import json

import pytest
from unittest.mock import patch

import storage.database as db_storage
from storage.database import get_db_connection, init_db


def test_sqlite_schema_cheatsheet_is_generated_from_db_storage():
    from agents.tools import get_sqlite_schema_cheatsheet

    text = get_sqlite_schema_cheatsheet()
    for table_name, columns in db_storage.STORAGE_SCHEMA_COLUMNS.items():
        assert f"`{table_name}`" in text
        for column_name in columns:
            assert f"`{column_name}`" in text
    assert "сгенерирован из `storage/database.py`" in text
    assert "Внутренние таблицы упоминай" not in text


@pytest.fixture(autouse=True)
def _temp_db_path(tmp_path):
    original = db_storage.DB_PATH
    db_storage.DB_PATH = str(tmp_path / "tools_test.db")
    init_db()
    yield
    db_storage.DB_PATH = original


def test_run_sql_select():
    from agents.tools import run_sql

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        (1, "f.xlsx", "2024-01-01"),
    )
    conn.commit()
    conn.close()

    result = run_sql.invoke(
        {"query": "SELECT file_id, filename FROM files WHERE file_id = 1"}
    )
    assert result["rows"] == [{"file_id": 1, "filename": "f.xlsx"}]
    assert result["returned_rows"] == 1
    assert result["truncated"] is False


def test_run_sql_export_csv_writes_full_result(tmp_path, monkeypatch):
    import agents.tools.sql as sql_tools
    from agents.tools import run_sql

    monkeypatch.setattr(sql_tools, "SQL_EXPORT_DIR", tmp_path / "exports")

    conn = get_db_connection()
    conn.executemany(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        [
            (11, "one.xlsx", "2026-01-01"),
            (12, "two.xlsx", "2026-01-02"),
        ],
    )
    conn.commit()
    conn.close()

    out = run_sql.invoke(
        {
            "query": "SELECT file_id, filename FROM files ORDER BY file_id",
            "export_csv": True,
            "preview_limit": 1,
        }
    )

    assert out["row_count"] == 2
    assert out["columns"] == ["file_id", "filename"]
    assert out["preview_rows"] == [{"file_id": 11, "filename": "one.xlsx"}]
    assert out["csv_url"].startswith("/exports/sql/sql_result_")
    csv_path = tmp_path / "exports" / out["csv_filename"]
    assert out["csv_path"] == str(csv_path)
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8-sig")
    assert "11,one.xlsx" in csv_text
    assert "12,two.xlsx" in csv_text


def test_run_sql_invalid_returns_error_dict():
    from agents.tools import run_sql

    out = run_sql.invoke({"query": "NOT A VALID STMT"})
    assert isinstance(out, dict)
    assert "error" in out


def test_run_sql_returns_sqlite_error_message():
    from agents.tools import run_sql

    query = "SELECT missing_column FROM files"
    out = run_sql.invoke({"query": query})

    assert out == {
        "error": "SQL query failed",
        "error_message": "no such column: missing_column",
        "query": query,
    }


def test_list_additional_objects_uses_exact_filters_and_preserves_duplicates():
    from agents.tools import list_additional_objects

    conn = get_db_connection()
    conn.executemany(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        [
            (10, "one.xlsx", "2026-01-01"),
            (11, "two.xlsx", "2026-01-02"),
        ],
    )
    long_sql = "SELECT order_id, payload FROM raw.orders " + "-- full sql\n" * 50
    conn.executemany(
        """
        INSERT INTO additional_objects
        (id, file_id, sheet_name, row_num, name, sql)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (100, 10, "Additional objects", 7, "mart.orders", long_sql),
            (101, 10, "Additional objects", 8, "mart.orders", long_sql),
            (102, 11, "Additional objects", 9, "mart.other", "SELECT 1"),
        ],
    )
    conn.commit()
    conn.close()

    result = list_additional_objects.invoke(
        {"file_id": 10, "name": " MART.ORDERS ", "limit": 10}
    )

    assert result["filters"] == {"file_id": 10, "name": "MART.ORDERS"}
    assert result["total_matches"] == 2
    assert result["returned_rows"] == 2
    assert result["truncated"] is False
    assert [row["additional_object_id"] for row in result["rows"]] == [100, 101]
    assert all(row["filename"] == "one.xlsx" for row in result["rows"])
    assert all(row["sql"] == long_sql for row in result["rows"])


def test_search_additional_objects_searches_full_sql_inside_file():
    from agents.tools import search_additional_objects

    conn = get_db_connection()
    conn.executemany(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        [
            (20, "one.xlsx", "2026-01-01"),
            (21, "two.xlsx", "2026-01-02"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO additional_objects
        (id, file_id, sheet_name, row_num, name, sql)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                200,
                20,
                "Additional objects",
                3,
                "mart.payments",
                "SELECT * FROM raw.payments LEFT JOIN raw.clients ON true",
            ),
            (
                201,
                21,
                "Additional objects",
                4,
                "mart.orders",
                "SELECT * FROM raw.orders LEFT JOIN raw.clients ON true",
            ),
        ],
    )
    conn.commit()
    conn.close()

    result = search_additional_objects.invoke(
        {
            "needle": "left join",
            "file_id": 20,
            "search_in": "sql",
            "limit": 10,
        }
    )

    assert result["query"] == "left join"
    assert result["file_id"] == 20
    assert result["searched_columns"] == ["sql"]
    assert result["total_matches"] == 1
    assert result["rows"][0]["additional_object_id"] == 200
    assert "LEFT JOIN" in result["rows"][0]["sql"]
    assert search_additional_objects.invoke({"needle": " "})["error"] == (
        "needle must be non-empty"
    )


def test_run_sql_rejects_write_queries():
    from agents.tools import run_sql

    out = run_sql.invoke(
        {"query": "INSERT INTO files (file_id, filename) VALUES ('bad', 'bad.xlsx')"}
    )
    assert isinstance(out, dict)
    assert "Only SELECT, WITH and EXPLAIN QUERY PLAN are allowed" in out["error"]

    result = run_sql.invoke(
        {"query": "SELECT file_id FROM files WHERE file_id = 'bad'"}
    )
    assert result["rows"] == []


def test_saved_sqlite_tool_result_is_queryable_with_bound_schema():
    from langchain_core.messages import ToolMessage

    from agents.tools import get_tools
    from agents.tools.saved_results import (
        bind_saved_result_schemas,
        persist_sqlite_tool_message,
        query_saved_result,
        saved_result_store_scope,
    )

    with saved_result_store_scope() as store:
        message = ToolMessage(
            content=json.dumps(
                {
                    "total": 3,
                    "rows": [
                        {"target_table": "t_a", "row_count": 2},
                        {"target_table": "t_b", "row_count": 5},
                    ],
                },
                ensure_ascii=False,
            ),
            tool_call_id="call-s2t",
            name="list_s2t_transformations",
        )

        enriched = persist_sqlite_tool_message(message)
        payload = json.loads(enriched.content)
        saved = payload["saved_result"]
        result_ref = saved["result_ref"]

        assert saved["source_tool"] == "list_s2t_transformations"
        assert saved["row_count"] == 2
        assert saved["source_total"] == 3
        assert saved["truncated"] is True
        assert saved["columns"] == [
            {"name": "target_table", "sqlite_type": "TEXT"},
            {"name": "row_count", "sqlite_type": "INTEGER"},
        ]
        assert store.descriptor(result_ref) is not None

        bound_tools = bind_saved_result_schemas(
            get_tools(),
            f"Посчитай строки в {result_ref}",
        )
        bound_query_tool = next(
            item for item in bound_tools if item.name == "query_saved_result"
        )
        assert result_ref in bound_query_tool.description
        assert '"target_table" TEXT' in bound_query_tool.description
        assert "truncated=true" in bound_query_tool.description

        queried = query_saved_result.invoke(
            {
                "result_ref": result_ref,
                "query": (
                    "SELECT target_table, row_count FROM result "
                    "WHERE row_count >= 5"
                ),
            }
        )
        assert queried["rows"] == [
            {"target_table": "t_b", "row_count": 5}
        ]
        assert queried["input_truncated"] is True

        forbidden = query_saved_result.invoke(
            {
                "result_ref": result_ref,
                "query": "SELECT name FROM sqlite_master",
            }
        )
        assert forbidden["error"] == "Saved result SQL query failed"


def test_query_saved_result_is_scoped_and_read_only():
    from agents.tools.saved_results import (
        query_saved_result,
        saved_result_store_scope,
    )

    with saved_result_store_scope() as store:
        descriptor = store.save_payload(
            source_tool="run_sql",
            payload={"columns": ["value"], "rows": [{"value": 1}]},
        )
        assert descriptor is not None
        rejected = query_saved_result.invoke(
            {
                "result_ref": descriptor.result_ref,
                "query": "DELETE FROM result",
            }
        )
        assert "Only SELECT" in rejected["error"]

    missing_scope = query_saved_result.invoke(
        {
            "result_ref": descriptor.result_ref,
            "query": "SELECT * FROM result",
        }
    )
    assert missing_scope["error"] == "No active saved-result store"


def test_previous_result_is_lazy_and_scoped_to_coordinator_run():
    from agents.contracts import WORKER_PREVIOUS_RESULTS_MARKER
    from agents.tools import get_tools
    from agents.tools.saved_results import (
        bind_saved_result_schemas,
        read_previous_result,
        saved_result_store_scope,
    )

    with saved_result_store_scope() as store:
        descriptor = store.save_payload(
            source_tool="run_sql",
            source_tool_call_id="call-first",
            payload={
                "columns": ["target_table", "row_count"],
                "rows": [{"target_table": "t_example", "row_count": 42}],
            },
        )
        assert descriptor is not None
        reference = store.register_previous_result(
            source_tool="run_sql",
            source_tool_call_id="call-first",
            content=json.dumps(
                {"rows": [{"target_table": "t_example", "row_count": 42}]}
            ),
            description="run_sql: t_example содержит 42 строки.",
            dataset_ref=descriptor.result_ref,
        )
        handoff = {
            "previous_results": [reference.model_dump(mode="json")]
        }
        task = (
            "Проверь точный прошлый результат."
            + WORKER_PREVIOUS_RESULTS_MARKER
            + "\n"
            + json.dumps(handoff, ensure_ascii=False)
        )

        resolved = read_previous_result.invoke(
            {"result_id": reference.result_id}
        )
        assert resolved["source_tool"] == "run_sql"
        assert resolved["result"]["rows"][0]["target_table"] == "t_example"

        bound_tools = bind_saved_result_schemas(get_tools(), task)
        bound_names = {item.name for item in bound_tools}
        assert "read_previous_result" in bound_names
        assert "query_saved_result" in bound_names
        bound_query_tool = next(
            item for item in bound_tools if item.name == "query_saved_result"
        )
        assert descriptor.result_ref in bound_query_tool.description
        assert '"target_table" TEXT' in bound_query_tool.description

        unrelated_tools = bind_saved_result_schemas(
            get_tools(),
            "Прочитай новые данные без прошлых результатов.",
        )
        unrelated_names = {item.name for item in unrelated_tools}
        assert "read_previous_result" not in unrelated_names
        assert "query_saved_result" not in unrelated_names

    missing = read_previous_result.invoke({"result_id": reference.result_id})
    assert missing["error"] == "No active saved-result store"


def test_search_excel_values_and_restore_source_row():
    from agents.tools import get_excel_row, search_excel_values

    headers = json.dumps(
        [
            {"index": 0, "flat": "Код клиента", "path": ["Код клиента"]},
            {"index": 1, "flat": "Описание", "path": ["Описание"]},
        ],
        ensure_ascii=False,
    )
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        (5, "values.xlsx", "2026-01-01"),
    )
    conn.execute(
        """
        INSERT INTO file_sheet_headers
        (file_id, sheet_name, skipped, header_start_row,
         header_rows_count, nested_structure, columns_count, headers_json)
        VALUES (?, ?, 0, 1, 1, 0, 2, ?)
        """,
        (5, "Клиенты", headers),
    )
    conn.executemany(
        """
        INSERT INTO data (file_id, table_name, row_num, column_id, value)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (5, "Клиенты", 3, 1, "КЛИЕНТ-42"),
            (5, "Клиенты", 3, 2, "Кредитный договор"),
        ],
    )
    conn.commit()
    conn.close()

    found = search_excel_values.invoke(
        {
            "needle": "клиент-42",
            "sheet_name": "клиенты",
            "column_name": "код клиента",
        }
    )
    assert found["total_matches"] == 1
    assert found["rows"][0]["column_name"] == "Код клиента"
    assert found["rows"][0]["excel_row_number"] == 6

    source_row = get_excel_row.invoke(
        {"file_id": 5, "sheet_name": "Клиенты", "row_num": 3}
    )
    assert source_row["filename"] == "values.xlsx"
    assert source_row["excel_row_number"] == 6
    assert source_row["cells"] == [
        {
            "column_id": 1,
            "column_index": 0,
            "column_name": "Код клиента",
            "value": "КЛИЕНТ-42",
        },
        {
            "column_id": 2,
            "column_index": 1,
            "column_name": "Описание",
            "value": "Кредитный договор",
        },
    ]


def test_semantic_search_descriptions_ranks_stored_embeddings(monkeypatch):
    from array import array

    from agents.tools import semantic_search_descriptions
    from services import embeddings

    vector = lambda values: array("f", values).tobytes()
    monkeypatch.setattr(embeddings, "embed_description", lambda text: vector([1.0, 0.0]))
    monkeypatch.setattr(embeddings, "embedding_model_name", lambda: "test-model")

    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO files
        (file_id, filename, upload_time, description, description_embedding)
        VALUES (?, ?, ?, ?, ?)
        """,
        (10, "catalog.xlsx", "2026-01-01", "Общий каталог", vector([0.0, 1.0])),
    )
    conn.execute(
        """
        INSERT INTO source_tables
        (id, file_id, sheet_name, row_num, table_name,
         description, description_embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (11, 10, "Source", 0, "src_contract", "Кредитные договоры", vector([1.0, 0.0])),
    )
    conn.execute(
        """
        INSERT INTO target_tables
        (id, file_id, sheet_name, row_num, table_name,
         description, description_embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (13, 10, "Target", 0, "t_client", "Клиенты", vector([0.8, 0.2])),
    )
    conn.execute(
        """
        INSERT INTO source_columns
        (id, file_id, sheet_name, row_num, table_name, column_name,
         data_type, primary_key, not_null, description, description_embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            15,
            10,
            "Source columns",
            0,
            "src_contract",
            "contract_number",
            "uuid",
            1,
            1,
            "Номер кредитного договора",
            vector([0.9, 0.1]),
        ),
    )
    conn.execute(
        """
        INSERT INTO target_columns
        (id, file_id, sheet_name, row_num, table_name, column_name,
         data_type, primary_key, not_null, description, description_embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            17,
            10,
            "Target columns",
            0,
            "t_client",
            "client_name",
            "text",
            0,
            0,
            "Имя клиента",
            vector([0.2, 0.8]),
        ),
    )
    conn.commit()
    conn.close()

    result = semantic_search_descriptions.invoke(
        {"query": "кредитные договоры", "limit": 3}
    )
    assert result["embedding_model"] == "test-model"
    assert result["total_candidates"] == 5
    assert result["rows"][0]["scope"] == "source_tables"
    assert result["rows"][0]["name"] == "src_contract"
    assert result["rows"][0]["score"] == 1.0

    expected_scopes = {
        "files": {"files"},
        "tables": {"source_tables", "target_tables"},
        "source_tables": {"source_tables"},
        "target_tables": {"target_tables"},
        "columns": {"source_columns", "target_columns"},
        "source_columns": {"source_columns"},
        "target_columns": {"target_columns"},
    }
    for scope, row_scopes in expected_scopes.items():
        scoped = semantic_search_descriptions.invoke(
            {"query": "кредитные договоры", "scope": scope, "limit": 10}
        )
        assert scoped["scope"] == scope
        assert {row["scope"] for row in scoped["rows"]} == row_scopes
        assert scoped["total_candidates"] == len(row_scopes)
    columns = semantic_search_descriptions.invoke(
        {"query": "номер договора", "scope": "source_columns", "limit": 10}
    )
    assert columns["rows"][0]["table_name"] == "src_contract"
    assert columns["rows"][0]["column_name"] == "contract_number"
    subset = semantic_search_descriptions.invoke(
        {
            "query": "номер договора",
            "scope": "source_columns",
            "file_id": 10,
            "table_name": "src_contract",
            "data_type": "uuid",
            "primary_key": True,
            "not_null": True,
            "limit": 10,
        }
    )
    assert subset["total_candidates"] == 1
    assert subset["subset"] == {
        "file_id": 10,
        "table_name": "src_contract",
        "data_type": "uuid",
        "primary_key": True,
        "not_null": True,
    }
    invalid_subset = semantic_search_descriptions.invoke(
        {"query": "договор", "scope": "tables", "primary_key": True}
    )
    assert invalid_subset["error"] == (
        "column subset filters require a column scope"
    )


def test_column_catalog_tools_support_exact_and_substring_subsets():
    from agents.tools import list_column_catalog, search_column_catalog

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (30, 'columns.xlsx', '2026-01-01')"
    )
    conn.executemany(
        """
        INSERT INTO source_columns
        (id, file_id, sheet_name, row_num, table_name, column_name, data_type,
         primary_key, not_null, description)
        VALUES (?, 30, 'Source columns', ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                31,
                0,
                "raw.contract",
                "contract_id",
                "uuid",
                1,
                1,
                "Идентификатор кредитного договора",
            ),
            (
                32,
                1,
                "raw.contract",
                "payload",
                "jsonb",
                0,
                0,
                "Тело сообщения",
            ),
        ],
    )
    conn.execute(
        """
        INSERT INTO target_columns
        (id, file_id, sheet_name, row_num, table_name, column_name, data_type,
         primary_key, not_null, description)
        VALUES (33, 30, 'Target columns', 0, 'mart.contract', 'contract_id',
                'uuid', 1, 1, 'Ключ договора')
        """
    )
    conn.commit()
    conn.close()

    exact = list_column_catalog.invoke(
        {
            "scope": "source_columns",
            "file_id": 30,
            "table_name": "RAW.CONTRACT",
            "column_name": "contract_id",
            "primary_key": True,
            "not_null": True,
            "columns": ["record_id", "table_name", "column_name", "data_type"],
        }
    )
    assert exact["total_matches"] == 1
    assert exact["rows"] == [
        {
            "record_id": 31,
            "table_name": "raw.contract",
            "column_name": "contract_id",
            "data_type": "uuid",
        }
    ]

    searched = search_column_catalog.invoke(
        {
            "needle": "договор",
            "scope": "all_tables",
            "file_id": 30,
            "data_type": "uuid",
            "primary_key": True,
        }
    )
    assert searched["query"] == "договор"
    assert searched["total_matches"] == 2
    assert {row["column_role"] for row in searched["rows"]} == {
        "source",
        "target",
    }
    description_search = search_column_catalog.invoke(
        {"needle": "сообщения", "scope": "source_columns"}
    )
    assert description_search["total_matches"] == 1
    assert description_search["rows"][0]["column_name"] == "payload"
    assert search_column_catalog.invoke(
        {"needle": " ", "scope": "all_tables"}
    )["error"] == (
        "needle must be non-empty"
    )
    invalid_columns = list_column_catalog.invoke(
        {
            "scope": "all_tables",
            "columns": ["column_name", "description_embedding"],
        }
    )
    assert "unknown columns" in invalid_columns["error"]


def test_list_column_catalog_uses_explicit_all_tables_scope():
    from agents.tools import list_column_catalog

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) "
        "VALUES (31, 'all-columns.xlsx', '2026-01-01')"
    )
    for table_name, record_id, role_table in (
        ("raw.all_columns", 41, "source_columns"),
        ("mart.all_columns", 42, "target_columns"),
    ):
        conn.execute(
            f"""
            INSERT INTO {role_table}
            (id, file_id, sheet_name, row_num, table_name, column_name,
             data_type, primary_key, not_null, description)
            VALUES (?, 31, 'Columns', 1, ?, 'required_id',
                    'uuid', 1, 1, 'Обязательный идентификатор')
            """,
            (record_id, table_name),
        )
    conn.commit()
    conn.close()

    result = list_column_catalog.invoke(
        {
            "scope": "all_tables",
            "file_id": 31,
            "not_null": True,
        }
    )
    assert "error" not in result
    assert result["scope"] == "all_tables"
    assert result["filters"] == {"file_id": 31, "not_null": True}
    assert {row["column_role"] for row in result["rows"]} == {
        "source",
        "target",
    }


def test_trace_transformation_path_combines_s2t_sql_and_additional_objects():
    from agents.tools import trace_transformation_path

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        (20, "path.xlsx", "2026-01-01"),
    )
    conn.execute(
        """
        INSERT INTO additional_objects
        (id, file_id, sheet_name, row_num, name, sql)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (21, 20, "Additional objects", 1, "view_orders", "SELECT id FROM raw.orders"),
    )
    conn.executemany(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num,
         source_table, source_field, target_table, target_field,
         transformation_rule)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (23, 20, "S2T", 1, "raw.orders", "id", "view_orders", "id", "-"),
            (25, 20, "S2T", 2, "view_orders", "id", "dwh.orders", "id", "SELECT id FROM view_orders"),
            (
                27,
                20,
                "S2T",
                3,
                "raw.agreement",
                "agreement_entityid_uid",
                "dwh.agreement",
                "agr_frame_id",
                "-",
            ),
        ],
    )
    conn.commit()
    conn.close()

    result = trace_transformation_path.invoke(
        {
            "table_name": "raw.orders",
            "column_name": "id",
            "direction": "downstream",
            "include_neo4j": False,
        }
    )
    assert result["returned_paths"] == 1
    path = result["paths"][0]
    assert path["depth"] == 2
    assert [step["transformation"]["kind"] for step in path["steps"]] == [
        "direct",
        "sql",
    ]
    assert path["steps"][1]["additional_objects"][0]["name"] == "view_orders"
    assert result["neo4j_evidence"] == {"included": False, "rows": []}
    assert "[raw.orders.id] --direct--> [view_orders.id]" in result[
        "text_diagram"
    ]
    assert "[view_orders.id] --sql--> [dwh.orders.id]" in result[
        "text_diagram"
    ]
    assert result["mermaid"].startswith("flowchart LR\n")
    assert [edge["transformation_id"] for edge in result["edges"]] == [23, 25]

    qualified = trace_transformation_path.invoke(
        {
            "table_name": "raw.orders",
            "column_name": "raw.orders.id",
            "direction": "downstream",
            "include_neo4j": False,
        }
    )
    assert qualified["column_name"] == "id"
    assert qualified["returned_paths"] == 1

    qualified_table = trace_transformation_path.invoke(
        {
            "table_name": "dwh.orders.id",
            "column_name": "id",
            "direction": "upstream",
            "include_neo4j": False,
        }
    )
    assert qualified_table["table_name"] == "dwh.orders"
    assert qualified_table["column_name"] == "id"
    assert qualified_table["returned_paths"] == 1
    assert "[raw.orders.id] --direct--> [view_orders.id]" in qualified_table[
        "text_diagram"
    ]
    assert "[view_orders.id] --sql--> [dwh.orders.id]" in qualified_table[
        "text_diagram"
    ]

    combined = trace_transformation_path.invoke(
        {
            "table_name": "dwh.agreement",
            "column_name": "agreement_entityid_uid",
            "direction": "both",
            "include_neo4j": False,
        }
    )
    assert combined["returned_paths"] == 1
    assert combined["paths"][0]["start"] == {
        "file_id": 20,
        "table": "raw.agreement",
        "column": "agreement_entityid_uid",
        "layer": None,
    }
    assert combined["paths"][0]["steps"][0]["transformation_id"] == 27
    assert combined["paths"][0]["steps"][0]["match_direction"] == "downstream"


def test_list_files_empty():
    from agents.tools import list_files

    assert list_files.invoke({}) == []


def test_list_files_returns_catalog_with_real_file_id():
    from agents.tools import list_files

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time, summary, description) VALUES (?, ?, ?, ?, ?)",
        (21, "summary.xlsx", "2026-01-01", "Business summary", "Short description"),
    )
    conn.execute(
        """INSERT INTO file_sheet_headers
        (file_id, sheet_name, skipped, header_start_row,
         header_rows_count, nested_structure, columns_count, headers_json)
        VALUES (?, ?, 0, 0, 1, 0, 0, '[]')""",
        (21, "Лист, который не должен попасть в каталог"),
    )
    conn.commit()
    conn.close()

    result = list_files.invoke({})
    assert result == [
        {
            "file_id": 21,
            "filename": "summary.xlsx",
            "description": "Short description",
            "upload_time": "2026-01-01",
        }
    ]
    assert "Лист, который не должен попасть в каталог" not in str(result)


def test_resolve_file_uses_exact_filename_case_insensitively():
    from agents.tools import resolve_file

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        (23, "Mapping.xlsx", "2026-01-01"),
    )
    conn.commit()
    conn.close()

    assert resolve_file.invoke({"filename": "mapping.xlsx"}) == {
        "file_id": 23,
        "filename": "Mapping.xlsx",
        "upload_time": "2026-01-01",
    }
    assert (
        resolve_file.invoke({"filename": "missing.xlsx"})["error"]
        == "Uploaded file not found"
    )


def test_registry_selects_exact_tools_by_name_and_allows_empty_selection():
    from agents.tools import get_tools_for_names

    selected = get_tools_for_names(("run_sql", "visualize_sql_lineage"))
    assert {tool.name for tool in selected} == {
        "run_sql",
        "visualize_sql_lineage",
    }

    assert get_tools_for_names(()) == ()

    with pytest.raises(ValueError, match="unknown"):
        get_tools_for_names(("unknown",))


def test_registered_tools_expose_annotation_derived_argument_schemas():
    from agents.tools import get_tools_by_name

    tools = get_tools_by_name()
    sql_schema = tools["run_sql"].args_schema.model_json_schema()
    assert sql_schema["required"] == ["query"]
    assert set(sql_schema["properties"]) == {
        "query",
        "export_csv",
        "preview_limit",
    }
    assert sql_schema["properties"]["export_csv"]["type"] == "boolean"
    assert sql_schema["properties"]["preview_limit"]["type"] == "integer"

    for tool_name in (
        "parse_sql_column_lineage",
        "parse_sql_table_lineage",
    ):
        lineage_parse_schema = tools[tool_name].args_schema.model_json_schema()
        assert lineage_parse_schema["required"] == ["query"]
        assert set(lineage_parse_schema["properties"]) == {"query", "dialect"}

    assert tools["search_excel_values"].args_schema.model_json_schema()["required"] == [
        "needle"
    ]
    assert tools["get_excel_row"].args_schema.model_json_schema()["required"] == [
        "file_id",
        "sheet_name",
        "row_num",
    ]
    assert tools[
        "semantic_search_descriptions"
    ].args_schema.model_json_schema()["required"] == ["query"]
    semantic_schema = tools[
        "semantic_search_descriptions"
    ].args_schema.model_json_schema()
    assert semantic_schema["properties"]["scope"]["enum"] == [
        "all",
        "files",
        "tables",
        "source_tables",
        "target_tables",
        "columns",
        "source_columns",
        "target_columns",
    ]
    assert set(semantic_schema["properties"]) == {
        "query",
        "scope",
        "limit",
        "file_id",
        "table_name",
        "column_name",
        "data_type",
        "primary_key",
        "not_null",
    }
    assert tools[
        "search_column_catalog"
    ].args_schema.model_json_schema()["required"] == ["needle", "scope"]
    column_list_schema = tools[
        "list_column_catalog"
    ].args_schema.model_json_schema()
    assert column_list_schema["required"] == ["scope"]
    assert column_list_schema["properties"]["scope"]["enum"] == [
        "all_tables",
        "source_columns",
        "target_columns",
    ]
    assert tools[
        "trace_transformation_path"
    ].args_schema.model_json_schema()["required"] == ["table_name"]
    path_description = tools["trace_transformation_path"].description
    assert "source_table + source_field" in path_description
    assert "target_table + target_field" in path_description
    assert "search_s2t_transformations" in path_description
    assert "text_diagram" in path_description
    assert "table_name.column_name" in path_description
    assert "Не составляй table_name из" in path_description
    assert "Производный SQL-анализ" not in path_description
    s2t_list_description = tools["list_s2t_transformations"].description
    assert "точными ролевыми фильтрами" in s2t_list_description
    assert "например только transformation_rule" in s2t_list_description
    assert "visualize_transformation_path" not in tools
    assert tools[
        "visualize_s2t_table_graph"
    ].args_schema.model_json_schema()["properties"] == {}

    cypher_schema = tools["run_cypher"].args_schema.model_json_schema()
    assert cypher_schema["required"] == ["query"]
    assert set(cypher_schema["properties"]) == {
        "query",
        "parameters",
        "limit",
    }
    assert cypher_schema["properties"]["limit"]["type"] == "integer"

    files_schema = tools["list_files"].args_schema.model_json_schema()
    assert files_schema["properties"] == {}
    assert tools["list_files"].return_direct is False
    assert tools["list_files"].invoke({}) == []

    summary_schema = tools["summarize_s2t_tables"].args_schema.model_json_schema()
    assert summary_schema["properties"]["group_by"]["enum"] == ["source", "target"]
    assert set(summary_schema["properties"]) == {
        "group_by",
        "min_related_tables",
        "limit",
    }
    list_s2t_schema = tools[
        "list_s2t_transformations"
    ].args_schema.model_json_schema()
    assert set(list_s2t_schema["properties"]) == {
        "limit",
        "columns",
        "target_table",
        "source_table",
        "target_field",
        "source_field",
    }
    columns_schema = list_s2t_schema["properties"]["columns"]
    assert columns_schema["default"] is None
    assert columns_schema["anyOf"][0] == {
        "type": "array",
        "items": {"type": "string"},
    }
    search_s2t_schema = tools[
        "search_s2t_transformations"
    ].args_schema.model_json_schema()
    assert set(search_s2t_schema["properties"]) == {"needle", "limit"}
    description_schema = tools["summarize_table_descriptions"].args_schema.model_json_schema()
    assert description_schema["required"] == ["table_name"]
    assert set(description_schema["properties"]) == {"table_name", "file_id", "limit"}
    file_description_schema = tools["get_file_description"].args_schema.model_json_schema()
    assert "Явный числовой идентификатор загрузки" in file_description_schema[
        "properties"
    ]["file_id"]["description"]

    plan_schema = tools["show_plan"].args_schema.model_json_schema()
    assert plan_schema["required"] == ["done", "to_do"]
    assert set(plan_schema["properties"]) == {"done", "to_do"}

    lineage_schema = tools["trace_neo4j_lineage"].args_schema.model_json_schema()
    assert lineage_schema["required"] == ["column_reference"]
    assert set(lineage_schema["properties"]) == {
        "column_reference",
        "file_id",
        "direction",
        "max_depth",
        "limit",
    }
    table_names_schema = tools[
        "list_s2t_table_names"
    ].args_schema.model_json_schema()
    assert table_names_schema["properties"]["set_operation"]["enum"] == [
        "sources",
        "targets",
        "intersection",
        "source_only",
        "target_only",
        "union",
    ]
    assert set(table_names_schema["properties"]) == {"set_operation", "limit"}
    assert table_names_schema["required"] == ["set_operation", "limit"]
    assert lineage_schema["properties"]["direction"]["enum"] == [
        "upstream",
        "downstream",
        "both",
    ]
    table_lineage_schema = tools[
        "trace_neo4j_table_lineage"
    ].args_schema.model_json_schema()
    assert table_lineage_schema["required"] == ["table_name"]
    assert set(table_lineage_schema["properties"]) == {
        "table_name",
        "file_id",
        "direction",
        "limit",
    }
    assert table_lineage_schema["properties"]["direction"]["enum"] == [
        "upstream",
        "downstream",
        "both",
    ]
    table_path_schema = tools[
        "trace_neo4j_table_path"
    ].args_schema.model_json_schema()
    assert table_path_schema["required"] == ["source_table", "target_table"]
    assert set(table_path_schema["properties"]) == {
        "source_table",
        "target_table",
        "file_id",
        "depth",
        "max_depth",
        "limit",
    }


@patch("agents.tools.neo4j.execute_neo4j_read")
def test_run_cypher_returns_limited_rows(mock_read):
    from agents.tools import run_cypher

    mock_read.return_value = [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "s"},
        {"source": "extra", "target": "extra"},
    ]
    query = """
        MATCH (source:ETLColumn)-[mapping:TRANSFORMS_TO]->
              (target:ETLColumn)
        WHERE mapping.file_id = $file_id
        RETURN source.name AS source, target.name AS target
    """

    result = run_cypher.invoke(
        {
            "query": query,
            "parameters": {"file_id": 7},
            "limit": 2,
        }
    )

    mock_read.assert_called_once_with(
        query.strip(),
        {"file_id": 7},
        row_limit=3,
    )
    assert result["columns"] == ["source", "target"]
    assert result["rows"] == [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "s"},
    ]
    assert result["returned_rows"] == 2
    assert result["truncated"] is True
    assert result["limit"] == 2


@patch("agents.tools.neo4j.execute_neo4j_read")
def test_run_cypher_decodes_transport_newlines_outside_literals(mock_read):
    from agents.tools import run_cypher

    mock_read.return_value = [{"value": "kept\\ninside"}]

    result = run_cypher.invoke(
        {
            "query": "MATCH (n)\\nWHERE n.name = 'kept\\ninside'\\nRETURN n",
            "parameters": {},
        }
    )

    mock_read.assert_called_once_with(
        "MATCH (n)\nWHERE n.name = 'kept\\ninside'\nRETURN n",
        {},
        row_limit=21,
    )
    assert result["rows"] == [{"value": "kept\\ninside"}]


@patch("agents.tools.neo4j.execute_neo4j_read")
def test_run_cypher_marks_neo4j_service_unavailable(mock_read):
    from neo4j.exceptions import ServiceUnavailable

    from agents.tools import run_cypher

    mock_read.side_effect = ServiceUnavailable(
        "Unable to retrieve routing information"
    )
    result = run_cypher.invoke(
        {"query": "MATCH (n) RETURN n", "limit": 2}
    )

    assert result["backend"] == "neo4j"
    assert result["unavailable"] is True
    assert "Neo4j недоступен" in result["error"]
    assert result["error_type"] == "ServiceUnavailable"


def test_parse_sql_column_lineage_resolves_columns_through_cte_and_join():
    from agents.tools import parse_sql_column_lineage

    result = parse_sql_column_lineage.invoke(
        {
            "query": """
                WITH prepared AS (
                    SELECT o.id, o.amount
                    FROM raw.orders o
                )
                SELECT
                    prepared.id,
                    prepared.amount + f.fee AS total_amount
                FROM prepared
                JOIN raw.fees f ON prepared.id = f.order_id
            """,
            "dialect": "greenplum",
        }
    )

    assert result["statement_type"] == "SELECT"
    assert result["target_table"] is None
    assert result["source_tables"] == ["raw.fees", "raw.orders"]
    lineage_by_target = {
        item["target_column"]: item
        for item in result["column_lineage"]
    }
    assert lineage_by_target["id"]["source_columns"] == [
        {"table": "raw.orders", "column": "id"}
    ]
    assert lineage_by_target["total_amount"]["source_columns"] == [
        {"table": "raw.fees", "column": "fee"},
        {"table": "raw.orders", "column": "amount"},
    ]
    assert lineage_by_target["total_amount"]["unresolved_source_columns"] == []


def test_parse_sql_column_lineage_maps_insert_target_columns_by_position():
    from agents.tools import parse_sql_column_lineage

    result = parse_sql_column_lineage.invoke(
        {
            "query": """
                INSERT INTO dwh.orders (order_id, doubled_amount)
                SELECT source.id, source.amount * 2
                FROM raw.orders source
            """
        }
    )

    assert result["statement_type"] == "INSERT"
    assert result["target_table"] == "dwh.orders"
    assert result["source_tables"] == ["raw.orders"]
    assert [
        item["target_column"]
        for item in result["column_lineage"]
    ] == ["order_id", "doubled_amount"]
    assert result["column_lineage"][1]["source_columns"] == [
        {"table": "raw.orders", "column": "amount"}
    ]


def test_parse_sql_column_lineage_restores_double_escaped_layout():
    from agents.tools import parse_sql_column_lineage

    query = (
        r"SELECT \n"
        r"    product.object_id_uid,\n"
        r"    info.idaccountnumber_uid AS agr_rko_id\n"
        r"FROM $$305stg.s305_0007_product AS product\n"
        r"LEFT JOIN $$305stg.s305_0004_nsoadditionalinfo AS info\n"
        r"    ON product.object_id_uid = info.idaccountnumber_uid;"
    )

    result = parse_sql_column_lineage.invoke({"query": query})

    assert "error" not in result
    assert r"\n" not in result["query"]
    assert result["source_tables"] == [
        "$$305stg.s305_0007_product",
        "$$305stg.s305_0004_nsoadditionalinfo",
    ]


def test_visualize_sql_lineage_preserves_exact_column_edges_in_html(
    tmp_path,
    monkeypatch,
):
    import agents.tools.sql_lineage as sql_lineage_module
    from agents.tools import visualize_sql_lineage

    monkeypatch.setattr(sql_lineage_module, "SQL_LINEAGE_EXPORT_DIR", tmp_path)
    result = visualize_sql_lineage.invoke(
        {
            "query": (
                "CREATE VIEW mart.order_customer AS "
                "SELECT o.id, c.name "
                "FROM raw.orders AS o "
                "JOIN raw.customers AS c ON c.id = o.customer_id"
            ),
            "dialect": "greenplum",
        }
    )

    assert "error" not in result
    assert result["visualization_type"] == "sqlglot_graph_html"
    assert result["visualization_url"].startswith("/exports/sql-lineage/")
    assert "text_diagram" not in result
    assert "mermaid" not in result
    assert result["target_table"] == "mart.order_customer"
    assert result["source_tables"] == ["raw.orders", "raw.customers"]
    lineage_by_target = {
        item["target_column"]: item
        for item in result["column_lineage"]
    }
    assert lineage_by_target["id"]["source_columns"] == [
        {"table": "raw.orders", "column": "id"}
    ]
    assert lineage_by_target["name"]["source_columns"] == [
        {"table": "raw.customers", "column": "name"}
    ]
    assert lineage_by_target["id"]["unresolved_source_columns"] == []
    assert lineage_by_target["name"]["unresolved_source_columns"] == []

    filename = result["visualization_url"].rsplit("/", 1)[-1]
    html = (tmp_path / filename).read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "vis-network" in html
    assert "raw.orders" in html
    assert "raw.customers" in html
    assert "o.id AS id" in html
    assert "c.name AS name" in html


def test_visualize_s2t_table_graph_aggregates_edges_and_writes_artifacts(
    tmp_path,
    monkeypatch,
):
    import agents.tools.s2t_graph as graph_module
    from agents.tools import visualize_s2t_table_graph

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        (80, "graph.xlsx", "2026-01-01"),
    )
    conn.executemany(
        """
        INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num,
         source_table, source_field, target_table, target_field,
         transformation_rule)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                801,
                80,
                "s2t",
                1,
                "stored.orders",
                "id",
                "dwh.orders",
                "id",
                "SELECT o.id, f.fee FROM raw.orders o JOIN raw.fees f ON f.id = o.id",
            ),
            (
                802,
                80,
                "s2t",
                2,
                "stored.orders",
                "amount",
                "dwh.orders",
                "amount",
                "SELECT o.id, f.fee FROM raw.orders o JOIN raw.fees f ON f.id = o.id",
            ),
            (
                803,
                80,
                "s2t",
                3,
                "raw.direct",
                "code",
                "dwh.direct",
                "code",
                "-",
            ),
        ],
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(graph_module, "S2T_TABLE_GRAPH_EXPORT_DIR", tmp_path)
    result = visualize_s2t_table_graph.invoke({})

    assert "error" not in result
    assert result["scope"] == "global"
    assert result["rows_analyzed"] == 3
    assert result["edge_count"] == 3
    assert result["visualization_url"].startswith("/exports/s2t-graphs/")
    assert result["data_url"].endswith(".json")

    html_name = result["visualization_url"].rsplit("/", 1)[-1]
    json_name = result["data_url"].rsplit("/", 1)[-1]
    html = (tmp_path / html_name).read_text(encoding="utf-8")
    details = json.loads((tmp_path / json_name).read_text(encoding="utf-8"))

    assert "vis-network" in html
    assert "raw.orders" in html
    assert "raw.fees" in html
    assert "raw.direct" in html
    edges = {
        (edge["source_table"], edge["target_table"]): edge
        for edge in details["edges"]
    }
    assert edges[("raw.orders", "dwh.orders")]["mapping_count"] == 2
    assert edges[("raw.fees", "dwh.orders")]["mapping_count"] == 2
    assert edges[("raw.direct", "dwh.direct")]["mapping_count"] == 1


def test_parse_sql_table_lineage_returns_only_physical_table_edges():
    from agents.tools import parse_sql_table_lineage

    result = parse_sql_table_lineage.invoke(
        {
            "query": """
                INSERT INTO dwh.order_totals (order_id, total_amount)
                WITH prepared AS (
                    SELECT o.id, o.amount
                    FROM raw.orders o
                )
                SELECT prepared.id, prepared.amount + f.fee
                FROM prepared
                JOIN raw.fees f ON prepared.id = f.order_id
            """
        }
    )

    assert result["target_table"] == "dwh.order_totals"
    assert result["source_tables"] == ["raw.fees", "raw.orders"]
    assert result["table_lineage"] == [
        {
            "source_table": "raw.fees",
            "target_table": "dwh.order_totals",
        },
        {
            "source_table": "raw.orders",
            "target_table": "dwh.order_totals",
        },
    ]
    assert "column_lineage" not in result


@pytest.mark.parametrize(
    "query, expected_error",
    [
        ("", "query must be non-empty"),
        ("SELECT 1; SELECT 2", "Exactly one SQL statement is allowed"),
        (
            "UPDATE target SET value = 1",
            "Only SELECT, INSERT ... SELECT and "
            "CREATE TABLE/VIEW ... AS SELECT are supported",
        ),
    ],
)
@pytest.mark.parametrize(
    "tool_name",
    ["parse_sql_column_lineage", "parse_sql_table_lineage"],
)
def test_sql_lineage_tools_reject_unsupported_input(
    query,
    expected_error,
    tool_name,
):
    from agents.tools import get_tools_by_name

    result = get_tools_by_name()[tool_name].invoke({"query": query})

    assert result["error"] == expected_error


@pytest.mark.parametrize(
    "query, expected_error",
    [
        (
            "MATCH (node) DELETE node",
            "Mutating or procedural Cypher is not allowed: DELETE",
        ),
        (
            "MATCH (node) SET node.value = 1 RETURN node",
            "Mutating or procedural Cypher is not allowed: SET",
        ),
        (
            "CALL db.labels()",
            "Only MATCH, OPTIONAL MATCH, WITH, UNWIND, RETURN, SHOW, "
            "EXPLAIN and PROFILE queries are allowed",
        ),
        (
            "MATCH (node) RETURN node; MATCH (other) RETURN other",
            "Exactly one Cypher statement is allowed",
        ),
    ],
)
@patch("agents.tools.neo4j.execute_neo4j_read")
def test_run_cypher_rejects_unsafe_queries(
    mock_read,
    query,
    expected_error,
):
    from agents.tools import run_cypher

    result = run_cypher.invoke({"query": query})

    mock_read.assert_not_called()
    assert result["error"] == expected_error
    assert result["query"] == query


@patch("agents.tools.neo4j.execute_neo4j_read")
def test_run_cypher_ignores_keywords_inside_literals_comments_and_identifiers(
    mock_read,
):
    from agents.tools import run_cypher

    mock_read.return_value = [{"value": "CREATE SET DELETE CALL"}]
    query = """
        // DELETE node
        MATCH (node:`CREATE`)
        WHERE node.rule = 'CREATE SET DELETE CALL;'
        RETURN node.rule AS `SET`
    """

    result = run_cypher.invoke({"query": query})

    mock_read.assert_called_once_with(
        query.strip(),
        {},
        row_limit=21,
    )
    assert result["rows"] == [{"value": "CREATE SET DELETE CALL"}]
    assert result["truncated"] is False


@patch("agents.tools.neo4j.execute_neo4j_read")
def test_trace_neo4j_lineage_uses_exact_names_and_scope(mock_read):
    from agents.tools import trace_neo4j_lineage

    mock_read.return_value = [
        {
            "state_index": 0,
            "file_id": 7,
            "transformation_id": 91,
            "source_table": "a_source",
            "source_layer": "B",
            "source_field": "client_id",
            "target_table": "b_target",
            "target_layer": "T",
            "target_field": "client_id",
            "matched_source_field": "client_id",
            "matched_target_field": "client_id",
            "match_direction": "downstream",
        }
    ]

    result = trace_neo4j_lineage.invoke(
        {
            "column_reference": "a_source.client_id",
            "file_id": 7,
            "direction": "downstream",
            "limit": 250,
        }
    )

    query, parameters = mock_read.call_args.args
    assert "MATCH (source:ETLProjection:ETLColumn)" in query
    assert "[mapping:TRANSFORMS_TO]" in query
    assert "source.table_name = state.table_name" in query
    assert "source.name = state.column_name" in query
    assert "target.table_name = state.table_name" in query
    assert "target.name = state.column_name" in query
    assert "MATCH (source:ETLProjection:ETLTable)" not in query
    assert "[mapping:TABLE_TRANSFORMS_TO]" not in query
    assert "[:COVERED_BY]" in query
    assert "[:EXPANDS_TO]" in query
    assert query.count("{name: '*'}") == 2
    assert "source.name = target.name" in query
    assert "transformation_rule" not in query
    assert "sheet_name" not in query
    assert parameters == {
        "states": [
            {
                "state_index": 0,
                "table_name": "a_source",
                "column_name": "client_id",
                "file_id": 7,
            }
        ],
        "direction": "downstream",
    }
    assert mock_read.call_args.kwargs == {"row_limit": 101}
    assert result["returned_rows"] == 1
    assert result["returned_paths"] == 1
    assert result["paths"][0]["depth"] == 1
    assert result["direction"] == "downstream"
    assert "wildcard_passthrough" not in result["rows"][0]
    assert "matched_source_field" not in result["rows"][0]
    assert "matched_target_field" not in result["rows"][0]


@patch("agents.tools.neo4j.execute_neo4j_read")
def test_trace_neo4j_lineage_splits_qualified_column_reference(mock_read):
    from agents.tools import trace_neo4j_lineage

    mock_read.return_value = []

    result = trace_neo4j_lineage.invoke(
        {
            "column_reference": "schema.layer.t_bus_srv.bus_srv_id",
            "direction": "upstream",
        }
    )

    parameters = mock_read.call_args.args[1]
    assert parameters["states"][0]["table_name"] == "schema.layer.t_bus_srv"
    assert parameters["states"][0]["column_name"] == "bus_srv_id"
    assert result["column_reference"] == "schema.layer.t_bus_srv.bus_srv_id"
    assert result["column_name"] == "bus_srv_id"


def test_trace_neo4j_lineage_contract_keeps_qualified_reference_atomic():
    from agents.tools import trace_neo4j_lineage

    schema = trace_neo4j_lineage.args_schema.model_json_schema()
    properties = schema["properties"]

    assert "одним атомарным аргументом" in trace_neo4j_lineage.description
    assert "Дословно скопируй" in trace_neo4j_lineage.description
    assert set(properties) == {
        "column_reference",
        "file_id",
        "direction",
        "max_depth",
        "limit",
    }
    assert "Дословная полная ссылка" in properties["column_reference"][
        "description"
    ]
    assert "Не сокращай" in properties["column_reference"]["description"]


@patch("agents.tools.neo4j.execute_neo4j_read")
def test_trace_neo4j_lineage_resolves_multilevel_wildcard(mock_read):
    from agents.tools import trace_neo4j_lineage

    def lineage_rows(_query, parameters, row_limit=None):
        state = parameters["states"][0]
        if state["table_name"] == "source_table":
            return [
                {
                    "state_index": 0,
                    "file_id": 7,
                    "transformation_id": 91,
                    "source_table": "source_table",
                    "source_layer": None,
                    "source_field": "object_id",
                    "target_table": "view_table",
                    "target_layer": "B",
                    "target_field": "object_id",
                    "matched_source_field": "object_id",
                    "matched_target_field": "object_id",
                    "match_direction": "downstream",
                }
            ]
        if state["table_name"] == "view_table":
            assert state["file_id"] == 7
            return [
                {
                    "state_index": 0,
                    "file_id": 7,
                    "transformation_id": 92,
                    "source_table": "view_table",
                    "source_layer": None,
                    "source_field": "object_id",
                    "target_table": "target_table",
                    "target_layer": "B",
                    "target_field": "object_id",
                    "matched_source_field": "*",
                    "matched_target_field": "*",
                    "match_direction": "downstream",
                }
            ]
        return []

    mock_read.side_effect = lineage_rows

    result = trace_neo4j_lineage.invoke(
        {
            "column_reference": "source_table.object_id",
            "direction": "downstream",
            "max_depth": 3,
        }
    )

    assert [path["depth"] for path in result["paths"]] == [1, 2]
    assert result["paths"][1]["end"] == {
        "table_name": "target_table",
        "column_name": "object_id",
    }
    wildcard_step = result["paths"][1]["steps"][1]
    assert wildcard_step["source_field"] == "*"
    assert wildcard_step["target_field"] == "*"
    assert "wildcard_passthrough" not in wildcard_step
    assert all(
        "wildcard_passthrough" not in row for row in result["rows"]
    )
    assert result["returned_rows"] == 2
    assert result["returned_paths"] == 2
    assert mock_read.call_count == 3


@patch("agents.tools.neo4j.execute_neo4j_read")
def test_trace_neo4j_lineage_allows_sqlglot_scope_depth(mock_read):
    from agents.tools import trace_neo4j_lineage

    mock_read.return_value = []

    result = trace_neo4j_lineage.invoke(
        {
            "column_reference": "source_table.object_id",
            "direction": "downstream",
            "max_depth": 999,
        }
    )

    assert result["max_depth"] == 50
    mock_read.assert_called_once()


@patch("agents.tools.neo4j.execute_neo4j_read")
def test_trace_neo4j_lineage_rejects_empty_column_reference(mock_read):
    from agents.tools import trace_neo4j_lineage

    result = trace_neo4j_lineage.invoke({"column_reference": "   "})

    mock_read.assert_not_called()
    assert result == {
        "error": "column_reference must be non-empty",
        "rows": [],
    }


@patch("agents.tools.neo4j.execute_neo4j_read")
def test_trace_neo4j_table_lineage_returns_sql_queries(mock_read):
    from agents.tools import trace_neo4j_table_lineage

    mock_read.return_value = [
        {
            "file_id": 7,
            "transformation_id": 91,
            "source_table": "a_source",
            "target_table": "b_target",
            "sql_query": "SELECT * FROM a_source",
            "match_direction": "downstream",
        }
    ]

    result = trace_neo4j_table_lineage.invoke(
        {
            "table_name": "a_source",
            "file_id": 7,
            "direction": "downstream",
            "limit": 250,
        }
    )

    query, parameters = mock_read.call_args.args
    assert "MATCH (source:ETLProjection:ETLTable)" in query
    assert "[mapping:TABLE_TRANSFORMS_TO]" in query
    assert "mapping.sql_query AS sql_query" in query
    assert "mapping.wildcard_passthrough" not in query
    assert parameters == {
        "table_name": "a_source",
        "file_id": 7,
        "direction": "downstream",
        "limit": 100,
    }
    assert result["returned_rows"] == 1
    assert result["table_exists"] is True
    assert result["rows"][0]["sql_query"] == "SELECT * FROM a_source"
    assert result["connection_count"] == 1
    assert result["connections"] == [
        {
            "direction": "downstream",
            "source_table": "a_source",
            "source_layer": None,
            "target_table": "b_target",
            "target_layer": None,
            "transformation_count": 1,
            "transformation_ids": [91],
        }
    ]
    serialized = json.dumps(result, ensure_ascii=False)
    assert serialized.index('"connections"') < serialized.index('"rows"')


@patch("agents.tools.neo4j.execute_neo4j_read")
def test_trace_neo4j_table_lineage_distinguishes_existing_node_without_edges(
    mock_read,
):
    from agents.tools import trace_neo4j_table_lineage

    mock_read.side_effect = [[], [{"table_exists": True}]]

    result = trace_neo4j_table_lineage.invoke(
        {"table_name": "isolated_table", "direction": "upstream"}
    )

    assert mock_read.call_count == 2
    existence_query, existence_parameters = mock_read.call_args_list[1].args
    assert "table.file_id = $file_id" in existence_query
    assert existence_parameters == {
        "table_name": "isolated_table",
        "file_id": None,
    }
    assert result["table_exists"] is True
    assert result["returned_rows"] == 0
    assert result["connections"] == []


@patch("agents.tools.neo4j.execute_neo4j_read")
def test_trace_neo4j_table_path_returns_exact_ordered_chain(mock_read):
    from agents.tools import trace_neo4j_table_path

    mock_read.return_value = [
        {
            "table_path": ["a_source", "b_middle", "c_target"],
            "depth": 2,
            "steps": [
                {"transformation_id": 10},
                {"transformation_id": 11},
            ],
        }
    ]

    result = trace_neo4j_table_path.invoke(
        {
            "source_table": "a_source",
            "target_table": "c_target",
            "file_id": 7,
            "depth": 2,
            "max_depth": 40,
            "limit": 250,
        }
    )

    query, parameters = mock_read.call_args.args
    assert "[:TABLE_TRANSFORMS_TO*1..50]" in query
    assert "[node IN nodes(path) | node.name] AS table_path" in query
    assert "length(path) = $depth" in query
    assert "collect(DISTINCT" in query
    assert "mappings: mappings" in query
    assert parameters == {
        "source_table": "a_source",
        "target_table": "c_target",
        "file_id": 7,
        "depth": 2,
        "max_depth": 2,
        "limit": 100,
    }
    assert result["path_count"] == 1
    assert result["chains"] == [
        {
            "depth": 2,
            "source": "a_source",
            "middle": ["b_middle"],
            "target": "c_target",
            "table_path": ["a_source", "b_middle", "c_target"],
        }
    ]
    assert result["paths"][0]["table_path"] == [
        "a_source",
        "b_middle",
        "c_target",
    ]
    serialized = json.dumps(result, ensure_ascii=False)
    assert serialized.index('"chains"') < serialized.index('"paths"')


def test_show_plan_returns_explicit_progress_without_side_effects():
    from agents.tools import show_plan

    assert show_plan.invoke(
        {
            "done": "Найден текущий файл.",
            "to_do": "Прочитать его S2T-трансформации.",
        }
    ) == {
        "done": "Найден текущий файл.",
        "to_do": "Прочитать его S2T-трансформации.",
    }


def test_tool_descriptions_separate_sqlite_and_neo4j_scenarios():
    from agents.tools import get_tools_by_name

    tools = get_tools_by_name()

    column_catalog_description = " ".join(
        tools["list_column_catalog"].description.split()
    )
    assert "``table.column`` всегда разделяй" in column_catalog_description
    column_catalog_schema = tools[
        "list_column_catalog"
    ].args_schema.model_json_schema()
    assert "без префикса table_name" in column_catalog_schema[
        "properties"
    ]["column_name"]["description"]
    assert "source/target table или field" in tools[
        "list_s2t_transformations"
    ].description
    normalized_s2t_list_description = " ".join(
        tools["list_s2t_transformations"].description.split()
    )
    assert "source_table, source_field, target_table" in (
        normalized_s2t_list_description
    )
    assert "только отдельными" in tools[
        "list_s2t_transformations"
    ].description
    assert "Подстрочный поиск этот tool не выполняет" in tools[
        "list_s2t_transformations"
    ].description
    assert "роль искомого значения неизвестна" in tools[
        "search_s2t_transformations"
    ].description
    assert "неполным или неквалифицированным именем" in tools[
        "search_s2t_transformations"
    ].description
    assert "разреши по результату точное полное" in tools[
        "search_s2t_transformations"
    ].description
    assert "используй list_s2t_transformations" in tools[
        "search_s2t_transformations"
    ].description
    assert "точные s2t-строки" in tools["run_sql"].description.casefold()
    assert "это сценарий Neo4j" in tools["run_sql"].description
    assert "не должны содержать" in tools["run_sql"].description
    assert "фильтр по file_id" in tools["run_sql"].description
    assert "Непустой source_table" in tools["run_sql"].description
    assert "TRIM(source_table) <> ''" in tools["run_sql"].description
    assert "не добавляй JOIN" in " ".join(
        tools["run_sql"].description.split()
    )

    assert "только для сложных графовых путей" in tools["run_cypher"].description
    assert "Для обычной таблицы S2T-трансформаций" in tools["run_cypher"].description
    assert "только когда пользователь просит lineage" in tools[
        "trace_neo4j_lineage"
    ].description
    assert "max_depth=1" in tools["trace_neo4j_lineage"].description
    assert "только прямых соседей" in tools[
        "trace_neo4j_lineage"
    ].description
    assert "глубину больше 1" in tools["trace_neo4j_lineage"].description
    assert "узлы ETLTable" in tools["run_cypher"].description
    assert "ETLTable хранится в свойстве name" in tools["run_cypher"].description
    assert "ETLColumn имя таблицы" in tools["run_cypher"].description
    assert "узлы ETLColumn" in tools["trace_neo4j_lineage"].description
    assert "trace_neo4j_table_lineage" in tools[
        "trace_neo4j_lineage"
    ].description
    assert "additional objects" in tools["trace_neo4j_lineage"].description
    assert "trace_transformation_path" in tools[
        "trace_neo4j_lineage"
    ].description
    assert "одним атомарным аргументом" in tools[
        "trace_neo4j_lineage"
    ].description
    assert "разделяй по последней точке" in tools[
        "trace_transformation_path"
    ].description
    assert "additional objects" in tools[
        "trace_transformation_path"
    ].description
    assert "готовую схему" in tools["trace_neo4j_lineage"].description
    assert "не передавай символы" in tools["run_sql"].description
    run_sql_schema = tools["run_sql"].args_schema.model_json_schema()
    assert "JSON-последовательностей" in (
        run_sql_schema["properties"]["query"]["description"]
    )
    assert "sql_query" in tools["trace_neo4j_table_lineage"].description
    assert "не ищет неизвестные таблицы" in tools[
        "trace_neo4j_table_lineage"
    ].description
    assert "двумя известными ETL-таблицами" in tools[
        "trace_neo4j_table_path"
    ].description
    assert "Для компактных списков" in tools[
        "summarize_s2t_tables"
    ].description
    assert "list_s2t_table_names" in tools["run_sql"].description
    assert "UNION ALL с сортировкой" in tools["run_sql"].description
    assert "не доказывает пересечение" in tools["run_sql"].description
    assert "transformation_id из Neo4j соответствует" in tools[
        "run_sql"
    ].description
    assert "s2t_transformations.id" in tools["run_sql"].description
    assert "не требует одной и той же строки" in tools[
        "list_s2t_table_names"
    ].description
    assert "не принимает file_id" in tools[
        "list_s2t_table_names"
    ].description
    column_parser_description = " ".join(
        tools["parse_sql_column_lineage"].description.split()
    )
    assert "lineage" in column_parser_description
    assert "ничего не выполняет" in column_parser_description
    assert "полный SQL уже дословно есть" in column_parser_description
    assert "JOIN/ON, WHERE, GROUP BY" in column_parser_description
    assert "planner должен анализировать такой текст напрямую" in (
        column_parser_description
    )
    assert "только исходные и целевую" in tools[
        "parse_sql_table_lineage"
    ].description
    assert "ничего не выполняет" in tools["parse_sql_table_lineage"].description
    assert "GraphHTML" in tools["visualize_sql_lineage"].description
    assert "visualization_url" in tools["visualize_sql_lineage"].description
    assert "не принимает и не применяет file_id" in tools[
        "visualize_s2t_table_graph"
    ].description
    assert "Не печатай DOT" in tools["visualize_s2t_table_graph"].description
    assert "не подставляет «последний файл»" in tools["resolve_file"].description
    assert "не пиши `FROM t_*`" in tools["run_sql"].description
    assert "мог не быть embedding" in tools[
        "semantic_search_descriptions"
    ].description
    assert "правила ``* -> *`` объединяются в один путь" in tools[
        "trace_neo4j_lineage"
    ].description
    assert "не доказательство отсутствия факта в SQLite" in tools[
        "run_cypher"
    ].description
    assert "Mermaid-код" in tools["trace_transformation_path"].description


def test_read_only_data_tool_contracts_describe_every_argument():
    from agents.tools import get_tools

    for tool in get_tools():
        if tool.name == "show_plan":
            continue
        assert tool.description.strip(), tool.name
        properties = (
            tool.args_schema.model_json_schema().get("properties") or {}
        )
        missing = [
            name
            for name, schema in properties.items()
            if not str(schema.get("description") or "").strip()
        ]
        assert missing == [], f"{tool.name}: {missing}"


def test_get_file_description_uses_cached_value():
    from agents.tools import get_file_description

    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO files (file_id, filename, upload_time, summary, description)
        VALUES (?, ?, ?, ?, ?)
        """,
        (31, "desc.xlsx", "2026-01-01", "Long summary", "Cached description"),
    )
    conn.commit()
    conn.close()

    out = get_file_description.invoke({"file_id": 31})

    assert out["description"] == "Cached description"
    assert out["description_present"] is True
    assert out["file"]["description"] == "Cached description"


def test_get_file_description_reports_missing_without_generation():
    from agents.tools import get_file_description

    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO files (file_id, filename, upload_time, summary, description)
        VALUES (?, ?, ?, ?, ?)
        """,
        (32, "desc_gen.xlsx", "2026-01-02", "Long summary", None),
    )
    conn.commit()
    conn.close()

    out = get_file_description.invoke({"file_id": 32})

    assert out["description"] is None
    assert out["summary"] == "Long summary"
    assert out["missing_description"] is True
    assert out["summary_present"] is True
    assert "явный запрос" in out["hint"]


def test_update_file_description_tool(mock_embeddings):
    from agents.tools import update_file_description

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        (33, "desc_upd.xlsx", "2026-01-03"),
    )
    conn.commit()
    conn.close()

    out = update_file_description.invoke(
        {"file_id": 33, "description": "Approved description"}
    )

    assert out["updated"] is True
    assert out["description"] == "Approved description"

    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT description, description_embedding
        FROM files
        WHERE file_id = ?
        """,
        (33,),
    ).fetchone()
    conn.close()
    assert row["description"] == "Approved description"
    assert row["description_embedding"] == b"embedding:Approved description"


@patch("agents.summarizer_agent.update_file_description_from_user_query")
def test_update_table_info_from_user_query_tool(mock_update):
    from agents.tools import update_table_info_from_user_query

    mock_update.return_value = "Updated from user query"
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        (34, "table_info.xlsx", "2026-01-04"),
    )
    conn.commit()
    conn.close()

    out = update_table_info_from_user_query.invoke(
        {
            "file_id": 34,
            "user_query": "Добавь, что это витрина по кредитным договорам",
        }
    )

    assert out["updated"] is True
    assert out["description"] == "Updated from user query"
    assert out["source"] == "user_query"
    assert out["sequence"] == [
        "verified_files_row",
        "ensured_generated_description",
        "updated_description_from_user_query",
    ]
    mock_update.assert_called_once_with(
        34,
        "Добавь, что это витрина по кредитным договорам",
        save=True,
    )


def test_list_sheets_and_columns_after_store():
    from storage.database import store_excel_data
    from agents.tools import list_columns, list_file_sheet_headers, list_sheets

    sheets = [
        {
            "sheet_name": "Sheet1",
            "skip_reason": None,
            "header": {"start_row": 0, "row_count": 1, "nested": False},
            "columns": ["ColA"],
            "data_rows": [[1]],
        }
    ]
    fh = store_excel_data("x.xlsx", "m", sheets)
    assert list_sheets.invoke({"file_id": fh}) == {
        "file_id": fh,
        "sheet_count": 1,
        "sheets": ["Sheet1"],
    }
    headers = list_file_sheet_headers.invoke({"file_id": fh})
    columns_result = list_columns.invoke(
        {"file_id": fh, "sheet_name": "sheet1"}
    )
    assert columns_result["column_count"] == 1
    assert columns_result["columns"][0]["name"] == "ColA"
    assert len(headers) == 1
    assert headers[0]["sheet_name"] == "Sheet1"
    assert headers[0]["columns_count"] == 1
    assert headers[0]["headers"][0]["flat"] == "ColA"


def test_list_columns_resolves_configured_sheet_group_alias():
    from agents.tools import list_columns
    from storage.database import store_excel_data

    file_id = store_excel_data(
        "aliases.xlsx",
        "m",
        [
            {
                "sheet_name": "pxf2a",
                "skip_reason": None,
                "header": {"start_row": 0, "row_count": 1, "nested": False},
                "columns": ["external_a_table"],
                "data_rows": [["ext_table"]],
            }
        ],
    )

    result = list_columns.invoke(
        {"file_id": file_id, "sheet_name": "pxf_to_a"}
    )

    assert result["sheet_name"] == "pxf2a"
    assert result["columns"][0]["name"] == "external_a_table"


@patch("agents.sheet_group_classifier.invoke_llm_plain_text")
def test_list_sheet_group_classifications_tool_does_not_call_llm(mock_llm):
    from storage.database import store_excel_data
    from agents.tools import list_sheet_group_classifications

    sheets = [
        {
            "sheet_name": "Unknown metadata sheet",
            "skip_reason": None,
            "header": {"start_row": 0, "row_count": 1, "nested": False},
            "columns": ["source column"],
            "data_rows": [["v"]],
        }
    ]
    fh = store_excel_data(
        "unknown.xlsx",
        "m",
        sheets,
    )

    out = list_sheet_group_classifications.invoke({"file_id": fh})

    mock_llm.assert_not_called()
    assert out["subagent"]["use_llm"] is False
    assert out["verification"]["status"] == "warning"
    assert out["verification"]["unmatched_sheets"] == ["Unknown metadata sheet"]


def test_search_s2t_transformations_uses_sql_table():
    from agents.tools import search_s2t_transformations

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        (41, "s2t_search.xlsx", "2026-01-01"),
    )
    conn.execute(
        """INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, target_table, target_field,
         source_table, source_field, transformation_rule)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            42,
            41,
            "s2t",
            0,
            "t_agr_cred",
            "agr_cred_id",
            "B700000025_AGR_CRED",
            "C_ID",
            "c_id",
        ),
    )
    conn.commit()
    conn.close()

    data = search_s2t_transformations.invoke(
        {"needle": "B700000025_AGR_CRED"}
    )
    assert data["searched_table"] == "s2t_transformations"
    assert data["scope"] == "global"
    assert "file_id" not in data
    assert "file" not in data
    assert data["total"] == 1
    assert data["rows"][0]["source_table"] == "B700000025_AGR_CRED"
    assert "table_transformation_sql" not in data["rows"][0]


def test_summarize_s2t_tables_groups_shared_targets_by_source():
    from agents.tools import summarize_s2t_tables

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        (51, "summary.xlsx", "2026-01-01"),
    )
    conn.executemany(
        """INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, target_table, target_field,
         source_table, source_field, transformation_rule)
        VALUES (?, 51, 'S2T', ?, ?, ?, ?, ?, ?)""",
        [
            (53, 1, "t_target_a", "a_id", "src_shared", "id", "direct"),
            (54, 2, "t_target_b", "b_id", "src_shared", "id", "direct"),
            (55, 3, "t_target_a", "a_code", "src_single", "code", ""),
        ],
    )
    conn.commit()
    conn.close()

    result = summarize_s2t_tables.invoke(
        {
            "group_by": "source",
            "min_related_tables": 2,
        }
    )

    assert result["group_by"] == "source"
    assert result["scope"] == "global"
    assert "file_id" not in result
    assert result["group_count"] == 1
    assert result["groups"] == [
        {
            "table_name": "src_shared",
            "layer": None,
            "mapping_count": 2,
            "field_count": 1,
            "related_table_count": 2,
            "mappings_with_rule": 2,
            "related_tables": ["t_target_a", "t_target_b"],
            "rule_coverage": 1.0,
        }
    ]


def test_list_s2t_table_names_supports_role_set_operations():
    from agents.tools import list_s2t_table_names

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        (52, "role_intersection.xlsx", "2026-01-01"),
    )
    conn.executemany(
        """INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, source_table, source_layer,
         source_field, target_table, target_layer, target_field, transformation_rule)
        VALUES (?, 52, 'S2T', ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (56, 1, " shared ", "B", "id", "downstream_a", "T", "id", "direct"),
            (57, 2, "shared", "B", "code", "downstream_b", "T", "code", "direct"),
            (58, 3, "upstream_a", "B", "id", "shared", "T", "id", "direct"),
            (59, 4, "upstream_b", "B", "code", "shared", "T", "code", "direct"),
            (60, 5, "source_only", "B", "id", "downstream_c", "T", "id", "direct"),
            (61, 6, "upstream_c", "B", "id", "target_only", "T", "id", "direct"),
        ],
    )
    conn.commit()
    conn.close()

    expected = {
        "sources": ["shared", "source_only", "upstream_a", "upstream_b", "upstream_c"],
        "targets": ["downstream_a", "downstream_b", "downstream_c", "shared", "target_only"],
        "intersection": ["shared"],
        "source_only": ["source_only", "upstream_a", "upstream_b", "upstream_c"],
        "target_only": ["downstream_a", "downstream_b", "downstream_c", "target_only"],
        "union": [
            "downstream_a",
            "downstream_b",
            "downstream_c",
            "shared",
            "source_only",
            "target_only",
            "upstream_a",
            "upstream_b",
            "upstream_c",
        ],
    }
    for operation, table_names in expected.items():
        result = list_s2t_table_names.invoke(
            {"set_operation": operation, "limit": 100}
        )
        assert result["columns"] == ["table_name"]
        assert [row["table_name"] for row in result["rows"]] == table_names
        assert result["returned_rows"] == len(table_names)
        assert result["set_operation"] == operation
        assert result["scope"] == "global"
        assert result["truncated"] is False

    limited = list_s2t_table_names.invoke(
        {"set_operation": "union", "limit": 2}
    )
    assert [row["table_name"] for row in limited["rows"]] == [
        "downstream_a",
        "downstream_b",
    ]
    assert limited["truncated"] is True



def test_summarize_table_descriptions_combines_roles_and_preserves_duplicates():
    from agents.tools import summarize_table_descriptions

    conn = get_db_connection()
    conn.executemany(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        [
            (71, "current.xlsx", "2026-01-01"),
            (72, "other.xlsx", "2026-01-02"),
        ],
    )
    conn.executemany(
        """INSERT INTO source_tables
        (id, file_id, sheet_name, row_num, table_name, description)
        VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (701, 71, "Sources", 0, "t_shared", "Описание источника"),
            (702, 71, "Sources", 1, "t_shared", "Описание источника"),
        ],
    )
    conn.executemany(
        """INSERT INTO target_tables
        (id, file_id, sheet_name, row_num, table_name, description)
        VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (703, 71, "Targets", 0, " T_SHARED ", "Описание приёмника"),
            (704, 72, "Targets", 0, "t_shared", "Описание другого файла"),
        ],
    )
    conn.commit()
    conn.close()

    result = summarize_table_descriptions.invoke(
        {"table_name": "t_shared", "file_id": 71}
    )

    assert result["searched_tables"] == ["source_tables", "target_tables"]
    assert result["total_matches"] == 3
    assert result["returned_matches"] == 3
    assert result["role_counts"] == {"source": 2, "target": 1}
    assert result["descriptions_present"] == 3
    assert [item["description"] for item in result["combined_descriptions"]] == [
        "Описание источника",
        "Описание источника",
        "Описание приёмника",
    ]
    assert [item["table_role"] for item in result["matches"]] == [
        "source",
        "source",
        "target",
    ]


def test_summarize_table_descriptions_does_not_guess_similar_name():
    from agents.tools import summarize_table_descriptions

    conn = get_db_connection()
    conn.execute(
        """INSERT INTO source_tables
        (id, file_id, sheet_name, row_num, table_name, description)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (705, 71, "Sources", 0, "t_credit", "Кредитная таблица"),
    )
    conn.commit()
    conn.close()

    result = summarize_table_descriptions.invoke({"table_name": "t_credi"})

    assert result["total_matches"] == 0
    assert result["matches"] == []
    assert result["combined_descriptions"] == []


def test_list_s2t_transformations_is_non_terminal_preview():
    from agents.tools import list_s2t_transformations

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        (61, "direct.xlsx", "2026-01-01"),
    )
    conn.executemany(
        """INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, target_table, target_field)
        VALUES (?, 61, 'S2T', ?, 't_target', ?)""",
        [(100 + index, index, f"column_{index}") for index in range(25)],
    )
    conn.commit()
    conn.close()

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO files (file_id, filename, upload_time) VALUES (?, ?, ?)",
        (63, "active-empty.xlsx", "2026-01-02"),
    )
    conn.commit()
    conn.close()

    result = list_s2t_transformations.invoke({"limit": 1000})
    assert result["total"] == 25
    assert result["limit"] == 20
    assert len(result["rows"]) == 20
    assert result["scope"] == "global"
    assert result["columns"] == [
        "row_num",
        "target_field",
        "source_field",
        "target_table",
        "source_table",
        "transformation_rule",
        "source_layer",
        "target_layer",
    ]
    assert "file_id" not in result
    assert "file" not in result


def test_get_s2t_rules_by_ids_maps_lineage_ids_without_sql():
    from agents.tools import get_s2t_rules_by_ids

    conn = get_db_connection()
    conn.executemany(
        """INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, source_table, source_field,
         target_table, target_field, transformation_rule)
        VALUES (?, 61, 'S2T', ?, 'source', 'c_closedate',
                'target', 'c_closedate', ?)""",
        [
            (118, 1, "UNION ALL"),
            (297, 2, "branch.c_closedate"),
        ],
    )
    conn.commit()
    conn.close()

    result = get_s2t_rules_by_ids.invoke(
        {"transformation_ids": [297, 999, 118, 297]}
    )

    assert result["requested_ids"] == [297, 999, 118]
    assert [row["id"] for row in result["rows"]] == [297, 118]
    assert result["rows"][1]["transformation_rule"] == "UNION ALL"
    assert result["missing_ids"] == [999]
    assert result["returned_rows"] == 2


def test_get_s2t_rules_by_ids_rejects_empty_input():
    from agents.tools import get_s2t_rules_by_ids

    result = get_s2t_rules_by_ids.invoke({"transformation_ids": []})

    assert result["error"].startswith("transformation_ids must contain")
    assert result["rows"] == []


def test_list_s2t_table_mapping_keeps_table_roles_unambiguous():
    from agents.tools import list_s2t_table_mapping

    conn = get_db_connection()
    conn.executemany(
        """INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, source_table, source_field,
         target_table, target_field, transformation_rule)
        VALUES (?, 61, 'S2T', ?, ?, ?, ?, ?, ?)""",
        [
            (
                231,
                1,
                "b3050000420005_paymentdetails",
                "object_id_uid",
                "t_optn",
                "optn_id",
                "source.object_id_uid",
            ),
            (
                232,
                2,
                "b3050000420005_paymentdetails",
                "other_id",
                "t_other",
                "other_id",
                "source.other_id",
            ),
        ],
    )
    conn.commit()
    conn.close()

    result = list_s2t_table_mapping.invoke(
        {
            "source_table": "b3050000420005_paymentdetails",
            "target_table": "t_optn",
        }
    )

    assert result["total"] == 1
    assert result["filters"] == {
        "target_table": "t_optn",
        "source_table": "b3050000420005_paymentdetails",
    }
    assert result["rows"][0]["source_field"] == "object_id_uid"
    assert result["rows"][0]["target_field"] == "optn_id"


def test_list_s2t_transformations_selects_requested_columns():
    from agents.tools import list_s2t_transformations

    conn = get_db_connection()
    conn.execute(
        """INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, target_table, transformation_rule)
        VALUES (201, 61, 'S2T', 7, 't_target', 'source.value')"""
    )
    conn.commit()
    conn.close()

    result = list_s2t_transformations.invoke(
        {"limit": 20, "columns": ["transformation_rule"]}
    )

    assert result["columns"] == ["transformation_rule"]
    assert result["rows"] == [{"transformation_rule": "source.value"}]


def test_list_s2t_transformations_filters_exact_target_table():
    from agents.tools import list_s2t_transformations

    conn = get_db_connection()
    conn.executemany(
        """INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, target_table, target_field)
        VALUES (?, 61, 'S2T', ?, ?, ?)""",
        [
            (211, 1, "t_target", "wanted"),
            (212, 2, "t_target_archive", "other"),
        ],
    )
    conn.commit()
    conn.close()

    result = list_s2t_transformations.invoke(
        {"limit": 3, "target_table": "t_target"}
    )

    assert result["total"] == 1
    assert result["filters"] == {"target_table": "t_target"}
    assert result["rows"][0]["target_field"] == "wanted"


def test_list_s2t_transformations_filters_exact_source_target_fields():
    from agents.tools import list_s2t_transformations

    conn = get_db_connection()
    conn.executemany(
        """INSERT INTO s2t_transformations
        (id, file_id, sheet_name, row_num, source_table, source_field,
         target_table, target_field)
        VALUES (?, 61, 'S2T', ?, ?, ?, ?, ?)""",
        [
            (221, 1, "s_exact", "source_id", "t_exact", "target_id"),
            (222, 2, "s_exact", "other_id", "t_exact", "target_id"),
        ],
    )
    conn.commit()
    conn.close()

    result = list_s2t_transformations.invoke(
        {
            "source_table": "s_exact",
            "source_field": "source_id",
            "target_table": "t_exact",
            "target_field": "target_id",
        }
    )

    assert result["total"] == 1
    assert result["rows"][0]["source_field"] == "source_id"
    assert result["filters"] == {
        "target_table": "t_exact",
        "source_table": "s_exact",
        "target_field": "target_id",
        "source_field": "source_id",
    }


def test_list_s2t_transformations_empty_result_is_global_not_file_error():
    from agents.tools import list_s2t_transformations

    result = list_s2t_transformations.invoke({})

    assert result == {
        "scope": "global",
        "total": 0,
        "limit": 20,
        "columns": [
            "row_num",
            "target_field",
            "source_field",
            "target_table",
            "source_table",
            "transformation_rule",
            "source_layer",
            "target_layer",
        ],
        "rows": [],
    }


def test_list_columns_schema_uses_gigachat_compatible_scalar_types():
    from agents.tools import list_columns

    schema = list_columns.args_schema.model_json_schema()

    assert schema["required"] == ["file_id", "sheet_name"]
    assert schema["properties"]["file_id"]["type"] == "integer"
    assert schema["properties"]["sheet_name"]["type"] == "string"
    assert "anyOf" not in schema["properties"]["file_id"]
    assert "anyOf" not in schema["properties"]["sheet_name"]
