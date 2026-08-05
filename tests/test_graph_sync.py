from unittest.mock import MagicMock, patch

from graph_storage.config import Neo4jSettings
from storage.database import get_db_connection


def _insert_graph_source_rows(file_id: int = 501) -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO files
            (file_id, filename, model_used, upload_time, summary, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                "mapping.xlsx",
                "test-model",
                "2026-07-27",
                "Summary",
                "Description",
            ),
        )
        conn.execute(
            """
            INSERT INTO file_sheet_headers
            (file_id, sheet_name, skipped, header_start_row,
             header_rows_count, nested_structure, columns_count,
             headers_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                "S2T",
                0,
                0,
                1,
                0,
                5,
                "[]",
            ),
        )
        conn.execute(
            """
            INSERT INTO source_tables
            (id, file_id, sheet_name, row_num, table_name, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (701, file_id, "S2T", 1, "Exact Source", "Source description"),
        )
        conn.execute(
            """
            INSERT INTO target_tables
            (id, file_id, sheet_name, row_num, table_name, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (801, file_id, "S2T", 2, "Exact Target", "Target description"),
        )
        conn.executemany(
            """
            INSERT INTO s2t_transformations
            (id, file_id, sheet_name, row_num,
             target_table, target_field, source_table, source_field,
             transformation_rule, source_layer, target_layer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    901,
                    file_id,
                    "S2T",
                    3,
                    "Exact Target",
                    "target_id",
                    "Exact Source",
                    "source_id",
                    "SELECT source_id FROM exact_source",
                    "B",
                    "T",
                ),
                (
                    902,
                    file_id,
                    "S2T",
                    4,
                    "Exact Target",
                    "target_id",
                    "Exact Source",
                    "source_id",
                    "SELECT source_id FROM exact_source WHERE source_id IS NOT NULL",
                    "B",
                    "T",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_file_graph_projection_preserves_exact_names_and_duplicate_rows(temp_db):
    from services.graph_sync import _build_file_graph_projection

    _insert_graph_source_rows()

    projection = _build_file_graph_projection(501)

    assert projection["file_id"] == 501
    assert {column["name"] for column in projection["columns"]} == {
        "source_id",
        "target_id",
    }
    assert {
        (column["table_name"], tuple(column["roles"]))
        for column in projection["columns"]
    } == {
        ("Exact Source", ("source",)),
        ("Exact Target", ("target",)),
    }
    assert [row["transformation_id"] for row in projection["lineage"]] == [
        901,
        902,
    ]
    assert {
        (table["name"], tuple(table["roles"]), tuple(table["layers"]))
        for table in projection["tables"]
    } == {
        ("Exact Source", ("source",), ("B",)),
        ("Exact Target", ("target",), ("T",)),
    }
    assert [
        (
            row["transformation_id"],
            row["sql_query"],
            row["wildcard_passthrough"],
        )
        for row in projection["table_lineage"]
    ] == [
        (901, "SELECT source_id FROM exact_source", False),
        (
            902,
            "SELECT source_id FROM exact_source WHERE source_id IS NOT NULL",
            False,
        ),
    ]


def test_file_graph_projection_keeps_wildcard_as_table_rule(temp_db):
    from services.graph_sync import _build_file_graph_projection

    _insert_graph_source_rows()
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO s2t_transformations
            (id, file_id, sheet_name, row_num,
             target_table, target_field, source_table, source_field,
             transformation_rule, source_layer, target_layer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                903,
                501,
                "additional_objects",
                5,
                "Wildcard Target",
                "*",
                "Exact Target",
                "*",
                "SELECT * FROM exact_target",
                None,
                "B",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    projection = _build_file_graph_projection(501)

    assert "*" not in {column["name"] for column in projection["columns"]}
    assert [row["transformation_id"] for row in projection["lineage"]] == [
        901,
        902,
    ]
    wildcard_edge = projection["table_lineage"][-1]
    assert wildcard_edge["transformation_id"] == 903
    assert wildcard_edge["wildcard_passthrough"] is True


def test_sync_file_graph_replaces_only_file_projection_in_one_transaction(temp_db):
    from services.graph_sync import sync_file_graph

    _insert_graph_source_rows()
    settings = Neo4jSettings(
        uri="neo4j://localhost:7687",
        username="neo4j",
        password="secret",
        database="neo4j",
    )
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    tx = MagicMock()
    tx.run.return_value.consume.return_value = None
    session.execute_write.side_effect = (
        lambda operation, projection: operation(tx, projection)
    )

    with patch(
        "services.graph_sync.load_neo4j_settings",
        return_value=settings,
    ), patch(
        "services.graph_sync.create_neo4j_driver",
        return_value=driver,
    ):
        report = sync_file_graph(501)

    assert report == {
        "file_id": 501,
        "columns": 2,
        "lineage_relationships": 2,
        "tables": 2,
        "table_lineage_relationships": 2,
    }
    driver.session.assert_called_once_with(database="neo4j")
    session.execute_write.assert_called_once()
    driver.close.assert_called_once_with()

    delete_query = tx.run.call_args_list[0].args[0]
    assert "MATCH (node:ETLProjection {file_id: $file_id})" in delete_query
    assert len(tx.run.call_args_list) == 5
    columns_call = tx.run.call_args_list[1]
    assert "CREATE (:ETLProjection:ETLColumn" in columns_call.args[0]
    assert len(columns_call.kwargs["rows"]) == 2

    tables_call = tx.run.call_args_list[2]
    assert "CREATE (:ETLProjection:ETLTable" in tables_call.args[0]
    assert len(tables_call.kwargs["rows"]) == 2
    assert "layers: row.layers" in tables_call.args[0]

    lineage_call = tx.run.call_args_list[3]
    assert "CREATE (source)-[:TRANSFORMS_TO" in lineage_call.args[0]
    assert "source_layer: row.source_layer" in lineage_call.args[0]
    assert [
        (row["transformation_id"], row["source_layer"], row["target_layer"])
        for row in lineage_call.kwargs["rows"]
    ] == [(901, "B", "T"), (902, "B", "T")]

    table_lineage_call = tx.run.call_args_list[4]
    assert "CREATE (source)-[:TABLE_TRANSFORMS_TO" in table_lineage_call.args[0]
    assert "wildcard_passthrough: row.wildcard_passthrough" in (
        table_lineage_call.args[0]
    )
    assert [
        (
            row["transformation_id"],
            row["source_layer"],
            row["target_layer"],
            row["sql_query"],
            row["wildcard_passthrough"],
        )
        for row in table_lineage_call.kwargs["rows"]
    ] == [
        (901, "B", "T", "SELECT source_id FROM exact_source", False),
        (
            902,
            "B",
            "T",
            "SELECT source_id FROM exact_source WHERE source_id IS NOT NULL",
            False,
        ),
    ]

    write_queries = "\n".join(
        call.args[0] for call in tx.run.call_args_list[1:]
    )
    for removed_label in (
        "ETLFile",
        "ExcelSheet",
        "TableCatalogEntry",
        "S2TTransformation",
    ):
        assert removed_label not in write_queries
    for removed_relationship in (
        "HAS_SHEET",
        "HAS_TABLE",
        "HAS_COLUMN",
        "HAS_TRANSFORMATION",
        "READS_FROM",
        "WRITES_TO",
    ):
        assert removed_relationship not in write_queries


def test_file_graph_projection_skips_edge_without_source_column(temp_db):
    from services.graph_sync import _build_file_graph_projection

    _insert_graph_source_rows()
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO s2t_transformations
            (id, file_id, sheet_name, row_num,
             target_table, target_field, source_table, source_field,
             transformation_rule)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                903,
                501,
                "S2T",
                5,
                "Exact Target",
                "target_only",
                None,
                None,
                "constant",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    projection = _build_file_graph_projection(501)

    assert {column["name"] for column in projection["columns"]} == {
        "source_id",
        "target_id",
        "target_only",
    }
    assert [row["transformation_id"] for row in projection["lineage"]] == [
        901,
        902,
    ]
    assert [row["transformation_id"] for row in projection["table_lineage"]] == [
        901,
        902,
    ]
    assert {table["name"] for table in projection["tables"]} == {
        "Exact Source",
        "Exact Target",
    }


def test_clear_graph_projection_deletes_all_application_nodes():
    from services.graph_sync import clear_graph_projection

    settings = Neo4jSettings(
        uri="neo4j://localhost:7687",
        username="neo4j",
        password="secret",
        database="neo4j",
    )
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    tx = MagicMock()
    summary = tx.run.return_value.consume.return_value
    summary.counters.nodes_deleted = 17
    session.execute_write.side_effect = lambda operation: operation(tx)

    with patch(
        "services.graph_sync.is_neo4j_configured",
        return_value=True,
    ), patch(
        "services.graph_sync.load_neo4j_settings",
        return_value=settings,
    ), patch(
        "services.graph_sync.create_neo4j_driver",
        return_value=driver,
    ):
        report = clear_graph_projection()

    assert report == {"nodes": 17}
    query = tx.run.call_args.args[0]
    assert "MATCH (node:ETLProjection)" in query
    assert "DETACH DELETE node" in query
    driver.session.assert_called_once_with(database="neo4j")
    driver.close.assert_called_once_with()
