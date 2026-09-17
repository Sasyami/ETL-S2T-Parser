"""Build table- and column-lineage Neo4j projections from committed SQLite facts."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from graph_storage import (
    close_neo4j_driver,
    create_neo4j_driver,
    is_neo4j_configured,
    load_neo4j_settings,
)
from storage.database import get_db_connection
from storage.graph_outbox import (
    get_graph_sync_state,
    list_pending_graph_syncs,
    mark_graph_sync_applied,
    mark_graph_sync_failed,
    request_graph_sync,
)


logger = logging.getLogger(__name__)


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _column_key(file_id: int, table_name: str, column_name: str) -> str:
    return json.dumps(
        [int(file_id), table_name, column_name],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _table_key(file_id: int, table_name: str) -> str:
    return json.dumps(
        [int(file_id), table_name],
        ensure_ascii=False,
        separators=(",", ":"),
    )


class StaleGraphSyncError(RuntimeError):
    """Raised when a projection snapshot no longer matches its outbox request."""


def _build_file_graph_projection(
    file_id: int,
    *,
    generation: Optional[int] = None,
    revision: Optional[int] = None,
) -> Dict[str, Any]:
    clean_file_id = int(file_id)
    conn = get_db_connection()
    try:
        conn.execute("BEGIN")
        if generation is not None or revision is not None:
            if generation is None or revision is None:
                raise ValueError("generation and revision must be provided together")
            state = conn.execute(
                """
                SELECT generation, desired_revision
                FROM graph_sync_outbox
                WHERE file_id = ?
                """,
                (clean_file_id,),
            ).fetchone()
            if (
                state is None
                or int(state["generation"]) != int(generation)
                or int(state["desired_revision"]) != int(revision)
            ):
                raise StaleGraphSyncError(
                    "graph projection request changed before its snapshot "
                    f"for file_id={clean_file_id}"
                )
        transformations = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, target_table, target_field,
                       target_layer, source_table, source_field,
                       source_layer, transformation_rule
                FROM s2t_transformations
                WHERE file_id = ?
                ORDER BY id
                """,
                (clean_file_id,),
            ).fetchall()
        ]
    finally:
        conn.close()

    columns_by_key: Dict[str, Dict[str, Any]] = {}
    tables_by_key: Dict[str, Dict[str, Any]] = {}

    def register_table(
        table_name: Any,
        role: str,
        layer: Any = None,
    ) -> Optional[str]:
        clean_table = _text(table_name)
        if clean_table is None:
            return None

        key = _table_key(clean_file_id, clean_table)
        table = tables_by_key.setdefault(
            key,
            {
                "key": key,
                "file_id": clean_file_id,
                "name": clean_table,
                "roles": set(),
                "layers": set(),
            },
        )
        table["roles"].add(role)
        clean_layer = _text(layer)
        if clean_layer is not None:
            table["layers"].add(clean_layer)
        return key

    def register_column(
        table_name: Any,
        column_name: Any,
        role: str,
    ) -> Optional[str]:
        clean_table = _text(table_name)
        clean_column = _text(column_name)
        if clean_table is None or clean_column is None:
            return None

        key = _column_key(clean_file_id, clean_table, clean_column)
        column = columns_by_key.setdefault(
            key,
            {
                "key": key,
                "file_id": clean_file_id,
                "table_name": clean_table,
                "name": clean_column,
                "roles": set(),
            },
        )
        column["roles"].add(role)
        return key

    lineage: List[Dict[str, Any]] = []
    table_lineage: List[Dict[str, Any]] = []
    wildcard_table_pairs: List[Tuple[str, str]] = []
    for row in transformations:
        source_field = _text(row.get("source_field"))
        target_field = _text(row.get("target_field"))
        source_table = _text(row.get("source_table"))
        target_table = _text(row.get("target_table"))
        source_table_key = register_table(
            source_table, "source", row.get("source_layer")
        )
        target_table_key = register_table(
            target_table, "target", row.get("target_layer")
        )
        # Wildcard is a logical column rule. Materialize it as ETLColumn("*")
        # so the complete column graph is represented by TRANSFORMS_TO edges.
        source_key = register_column(
            source_table,
            source_field,
            "source",
        )
        target_key = register_column(
            target_table,
            target_field,
            "target",
        )
        if (
            source_field == "*"
            and target_field == "*"
            and source_table is not None
            and target_table is not None
        ):
            wildcard_table_pairs.append((source_table, target_table))
        if source_key is not None and target_key is not None:
            lineage.append(
                {
                    "file_id": clean_file_id,
                    "transformation_id": int(row["id"]),
                    "source_column_key": source_key,
                    "target_column_key": target_key,
                    "source_layer": _text(row.get("source_layer")),
                    "target_layer": _text(row.get("target_layer")),
                }
            )
        sql_query = _text(row.get("transformation_rule"))
        if (
            source_table_key is not None
            and target_table_key is not None
        ):
            table_lineage.append(
                {
                    "file_id": clean_file_id,
                    "transformation_id": int(row["id"]),
                    "source_table_key": source_table_key,
                    "target_table_key": target_table_key,
                    "source_layer": _text(row.get("source_layer")),
                    "target_layer": _text(row.get("target_layer")),
                    "sql_query": sql_query,
                    "rule_status": (
                        "present" if sql_query is not None else "missing"
                    ),
                }
            )

    # Propagate all currently known concrete column names through wildcard
    # table components. This materializes the same-name columns on both sides
    # without producing false cross-column paths such as id -> name.
    concrete_names_by_table: Dict[str, set[str]] = {}
    for column in columns_by_key.values():
        if column["name"] == "*":
            continue
        concrete_names_by_table.setdefault(column["table_name"], set()).add(
            column["name"]
        )
    changed = True
    while changed:
        changed = False
        for source_table, target_table in wildcard_table_pairs:
            shared_names = (
                concrete_names_by_table.get(source_table, set())
                | concrete_names_by_table.get(target_table, set())
            )
            for table_name in (source_table, target_table):
                table_names = concrete_names_by_table.setdefault(
                    table_name,
                    set(),
                )
                before = len(table_names)
                table_names.update(shared_names)
                changed = changed or len(table_names) != before

    for source_table, target_table in wildcard_table_pairs:
        shared_names = (
            concrete_names_by_table.get(source_table, set())
            | concrete_names_by_table.get(target_table, set())
        )
        for column_name in shared_names:
            register_column(source_table, column_name, "source")
            register_column(target_table, column_name, "target")

    wildcard_memberships_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    wildcard_tables = {
        table_name
        for pair in wildcard_table_pairs
        for table_name in pair
    }
    for table_name in wildcard_tables:
        wildcard_key = _column_key(clean_file_id, table_name, "*")
        if wildcard_key not in columns_by_key:
            continue
        for column_name in sorted(concrete_names_by_table.get(table_name, set())):
            column_key = _column_key(clean_file_id, table_name, column_name)
            if column_key not in columns_by_key:
                continue
            wildcard_memberships_by_key[(column_key, wildcard_key)] = {
                "file_id": clean_file_id,
                "column_key": column_key,
                "wildcard_key": wildcard_key,
            }

    columns = [
        {**column, "roles": sorted(column["roles"])}
        for column in columns_by_key.values()
    ]
    tables = [
        {
            **table,
            "roles": sorted(table["roles"]),
            "layers": sorted(table["layers"]),
        }
        for table in tables_by_key.values()
    ]
    return {
        "file_id": clean_file_id,
        "generation": int(generation or 0),
        "revision": int(revision or 0),
        "columns": columns,
        "lineage": lineage,
        "wildcard_memberships": list(wildcard_memberships_by_key.values()),
        "tables": tables,
        "table_lineage": table_lineage,
    }


def _replace_file_graph(tx, projection: Dict[str, Any]) -> bool:
    file_id = int(projection["file_id"])
    generation = int(projection.get("generation") or 0)
    revision = int(projection.get("revision") or 0)

    fence_record = tx.run(
        """
        MERGE (fence:ETLProjectionFence {file_id: $file_id})
        ON CREATE SET fence.generation = -1,
                      fence.revision = -1,
                      fence._lock = 0
        SET fence._lock = coalesce(fence._lock, 0) + 1
        WITH fence,
             ($generation > fence.generation OR
              ($generation = fence.generation AND
               $revision >= fence.revision)) AS accepted
        FOREACH (_ IN CASE WHEN accepted THEN [1] ELSE [] END |
            SET fence.generation = $generation,
                fence.revision = $revision
        )
        RETURN accepted
        """,
        file_id=file_id,
        generation=generation,
        revision=revision,
    ).single()
    if fence_record is None or not bool(fence_record.get("accepted", False)):
        return False

    tx.run(
        "MATCH (node:ETLProjection {file_id: $file_id}) DETACH DELETE node",
        file_id=file_id,
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        CREATE (:ETLProjection:ETLColumn {
            file_id: $file_id,
            projection_generation: $generation,
            projection_revision: $revision,
            key: row.key,
            table_name: row.table_name,
            name: row.name,
            roles: row.roles
        })
        """,
        file_id=file_id,
        generation=generation,
        revision=revision,
        rows=projection["columns"],
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        CREATE (:ETLProjection:ETLTable {
            file_id: $file_id,
            projection_generation: $generation,
            projection_revision: $revision,
            key: row.key,
            name: row.name,
            roles: row.roles,
            layers: row.layers
        })
        """,
        file_id=file_id,
        generation=generation,
        revision=revision,
        rows=projection["tables"],
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (source:ETLProjection:ETLColumn {
            file_id: $file_id,
            key: row.source_column_key
        })
        MATCH (target:ETLProjection:ETLColumn {
            file_id: $file_id,
            key: row.target_column_key
        })
        CREATE (source)-[:TRANSFORMS_TO {
            file_id: $file_id,
            transformation_id: row.transformation_id,
            source_layer: row.source_layer,
            target_layer: row.target_layer
        }]->(target)
        """,
        file_id=file_id,
        rows=projection["lineage"],
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (column:ETLProjection:ETLColumn {
            file_id: $file_id,
            key: row.column_key
        })
        MATCH (wildcard:ETLProjection:ETLColumn {
            file_id: $file_id,
            key: row.wildcard_key
        })
        CREATE (column)-[:COVERED_BY {file_id: $file_id}]->(wildcard)
        CREATE (wildcard)-[:EXPANDS_TO {file_id: $file_id}]->(column)
        """,
        file_id=file_id,
        rows=projection["wildcard_memberships"],
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (source:ETLProjection:ETLTable {
            file_id: $file_id,
            key: row.source_table_key
        })
        MATCH (target:ETLProjection:ETLTable {
            file_id: $file_id,
            key: row.target_table_key
        })
        CREATE (source)-[:TABLE_TRANSFORMS_TO {
            file_id: $file_id,
            transformation_id: row.transformation_id,
            source_layer: row.source_layer,
            target_layer: row.target_layer,
            sql_query: row.sql_query,
            rule_status: row.rule_status
        }]->(target)
        """,
        file_id=file_id,
        rows=projection["table_lineage"],
    ).consume()
    return True


def _sync_file_graph_revision(
    file_id: int,
    revision: int,
    generation: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply one requested revision and confirm it only after Neo4j commits."""
    clean_file_id = int(file_id)
    clean_revision = int(revision)
    if generation is None:
        state = get_graph_sync_state(clean_file_id)
        if state is None:
            raise StaleGraphSyncError(
                f"graph outbox row is missing for file_id={clean_file_id}"
            )
        clean_generation = int(state["generation"])
    else:
        clean_generation = int(generation)
    try:
        projection = _build_file_graph_projection(
            clean_file_id,
            generation=clean_generation,
            revision=clean_revision,
        )
        settings = load_neo4j_settings()
        driver = create_neo4j_driver(settings)
        try:
            with driver.session(database=settings.database) as session:
                accepted = session.execute_write(_replace_file_graph, projection)
                if not accepted:
                    raise StaleGraphSyncError(
                        "Neo4j rejected a stale graph projection "
                        f"for file_id={clean_file_id} generation="
                        f"{clean_generation} revision={clean_revision}"
                    )
        finally:
            close_neo4j_driver(driver)
    except Exception as exc:
        try:
            mark_graph_sync_failed(
                clean_file_id,
                clean_revision,
                str(exc),
                generation=clean_generation,
            )
        except Exception:
            logger.exception(
                "Failed to record graph outbox error for file_id=%s revision=%s",
                clean_file_id,
                clean_revision,
            )
        raise

    mark_graph_sync_applied(
        clean_file_id,
        clean_revision,
        generation=clean_generation,
    )
    state = get_graph_sync_state(clean_file_id) or {}

    return {
        "file_id": clean_file_id,
        "desired_revision": int(
            state.get("desired_revision", clean_revision)
        ),
        "applied_revision": int(
            state.get("applied_revision", clean_revision)
        ),
        "columns": len(projection["columns"]),
        "lineage_relationships": len(projection["lineage"]),
        "wildcard_membership_relationships": (
            2 * len(projection["wildcard_memberships"])
        ),
        "tables": len(projection["tables"]),
        "table_lineage_relationships": len(projection["table_lineage"]),
    }


def sync_file_graph(file_id: int) -> Dict[str, Any]:
    """Deliver the latest requested projection revision for one file."""
    clean_file_id = int(file_id)
    state = get_graph_sync_state(clean_file_id)
    if state is None or int(state["desired_revision"]) <= int(
        state["applied_revision"]
    ):
        request_graph_sync(clean_file_id)
        state = get_graph_sync_state(clean_file_id)
    if state is None:
        raise RuntimeError(f"Graph outbox row is missing for file_id={clean_file_id}")
    return _sync_file_graph_revision(
        clean_file_id,
        int(state["desired_revision"]),
        int(state["generation"]),
    )


def sync_pending_graph_projections(limit: int = 100) -> Dict[str, Any]:
    """Retry pending outbox rows, retaining failed revisions for later runs."""
    rows = list_pending_graph_syncs(limit=limit)
    applied: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for row in rows:
        file_id = int(row["file_id"])
        revision = int(row["desired_revision"])
        generation = int(row["generation"])
        try:
            applied.append(
                _sync_file_graph_revision(file_id, revision, generation)
            )
        except Exception as exc:
            logger.exception(
                "Pending Neo4j projection failed for file_id=%s revision=%s",
                file_id,
                revision,
            )
            errors.append(
                {
                    "file_id": file_id,
                    "desired_revision": revision,
                    "error": str(exc),
                }
            )
    return {
        "pending": len(rows),
        "applied": applied,
        "errors": errors,
    }


def _clear_graph_projection(
    tx,
    generation: int,
    requests: Sequence[Mapping[str, Any]],
) -> int:
    tx.run(
        """
        MATCH (fence:ETLProjectionFence)
        SET fence._lock = coalesce(fence._lock, 0) + 1
        WITH fence, fence.generation AS old_generation
        SET
            fence.generation = CASE
                WHEN old_generation < $generation
                THEN $generation
                ELSE old_generation
            END,
            fence.revision = CASE
                WHEN old_generation < $generation
                THEN 0
                ELSE fence.revision
            END
        """,
        generation=int(generation),
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (fence:ETLProjectionFence {file_id: row.file_id})
        ON CREATE SET fence.generation = row.generation,
                      fence.revision = row.revision,
                      fence._lock = 0
        SET fence._lock = coalesce(fence._lock, 0) + 1
        WITH fence, row, fence.generation AS old_generation,
             fence.revision AS old_revision
        SET
            fence.generation = CASE
                WHEN old_generation < row.generation
                THEN row.generation
                ELSE old_generation
            END,
            fence.revision = CASE
                WHEN old_generation < row.generation
                THEN row.revision
                WHEN old_generation = row.generation AND
                     old_revision < row.revision
                THEN row.revision
                ELSE old_revision
            END
        """,
        rows=[dict(request) for request in requests],
    ).consume()
    summary = tx.run(
        """
        MATCH (node:ETLProjection)
        WHERE node.projection_generation IS NULL
           OR node.projection_generation < $generation
           OR any(row IN $rows WHERE
                row.file_id = node.file_id AND
                row.generation = node.projection_generation AND
                node.projection_revision <= row.revision)
        DETACH DELETE node
        """,
        generation=int(generation),
        rows=[dict(request) for request in requests],
    ).consume()
    return int(summary.counters.nodes_deleted)


def clear_graph_projection(
    *,
    generation: int,
    requests: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Delete the complete application-owned Neo4j projection."""
    if not is_neo4j_configured():
        return {
            "nodes": 0,
            "skipped": True,
            "reason": "Neo4j не настроен",
        }
    settings = load_neo4j_settings()
    driver = create_neo4j_driver(settings)
    try:
        with driver.session(database=settings.database) as session:
            deleted = session.execute_write(
                _clear_graph_projection,
                int(generation),
                tuple(requests),
            )
    finally:
        close_neo4j_driver(driver)
    return {"nodes": int(deleted)}
