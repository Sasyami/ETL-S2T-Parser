"""Durable SQLite outbox for rebuilding per-file Neo4j projections."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .database import get_db_connection


def _now() -> str:
    return datetime.now().isoformat()


def _current_generation(cursor: sqlite3.Cursor) -> int:
    row = cursor.execute(
        """
        SELECT generation
        FROM graph_sync_generation
        WHERE singleton_id = 1
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("Graph sync generation is not initialized")
    return int(row[0])


def enqueue_graph_sync(cursor: sqlite3.Cursor, file_id: int) -> int:
    """Increment the desired projection revision inside the caller transaction."""
    clean_file_id = int(file_id)
    generation = _current_generation(cursor)
    cursor.execute(
        """
        INSERT INTO graph_sync_outbox
        (file_id, generation, desired_revision, applied_revision, attempts,
         last_error, updated_at, applied_at)
        VALUES (?, ?, 1, 0, 0, NULL, ?, NULL)
        ON CONFLICT(file_id) DO UPDATE SET
            generation = excluded.generation,
            desired_revision = CASE
                WHEN graph_sync_outbox.generation = excluded.generation
                THEN graph_sync_outbox.desired_revision + 1
                ELSE 1
            END,
            applied_revision = CASE
                WHEN graph_sync_outbox.generation = excluded.generation
                THEN graph_sync_outbox.applied_revision
                ELSE 0
            END,
            attempts = CASE
                WHEN graph_sync_outbox.generation = excluded.generation
                THEN graph_sync_outbox.attempts
                ELSE 0
            END,
            last_error = NULL,
            updated_at = excluded.updated_at,
            applied_at = CASE
                WHEN graph_sync_outbox.generation = excluded.generation
                THEN graph_sync_outbox.applied_at
                ELSE NULL
            END
        """,
        (clean_file_id, generation, _now()),
    )
    row = cursor.execute(
        """
        SELECT desired_revision
        FROM graph_sync_outbox
        WHERE file_id = ?
        """,
        (clean_file_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Graph outbox row was not created for file_id={clean_file_id}")
    return int(row[0])


def request_graph_sync(file_id: int) -> int:
    """Create a durable graph-sync request in its own SQLite transaction."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN")
        revision = enqueue_graph_sync(cursor, int(file_id))
        conn.commit()
        return revision
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_graph_sync_state(file_id: int) -> Optional[Dict[str, Any]]:
    """Return the durable revision state for one file."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT file_id, generation, desired_revision, applied_revision, attempts,
                   last_error, updated_at, applied_at
            FROM graph_sync_outbox
            WHERE file_id = ?
            """,
            (int(file_id),),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def list_pending_graph_syncs(limit: int = 100) -> List[Dict[str, Any]]:
    """Return revisions that have not yet been confirmed in Neo4j."""
    clean_limit = max(1, min(int(limit), 1000))
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT file_id, generation, desired_revision, applied_revision, attempts,
                   last_error, updated_at, applied_at
            FROM graph_sync_outbox
            WHERE desired_revision > applied_revision
            ORDER BY updated_at, file_id
            LIMIT ?
            """,
            (clean_limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _resolve_generation(
    conn: sqlite3.Connection,
    file_id: int,
    generation: Optional[int],
) -> Optional[int]:
    if generation is not None:
        return int(generation)
    row = conn.execute(
        "SELECT generation FROM graph_sync_outbox WHERE file_id = ?",
        (int(file_id),),
    ).fetchone()
    return int(row[0]) if row is not None else None


def mark_graph_sync_applied(
    file_id: int,
    revision: int,
    generation: Optional[int] = None,
) -> bool:
    """Confirm a revision only after the Neo4j transaction succeeded."""
    conn = get_db_connection()
    try:
        clean_generation = _resolve_generation(conn, file_id, generation)
        if clean_generation is None:
            return False
        cursor = conn.execute(
            """
            UPDATE graph_sync_outbox
            SET applied_revision = CASE
                    WHEN applied_revision < ? THEN ?
                    ELSE applied_revision
                END,
                attempts = CASE
                    WHEN desired_revision = ? THEN 0
                    ELSE attempts
                END,
                last_error = CASE
                    WHEN desired_revision = ? THEN NULL
                    ELSE last_error
                END,
                applied_at = ?
            WHERE file_id = ?
              AND generation = ?
              AND desired_revision >= ?
            """,
            (
                int(revision),
                int(revision),
                int(revision),
                int(revision),
                _now(),
                int(file_id),
                clean_generation,
                int(revision),
            ),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def mark_graph_sync_failed(
    file_id: int,
    revision: int,
    error: str,
    generation: Optional[int] = None,
) -> bool:
    """Record a failed delivery without advancing applied_revision."""
    conn = get_db_connection()
    try:
        clean_generation = _resolve_generation(conn, file_id, generation)
        if clean_generation is None:
            return False
        cursor = conn.execute(
            """
            UPDATE graph_sync_outbox
            SET attempts = attempts + 1,
                last_error = ?,
                updated_at = ?
            WHERE file_id = ?
              AND generation = ?
              AND desired_revision = ?
              AND applied_revision < ?
            """,
            (
                str(error),
                _now(),
                int(file_id),
                clean_generation,
                int(revision),
                int(revision),
            ),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def mark_graph_syncs_applied(
    requests: Sequence[Mapping[str, Any]],
) -> int:
    """Confirm only the generation/revisions included in one clear snapshot."""
    conn = get_db_connection()
    try:
        now = _now()
        updated = 0
        for request in requests:
            file_id = int(request["file_id"])
            generation = int(request["generation"])
            revision = int(request["revision"])
            cursor = conn.execute(
                """
                UPDATE graph_sync_outbox
                SET applied_revision = CASE
                        WHEN applied_revision < ? THEN ?
                        ELSE applied_revision
                    END,
                    attempts = CASE
                        WHEN desired_revision = ? THEN 0
                        ELSE attempts
                    END,
                    last_error = CASE
                        WHEN desired_revision = ? THEN NULL
                        ELSE last_error
                    END,
                    applied_at = ?
                WHERE file_id = ?
                  AND generation = ?
                  AND desired_revision >= ?
                """,
                (
                    revision,
                    revision,
                    revision,
                    revision,
                    now,
                    file_id,
                    generation,
                    revision,
                ),
            )
            updated += int(cursor.rowcount == 1)
        conn.commit()
        return updated
    finally:
        conn.close()


__all__ = [
    "enqueue_graph_sync",
    "get_graph_sync_state",
    "list_pending_graph_syncs",
    "mark_graph_sync_applied",
    "mark_graph_sync_failed",
    "mark_graph_syncs_applied",
    "request_graph_sync",
]
