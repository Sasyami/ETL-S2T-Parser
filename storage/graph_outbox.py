"""Durable SQLite outbox for rebuilding per-file Neo4j projections."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from .database import get_db_connection


def _now() -> str:
    return datetime.now().isoformat()


def enqueue_graph_sync(cursor: sqlite3.Cursor, file_id: int) -> int:
    """Increment the desired projection revision inside the caller transaction."""
    clean_file_id = int(file_id)
    cursor.execute(
        """
        INSERT INTO graph_sync_outbox
        (file_id, desired_revision, applied_revision, attempts,
         last_error, updated_at, applied_at)
        VALUES (?, 1, 0, 0, NULL, ?, NULL)
        ON CONFLICT(file_id) DO UPDATE SET
            desired_revision = graph_sync_outbox.desired_revision + 1,
            last_error = NULL,
            updated_at = excluded.updated_at
        """,
        (clean_file_id, _now()),
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
            SELECT file_id, desired_revision, applied_revision, attempts,
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
            SELECT file_id, desired_revision, applied_revision, attempts,
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


def mark_graph_sync_applied(file_id: int, revision: int) -> None:
    """Confirm a revision only after the Neo4j transaction succeeded."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            UPDATE graph_sync_outbox
            SET applied_revision = CASE
                    WHEN applied_revision < ? THEN ?
                    ELSE applied_revision
                END,
                attempts = 0,
                last_error = NULL,
                applied_at = ?
            WHERE file_id = ?
            """,
            (int(revision), int(revision), _now(), int(file_id)),
        )
        conn.commit()
    finally:
        conn.close()


def mark_graph_sync_failed(file_id: int, revision: int, error: str) -> None:
    """Record a failed delivery without advancing applied_revision."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            UPDATE graph_sync_outbox
            SET attempts = attempts + 1,
                last_error = ?,
                updated_at = ?
            WHERE file_id = ? AND desired_revision >= ?
            """,
            (str(error), _now(), int(file_id), int(revision)),
        )
        conn.commit()
    finally:
        conn.close()


def mark_all_graph_syncs_applied() -> None:
    """Confirm all pending empty projections after a successful global clear."""
    conn = get_db_connection()
    try:
        now = _now()
        conn.execute(
            """
            UPDATE graph_sync_outbox
            SET applied_revision = desired_revision,
                attempts = 0,
                last_error = NULL,
                applied_at = ?
            WHERE desired_revision > applied_revision
            """,
            (now,),
        )
        conn.commit()
    finally:
        conn.close()


__all__ = [
    "enqueue_graph_sync",
    "get_graph_sync_state",
    "list_pending_graph_syncs",
    "mark_all_graph_syncs_applied",
    "mark_graph_sync_applied",
    "mark_graph_sync_failed",
    "request_graph_sync",
]
