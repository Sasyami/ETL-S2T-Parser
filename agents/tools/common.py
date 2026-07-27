"""Shared helpers for agent tools."""

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def clamped_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def file_meta(file_id: int) -> dict[str, Any]:
    """Return minimal metadata for one uploaded file."""
    from storage.database import get_db_connection

    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT file_id, filename, upload_time FROM files WHERE file_id = ?",
            (file_id,),
        ).fetchone()
        return dict(row) if row else {"file_id": file_id}
    finally:
        conn.close()
