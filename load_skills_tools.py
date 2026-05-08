import os
import json
import logging
from typing import List, Dict, Any, Callable

logger = logging.getLogger(__name__)

# ============================================
# MARKDOWN LOADERS (unchanged)
# ============================================
def load_skills() -> str:
    """Load skills.md from the project root."""
    try:
        with open("skills.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def load_tools() -> str:
    """Load tools.md from the project root."""
    try:
        with open("tools.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

# ============================================
# TOOL IMPLEMENTATIONS
# ============================================
# Registry: tool name -> callable
TOOL_FUNCTIONS: Dict[str, Callable] = {}

def register_tool(name: str, func: Callable):
    TOOL_FUNCTIONS[name] = func

# ----------------------------------------------------------------------
# Tool: run_sql
# ----------------------------------------------------------------------
def run_sql(query: str) -> List[Dict[str, Any]]:
    """Execute a read‑only SQL query on the internal database."""
    import sqlite3
    from db_storage import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"SQL error: {e}")
        result = {"error": str(e)}
    finally:
        conn.close()
    return result

register_tool("run_sql", run_sql)

# ----------------------------------------------------------------------
# Tool: get_lineage
# ----------------------------------------------------------------------
def get_lineage(column_id: str) -> List[Dict[str, Any]]:
    from db_storage import get_lineage as db_get_lineage
    return db_get_lineage(column_id)

register_tool("get_lineage", get_lineage)

# ----------------------------------------------------------------------
# Tool: get_upstream_sources
# ----------------------------------------------------------------------
def get_upstream_sources(column_id: str) -> List[str]:
    from db_storage import get_upstream_sources as db_upstream
    return db_upstream(column_id)

register_tool("get_upstream_sources", get_upstream_sources)

# ----------------------------------------------------------------------
# Tool: get_downstream_targets
# ----------------------------------------------------------------------
def get_downstream_targets(column_id: str) -> List[str]:
    from db_storage import get_downstream_targets as db_downstream
    return db_downstream(column_id)

register_tool("get_downstream_targets", get_downstream_targets)

# ----------------------------------------------------------------------
# Tool: similarity_search
# ----------------------------------------------------------------------
def similarity_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    from semantic_layer import similarity_search as sem_search
    return sem_search(query, top_k)

register_tool("similarity_search", similarity_search)

# ----------------------------------------------------------------------
# Tool: find_similar_columns
# ----------------------------------------------------------------------
def find_similar_columns(name: str) -> List[Dict[str, Any]]:
    from semantic_layer import find_similar_columns as sem_find
    return sem_find(name)

register_tool("find_similar_columns", find_similar_columns)

# ----------------------------------------------------------------------
# Tool: list_files
# ----------------------------------------------------------------------
def list_files() -> List[Dict[str, Any]]:
    from db_storage import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT file_hash, filename, upload_time FROM files ORDER BY upload_time DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

register_tool("list_files", list_files)

# ----------------------------------------------------------------------
# Tool: list_sheets
# ----------------------------------------------------------------------
def list_sheets(file_hash: str) -> List[str]:
    from db_storage import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT sheet_name FROM sheets WHERE file_hash = ? ORDER BY sheet_name", (file_hash,))
    rows = cursor.fetchall()
    conn.close()
    return [row["sheet_name"] for row in rows]

register_tool("list_sheets", list_sheets)

# ----------------------------------------------------------------------
# Tool: list_columns
# ----------------------------------------------------------------------
def list_columns(sheet_id: str) -> List[Dict[str, str]]:
    """
    sheet_id can be sheet_hash or sheet_name (if sheet_name is unique across all files).
    For uniqueness, prefer sheet_hash.
    """
    from db_storage import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()
    # Try as sheet_hash first
    cursor.execute("""
        SELECT column_hash, column_name_flat, column_index
        FROM columns
        WHERE sheet_hash = ?
        ORDER BY column_index
    """, (sheet_id,))
    rows = cursor.fetchall()
    if not rows:
        # Maybe it's a sheet_name? Need to join with sheets table
        cursor.execute("""
            SELECT c.column_hash, c.column_name_flat, c.column_index
            FROM columns c
            JOIN sheets s ON c.sheet_hash = s.sheet_hash
            WHERE s.sheet_name = ?
            ORDER BY c.column_index
        """, (sheet_id,))
        rows = cursor.fetchall()
    conn.close()
    return [{"column_hash": row["column_hash"], "name": row["column_name_flat"], "index": row["column_index"]} for row in rows]

register_tool("list_columns", list_columns)

# ----------------------------------------------------------------------
# Additional convenience: get_column_id_by_name (not a tool, but helper)
# ----------------------------------------------------------------------
def get_column_id_by_name(sheet_hash: str, column_name: str) -> str:
    from db_storage import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT column_hash FROM columns
        WHERE sheet_hash = ? AND column_name_flat = ?
    """, (sheet_hash, column_name))
    row = cursor.fetchone()
    conn.close()
    return row["column_hash"] if row else None

# ============================================
# Public API
# ============================================
def get_tool_functions() -> Dict[str, Callable]:
    """Return the registry of tool name -> callable."""
    return TOOL_FUNCTIONS