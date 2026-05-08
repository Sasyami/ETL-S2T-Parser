import sqlite3
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = "excel_data.db"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def generate_id(*args) -> str:
    """Generate a deterministic ID from any number of strings."""
    combined = "|".join(str(a) for a in args if a is not None)
    return hashlib.md5(combined.encode()).hexdigest()


def init_db():
    """Initialize all tables with migrations for existing databases."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # files table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_hash TEXT PRIMARY KEY,
            filename TEXT,
            model_used TEXT,
            upload_time TEXT,
            summary TEXT,
            result_json TEXT
        )
    """)
    cursor.execute("PRAGMA table_info(files)")
    cols = [c[1] for c in cursor.fetchall()]
    if "upload_time" not in cols:
        cursor.execute("ALTER TABLE files ADD COLUMN upload_time TEXT")
    if "summary" not in cols:
        cursor.execute("ALTER TABLE files ADD COLUMN summary TEXT")
    if "result_json" not in cols:
        cursor.execute("ALTER TABLE files ADD COLUMN result_json TEXT")

    # sheets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sheets (
            sheet_hash TEXT PRIMARY KEY,
            file_hash TEXT,
            sheet_name TEXT,
            header_start_row INTEGER,
            header_rows_count INTEGER,
            nested_structure INTEGER,
            skipped INTEGER DEFAULT 0,
            skip_reason TEXT
        )
    """)
    cursor.execute("PRAGMA table_info(sheets)")
    sheet_cols = [c[1] for c in cursor.fetchall()]
    if "skipped" not in sheet_cols:
        cursor.execute("ALTER TABLE sheets ADD COLUMN skipped INTEGER DEFAULT 0")
    if "skip_reason" not in sheet_cols:
        cursor.execute("ALTER TABLE sheets ADD COLUMN skip_reason TEXT")

    # columns table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS columns (
            column_hash TEXT PRIMARY KEY,
            sheet_hash TEXT,
            column_index INTEGER,
            column_name_flat TEXT,
            column_header TEXT
        )
    """)

    # data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data (
            id TEXT PRIMARY KEY,
            sheet_hash TEXT,
            row_num INTEGER,
            column_hash TEXT,
            value TEXT
        )
    """)

    # graph relationships table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            from_id TEXT,
            to_id TEXT,
            relation_type TEXT,
            metadata TEXT
        )
    """)

    # embeddings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            entity_id TEXT,
            entity_type TEXT,
            vector TEXT
        )
    """)

    # target schema tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS source_tables (
            id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            system_code TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS target_tables (
            id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS column_mappings (
            id TEXT PRIMARY KEY,
            target_table_id TEXT,
            target_column TEXT,
            source_table_id TEXT,
            source_column TEXT,
            transformation_rule TEXT,
            data_type TEXT,
            is_primary_key INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS additions (
            id TEXT PRIMARY KEY,
            table_name TEXT,
            table_description TEXT,
            source_tables_name TEXT,
            sql TEXT,
            description TEXT
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized with all tables")


# ============================================
# ORIGINAL STORAGE FUNCTIONS
# ============================================
def store_excel_data(file_bytes: bytes, filename: str, model_used: str,
                     sheets_info: List[Dict], data_rows_by_sheet: Dict[str, List],
                     max_rows_per_sheet: int = 1000) -> str:
    """
    Store parsed Excel data into SQLite.
    Returns file_hash.
    """
    file_hash = hashlib.md5(file_bytes).hexdigest()
    upload_time = datetime.now().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert file record
    cursor.execute("""
        INSERT OR REPLACE INTO files (file_hash, filename, model_used, upload_time)
        VALUES (?, ?, ?, ?)
    """, (file_hash, filename, model_used, upload_time))

    for sheet in sheets_info:
        sheet_name = sheet["sheet_name"]
        skipped = sheet.get("skipped", False)
        skip_reason = sheet.get("skip_reason", "")

        if skipped:
            sheet_hash = generate_id(file_hash, sheet_name)
            cursor.execute("""
                INSERT OR REPLACE INTO sheets
                (sheet_hash, file_hash, sheet_name, skipped, skip_reason)
                VALUES (?, ?, ?, ?, ?)
            """, (sheet_hash, file_hash, sheet_name, 1, skip_reason))
            continue

        # Not skipped
        header_start_row = sheet["ai_decision"]["header_start_row"]
        header_rows_count = sheet["ai_decision"]["header_rows_count"]
        nested = 1 if sheet["ai_decision"]["nested_structure"] else 0
        sheet_hash = generate_id(file_hash, sheet_name)
        cursor.execute("""
            INSERT OR REPLACE INTO sheets
            (sheet_hash, file_hash, sheet_name, header_start_row, header_rows_count, nested_structure, skipped)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (sheet_hash, file_hash, sheet_name, header_start_row, header_rows_count, nested, 0))

        # Insert columns
        columns = sheet.get("columns", [])
        for idx, col in enumerate(columns):
            if isinstance(col, list):
                flat_name = " > ".join(str(c) for c in col if c)
                col_hash = generate_id(sheet_hash, idx, flat_name)
                cursor.execute("""
                    INSERT OR REPLACE INTO columns
                    (column_hash, sheet_hash, column_index, column_name_flat, column_header)
                    VALUES (?, ?, ?, ?, ?)
                """, (col_hash, sheet_hash, idx, flat_name, json.dumps(col, ensure_ascii=False)))
            else:
                flat_name = str(col) if col is not None else f"Column_{idx + 1}"
                col_hash = generate_id(sheet_hash, idx, flat_name)
                cursor.execute("""
                    INSERT OR REPLACE INTO columns
                    (column_hash, sheet_hash, column_index, column_name_flat, column_header)
                    VALUES (?, ?, ?, ?, ?)
                """, (col_hash, sheet_hash, idx, flat_name, flat_name))

        # Insert data rows
        data_rows = data_rows_by_sheet.get(sheet_name, [])
        for row_num, row in enumerate(data_rows[:max_rows_per_sheet]):
            for col_idx, value in enumerate(row):
                if value is None:
                    continue
                cursor.execute("SELECT column_hash FROM columns WHERE sheet_hash = ? AND column_index = ?",
                               (sheet_hash, col_idx))
                col_row = cursor.fetchone()
                if col_row:
                    col_hash = col_row["column_hash"]
                else:
                    col_hash = generate_id(sheet_hash, col_idx, str(value)[:50])
                data_id = generate_id(sheet_hash, row_num, col_hash)
                cursor.execute("""
                    INSERT OR IGNORE INTO data (id, sheet_hash, row_num, column_hash, value)
                    VALUES (?, ?, ?, ?, ?)
                """, (data_id, sheet_hash, row_num, col_hash, str(value)[:1000]))

    conn.commit()
    conn.close()
    return file_hash


def update_file_result_json(file_hash: str, result_json: str):
    """Store the full JSON response from /upload in the files table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE files SET result_json = ? WHERE file_hash = ?", (result_json, file_hash))
    conn.commit()
    conn.close()


def update_file_summary(file_hash: str, summary: str):
    """Store the generated summary in the files table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE files SET summary = ? WHERE file_hash = ?", (summary, file_hash))
    conn.commit()
    conn.close()


# ============================================
# GRAPH / LINEAGE EXTENSIONS
# ============================================
def add_relationship(from_id: str, to_id: str, relation_type: str, metadata: str = None) -> str:
    """Store a directed relationship between two entities."""
    rel_id = generate_id(from_id, to_id, relation_type)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (id, from_id, to_id, relation_type, metadata)
        VALUES (?, ?, ?, ?, ?)
    """, (rel_id, from_id, to_id, relation_type, metadata))
    conn.commit()
    conn.close()
    return rel_id


def get_lineage(column_id: str) -> List[Dict[str, Any]]:
    """Get all relationships (both incoming and outgoing) for a column."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT from_id, to_id, relation_type, metadata
        FROM relationships
        WHERE from_id = ? OR to_id = ?
    """, (column_id, column_id))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_upstream_sources(column_id: str) -> List[str]:
    """Return IDs of columns/tables that directly feed into this column."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT from_id FROM relationships
        WHERE to_id = ? AND relation_type IN ('DERIVED_FROM', 'MAPS_TO')
    """, (column_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row["from_id"] for row in rows]


def get_downstream_targets(column_id: str) -> List[str]:
    """Return IDs of columns/tables that depend on this column."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT to_id FROM relationships
        WHERE from_id = ? AND relation_type IN ('MAPS_TO', 'DERIVED_FROM')
    """, (column_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row["to_id"] for row in rows]


def get_column_id_by_name(sheet_hash: str, column_name: str) -> str:
    """
    Find column hash by name, supporting nested headers (e.g., "Parent > Child").
    Tries exact match, then last part after " > ", then partial match.
    """
    if not column_name:
        return None
    column_name = column_name.strip()
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Exact match (case‑insensitive)
    cursor.execute("""
        SELECT column_hash FROM columns
        WHERE sheet_hash = ? AND LOWER(column_name_flat) = LOWER(?)
    """, (sheet_hash, column_name))
    row = cursor.fetchone()
    if row:
        conn.close()
        return row["column_hash"]

    # 2. Try to match the last part of a nested header (after " > ")
    last_part = column_name
    if " > " in column_name:
        last_part = column_name.split(" > ")[-1]
    cursor.execute("""
        SELECT column_hash FROM columns
        WHERE sheet_hash = ? AND LOWER(column_name_flat) LIKE LOWER(?)
        AND column_name_flat LIKE '% > %'
    """, (sheet_hash, '%' + last_part))
    row = cursor.fetchone()
    if row:
        conn.close()
        return row["column_hash"]

    # 3. Partial match (contains) – last resort
    cursor.execute("""
        SELECT column_hash FROM columns
        WHERE sheet_hash = ? AND LOWER(column_name_flat) LIKE LOWER(?)
        LIMIT 1
    """, (sheet_hash, '%' + column_name + '%'))
    row = cursor.fetchone()
    conn.close()
    return row["column_hash"] if row else None


def get_target_column_id(target_table_name: str, target_column: str) -> str:
    """Generate deterministic ID for a target column."""
    return generate_id(target_table_name, target_column)


def get_sheet_hash(file_hash: str, sheet_name: str) -> str:
    """Get sheet hash from file_hash and sheet_name."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT sheet_hash FROM sheets WHERE file_hash = ? AND sheet_name = ?", (file_hash, sheet_name))
    row = cursor.fetchone()
    conn.close()
    return row["sheet_hash"] if row else None


# ============================================
# METADATA QUERY HELPERS
# ============================================
def get_all_files() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT file_hash, filename, upload_time FROM files ORDER BY upload_time DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_sheets_by_file(file_hash: str) -> List[str]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT sheet_name FROM sheets WHERE file_hash = ? ORDER BY sheet_name", (file_hash,))
    rows = cursor.fetchall()
    conn.close()
    return [row["sheet_name"] for row in rows]


def get_columns_by_sheet(sheet_hash: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT column_hash, column_name_flat, column_index
        FROM columns
        WHERE sheet_hash = ?
        ORDER BY column_index
    """, (sheet_hash,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]