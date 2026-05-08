import os
import re
import json
import logging
from typing import Any, List, Dict, Callable, Optional

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


def user_wants_stored_file_inventory(query: str) -> bool:
    """
    True when the user asks for an inventory of uploads in `files`.
    Used to bypass the LLM so answers cannot invent hashes or names.
    """
    q = (query or "").strip().lower()
    if not re.search(r"\bfiles?\b", q):
        return False
    # Analytical questions about file *content* / columns, not a full listing
    if re.search(
        r"\b(what|which)\s+files?\s+(have|contain|use|with|include|hold)\b", q
    ):
        return False
    if re.search(r"\b(column|sheet|mapping|lineage|similar)\b", q):
        return False
    if re.search(r"\b(what|which)\s+files?\b", q):
        if re.search(
            r"\b(map|maps|connected|relations?|related|linked|joined|contain|"
            r"having|have|hold|use|includes?|including)\b",
            q,
        ):
            return False
    patterns = (
        r"\b(all|every|complete)\b.{0,100}\bfiles?\b",
        r"\b(list|show|give|display|enumerate)\b.{0,80}\b(all|every)\b",
        r"\bfiles?\b.{0,50}\b(stored|uploaded|saved|in\s+the\s+db|in\s+sqlite|database)\b",
        r"\b(stored|uploaded|saved)\b.{0,50}\bfiles?\b",
        r"\bgive\s+me\b.{0,60}\bfiles?\b",
        r"\bwhat\s+files?\b",
        r"\bwhich\s+files?\b",
    )
    return any(re.search(p, q) for p in patterns)


def format_stored_files_answer() -> str:
    """Markdown + JSON from real `list_files()` (no LLM)."""
    rows = list_files()
    raw = json.dumps(rows, ensure_ascii=False, indent=2)
    if not rows:
        return (
            "**Stored files (from database via `list_files`):** none — the `files` table is empty.\n\n"
            f"```json\n{raw}\n```"
        )
    lines = [
        f"| {r.get('filename', '')} | `{r.get('file_hash', '')}` | {r.get('upload_time', '')} |"
        for r in rows
    ]
    md = (
        "**Stored files (from database via `list_files`):**\n\n"
        "| filename | file_hash | upload_time |\n|:---|:---|:---|\n"
        + "\n".join(lines)
        + "\n\n```json\n"
        + raw
        + "\n```"
    )
    return md


def extract_workbook_filenames_from_query(query: str) -> List[str]:
    """Return unique workbook filenames (with extension) mentioned in natural language."""
    text = query or ""
    found = re.findall(
        r"\b([A-Za-z0-9][A-Za-z0-9_.\-]*\.(?:xlsx|xlsm|xls|csv))\b",
        text,
        flags=re.I,
    )
    out: List[str] = []
    seen: set[str] = set()
    for f in found:
        key = f.lower()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def user_wants_sheets_for_named_workbook(query: str) -> bool:
    """
    True when the user asks which sheet tabs exist for a workbook named in the query
    (e.g. "... sheets in foo.xlsx ...").
    """
    q_raw = query or ""
    q = q_raw.strip().lower()
    if not extract_workbook_filenames_from_query(q_raw):
        return False
    if not re.search(r"\b(sheets?|tabs?|worksheets?)\b", q):
        return False
    if re.search(r"\bcolumns?\b", q) and not re.search(r"\bsheets?\b", q):
        return False
    return True


def find_file_rows_for_workbook_filename(filename: str) -> List[Dict[str, Any]]:
    needle = filename.strip()
    from db_storage import get_db_connection

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT file_hash, filename, upload_time FROM files
           WHERE filename = ? OR lower(filename) = lower(?)
           ORDER BY upload_time DESC""",
        (needle, needle),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    if not rows and (os.sep in needle or "/" in needle):
        base = os.path.basename(needle.replace("\\", "/"))
        cursor.execute(
            """SELECT file_hash, filename, upload_time FROM files
               WHERE filename = ? OR lower(filename) = lower(?)
               ORDER BY upload_time DESC""",
            (base, base),
        )
        rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def format_sheets_for_workbook_filenames(filenames: List[str]) -> str:
    """Markdown + JSON with sheet tab names per `file_hash` (uses `list_sheets`)."""
    workbooks_out: List[Dict[str, Any]] = []
    md_lines: List[str] = []

    for fname in filenames:
        matches = find_file_rows_for_workbook_filename(fname)
        if not matches:
            workbooks_out.append({"query": fname, "status": "not_found", "uploads": []})
            md_lines.append(f"**`{fname}`:** no matching row in `files`.")
            continue
        uploads_block: List[Dict[str, Any]] = []
        for m in matches:
            fh = m["file_hash"]
            names = list_sheets(fh)
            uploads_block.append(
                {
                    "file_hash": fh,
                    "filename": m["filename"],
                    "upload_time": m.get("upload_time"),
                    "sheet_names": names,
                }
            )
            if names:
                listed = ", ".join(f"`{n}`" for n in names)
            else:
                listed = "*(no rows in `sheets` for this upload)*"
            md_lines.append(
                f"**`{m['filename']}`** (`file_hash` `{fh}`): {listed}"
            )
        workbooks_out.append({"query": fname, "status": "ok", "uploads": uploads_block})

    raw = json.dumps(workbooks_out, ensure_ascii=False, indent=2)
    header = "**Sheets (from database: `files` → `list_sheets`):**\n\n"
    return header + "\n\n".join(md_lines) + "\n\n```json\n" + raw + "\n```"


def try_answer_sheets_for_workbook_query(query: str) -> Optional[str]:
    """
    If the query asks for sheet tabs for a named workbook, return a DB-backed answer.
    Otherwise return None (caller may use the LLM).
    """
    q = (query or "").strip()
    if not user_wants_sheets_for_named_workbook(q):
        return None
    names = extract_workbook_filenames_from_query(q)
    if not names:
        return None
    return format_sheets_for_workbook_filenames(names)

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
# Tool: search_column_mappings
# ----------------------------------------------------------------------
def search_column_mappings(needle: str, limit: int = 50) -> Any:
    """
    Find rows where the needle appears in mapping or table identifiers.
    Joins human-readable names from source_tables / target_tables.
    """
    from db_storage import get_db_connection

    text = (needle or "").strip()
    if not text:
        return {"error": "needle must be non-empty"}
    if len(text) > 200:
        return {"error": "needle too long"}

    lim = max(1, min(int(limit), 100))
    pat = f"%{text}%"

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT
                cm.id AS mapping_id,
                cm.target_table_id,
                cm.target_column,
                cm.source_table_id,
                cm.source_column,
                cm.transformation_rule,
                cm.data_type,
                tt.name AS target_table_name,
                st.name AS source_table_name
            FROM column_mappings cm
            LEFT JOIN target_tables tt ON cm.target_table_id = tt.id
            LEFT JOIN source_tables st ON cm.source_table_id = st.id
            WHERE cm.target_column LIKE ?
               OR cm.source_column LIKE ?
               OR cm.target_table_id LIKE ?
               OR cm.source_table_id LIKE ?
               OR cm.id LIKE ?
               OR IFNULL(tt.name, '') LIKE ?
               OR IFNULL(st.name, '') LIKE ?
            ORDER BY cm.target_table_id, cm.target_column
            LIMIT ?
            """,
            (pat, pat, pat, pat, pat, pat, pat, lim),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


register_tool("search_column_mappings", search_column_mappings)


def extract_target_mapping_search_needle(query: str) -> Optional[str]:
    q = (query or "").strip()
    if not q:
        return None
    m = re.search(
        r"(?i)\b(?:uses?|contains?|has|holds|for|matching|named|called|like)\s+(?:the\s+)?([a-z0-9_.\-]{8,})\b",
        q,
    )
    if m:
        cand = m.group(1).rstrip(".,;:")
        junk = frozenset(
            {"column", "columns", "table", "tables", "mapping", "mappings", "target", "source"}
        )
        if cand.lower() not in junk:
            return cand
    if re.search(r"(?i)\btarget\s+table\b|\bcolumn_mappings\b", q):
        m2 = re.search(r"\b([a-z]\d+[a-z0-9_.\-]{4,})\b", q, flags=re.I)
        if m2:
            return m2.group(1)
    return None


def user_wants_target_table_uses_identifier(query: str) -> bool:
    qi = query or ""
    if not qi.strip():
        return False
    if not re.search(
        r"(?i)(\btarget\s+table\b|\bwhich\s+table\b|\bwhat\s+(?:target\s+)?table\b)",
        qi,
    ):
        return False
    return extract_target_mapping_search_needle(qi) is not None


def format_target_table_mapping_answer(needle: str) -> str:
    rows = search_column_mappings(needle, limit=100)
    if isinstance(rows, dict):
        err = rows.get("error", "unknown error")
        return f"Could not search `column_mappings` for `{needle}`: {err}"
    raw = json.dumps(rows, ensure_ascii=False, indent=2)
    if not rows:
        return (
            f"No `column_mappings` rows matched substring **`{needle}`** "
            "(checked target/source columns and table ids, joined names).\n\n"
            f"```json\n{raw}\n```"
        )

    uniq_targets: Dict[str, str] = {}
    for r in rows:
        tid = r.get("target_table_id") or ""
        tname = r.get("target_table_name") or ""
        key = tid or "(null target_table_id)"
        if key not in uniq_targets:
            uniq_targets[key] = tname

    bullets = []
    for tid, nm in uniq_targets.items():
        label = f"`{tid}`"
        if nm:
            label += f" — **{nm}**"
        bullets.append(f"- {label}")

    md = (
        f"**Target table(s) tied to `{needle}`** (from `column_mappings` + "
        f"`target_tables`):\n\n"
        + "\n".join(bullets)
        + "\n\n**Matching rows:**\n\n"
        "| target_table_name | target_table_id | target_column | source_column | mapping_id |\n"
        "|---|---|---|---|---|\n"
    )
    for r in rows:
        md += (
            f"| {r.get('target_table_name') or ''} | `{r.get('target_table_id')}` | "
            f"`{r.get('target_column')}` | `{r.get('source_column')}` | `{r.get('mapping_id')}` |\n"
        )
    md += "\n```json\n" + raw + "\n```"
    return md


def try_answer_target_table_for_mapping_identifier(query: str) -> Optional[str]:
    q = (query or "").strip()
    if not user_wants_target_table_uses_identifier(q):
        return None
    needle = extract_target_mapping_search_needle(q)
    if not needle:
        return None
    return format_target_table_mapping_answer(needle)

# ----------------------------------------------------------------------
# Tool: mapping_overview (S2T target schema snapshot for insights UI)
# ----------------------------------------------------------------------
def mapping_overview(limit: int = 15) -> Dict[str, Any]:
    """
    Snapshot of finalized S2T tables: row counts and small samples.
    Read-only; safe table names only.
    """
    from db_storage import get_db_connection

    tables = ("source_tables", "target_tables", "column_mappings", "additions")
    limit = max(1, min(int(limit), 100))
    conn = get_db_connection()
    cursor = conn.cursor()
    out: Dict[str, Any] = {}
    try:
        for t in tables:
            cursor.execute(f"SELECT COUNT(*) AS n FROM {t}")
            n = int(cursor.fetchone()["n"])
            cursor.execute(f"SELECT * FROM {t} LIMIT ?", (limit,))
            sample = [dict(row) for row in cursor.fetchall()]
            out[t] = {"count": n, "sample": sample}
        cursor.execute(
            """
            SELECT relation_type AS type, COUNT(*) AS n
            FROM relationships
            GROUP BY relation_type
            ORDER BY n DESC
            """
        )
        out["relationships_by_type"] = [dict(row) for row in cursor.fetchall()]
        cursor.execute("SELECT COUNT(*) AS n FROM relationships")
        out["relationships_total"] = int(cursor.fetchone()["n"])
    finally:
        conn.close()
    return out


register_tool("mapping_overview", mapping_overview)

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