import os
import re
import json
import logging
from typing import Any, List, Dict, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

SQLITE_MAPPING_SCHEMA_CHEATSHEET = """

---

## SQLite column names — use exactly this (avoid SQL errors)

| Table | Columns (real names) |
|-------|---------------------|
| `target_tables` | `id`, `name`, `description` — **`name`**, NOT `table_name` |
| `source_tables` | `id`, `name`, `description`, `system_code` — **`name`**, NOT `table_name` |
| `column_mappings` | `id`, `target_table_id`, **`target_column`**, **`column_description`** (catalog text), `source_table_id`, **`source_column`**, `transformation_rule`, `data_type`, `is_primary_key` — NOT `target_column_name` |
| `additions` | `id`, `table_name`, `table_description`, `source_tables_name`, `sql`, `description` |

**Logical target models** (names like `t_agr_frame`) are **not** separate SQLite tables: their fields appear as rows in **`column_mappings`** (`target_column`). Use tool **`list_target_table_columns`** instead of `PRAGMA t_agr_frame`.

Join pattern: `column_mappings cm` → `JOIN target_tables tt ON cm.target_table_id = tt.id` and select **`tt.name`** as the logical target table title.

"""


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
    """Load tools.md plus mandatory SQLite naming hints agents often get wrong."""
    try:
        with open("tools.md", "r", encoding="utf-8") as f:
            body = f.read()
    except FileNotFoundError:
        body = ""
    return body + SQLITE_MAPPING_SCHEMA_CHEATSHEET

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
                cm.column_description,
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
               OR IFNULL(cm.column_description, '') LIKE ?
            ORDER BY cm.target_table_id, cm.target_column
            LIMIT ?
            """,
            (pat, pat, pat, pat, pat, pat, pat, pat, lim),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


register_tool("search_column_mappings", search_column_mappings)


def list_target_table_columns(table_identifier: str) -> Dict[str, Any]:
    """
    Logical target catalog: list column_mappings rows for a target table id or name.
    There is no SQLite table per logical name (e.g. t_agr_frame); fields are target_column values.
    """
    from db_storage import get_db_connection

    raw = (table_identifier or "").strip()
    if not raw:
        return {"error": "empty table_identifier"}
    if len(raw) > 200:
        return {"error": "table_identifier too long"}

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT id, name, description FROM target_tables
               WHERE id = ? OR lower(id) = lower(?)
                  OR lower(name) = lower(?)""",
            (raw, raw, raw),
        )
        tt = [dict(r) for r in cursor.fetchall()]
        ids: List[str] = []
        for r in tt:
            if r["id"] and r["id"] not in ids:
                ids.append(r["id"])
        if not ids:
            cursor.execute(
                """SELECT DISTINCT target_table_id FROM column_mappings
                   WHERE target_table_id = ? OR lower(target_table_id) = lower(?)""",
                (raw, raw),
            )
            ids = [
                row["target_table_id"]
                for row in cursor.fetchall()
                if row["target_table_id"]
            ]
        if not ids:
            return {
                "error": (
                    f"No row for «{raw}» in target_tables and no column_mappings "
                    f"with target_table_id matching that id/name."
                ),
                "table_identifier": raw,
                "hint": (
                    "Names like t_agr_frame are logical target tables: metadata lives in "
                    "`target_tables` (id, name) and field list in `column_mappings.target_column`. "
                    "They are not separate SQLite tables — do not use PRAGMA on t_agr_frame."
                ),
            }

        if not tt:
            ph = ",".join("?" * len(ids))
            cursor.execute(
                f"SELECT id, name, description FROM target_tables WHERE id IN ({ph})",
                ids,
            )
            tt = [dict(r) for r in cursor.fetchall()]

        ph2 = ",".join("?" * len(ids))
        cursor.execute(
            f"""
            SELECT id AS mapping_id, target_table_id, target_column, data_type,
                   is_primary_key, source_table_id, source_column, transformation_rule,
                   column_description
            FROM column_mappings
            WHERE target_table_id IN ({ph2})
            ORDER BY target_column
            """,
            ids,
        )
        cols = [dict(r) for r in cursor.fetchall()]
        return {
            "table_identifier_query": raw,
            "target_tables": tt,
            "columns": cols,
            "column_count": len(cols),
            "note": (
                "Logical fields = values in column_mappings.target_column for this target_table_id."
            ),
        }
    finally:
        conn.close()


register_tool("list_target_table_columns", list_target_table_columns)


def extract_target_table_identifier_from_fields_query(query: str) -> Optional[str]:
    q = (query or "").strip()
    if not q:
        return None
    m = re.search(r"`([A-Za-z][A-Za-z0-9_]*)`", q)
    if m:
        return m.group(1)
    m = re.search(r"(?i)\bтаблиц[а-яё]+\s+([A-Za-z][A-Za-z0-9_]*)\b", q)
    if m:
        return m.group(1)
    m = re.search(
        r"(?i)\b(?:у|для)\s+таблиц[а-яё]+\s+([A-Za-z][A-Za-z0-9_]*)\b",
        q,
    )
    if m:
        return m.group(1)
    m = re.search(r"(?i)\btable\s+([A-Za-z][A-Za-z0-9_]*)\b", q)
    if m:
        return m.group(1)
    return None


def user_wants_logical_target_table_fields(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    ru_list = bool(
        re.search(r"(?i)\b(поля|полей|колонки|структур[аы]?|атрибуты)\b", q)
    ) and bool(re.search(r"(?i)\bтаблиц", q))
    ru_desc = bool(re.search(r"(?i)\bописани", q)) and bool(
        re.search(r"(?i)\b(полей|поля|колонок|колонки|столбц)\b", q)
    ) and bool(re.search(r"(?i)\bтаблиц", q))
    en_list = bool(re.search(r"(?i)\b(columns|fields|attributes)\b", q)) and bool(
        re.search(r"(?i)\btable\b", q)
    )
    en_desc = bool(re.search(r"(?i)\bdescriptions?\b", q)) and bool(
        re.search(r"(?i)\b(columns|fields)\b", q)
    ) and bool(re.search(r"(?i)\btable\b", q))
    if not (ru_list or ru_desc or en_list or en_desc):
        return False
    return extract_target_table_identifier_from_fields_query(q) is not None


def _markdown_table_cell(value: Any, max_len: int = 400) -> str:
    if value is None:
        return ""
    s = str(value).replace("\n", " ").replace("|", "\\|").strip()
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def _aggregate_catalog_columns(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Merge multiple column_mappings rows that share target_column into one displayed row.
    Duplicate mapping_ids usually differ by source_* or reload history.
    """
    from collections import OrderedDict

    groups: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    for r in rows:
        tc = str(r.get("target_column") or "")
        if tc not in groups:
            groups[tc] = []
        groups[tc].append(r)

    merged_note = any(len(gr) > 1 for gr in groups.values())

    agg: List[Dict[str, Any]] = []
    for tc, grp in groups.items():
        descriptions = []
        for g in grp:
            cd = g.get("column_description")
            if cd and str(cd).strip() and str(cd).strip() not in descriptions:
                descriptions.append(str(cd).strip())
        srcs = []
        for g in grp:
            sc = g.get("source_column")
            if sc and str(sc).strip() and str(sc).strip() not in srcs:
                srcs.append(str(sc).strip())
        trans = []
        for g in grp:
            tr = g.get("transformation_rule")
            if tr and str(tr).strip() and str(tr).strip() not in trans:
                trans.append(str(tr).strip())
        pkv = [g.get("is_primary_key") for g in grp]
        pk_show = pkv[0] if len({repr(x) for x in pkv}) == 1 else "…"

        dts_uniq = []
        for g in grp:
            dt = str(g.get("data_type") or "").strip()
            if dt and dt not in dts_uniq:
                dts_uniq.append(dt)
        dt_show = dts_uniq[0] if len(dts_uniq) == 1 else ", ".join(dts_uniq)

        mids = ", ".join(
            str(g.get("mapping_id")) for g in grp if g.get("mapping_id") not in (None, "")
        )
        agg.append(
            {
                "target_column": tc,
                "column_description": " · ".join(descriptions),
                "source_column": ", ".join(srcs),
                "transformation_rule": "; ".join(trans),
                "data_type": dt_show,
                "is_primary_key": pk_show,
                "mapping_id": mids,
            }
        )
    return agg, merged_note


def _format_mapping_ids_markdown(mapping_id: Any) -> str:
    """Wrap plain mapping id list for markdown tables (CSV keeps plain ids)."""
    s = str(mapping_id or "").strip()
    if not s:
        return ""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return ", ".join(f"`{p}`" for p in parts)


def get_aggregated_target_table_catalog(table_identifier: str) -> Dict[str, Any]:
    """
    Full logical-target catalog with one row per target_column when possible.
    Intended for Streamlit / scripts; merges duplicate mappings like format_target_table_fields_answer.
    """
    data = list_target_table_columns(table_identifier)
    if data.get("error"):
        return data
    cols = data.get("columns") or []
    if not cols:
        return {
            **data,
            "aggregated": [],
            "had_duplicate_target_columns": False,
        }
    agg, merged = _aggregate_catalog_columns(cols)
    return {
        **data,
        "aggregated": agg,
        "had_duplicate_target_columns": merged,
    }


def format_target_table_fields_answer(data: Dict[str, Any]) -> str:
    if "error" in data:
        body = (
            f"**Нет данных по логической таблице `{data.get('table_identifier', '')}`** "
            f"(в этой SQLite нет отдельной физической таблицы с таким именем).\n\n"
            f"{data.get('error', '')}\n\n"
            f"{data.get('hint', '')}"
        )
        return body + "\n\n```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```"

    cols = data.get("columns") or []
    tt = data.get("target_tables") or []
    raw = json.dumps(data, ensure_ascii=False, indent=2)
    if not cols:
        head = "**Целевая таблица найдена, но строк в column_mappings для неё нет** (поля каталога не загружены).\n\n"
        return head + f"```json\n{raw}\n```"

    display_rows, merged_any = _aggregate_catalog_columns(cols)

    lines = []
    if tt:
        for r in tt:
            lines.append(
                f"- `target_tables`: id=`{r.get('id')}`, name=`{r.get('name')}`"
            )
    md = (
        "**Поля целевой таблицы** (`column_mappings`). "
        "Если **`column_description`** пуст, смотрите **`source_column`** и **`transformation_rule`**.\n\n"
        "| target_column | column_description | source_column | transformation_rule | "
        "data_type | PK | mapping_id |\n"
        "|---|---|---|---|---|:---:|---|\n"
    )
    for r in display_rows:
        pk = r.get("is_primary_key")
        md += (
            f"| `{r.get('target_column')}` | {_markdown_table_cell(r.get('column_description'))} | "
            f"{_markdown_table_cell(r.get('source_column'))} | "
            f"{_markdown_table_cell(r.get('transformation_rule'))} | "
            f"{_markdown_table_cell(r.get('data_type'), 80)} | {pk} | "
            f"{_format_mapping_ids_markdown(r.get('mapping_id'))} |\n"
        )
    if merged_any:
        md += (
            "\n(*Несколько записей на одно `target_column` сведены в одну строку; несколько **`mapping_id`** — "
            "разные строки источника или повторная загрузка. Полная детализация — в JSON ниже.)*\n"
        )
    md += "\n" + ("\n".join(lines) + "\n\n" if lines else "")
    md += "```json\n" + raw + "\n```"
    return md


def try_answer_target_table_fields_query(query: str) -> Optional[str]:
    q = (query or "").strip()
    if not user_wants_logical_target_table_fields(q):
        return None
    tid = extract_target_table_identifier_from_fields_query(q)
    if not tid:
        return None
    data = list_target_table_columns(tid)
    return format_target_table_fields_answer(data)


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
        "| target_table_name | target_table_id | target_column | column_description | source_column | mapping_id |\n"
        "|---|---|---|---|---|---|\n"
    )
    for r in rows:
        md += (
            f"| {r.get('target_table_name') or ''} | `{r.get('target_table_id')}` | "
            f"`{r.get('target_column')}` | {_markdown_table_cell(r.get('column_description'))} | "
            f"`{r.get('source_column')}` | `{r.get('mapping_id')}` |\n"
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