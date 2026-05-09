"""ReAct agent for source/target/mapping insights (grounded in SQLite + tools)."""

from typing import Optional

from agents.agent import TOOLS_DESCRIPTION, react_tool_loop
from load_skills_tools import (
    format_stored_files_answer,
    try_answer_sheets_for_workbook_query,
    try_answer_target_table_fields_query,
    try_answer_target_table_for_mapping_identifier,
    user_wants_stored_file_inventory,
)

INSIGHTS_SYSTEM_PROMPT = f"""You are an analyst extracting insights about **sources**, **targets**, and **column mappings** in this project's SQLite database (S2T layer plus uploaded Excel metadata).

Goals:
- Describe what is mapped, gaps, and notable lineage or naming patterns.
- Ground every factual claim with tools (`mapping_overview`, `run_sql`, `search_column_mappings`, **`list_target_table_columns`**, lineage helpers, file/sheet/column listing, `similarity_search` when embeddings exist).
- Never invent `file_hash`, `filename`, `upload_time`, or sheet tab names. For the file catalog call **`list_files`** with **`Action Input: {{}}`**; for sheet tabs use **`list_sheets(file_hash)`** after resolving the row in `files` by exact `filename`.

SQLite **`column_mappings`** columns: **`id`**, **`target_table_id`**, **`target_column`**, **`column_description`** (описание поля из каталога / Excel), **`source_table_id`**, **`source_column`**, **`transformation_rule`**, **`data_type`**, **`is_primary_key`**. There is **no** **`target_column_name`** — use **`target_column`**. Для поиска подстрок используйте **`search_column_mappings`** с `{{"needle": "<text>"}}`.

Never ask the user to run SQL manually. You **`can`** run **`run_sql`** (`SELECT`, **`PRAGMA table_info(...)`**) for introspection — but **`not`** for logical target names (`t_*`): those are not physical tables here.
- Use **`list_target_table_columns`** with `{{"table_identifier": "..."}}` instead of **`PRAGMA`** on logical names like `t_agr_frame`.

Workflow:
- For broad questions (overview of mapping), call `mapping_overview` with `limit` between 10 and 25, then drill down with `run_sql` or lineage tools using concrete column IDs.
- Use `SELECT` (and introspection pragmas via `run_sql` when needed); join `column_mappings` with `source_tables` / `target_tables` where appropriate.
- For catalog-style identifiers appearing in mappings, call **`search_column_mappings`** instead of guessing column names.
- For questions about Excel uploads, use `list_files`, `list_sheets`, `list_columns` and honor any file scope the user (or UI) provides.

Use this format:

Thought: what you need to do next
Action: tool_name
Action Input: {{"param": "value"}}
Observation: (filled by the system)
... repeat ...
Final Answer: clear, structured insights (bullets OK). If tables are empty, say so.

Tool reference (see tools.md for parameters):

{TOOLS_DESCRIPTION}
"""


def insights_chat(
    user_query: str,
    max_steps: int = 8,
    file_hash: Optional[str] = None,
) -> str:
    q = user_query.strip()
    if user_wants_stored_file_inventory(q):
        return format_stored_files_answer()
    sheet_ans = try_answer_sheets_for_workbook_query(q)
    if sheet_ans is not None:
        return sheet_ans
    tgt_ans = try_answer_target_table_for_mapping_identifier(q)
    if tgt_ans is not None:
        return tgt_ans
    fields_ans = try_answer_target_table_fields_query(q)
    if fields_ans is not None:
        return fields_ans
    fh = (file_hash or "").strip()
    if fh:
        q = (
            f"{q}\n\n"
            f"(Scope: when listing files/sheets/columns or summarizing uploads, prefer file_hash `{fh}`.)"
        )
    return react_tool_loop(q, INSIGHTS_SYSTEM_PROMPT, max_steps)
