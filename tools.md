# Available Tools

## `get_header_decision(sheet_name, preview_rows)`
- **Input:** `sheet_name` (str), `preview_rows` (list of lists, first 10 rows)
- **Output:** `(header_start_row: int, header_rows_count: int, nested: bool)`
- **Purpose:** Call GigaChat to decide header structure.

## `store_excel_data(file_bytes, filename, model_used, sheets_info, data_rows_by_sheet, max_rows_per_sheet)`
- **Input:** 
  - `file_bytes`: raw Excel file content
  - `filename`: original file name
  - `model_used`: which LLM was used
  - `sheets_info`: list of parsed sheet metadata
  - `data_rows_by_sheet`: dict mapping sheet name → list of data rows
  - `max_rows_per_sheet`: limit rows stored (default 1000)
- **Output:** `file_hash` (str)
- **Purpose:** Insert into SQLite (files, sheets, columns, data tables).

## `summarize_file(file_hash, save=True)`
- **Input:** `file_hash` (str), `save` (bool)
- **Output:** `summary` (str, Russian business description)
- **Purpose:** Generate high‑level summary using GigaChat.

## `compare_with_target(excel_json)`
- **Input:** `excel_json` (dict, as returned by `/upload`)
- **Output:** similarity report (dict with `similarity_score`, `mapping_suggestions`, etc.)
- **Purpose:** Match Excel sheets to target database schema.

## `parse_excel_with_decisions(file_bytes, corrections=None, skip_sheets=None)`
- **Input:** file bytes, optional corrections dict, optional list of sheets to skip
- **Output:** `(sheets_info, data_rows_by_sheet)`
- **Purpose:** Core parsing logic, used both for initial upload and after corrections.

## `preview_headers(file_bytes, sheet_name, start_row, header_rows)`
- **Input:** file bytes, sheet name, start row index, number of header rows
- **Output:** list of column headers (flattened if nested)
- **Purpose:** Show user what headers would look like before applying changes.

## `run_sql(query)`
- **Input:** `query` (str) – read‑only SQL query (SELECT only)
- **Output:** list of dicts (rows)
- **Purpose:** Execute a SELECT query on the internal database.

## `get_lineage(column_id)`
- **Input:** `column_id` (str)
- **Output:** list of relationships (each with from_id, to_id, relation_type, metadata)
- **Purpose:** Retrieve upstream and downstream lineage for a column.

## `get_upstream_sources(column_id)`
- **Input:** `column_id` (str)
- **Output:** list of source column IDs/names
- **Purpose:** Get all sources that feed into a column.

## `get_downstream_targets(column_id)`
- **Input:** `column_id` (str)
- **Output:** list of target column IDs/names
- **Purpose:** Get all targets that depend on a column.

## `similarity_search(query)`
- **Input:** `query` (str)
- **Output:** list of entities (id, type, similarity score)
- **Purpose:** Find columns or other entities semantically similar to the query.

## `find_similar_columns(name)`
- **Input:** `name` (str)
- **Output:** list of similar column names with scores
- **Purpose:** Convenience wrapper around similarity_search for columns.

## `list_files()`
- **Input:** none
- **Output:** list of dicts (file_hash, filename, upload_time)
- **Purpose:** List all uploaded files.

## `list_sheets(file_hash)`
- **Input:** `file_hash` (str)
- **Output:** list of sheet names
- **Purpose:** List sheets in a given file.

## `list_columns(sheet_id)`
- **Input:** `sheet_id` (str) – can be sheet_hash or sheet_name (prefer sheet_hash)
- **Output:** list of column names (with their hashes if needed)
- **Purpose:** List columns in a specific sheet.