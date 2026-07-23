import json
import logging
import sys
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)

_json_parser = JsonOutputParser()

_SCHEMA_MATCHER_SYSTEM = "Ты – аналитик данных. Отвечай только JSON."


def invoke_llm_plain_text(prompt: str) -> str:
    """SYSTEM (JSON-only) + USER prompt → assistant text (patchable in tests)."""
    from . import agent as _agent

    messages = [
        SystemMessage(content=_SCHEMA_MATCHER_SYSTEM),
        HumanMessage(content=prompt),
    ]
    return (_agent.chat_model | StrOutputParser()).invoke(messages)


def _dispatch_invoke_llm_plain_text(prompt: str) -> str:
    """Resolve ``invoke_llm_plain_text`` at runtime so patched tests see the stub."""
    return sys.modules[__name__].invoke_llm_plain_text(prompt)


TARGET_SCHEMA = {
    "database": "SQLite",
    "version": "1.0",
    "tables": [
        {
            "name": "source_tables",
            "description": "Справочник исходных таблиц",
            "columns": ["name", "description", "system_code"]
        },
        {
            "name": "target_tables",
            "description": "Целевые таблицы в хранилище",
            "columns": ["name", "description"]
        },
        {
            "name": "column_mappings",
            "description": "Правила маппинга исходных колонок в целевые",
            "columns": ["target_table_name", "target_column", "target_column_description",
                        "source_table_name",
                        "source_column", "transformation_rule", "data_type", "is_primary_key"]
        },
        {
            "name": "additions",
            "description": "Дополнительные правила и объекты трасформации",
            "columns": ["table_name", "table_description", "source_tables_name",
                        "sql", "description"]
        }
    ]
}


def _parse_model_json(raw_text: str, expected_type: type):
    """Parse model JSON; fall back to brace extraction when the parser rejects prose wrappers."""
    from .agent import extract_json_payload

    parsed = None
    try:
        parsed = _json_parser.parse(raw_text)
    except Exception:
        parsed = None
    if not isinstance(parsed, expected_type):
        parsed = json.loads(extract_json_payload(raw_text))
    if not isinstance(parsed, expected_type):
        raise ValueError(
            f"Unexpected JSON type: expected {expected_type.__name__}, got {type(parsed).__name__}"
        )
    return parsed


# ---------- Sheet matching (prompt + LC chat model)
def build_sheet_matching_prompt(excel_sheets: str, target_tables: str) -> str:
    return f"""Ты – эксперт по интеграции данных. Сопоставь каждый лист Excel (его название и колонки) с наиболее подходящей таблицей из целевой схемы.

Excel sheets data:
{excel_sheets}

Target schema tables:
{target_tables}

Для каждого листа укажи:
- наиболее вероятную целевую таблицу (или null, если нет подходящей)
- степень схожести (high, medium, low)
- краткое объяснение

Формат ответа – JSON список:
[
    {{
        "sheet_name": "название листа",
        "target_table": "имя таблицы или null",
        "similarity": "high/medium/low",
        "reason": "почему"
    }}
]
"""


sheet_matching_chain = (
    RunnableLambda(
        lambda p: build_sheet_matching_prompt(p["excel_sheets"], p["target_tables"])
    )
    | RunnableLambda(_dispatch_invoke_llm_plain_text)
    | RunnableLambda(lambda text: _parse_model_json(text, list))
)


def match_sheets_to_tables(excel_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        sheets_data = []
        for sheet in excel_json.get("sheets", []):
            if sheet.get("skipped", False):
                continue
            sheet_name = sheet["sheet_name"]
            columns = sheet.get("columns", [])
            flat_cols = []
            for col in columns:
                if isinstance(col, list):
                    flat_cols.append(" > ".join(str(c) for c in col if c))
                else:
                    flat_cols.append(str(col))
            sheets_data.append({
                "sheet_name": sheet_name,
                "columns": flat_cols[:20]
            })
        target_tables = [{"name": t["name"], "description": t["description"], "columns": t["columns"]}
                         for t in TARGET_SCHEMA["tables"]]
        excel_sheets_str = json.dumps(sheets_data, ensure_ascii=False, indent=2)
        target_tables_str = json.dumps(target_tables, ensure_ascii=False, indent=2)

        raw_result = sheet_matching_chain.invoke(
            {"excel_sheets": excel_sheets_str, "target_tables": target_tables_str}
        )
        logger.info("Sheet matching completed")
        return raw_result
    except Exception as e:
        logger.exception(f"Sheet matching failed: {e}")
        return []


# ---------- Column mapping (prompt + LC chat model)
def build_column_mapping_prompt(target_table: str, target_columns: List[str],
                                sheet_name: str, excel_columns: List[str]) -> str:
    return f"""Ты – эксперт по маппингу данных. Для таблицы "{target_table}" (колонки: {target_columns}) 
сопоставь колонки из листа Excel "{sheet_name}" (список колонок: {excel_columns}) 
с целевыми колонками. Определи, какие Excel-колонки соответствуют каким целевым колонкам.

Верни JSON-объект, где ключ – целевая колонка, значение – Excel-колонка (или null, если нет соответствия). Также добавь поле "similarity" (high/medium/low) для общего соответствия.

Пример:
{{
    "mapping": {{
        "name": "Название таблицы",
        "system_code": "Код СИ"
    }},
    "similarity": "high"
}}
"""


def _finalize_column_mapping_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(d)
    if "mapping" not in m or m.get("mapping") is None:
        m["mapping"] = {}
    if "similarity" not in m:
        m["similarity"] = "low"
    return m


column_mapping_chain = (
    RunnableLambda(
        lambda p: build_column_mapping_prompt(
            p["target_table"],
            p["target_columns"],
            p["sheet_name"],
            p["excel_columns"],
        )
    )
    | RunnableLambda(_dispatch_invoke_llm_plain_text)
    | RunnableLambda(lambda text: _parse_model_json(text, dict))
    | RunnableLambda(_finalize_column_mapping_dict)
)


def map_columns_for_table(excel_json: Dict[str, Any], sheet_name: str, target_table_name: str) -> Dict[str, Any]:
    target_columns: List[str] = []
    excel_columns: List[str] = []
    try:
        sheet = None
        for s in excel_json.get("sheets", []):
            if s.get("sheet_name") == sheet_name and not s.get("skipped"):
                sheet = s
                break
        if not sheet:
            return {"error": f"Sheet '{sheet_name}' not found", "mapping": {}, "similarity": "low"}

        target_table = next((t for t in TARGET_SCHEMA["tables"] if t["name"] == target_table_name), None)
        if not target_table:
            return {"error": f"Target table '{target_table_name}' not found", "mapping": {}, "similarity": "low"}

        target_columns = target_table["columns"]
        for col in sheet.get("columns", []):
            if isinstance(col, list):
                excel_columns.append(" > ".join(str(c) for c in col if c))
            else:
                excel_columns.append(str(col))

        raw_result = column_mapping_chain.invoke(
            {
                "target_table": target_table_name,
                "target_columns": target_columns,
                "sheet_name": sheet_name,
                "excel_columns": excel_columns[:30],
            }
        )
        logger.info(f"Column mapping completed for {sheet_name} -> {target_table_name}")
        return raw_result
    except Exception as e:
        logger.exception(f"Column mapping failed for {target_table_name}: {e}")
        # Fallback: simple substring match
        mapping = {}
        for tcol in target_columns:
            for ecol in excel_columns:
                if tcol.lower() in ecol.lower() or ecol.lower() in tcol.lower():
                    mapping[tcol] = ecol
                    break
        return {"mapping": mapping, "similarity": "low", "error": str(e)}


# ---------- Helper to create graph edges from column mappings ----------
def create_graph_edges_from_mapping(file_hash: str, sheet_name: str, target_table_name: str,
                                    column_mapping: Dict[str, str]) -> None:
    """Create relationships in the graph for each column mapping."""
    try:
        from db_storage import get_db_connection, add_relationship
        from db_storage import get_column_id_by_name, get_target_column_id

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT sheet_hash FROM sheets WHERE file_hash = ? AND sheet_name = ?", (file_hash, sheet_name))
        row = cursor.fetchone()
        conn.close()
        if not row:
            logger.warning(f"Cannot find sheet_hash for {file_hash}/{sheet_name}")
            return
        sheet_hash = row["sheet_hash"]

        for target_col, source_col in column_mapping.items():
            if not source_col:
                logger.warning(f"Empty source column for target '{target_col}', skipping")
                continue
            source_col_id = get_column_id_by_name(sheet_hash, source_col)
            if not source_col_id:
                logger.warning(f"Source column '{source_col}' not found in DB")
                continue
            target_col_id = get_target_column_id(target_table_name, target_col)
            if not target_col_id:
                logger.warning(f"Cannot build target column id for '{target_table_name}.{target_col}'")
                continue
            add_relationship(source_col_id, target_col_id, "MAPS_TO",
                             metadata=json.dumps({"transformation": "direct"}))
            add_relationship(target_col_id, source_col_id, "DERIVED_FROM")
            logger.debug(f"Created edge: {source_col_id} -> {target_col_id} (MAPS_TO)")
    except Exception as e:
        logger.error(f"Failed to create graph edges: {e}")


# ---------- Main Comparison ----------
def compare_with_target(excel_json: Dict[str, Any], file_hash: str = None) -> Dict[str, Any]:
    """Match Excel sheets to target schema and optionally create graph edges."""
    if file_hash is None:
        file_hash = excel_json.get("file_hash")

    sheet_matches = match_sheets_to_tables(excel_json)
    mapping_suggestions = []
    for match in sheet_matches:
        target_table = match.get("target_table")
        if target_table and target_table != "null":
            column_mapping = map_columns_for_table(excel_json, match["sheet_name"], target_table)
            mapping_suggestions.append({
                "excel_sheet": match["sheet_name"],
                "target_table": target_table,
                "similarity": match.get("similarity", "low"),
                "explanation": match.get("reason", ""),
                "column_mapping": column_mapping.get("mapping", {}),
                "mapping_similarity": column_mapping.get("similarity", "low")
            })
            # Create graph edges if file_hash is available and mapping is not empty
            if file_hash and column_mapping.get("mapping"):
                create_graph_edges_from_mapping(
                    file_hash, match["sheet_name"], target_table,
                    column_mapping["mapping"]
                )
        else:
            mapping_suggestions.append({
                "excel_sheet": match["sheet_name"],
                "target_table": None,
                "similarity": "none",
                "explanation": match.get("reason", "No match"),
                "column_mapping": {}
            })

    score_map = {"high": 3, "medium": 2, "low": 1, "none": 0}
    # Prefer column-level confidence where available.
    total = sum(score_map.get(m.get("mapping_similarity", m["similarity"]), 0) for m in mapping_suggestions)
    count = len([m for m in mapping_suggestions if m["target_table"]])
    avg_score = (total / (count * 3) * 100) if count > 0 else 0

    unmatched_excel = [m["excel_sheet"] for m in mapping_suggestions if not m["target_table"]]
    matched_tables = set(m["target_table"] for m in mapping_suggestions if m["target_table"])
    all_target_tables = set(t["name"] for t in TARGET_SCHEMA["tables"])
    unmatched_target = list(all_target_tables - matched_tables)

    return {
        "similarity_score": round(avg_score),
        "mapping_suggestions": mapping_suggestions,
        "unmatched_excel_sheets": unmatched_excel,
        "unmatched_target_tables": unmatched_target,
        "recommendations": (
            "Рекомендуется использовать column_mapping для загрузки данных. "
            "Для таблиц без маппинга потребуется ручная настройка."
        )
    }