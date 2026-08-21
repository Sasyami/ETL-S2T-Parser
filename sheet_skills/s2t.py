"""Configured S2T inspection, column matching, writing and verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from agents.llm_factory import create_chat_model
from agents.sheet_group_classifier import classify_file_sheet_groups
from config.column_mapping import get_sheet_column_mapping
from config.table_layers import resolve_sheet_layers
from config.useful_columns import get_usefull_col_extraction_target
from sheet_skills.column_matching import (
    candidate_sheets,
    clean_value,
    column_evidence,
    load_columns,
    load_row_values,
    match_fields,
    persist_mapping_aliases,
)
from storage.s2t import insert_s2t_transformations, verify_s2t_transformations

USEFULL_COL_EXTRACTION_TARGET = "s2t_transformations"
USEFULL_COL_EXTRACTION_SUBAGENT = "usefull_col_extraction"


def _load_runtime_target_config() -> Dict[str, Any]:
    config = get_usefull_col_extraction_target(USEFULL_COL_EXTRACTION_TARGET)
    if not config:
        raise RuntimeError(f"Missing target: {USEFULL_COL_EXTRACTION_TARGET}")
    return config


_TARGET_CONFIG = _load_runtime_target_config()
USEFULL_SHEET_GROUP = str(_TARGET_CONFIG.get("sheet_group") or "").strip()
S2T_FIELDS = tuple(
    str(field).strip() for field in _TARGET_CONFIG.get("fields") or [] if str(field).strip()
)
if not USEFULL_SHEET_GROUP or not S2T_FIELDS:
    raise RuntimeError(f"Invalid target: {USEFULL_COL_EXTRACTION_TARGET}")


class S2TExtractionError(RuntimeError):
    def __init__(self, message: str, report: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.report = report or {"status": "error", "error": message}


class S2TRowValidationError(S2TExtractionError):
    """Raised before writing when a source row has no target_table."""


def _inspect_candidate_sheets(
    file_id: int, sheet_group_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    analysis = sheet_group_analysis or classify_file_sheet_groups(
        file_id, use_llm=True, persist_aliases=False
    )
    sheets = [
        {
            **sheet,
            "columns": load_columns(
                file_id, sheet["sheet_name"], sample_limit=5
            ),
        }
        for sheet in candidate_sheets(file_id, analysis, USEFULL_SHEET_GROUP)
    ]
    return {"file_id": file_id, "sheet_group_analysis": analysis, "sheets": sheets}


def _deterministic_sheet_mapping(
    sheet: Dict[str, Any], threshold: float = 0.70
) -> Dict[str, Any]:
    return match_fields(sheet, USEFULL_SHEET_GROUP, S2T_FIELDS, threshold)


def _validate_sheet_mappings(
    file_id: int, mappings: List[Dict[str, Any]], inspection: Dict[str, Any]
) -> List[Dict[str, Any]]:
    del file_id
    sheets = {
        str(sheet["sheet_name"]).casefold(): sheet
        for sheet in inspection.get("sheets", [])
    }
    result = []
    for mapping in mappings:
        sheet_name = str(mapping.get("sheet_name") or "")
        sheet = sheets.get(sheet_name.casefold())
        selected_raw = mapping.get("field_column_ids")
        if not sheet:
            raise ValueError(f"Unknown configured sheet_name: {sheet_name}")
        if not isinstance(selected_raw, dict):
            raise ValueError(f"{sheet_name}: missing field_column_ids object")
        unknown = sorted(set(selected_raw) - set(S2T_FIELDS))
        if unknown:
            raise ValueError(f"{sheet_name}: unknown configured fields {unknown}")

        selected = {field: selected_raw.get(field) or None for field in S2T_FIELDS}
        valid_ids = {column["column_id"] for column in sheet.get("columns", [])}
        invalid = {
            field: column_id
            for field, column_id in selected.items()
            if column_id and column_id not in valid_ids
        }
        if invalid:
            raise ValueError(
                f"{sheet_name}: fields point to unknown column_id values {invalid}"
            )
        assigned = [value for value in selected.values() if value]
        if len(assigned) != len(set(assigned)):
            raise ValueError(
                f"{sheet_name}: one physical column is assigned to multiple fields"
            )
        if not selected.get("target_table"):
            raise ValueError(f"{sheet_name}: required target_table is not mapped")

        evidence = dict(mapping.get("evidence") or {})
        missing_evidence = [
            field for field, column_id in selected.items() if column_id and field not in evidence
        ]
        if missing_evidence:
            raise ValueError(f"{sheet_name}: missing evidence for {missing_evidence}")
        result.append(
            {
                "sheet_name": sheet["sheet_name"],
                "field_column_ids": selected,
                "evidence": evidence,
            }
        )
    return result


def write_s2t_transformations_from_plan(
    file_id: int,
    sheet_mappings: List[Dict[str, Any]],
    inspection: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate the plan and transactionally replace S2T results."""
    source = inspection or _inspect_candidate_sheets(file_id)
    mappings = _validate_sheet_mappings(file_id, sheet_mappings, source)
    records, row_errors = [], []
    sheet_results = []
    for mapping in mappings:
        selected = mapping["field_column_ids"]
        sheet_result = {
            "sheet_name": mapping["sheet_name"],
            "rows_written": 0,
            "empty_target_columns_count": 0,
        }
        sheet_results.append(sheet_result)
        for row_num, row in load_row_values(
            file_id, mapping["sheet_name"]
        ).items():
            values = {
                field: clean_value(row.get(selected.get(field))) for field in S2T_FIELDS
            }
            if not any(values.values()):
                continue
            layers = resolve_sheet_layers(
                mapping["sheet_name"], sheet_group=USEFULL_SHEET_GROUP
            )
            if not values.get("source_table"):
                layers["source_layer"] = None
            if not values.get("target_table"):
                layers["target_layer"] = None
            if not values.get("target_table"):
                row_errors.append(
                    {
                        "file_id": file_id,
                        "sheet_name": mapping["sheet_name"],
                        "row_num": row_num,
                        "field": "target_table",
                        "error": "В строке S2T не заполнена целевая таблица",
                    }
                )
            else:
                records.append(
                    {
                        "file_id": file_id,
                        "sheet_name": mapping["sheet_name"],
                        "row_num": row_num,
                        **values,
                        **layers,
                    }
                )
                sheet_result["rows_written"] += 1
                if not values.get("target_field"):
                    sheet_result["empty_target_columns_count"] += 1
    if row_errors:
        report = {
            "status": "error",
            "stage": "validate_rows",
            "file_id": file_id,
            "error": "Найдены строки S2T без целевой таблицы",
            "validation_errors": row_errors,
        }
        raise S2TRowValidationError(report["error"], report)
    return {
        **insert_s2t_transformations(file_id, records),
        "sheet_mappings": mappings,
        "sheet_results": sheet_results,
        "empty_target_columns_count": sum(
            int(item["empty_target_columns_count"])
            for item in sheet_results
        ),
    }


def _invoke_llm_plain_text(prompt: str) -> str:
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
    return (create_chat_model() | StrOutputParser()).invoke(prompt)


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _short_samples(values: Iterable[Any]) -> List[str]:
    return [
        text if len(text) <= 120 else text[:119] + "..."
        for text in (str(value) for value in list(values or [])[:3])
    ]


def _llm_column_context(
    sheet: Dict[str, Any], draft: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    columns = list(sheet.get("columns", []))
    selected = (draft or {}).get("field_column_ids") or {}
    indices = {
        int(column["column_id"]): int(column.get("column_index") or 0)
        for column in columns
    }
    anchors = {
        "source": [
            indices[int(selected[field])]
            for field in ("source_table", "source_field")
            if selected.get(field) and int(selected[field]) in indices
        ],
        "target": [
            indices[int(selected[field])]
            for field in ("target_table", "target_field")
            if selected.get(field) and int(selected[field]) in indices
        ],
    }
    occurrences: Dict[str, int] = {}
    result = []
    for position, column in enumerate(columns):
        name = str(column.get("column_name_flat") or "").strip()
        occurrences[name] = occurrences.get(name, 0) + 1
        distances = {
            role: min(
                (abs(position - anchor) for anchor in role_anchors),
                default=10**6,
            )
            for role, role_anchors in anchors.items()
        }
        side_hint = (
            "unknown"
            if distances["source"] == distances["target"]
            else min(distances, key=distances.get)
        )
        result.append(
            {
                "column_name": name,
                "occurrence": occurrences[name],
                "side_hint": side_hint,
                "previous_column": (
                    columns[position - 1].get("column_name_flat")
                    if position > 0
                    else None
                ),
                "next_column": (
                    columns[position + 1].get("column_name_flat")
                    if position + 1 < len(columns)
                    else None
                ),
                "sample_values": _short_samples(column.get("sample_values") or []),
            }
        )
    return result


def _build_sheet_llm_prompt(
    sheet: Dict[str, Any], draft: Optional[Dict[str, Any]] = None
) -> str:
    mapping = {
        field: aliases
        for field, aliases in get_sheet_column_mapping(USEFULL_SHEET_GROUP).items()
        if field in set(S2T_FIELDS)
    }
    payload = {
        "sheet_name": sheet.get("sheet_name"),
        "column_mapping_json": {USEFULL_SHEET_GROUP: mapping},
        "columns": _llm_column_context(sheet, draft),
    }
    return (
        "Сопоставь полезные колонки одного настроенного листа с доступными полями "
        f"группы column_mapping_json {USEFULL_SHEET_GROUP!r}. "
        "Для каждого входного column_name и occurrence верни один mapping_field из "
        "column_mapping_json или null. Не придумывай значения и не назначай один "
        "mapping_field нескольким колонкам. Одинаковые заголовки различай по "
        "occurrence, соседним колонкам и side_hint: source относится к блоку "
        "источника, target — к блоку приёмника. Верни только JSON без markdown: "
        '{"sheet_name":"...","column_roles":[{"column_name":"...",'
        '"occurrence":1,"mapping_field":"target_table"},{"column_name":"...",'
        '"occurrence":1,"mapping_field":null}]}.\n\n'
        + json.dumps(payload, ensure_ascii=False, default=str)
    )


def _normalise_llm_field(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text not in S2T_FIELDS:
        raise ValueError(f"Unknown configured mapping_field: {text}")
    return text


def _sheet_mapping_from_column_roles(
    parsed: Dict[str, Any], sheet: Dict[str, Any]
) -> Dict[str, Any]:
    sheet_name = str(sheet.get("sheet_name") or "")
    if parsed.get("sheet_name") and parsed["sheet_name"] != sheet.get("sheet_name"):
        raise ValueError(f"{sheet_name}: LLM returned wrong sheet_name")
    roles = parsed.get("column_roles")
    if not isinstance(roles, list):
        raise ValueError(f"{sheet_name}: missing JSON list field column_roles")

    columns: Dict[tuple[str, int], Dict[str, Any]] = {}
    occurrences: Dict[str, int] = {}
    for column in sheet.get("columns", []):
        name = str(column.get("column_name_flat") or "").strip()
        if name:
            occurrences[name] = occurrences.get(name, 0) + 1
            columns[(name, occurrences[name])] = column

    selected, evidence, seen = {field: None for field in S2T_FIELDS}, {}, set()
    for item in roles:
        if not isinstance(item, dict):
            raise ValueError(f"{sheet_name}: column_roles items must be objects")
        name = str(item.get("column_name") or "").strip()
        try:
            occurrence = int(item.get("occurrence") or 1)
        except (TypeError, ValueError):
            raise ValueError(
                f"{sheet_name}: invalid occurrence for column_name {name!r}"
            )
        key = (name, occurrence)
        if key not in columns or key in seen:
            raise ValueError(
                f"{sheet_name}: unknown or duplicate column occurrence: {key!r}"
            )
        seen.add(key)
        field = _normalise_llm_field(item.get("mapping_field"))
        if field is None:
            continue
        if selected[field]:
            raise ValueError(
                f"{sheet_name}: mapping_field {field} is assigned to multiple columns"
            )
        column = columns[key]
        selected[field] = column["column_id"]
        evidence[field] = column_evidence(column, "llm", field)
    missing = sorted(set(columns) - seen)
    if missing:
        raise ValueError(
            f"{sheet_name}: column_roles must include every column_name, missing {missing}"
        )
    return {
        "sheet_name": sheet.get("sheet_name"),
        "field_column_ids": selected,
        "evidence": evidence,
    }


def _resolve_sheet_mapping(
    file_id: int, sheet: Dict[str, Any], inspection: Dict[str, Any]
) -> Dict[str, Any]:
    draft = _deterministic_sheet_mapping(sheet)
    if all(draft["field_column_ids"].get(field) for field in S2T_FIELDS):
        mapping = _validate_sheet_mappings(file_id, [draft], inspection)[0]
        return {"mapping": mapping, "attempts": 0, "method": "deterministic"}
    try:
        candidate = _sheet_mapping_from_column_roles(
            _extract_json_object(
                _invoke_llm_plain_text(_build_sheet_llm_prompt(sheet, draft))
            ),
            sheet,
        )
        mapping = _validate_sheet_mappings(file_id, [candidate], inspection)[0]
        return {"mapping": mapping, "attempts": 1, "method": "llm"}
    except Exception as exc:
        return {
            "mapping": None,
            "attempts": 1,
            "method": "llm",
            "error": str(exc) or "useful-column matching failed",
        }


def _make_report(
    file_id: int,
    status: str,
    attempts: int = 0,
    target: str = USEFULL_COL_EXTRACTION_TARGET,
    **details: Any,
) -> Dict[str, Any]:
    return {
        "status": status,
        "file_id": file_id,
        "subagent": USEFULL_COL_EXTRACTION_SUBAGENT,
        "target": target,
        "attempts": attempts,
        **details,
    }


def run_s2t_extraction_subagent(
    file_id: int, sheet_group_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Classify sheets, resolve mappings, write rows and verify the database."""
    analysis = sheet_group_analysis or classify_file_sheet_groups(
        file_id, use_llm=True, persist_aliases=True
    )
    inspection = _inspect_candidate_sheets(file_id, analysis)
    attempts, mappings, sheets = 0, [], []
    try:
        for sheet in inspection["sheets"]:
            resolved = _resolve_sheet_mapping(file_id, sheet, inspection)
            attempts += resolved["attempts"]
            if resolved.get("mapping") is None:
                report = _make_report(
                    file_id, "error", attempts, sheet_name=sheet["sheet_name"],
                    attempt=resolved["attempts"], error=resolved["error"],
                )
                raise S2TExtractionError(report["error"], report)
            mappings.append(resolved["mapping"])
            sheets.append(
                {
                    "sheet_name": sheet["sheet_name"],
                    "method": resolved["method"],
                    "attempts": resolved["attempts"],
                }
            )

        aliases_added = persist_mapping_aliases(USEFULL_SHEET_GROUP, mappings)
        write_report = write_s2t_transformations_from_plan(
            file_id, mappings, inspection=inspection
        )
        sheet_results_by_name = {
            str(item["sheet_name"]).casefold(): item
            for item in write_report["sheet_results"]
        }
        for sheet in sheets:
            sheet_result = sheet_results_by_name.get(
                str(sheet["sheet_name"]).casefold(),
                {},
            )
            sheet["rows_written"] = int(sheet_result.get("rows_written", 0))
            sheet["empty_target_columns_count"] = int(
                sheet_result.get("empty_target_columns_count", 0)
            )
        verification = verify_s2t_transformations(file_id)
        return _make_report(
            file_id, "ok", attempts, sheets=sheets, aliases_added=aliases_added,
            written=int(write_report["count"]),
            empty_target_columns_count=int(
                write_report["empty_target_columns_count"]
            ),
            verification=verification,
            sheet_mappings=write_report["sheet_mappings"],
        )
    except S2TExtractionError:
        raise
    except Exception as exc:
        report = _make_report(
            file_id, "error", attempts,
            error=str(exc) or "Useful-column extraction failed",
        )
        raise S2TExtractionError(report["error"], report) from exc
