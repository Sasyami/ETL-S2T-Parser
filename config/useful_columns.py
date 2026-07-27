import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from .column_mapping import load_column_mapping
from .sheet_groups import load_sheet_groups


USEFULL_COL_EXTRACTION_PATH = Path(__file__).with_name("usefull_col_extraction.json")


@lru_cache(maxsize=8)
def load_usefull_col_extraction_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else USEFULL_COL_EXTRACTION_PATH
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def clear_usefull_col_extraction_cache() -> None:
    load_usefull_col_extraction_config.cache_clear()


def _clean_group_name(value: Any) -> str:
    return str(value or "").strip()


def _normalise_fields(raw_fields: Any) -> List[str]:
    if not isinstance(raw_fields, list):
        return []

    seen: set[str] = set()
    fields: List[str] = []
    for raw_field in raw_fields:
        field = _clean_group_name(raw_field)
        if not field or field in seen:
            continue
        seen.add(field)
        fields.append(field)
    return fields


def get_usefull_col_extraction_target(target_name: str, path: Optional[str] = None) -> Dict[str, Any]:
    target = load_usefull_col_extraction_config(path).get(target_name, {})
    if not isinstance(target, dict):
        return {}

    sheet_group = _clean_group_name(target.get("sheet_group"))
    fields = _normalise_fields(target.get("fields"))

    if not sheet_group:
        raise ValueError(f"usefull_col_extraction target '{target_name}' must define sheet_group")
    if not fields:
        raise ValueError(f"usefull_col_extraction target '{target_name}' must define fields as a list")

    known_sheet_groups = load_sheet_groups()
    if sheet_group not in known_sheet_groups:
        raise ValueError(
            f"usefull_col_extraction target '{target_name}' references unknown sheet_group '{sheet_group}'"
        )

    known_column_groups = load_column_mapping()
    if sheet_group not in known_column_groups:
        raise ValueError(
            f"usefull_col_extraction target '{target_name}' sheet_group "
            f"'{sheet_group}' has no column_mapping group"
        )

    valid_mapping_fields = known_column_groups[sheet_group]
    missing_fields = [field for field in fields if field not in valid_mapping_fields]
    if missing_fields:
        missing_str = ", ".join(missing_fields)
        raise ValueError(
            f"usefull_col_extraction target '{target_name}' references unknown mapping fields: {missing_str}"
        )

    return {
        "sheet_group": sheet_group,
        "fields": fields,
    }
