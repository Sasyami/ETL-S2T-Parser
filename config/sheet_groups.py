import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


SHEET_GROUPS_PATH = Path(__file__).with_name("sheet_groups.json")


def normalize_sheet_name(value: str) -> str:
    text = "" if value is None else str(value)
    text = text.casefold().replace("ё", "е")
    text = re.sub(r"\s*->\s*", "->", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@lru_cache(maxsize=1)
def load_sheet_groups() -> Dict[str, List[str]]:
    with SHEET_GROUPS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("sheet_groups.json must contain an object")

    groups: Dict[str, List[str]] = {}
    for group, aliases in data.items():
        if not isinstance(group, str) or not group.strip():
            raise ValueError("sheet group names must be non-empty strings")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(f"sheet group '{group}' must have a non-empty aliases list")
        clean_aliases = []
        for alias in aliases:
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(f"sheet group '{group}' contains an empty alias")
            clean_aliases.append(alias.strip())
        groups[group.strip()] = clean_aliases
    return groups


def clear_sheet_groups_cache() -> None:
    load_sheet_groups.cache_clear()


def add_sheet_group_alias(group: str, alias: Any, path: Optional[str] = None) -> List[str]:
    """Append a sheet-name alias to sheet_groups.json, deduplicating by normalized text."""
    text = "" if alias is None else str(alias).strip()
    if not group or not text:
        return []

    mapping_path = Path(path) if path else SHEET_GROUPS_PATH
    with mapping_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("sheet_groups.json must contain an object")
    if group not in data:
        raise ValueError(f"Unknown sheet group: {group}")

    aliases = data.get(group)
    if not isinstance(aliases, list):
        aliases = []
        data[group] = aliases

    normalized = normalize_sheet_name(text)
    for existing_group, existing_aliases in data.items():
        existing_values = existing_aliases if isinstance(existing_aliases, list) else []
        for existing_alias in existing_values + [existing_group]:
            if normalize_sheet_name(existing_alias) == normalized:
                if existing_group == group:
                    return []
                raise ValueError(
                    f"Alias {text!r} conflicts with existing sheet group {existing_group!r}"
                )

    aliases.append(text)
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    clear_sheet_groups_cache()
    return [text]


def aliases_for_group(group: str, groups: Optional[Dict[str, List[str]]] = None) -> Set[str]:
    source = groups if groups is not None else load_sheet_groups()
    aliases = set(source.get(group, []))
    aliases.add(group)
    return {normalize_sheet_name(alias) for alias in aliases}


def iter_group_aliases(
    groups: Optional[Dict[str, List[str]]] = None,
) -> Iterator[Tuple[str, str, str]]:
    source = groups if groups is not None else load_sheet_groups()
    for group, aliases in source.items():
        for alias in aliases + [group]:
            yield group, alias, normalize_sheet_name(alias)


def find_sheet_group_alias(
    sheet_name: str,
    groups: Optional[Dict[str, List[str]]] = None,
) -> Optional[Dict[str, str]]:
    raw = "" if sheet_name is None else str(sheet_name).strip()
    for group, alias, normalized_alias in iter_group_aliases(groups):
        if raw == alias:
            return {
                "group": group,
                "alias": alias,
                "normalized_alias": normalized_alias,
            }

    normalized = normalize_sheet_name(sheet_name)
    for group, alias, normalized_alias in iter_group_aliases(groups):
        if normalized == normalized_alias:
            return {
                "group": group,
                "alias": alias,
                "normalized_alias": normalized_alias,
            }
    return None


def sheet_name_in_group(sheet_name: str, group: str, groups: Optional[Dict[str, List[str]]] = None) -> bool:
    return normalize_sheet_name(sheet_name) in aliases_for_group(group, groups)


def group_for_sheet(sheet_name: str, groups: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
    match = find_sheet_group_alias(sheet_name, groups)
    return match["group"] if match else None
