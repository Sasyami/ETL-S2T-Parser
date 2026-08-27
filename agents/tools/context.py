"""Runtime skills and lazily selected data schemas for the chat agent."""

import json
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from .common import PROJECT_ROOT

PROMPTS_DIR = PROJECT_ROOT / "agents" / "prompts"
CONFIG_DIR = PROJECT_ROOT / "config"

SchemaName = Literal[
    "SQLite ETL",
    "S2T-маппинг",
    "Excel-маппинги",
    "Neo4j lineage",
]

SCHEMA_CATALOG: Dict[str, str] = {
    "SQLite ETL": (
        "Реальные колонки публичных SQLite-таблиц; нужна для составления "
        "произвольного run_sql (только read-only) и проверки физической "
        "структуры хранения."
    ),
    "S2T-маппинг": (
        "Поля S2T-кортежа, source/target-роли, алиасы заголовков S2T-листа "
        "и правила ETL-слоёв. Нужна для работы с сырой схемой/конфигурацией; "
        "готовые S2T-tools имеют собственные контракты."
    ),
    "Excel-маппинги": (
        "Группы и алиасы Excel-листов, роли их физических заголовков и "
        "настроенные цели извлечения."
    ),
    "Neo4j lineage": (
        "Labels, свойства и направления связей ETLTable/ETLColumn в Neo4j; "
        "нужна для произвольного run_cypher или ручного анализа графа, но не "
        "для готовых trace-tools."
    ),
}


_DOWNSTREAM_TABLE_DESCRIPTIONS: Dict[str, str] = {
    "files": "загруженные Excel-файлы и их сохранённые описания",
    "file_sheet_headers": "листы файлов и распознанные заголовки",
    "source_tables": "исходные логические таблицы и бизнес-описания таблиц",
    "target_tables": "целевые логические таблицы и бизнес-описания таблиц",
    "source_columns": "исходные колонки, их таблицы, типы и описания полей",
    "target_columns": "целевые колонки, их таблицы, типы и описания полей",
    "additional_objects": "Additional objects с точным именем и полным SQL",
    "pxf_to_a": "соответствия external, materialized и replica-таблиц",
    "s2t_transformations": "точные source→target таблицы, поля и текст правила",
    "data": "сырые значения ячеек Excel с координатами происхождения",
}


def get_downstream_capability_context() -> str:
    """Return compact planning capabilities without concrete tool names."""
    return "\n".join(
        [
            "Доступные возможности чтения (описывай нужные данные, не инструмент):",
            "- точные фильтры и списки файлов, листов, таблиц и колонок;",
            "- буквальный поиск таблиц и колонок по явно данному фрагменту;",
            "- смысловой поиск по описаниям, когда точное имя неизвестно;",
            "- точные и частичные S2T-строки, пары таблиц, правила и агрегации;",
            "- Additional objects по атрибутам или фрагменту, включая полный SQL;",
            "- lineage колонок и таблиц, пути, влияние и разбор явно данного SQL;",
            "- сырые значения Excel и read-only срезы хранилища;",
            "- полные сохранённые результаты прошлых workers и SQL-анализ их строк.",
        ]
    )


def get_downstream_table_context() -> str:
    """Return exact storage table names with compact planning descriptions."""
    from storage.database import USER_FACING_TABLES

    configured = set(_DOWNSTREAM_TABLE_DESCRIPTIONS)
    actual = set(USER_FACING_TABLES)
    if configured != actual:
        raise RuntimeError(
            "Downstream table descriptions are out of sync: "
            f"missing={sorted(actual - configured)}, "
            f"extra={sorted(configured - actual)}"
        )
    return "\n".join(
        [
            "Реальные таблицы хранилища (справка, не список шагов; "
            "наличие таблицы не требует её чтения):"
        ]
        + [
            f"- `{name}` — {_DOWNSTREAM_TABLE_DESCRIPTIONS[name]}."
            for name in USER_FACING_TABLES
        ]
    )


def _prompt_text(filename: str) -> str:
    try:
        return (PROMPTS_DIR / filename).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _format_backtick_list(names: Tuple[str, ...]) -> str:
    quoted = [f"`{name}`" for name in names]
    if len(quoted) <= 1:
        return "".join(quoted)
    return ", ".join(quoted[:-1]) + f" и {quoted[-1]}"


def get_sqlite_schema_cheatsheet() -> str:
    """Собрать блок схемы SQLite для prompt-ов агентов из storage/database.py."""
    from storage.database import (
        INTERNAL_TABLES,
        STORAGE_SCHEMA_COLUMNS,
        STORAGE_SCHEMA_TABLE_ORDER,
        S2T_RECORD_FIELDS,
        USER_FACING_TABLES,
    )

    rows = []
    for table_name in STORAGE_SCHEMA_TABLE_ORDER:
        columns = STORAGE_SCHEMA_COLUMNS[table_name]
        role = "публичная" if table_name in USER_FACING_TABLES else "внутренняя"
        rows.append(
            f"| `{table_name}` | {role} | "
            + ", ".join(f"`{column}`" for column in columns)
            + " |"
        )

    public_tables = _format_backtick_list(USER_FACING_TABLES)
    internal_tables = _format_backtick_list(INTERNAL_TABLES)
    internal_guidance = (
        f"- Внутренние таблицы упоминай только для явных вопросов про хранение или debug: {internal_tables}.\n"
        if INTERNAL_TABLES
        else ""
    )
    s2t_display_columns = _format_backtick_list(("row_num", *S2T_RECORD_FIELDS))
    return (
        "## Актуальная схема SQLite\n\n"
        "Блок с таблицами и колонками сгенерирован из `storage/database.py`; не подменяй его устаревшей документацией.\n\n"
        "| Таблица | Роль | Колонки (реальные имена) |\n"
        "|---------|------|--------------------------|\n"
        + "\n".join(rows)
        + "\n\n"
        "## Публичная DDL-схема для обычных вопросов в чате\n"
        "- Вопросы пользователя про \"таблицы\", \"DDL\" и \"схему\" трактуй как вопросы про публичный слой ETL/S2T, а не про все внутренние SQLite-таблицы.\n"
        f"- По умолчанию показывай только публичные таблицы: {public_tables}.\n"
        + internal_guidance
        + f"- Для `s2t_transformations` по умолчанию показывай только {s2t_display_columns}, если пользователь явно не просит сырой DDL.\n"
        "- Не перечисляй `sqlite_master` и служебные таблицы, если пользователь прямо не спрашивает про внутреннюю реализацию БД.\n"
    )


def _config_object(filename: str) -> Dict[str, object]:
    """Read one checked-in JSON config used as a schema source of truth."""
    return json.loads((CONFIG_DIR / filename).read_text(encoding="utf-8"))


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def get_s2t_mapping_schema_cheatsheet() -> str:
    """Build the current S2T tuple and upload-column mapping schema."""
    from storage.database import S2T_RECORD_FIELDS

    column_mapping = _config_object("column_mapping.json")
    extraction = _config_object("usefull_col_extraction.json")
    table_layers = _config_object("table_layers.json")
    return (
        "## Схема S2T-маппинга\n\n"
        "Источник истины — текущие config JSON и storage/database.py.\n"
        "- Один сохранённый кортеж: "
        + ", ".join(f"`{field}`" for field in S2T_RECORD_FIELDS)
        + ".\n"
        "- `source_*` описывает вход, `target_*` — результат; "
        "`transformation_rule` хранит правило или SQL как текст.\n"
        "- `s2t_transformations` глобальна: её нельзя автоматически фильтровать "
        "по `file_id`. Пустой `target_table` недопустим при загрузке.\n"
        "- Алиасы заголовков S2T-листа: "
        + _compact_json(column_mapping.get("s2t", {}))
        + "\n- Цель извлечения S2T: "
        + _compact_json(extraction.get("s2t_transformations", {}))
        + "\n- Правила слоёв: "
        + _compact_json(table_layers)
    )


def get_excel_mapping_schema_cheatsheet() -> str:
    """Build current non-S2T sheet and column matching schemas."""
    column_mapping = _config_object("column_mapping.json")
    column_mapping.pop("s2t", None)
    extraction = _config_object("usefull_col_extraction.json")
    extraction.pop("s2t_transformations", None)
    return (
        "## Схемы Excel-маппингов\n\n"
        "Источник истины — текущие config JSON; имена ниже являются "
        "настроенными ролями и алиасами, а не найденными строками файла.\n"
        "- Группы и алиасы листов: "
        + _compact_json(_config_object("sheet_groups.json"))
        + "\n- Алиасы колонок по группам: "
        + _compact_json(column_mapping)
        + "\n- Целевые поля извлечения: "
        + _compact_json(extraction)
    )


def get_neo4j_schema_cheatsheet() -> str:
    """Return the public graph projection schema used by lineage tools."""
    return (
        "## Схема Neo4j lineage\n\n"
        "- Узел `ETLTable`: точное имя таблицы в свойстве `name`.\n"
        "- Узел `ETLColumn`: `key`, `table_name`, `name`; wildcard хранится "
        "как отдельная колонка с `name=\"*\"`.\n"
        "- `(:ETLColumn)-[:TRANSFORMS_TO]->(:ETLColumn)` — направленная "
        "колонковая связь.\n"
        "- `(:ETLTable)-[:TABLE_TRANSFORMS_TO]->(:ETLTable)` — направленная "
        "табличная связь; SQL правила может находиться на ребре.\n"
        "- Все узлы проекции имеют label `ETLProjection`; исходные факты "
        "остаются в SQLite."
    )


def load_schemas(sections: Iterable[str]) -> str:
    """Load only the exact data schemas selected by the router."""
    loaders = {
        "SQLite ETL": get_sqlite_schema_cheatsheet,
        "S2T-маппинг": get_s2t_mapping_schema_cheatsheet,
        "Excel-маппинги": get_excel_mapping_schema_cheatsheet,
        "Neo4j lineage": get_neo4j_schema_cheatsheet,
    }
    selected: List[str] = []
    for section in dict.fromkeys(str(item) for item in sections):
        loader = loaders.get(section)
        if loader is not None:
            selected.append(loader().strip())
    return "\n\n---\n\n".join(part for part in selected if part)


def load_skills(sections: Optional[Iterable[str]] = None) -> str:
    """Загрузить все либо только выбранные разделы runtime skills."""
    text = _prompt_text("skills.md")
    if sections is None or not text:
        return text

    requested = {section.strip().casefold() for section in sections}
    lines = text.splitlines()
    preamble: List[str] = []
    blocks: List[Tuple[str, List[str]]] = []
    current_name: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_name is not None:
                blocks.append((current_name, current_lines))
            current_name = line[3:].strip()
            current_lines = [line]
        elif current_name is None:
            preamble.append(line)
        else:
            current_lines.append(line)

    if current_name is not None:
        blocks.append((current_name, current_lines))

    selected_lines = list(preamble)
    for name, block_lines in blocks:
        if name.casefold() in requested:
            if selected_lines and selected_lines[-1] != "":
                selected_lines.append("")
            selected_lines.extend(block_lines)

    return "\n".join(selected_lines).strip()


def load_chat_agent_context() -> str:
    """Загрузить runtime-контекст для Flask chat-agent."""
    return _prompt_text("chat_agent.md")


def load_upstream_analysis_context() -> str:
    """Загрузить постоянные правила анализа для upstream coordinator."""
    return _prompt_text("upstream_analysis.md")
