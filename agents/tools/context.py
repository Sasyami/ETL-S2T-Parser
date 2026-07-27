"""Runtime prompt and SQLite schema context for the chat agent."""

from typing import Tuple

from .common import PROJECT_ROOT

PROMPTS_DIR = PROJECT_ROOT / "agents" / "prompts"


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
        LEGACY_TABLES,
        REMOVED_WORKBOOK_TABLES,
        STORAGE_SCHEMA_COLUMNS,
        STORAGE_SCHEMA_TABLE_ORDER,
        S2T_FIELDS,
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
    removed_tables = ", ".join(
        f"`{name}`" for name in (LEGACY_TABLES + REMOVED_WORKBOOK_TABLES)
    )
    s2t_display_columns = _format_backtick_list(("row_num", *S2T_FIELDS))
    return (
        "\n\n---\n\n"
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
        f"- Устаревшие catalog/lineage-таблицы удалены: {removed_tables}.\n"
    )


def load_skills() -> str:
    """Загрузить runtime skills из каталога prompts."""
    return _prompt_text("skills.md")


def load_chat_agent_context() -> str:
    """Загрузить runtime-контекст для Flask chat-agent."""
    return _prompt_text("chat_agent.md")
