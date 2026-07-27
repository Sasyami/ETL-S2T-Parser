"""Read-only tools for the synchronized Neo4j projection."""

import logging
import re
from typing import Any, Dict, Literal, Optional

from langchain_core.tools import tool

from graph_storage import execute_neo4j_read

from .common import clamped_int

logger = logging.getLogger(__name__)

MAX_CYPHER_ROWS = 100
_READONLY_CYPHER_START = re.compile(
    r"^(?:MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|RETURN|SHOW|EXPLAIN|PROFILE)\b",
    re.IGNORECASE,
)
_MUTATING_CYPHER_CLAUSE = re.compile(
    r"(?<![.:])\b(?:"
    r"CREATE|MERGE|INSERT|DELETE|DETACH|SET|REMOVE|DROP|ALTER|RENAME|"
    r"GRANT|DENY|REVOKE|TERMINATE|START|STOP|FOREACH|CALL"
    r")\b|\bLOAD\s+CSV\b",
    re.IGNORECASE,
)


def _read_rows(
    query: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        return {"rows": execute_neo4j_read(query, parameters)}
    except KeyError as exc:
        return {
            "error": f"Neo4j setting is missing: {exc.args[0]}",
            "rows": [],
        }
    except Exception:
        logger.exception("Neo4j read failed")
        return {
            "error": "Neo4j read failed",
            "rows": [],
        }


def _strip_cypher_literals_and_comments(query: str) -> str:
    """Hide literals, quoted identifiers and comments before clause validation."""
    output = []
    index = 0
    state = "text"
    while index < len(query):
        char = query[index]
        following = query[index + 1] if index + 1 < len(query) else ""

        if state == "text":
            if char == "/" and following == "/":
                output.extend((" ", " "))
                state = "line_comment"
                index += 2
                continue
            if char == "/" and following == "*":
                output.extend((" ", " "))
                state = "block_comment"
                index += 2
                continue
            if char in {"'", '"', "`"}:
                output.append(" ")
                state = {
                    "'": "single_quote",
                    '"': "double_quote",
                    "`": "backtick",
                }[char]
                index += 1
                continue
            output.append(char)
            index += 1
            continue

        if state == "line_comment":
            output.append("\n" if char in "\r\n" else " ")
            if char in "\r\n":
                state = "text"
            index += 1
            continue

        if state == "block_comment":
            output.append(" ")
            if char == "*" and following == "/":
                output.append(" ")
                state = "text"
                index += 2
            else:
                index += 1
            continue

        output.append(" ")
        closing = {
            "single_quote": "'",
            "double_quote": '"',
            "backtick": "`",
        }[state]
        if char == "\\" and state != "backtick" and following:
            output.append(" ")
            index += 2
            continue
        if char == closing:
            if following == closing:
                output.append(" ")
                index += 2
                continue
            state = "text"
        index += 1

    return "".join(output)


def _validate_readonly_cypher(query: str) -> Optional[str]:
    text = query.strip()
    if not text:
        return "query must be non-empty"

    stripped = _strip_cypher_literals_and_comments(text).strip()
    statements = [
        statement.strip()
        for statement in stripped.rstrip(";").split(";")
        if statement.strip()
    ]
    if len(statements) != 1:
        return "Exactly one Cypher statement is allowed"
    if not _READONLY_CYPHER_START.match(statements[0]):
        return (
            "Only MATCH, OPTIONAL MATCH, WITH, UNWIND, RETURN, SHOW, "
            "EXPLAIN and PROFILE queries are allowed"
        )
    forbidden = _MUTATING_CYPHER_CLAUSE.search(statements[0])
    if forbidden:
        return f"Mutating or procedural Cypher is not allowed: {forbidden.group(0)}"
    return None


@tool(parse_docstring=True)
def run_cypher(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Выполнить один свободный read-only Cypher-запрос к колонкам Neo4j.

    Используй только для сложных графовых путей, цепочек зависимостей, обходов
    соседних колонок и impact analysis. В графе есть только узлы ETLColumn и
    рёбра TRANSFORMS_TO; сведения о файлах, листах, таблицах, правилах и строках
    S2T получай из SQLite. Для обычной таблицы S2T-трансформаций, строк,
    маппингов, правил и source → target используй SQLite-tools, а не этот
    инструмент. Поддерживаются MATCH, OPTIONAL MATCH, WITH, UNWIND, RETURN,
    SHOW, EXPLAIN и PROFILE. Изменяющие конструкции, несколько выражений и
    процедурный CALL отклоняются до обращения к Neo4j.

    Args:
        query: Полный текст одного read-only Cypher-запроса.
        parameters: Именованные параметры Cypher без символа $ в ключах.
        limit: Максимальное число строк в ответе, от 1 до 100.
    """
    text = (query or "").strip()
    validation_error = _validate_readonly_cypher(text)
    if validation_error:
        return {
            "error": validation_error,
            "query": text,
        }

    clean_parameters = dict(parameters or {})
    clean_limit = clamped_int(limit, 20, 1, MAX_CYPHER_ROWS)
    try:
        rows = execute_neo4j_read(
            text,
            clean_parameters,
            row_limit=clean_limit + 1,
        )
    except KeyError as exc:
        return {
            "error": f"Neo4j setting is missing: {exc.args[0]}",
            "query": text,
        }
    except Exception:
        logger.exception("Cypher execution failed")
        return {
            "error": "Cypher query failed",
            "query": text,
        }

    truncated = len(rows) > clean_limit
    visible_rows = rows[:clean_limit]
    return {
        "query": text,
        "parameters": clean_parameters,
        "columns": list(visible_rows[0]) if visible_rows else [],
        "rows": visible_rows,
        "returned_rows": len(visible_rows),
        "truncated": truncated,
        "limit": clean_limit,
    }


@tool(parse_docstring=True)
def trace_neo4j_lineage(
    table_name: str,
    column_name: Optional[str] = None,
    file_id: Optional[int] = None,
    direction: Literal["upstream", "downstream", "both"] = "both",
    limit: int = 50,
) -> Dict[str, Any]:
    """Найти непосредственный upstream/downstream lineage колонок.

    Используй только когда пользователь просит lineage, upstream, downstream,
    зависимые колонки или непосредственную связь в графе. Для обычного показа
    таблицы S2T, строк, маппингов и правил используй SQLite-tools.
    Граф содержит только узлы ETLColumn и прямые рёбра TRANSFORMS_TO. Имена
    таблиц и колонок сравниваются точно, без нормализации и подстановки похожих
    значений.
    Если column_name отсутствует, возвращаются связи всех колонок указанной
    таблицы. Подробности трансформации получай из SQLite по transformation_id.

    Args:
        table_name: Точное имя исходной или целевой логической ETL-таблицы.
        column_name: Точное имя колонки либо null для lineage всей таблицы.
        file_id: Опциональный идентификатор файла для ограничения графа.
        direction: upstream, downstream или оба направления both.
        limit: Максимальное число найденных связей колонок, от 1 до 100.
    """
    clean_table_name = str(table_name or "").strip()
    if not clean_table_name:
        return {
            "error": "table_name must be non-empty",
            "rows": [],
        }
    clean_column_name = (
        str(column_name).strip()
        if column_name is not None and str(column_name).strip()
        else None
    )
    clean_file_id = int(file_id) if file_id is not None else None
    clean_limit = clamped_int(limit, 50, 1, 100)
    result = _read_rows(
        """
        MATCH (source:ETLProjection:ETLColumn)
              -[mapping:TRANSFORMS_TO]->
              (target:ETLProjection:ETLColumn)
        WHERE
            ($file_id IS NULL OR mapping.file_id = $file_id)
            AND (
                (
                    $column_name IS NULL
                    AND (
                        (
                            $direction IN ['downstream', 'both']
                            AND source.table_name = $table_name
                        )
                        OR (
                            $direction IN ['upstream', 'both']
                            AND target.table_name = $table_name
                        )
                    )
                )
                OR (
                    $column_name IS NOT NULL
                    AND (
                        (
                            $direction IN ['downstream', 'both']
                            AND source.table_name = $table_name
                            AND source.name = $column_name
                        )
                        OR (
                            $direction IN ['upstream', 'both']
                            AND target.table_name = $table_name
                            AND target.name = $column_name
                        )
                    )
                )
            )
        RETURN
            mapping.file_id AS file_id,
            mapping.transformation_id AS transformation_id,
            source.table_name AS source_table,
            source.name AS source_field,
            target.table_name AS target_table,
            target.name AS target_field,
            CASE
                WHEN source.table_name = $table_name
                     AND (
                         $column_name IS NULL
                         OR source.name = $column_name
                     )
                THEN 'downstream'
                ELSE 'upstream'
            END AS match_direction
        ORDER BY mapping.file_id, mapping.transformation_id
        LIMIT $limit
        """,
        {
            "table_name": clean_table_name,
            "column_name": clean_column_name,
            "file_id": clean_file_id,
            "direction": direction,
            "limit": clean_limit,
        },
    )
    result.update(
        {
            "table_name": clean_table_name,
            "column_name": clean_column_name,
            "file_id": clean_file_id,
            "direction": direction,
            "limit": clean_limit,
            "returned_rows": len(result["rows"]),
        }
    )
    return result
