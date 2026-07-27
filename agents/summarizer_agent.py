import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from storage.database import (
    get_db_connection,
    update_file_description,
    update_file_summary,
)
from .agent import chat_model

try:
    from langfuse import observe
    from .observability import get_callback_handler

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

    def observe(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def get_callback_handler():
        return None


logger = logging.getLogger(__name__)

SUMMARY_ROWS_PER_SHEET = 5
SUMMARY_CELL_CHAR_LIMIT = 300
SYSTEM_PROMPT = "Сделай краткое саммари на русском языке по переданным табличным данным."
SUMMARY_OUTPUT_REQUIREMENTS = """
Сформируй один цельный абзац из 3–5 предложений объёмом не более 1200 символов.
Опиши назначение данных, основные сущности и только те связи, маппинги или правила,
которые прямо подтверждаются названиями колонок и примерами строк.
Не придумывай отсутствующие факты и не перечисляй механически все колонки.
Сразу начни с предметной области или назначения данных, без фраз «документ описывает»
и «данные содержат». Не упоминай запрос пользователя, JSON, Excel, файл, документ,
листы, выборку строк или процесс анализа.
Не используй Markdown-заголовки, списки, вступления и заключения.
""".strip()
SUMMARY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "business_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1200,
                }
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
}


def _summarizer_messages(inp: Dict[str, str]) -> List[BaseMessage]:
    return [
        SystemMessage(content=f"{SYSTEM_PROMPT}\n\n{SUMMARY_OUTPUT_REQUIREMENTS}"),
        HumanMessage(
            content=(
                f"{SUMMARY_OUTPUT_REQUIREMENTS}\n\n"
                "Верни только итоговое саммари без пояснений.\n\n"
                f"Данные:\n{inp['user_content']}"
            )
        ),
    ]


_summarizer_llm_chain = (
    RunnableLambda(_summarizer_messages)
    | chat_model
    | StrOutputParser()
)


def call_gigachat(user_content: str) -> str:
    """Invoke the configured chat model for summary-related text."""
    if getattr(chat_model, "supports_json_schema", False) is True:
        reply = chat_model.invoke(
            _summarizer_messages({"user_content": user_content}),
            response_format=SUMMARY_RESPONSE_FORMAT,
        )
        raw = StrOutputParser().invoke(reply).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM returned invalid structured summary JSON") from exc
        summary = payload.get("summary") if isinstance(payload, dict) else None
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("LLM returned an empty structured summary")
        return summary.strip()
    return _summarizer_llm_chain.invoke({"user_content": user_content}).strip()


def _file_text_fields(file_id: int) -> Dict[str, Any]:
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT file_id, filename, summary, description FROM files WHERE file_id = ?",
            (file_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise ValueError(f"File {file_id} not found")
    return dict(row)


def _sheet_columns(sheet_id: int, headers_json: Optional[str]) -> List[Dict[str, Any]]:
    try:
        parsed_headers = json.loads(headers_json or "[]")
    except (TypeError, json.JSONDecodeError):
        parsed_headers = []

    columns: List[Dict[str, Any]] = []
    for position, item in enumerate(parsed_headers):
        if not isinstance(item, dict):
            continue
        flat_name = str(item.get("flat") or "").strip()
        if not flat_name:
            path = item.get("path")
            if isinstance(path, list):
                flat_name = " > ".join(str(part) for part in path if part)
        if not flat_name:
            continue
        try:
            column_index = int(item.get("index", position))
        except (TypeError, ValueError):
            column_index = position
        columns.append(
            {
                "index": column_index,
                "name": flat_name,
                "column_id": column_index + 1,
            }
        )
    return sorted(columns, key=lambda column: column["index"])


def _first_sheet_rows(cursor: Any, sheet_id: int, columns: List[Dict[str, Any]]) -> List[List[Any]]:
    cursor.execute(
        """
        SELECT row_num, column_id, value
        FROM data
        WHERE sheet_id = ?
          AND row_num IN (
              SELECT row_num
              FROM data
              WHERE sheet_id = ?
              GROUP BY row_num
              ORDER BY row_num
              LIMIT ?
          )
        ORDER BY row_num, id
        """,
        (sheet_id, sheet_id, SUMMARY_ROWS_PER_SHEET),
    )
    values_by_row: Dict[int, Dict[str, Any]] = {}
    for cell in cursor.fetchall():
        values_by_row.setdefault(cell["row_num"], {})[cell["column_id"]] = cell["value"]

    return [
        [row_values.get(column["column_id"]) for column in columns]
        for _, row_values in sorted(values_by_row.items())
    ]


def fetch_file_data(file_id: int) -> Dict[str, Any]:
    """Return sheet names, column names, and at most five aligned rows per sheet."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM files WHERE file_id = ?", (file_id,))
        if not cursor.fetchone():
            raise ValueError(f"File {file_id} not found")

        cursor.execute(
            """
            SELECT sheet_id, sheet_name, headers_json
            FROM file_sheet_headers
            WHERE file_id = ? AND IFNULL(skipped, 0) = 0
            ORDER BY sheet_name
            """,
            (file_id,),
        )
        header_rows = cursor.fetchall()

        sheets = []
        for header_row in header_rows:
            sheet_id = header_row["sheet_id"]
            columns = _sheet_columns(sheet_id, header_row["headers_json"])
            sheets.append(
                {
                    "sheet_name": header_row["sheet_name"],
                    "columns": [column["name"] for column in columns],
                    "rows": _first_sheet_rows(cursor, sheet_id, columns),
                }
            )
    finally:
        conn.close()

    return {"sheets": sheets, "final_summary": ""}


def _compact_summary_value(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).replace("\r\n", "\n").strip()
    if len(text) <= SUMMARY_CELL_CHAR_LIMIT:
        return text
    return f"{text[:SUMMARY_CELL_CHAR_LIMIT - 1]}…"


def _summary_payload_sheets(sheets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "sheet_name": sheet["sheet_name"],
            "columns": [_compact_summary_value(column) for column in sheet["columns"]],
            "rows": [
                [_compact_summary_value(value) for value in row]
                for row in sheet["rows"][:SUMMARY_ROWS_PER_SHEET]
            ],
        }
        for sheet in sheets
    ]


@observe()
def summarize_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps(
        {"sheets": _summary_payload_sheets(state["sheets"])},
        ensure_ascii=False,
        default=str,
    )
    state["final_summary"] = call_gigachat(payload)
    return state


summarizer_chain = (
    RunnableLambda(fetch_file_data)
    | RunnableLambda(summarize_snapshot)
)


@observe()
def generate_summary(file_id: int) -> str:
    handler = get_callback_handler()
    if handler:
        chain_with_config = summarizer_chain.with_config(
            {
                "callbacks": [handler],
                "run_name": f"summarize_{file_id}",
            }
        )
        result = chain_with_config.invoke(file_id)
    else:
        result = summarizer_chain.invoke(file_id)
    return result["final_summary"]


def summarize_file(file_id: int, save: bool = True) -> str:
    summary = generate_summary(file_id)
    if save:
        update_file_summary(file_id, summary)
    return summary


def generate_description_from_summary(summary: str) -> str:
    prompt = f"""
Сформируй краткое описание данных на русском языке по готовому бизнес-саммари ниже.

Требования:
- 2–3 предложения, один короткий абзац;
- пиши только о предметной области, ключевых сущностях, видимых правилах и назначении данных;
- не упоминай Excel, файл, листы, загрузку, документ или рабочую книгу;
- не добавляй вводных фраз вида «данный файл содержит».

Готовое бизнес-саммари:
{summary}
"""
    return call_gigachat(prompt).strip()


def ensure_file_description(
    file_id: int,
    refresh: bool = False,
    save: bool = True,
    summary_override: Optional[str] = None,
) -> str:
    fields = _file_text_fields(file_id)
    cached_description = str(fields.get("description") or "").strip()
    if cached_description and not refresh:
        return cached_description

    summary = str(summary_override or "").strip() or str(fields.get("summary") or "").strip()
    if not summary:
        summary = summarize_file(file_id, save=save)

    description = generate_description_from_summary(summary)
    if save:
        update_file_description(file_id, description)
    return description


def generate_description_update_from_user_query(
    current_description: str,
    summary: str,
    user_query: str,
) -> str:
    prompt = f"""
Обнови краткое описание данных по уточнению пользователя.

Текущее краткое описание:
{current_description}

Текущее бизнес-саммари:
{summary}

Запрос пользователя:
{user_query}

Требования:
- верни только обновлённое краткое описание на русском языке;
- 2–4 предложения, один короткий абзац;
- опирайся на сохранённое описание, бизнес-саммари и факты из запроса пользователя;
- если пользователь уточняет или исправляет акцент описания, учти это;
- не упоминай Excel, файл, листы, загрузку, документ или рабочую книгу;
- не придумывай факты, которых нет в саммари или запросе пользователя.
"""
    return call_gigachat(prompt).strip()


def update_file_description_from_user_query(
    file_id: int,
    user_query: str,
    save: bool = True,
) -> str:
    request_text = str(user_query or "").strip()
    if not request_text:
        raise ValueError("user_query must be non-empty")

    base_description = ensure_file_description(file_id, refresh=False, save=save)
    fields = _file_text_fields(file_id)
    summary = str(fields.get("summary") or "").strip()
    updated_description = generate_description_update_from_user_query(
        current_description=base_description,
        summary=summary,
        user_query=request_text,
    )
    if save:
        update_file_description(file_id, updated_description)
    return updated_description
