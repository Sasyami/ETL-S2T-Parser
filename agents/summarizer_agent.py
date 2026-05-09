import logging
import json
import re
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda
from db_storage import get_db_connection, update_file_summary
from .agent import chat_model
from load_skills_tools import load_skills, load_tools

# Langfuse imports
try:
    from langfuse import observe
    from langfuse_setup import get_callback_handler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    def observe(*args, **kwargs):
        def decorator(func): return func
        return decorator
    def get_callback_handler(): return None

logger = logging.getLogger(__name__)

# Columns whose headers suggest human-authored descriptions (S2T / data-dictionary sheets).
_DESCRIPTION_HEADER_HINTS = (
    "description",
    "описание",
    "опис",
    "attribute note",
    "column attribute",
    "note",
    "comment",
    "примечание",
    "назначение",
    "definition",
    "table desc",
    "memo",
    "summary",
)


def _column_looks_like_description(flat_name: str) -> bool:
    if not flat_name:
        return False
    n = flat_name.lower()
    return any(h in n for h in _DESCRIPTION_HEADER_HINTS)


SYSTEM_PROMPT = f"""
{load_skills()}

{load_tools()}

You analyze **business meaning in source and column metadata** (descriptions, notes, domain-oriented names).
**Do not** discuss the workbook as an artifact: no Excel, sheets, tabs, file name, upload, “this document”, page counts,
or S2T/mapping “as a document type”. Speak only about the **subject domain, data objects, and processes** implied by field content.
Use the skills and tools above where relevant.
"""

_schema_json_parser = JsonOutputParser()


def _summarizer_messages(inp: Dict[str, str]) -> List[BaseMessage]:
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=inp["user_content"]),
    ]


_summarizer_llm_chain = (
    RunnableLambda(_summarizer_messages)
    | chat_model
    | StrOutputParser()
)


def call_gigachat(user_content: str) -> str:
    """Call GigaChat with system prompt and user content (LCEL)."""
    return _summarizer_llm_chain.invoke({"user_content": user_content}).strip()


def fetch_file_data(file_hash: str) -> Dict[str, Any]:
    """Fetch file metadata, sheets, columns, sample rows, and important values."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM files WHERE file_hash = ?", (file_hash,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"File hash {file_hash} not found")
    filename = row["filename"]

    cursor.execute("""
        SELECT s.sheet_name, s.header_start_row, s.header_rows_count, s.nested_structure,
               c.column_index, c.column_name_flat, c.column_header
        FROM sheets s
        JOIN columns c ON s.sheet_hash = c.sheet_hash
        WHERE s.file_hash = ?
        ORDER BY s.sheet_name, c.column_index
    """, (file_hash,))
    rows = cursor.fetchall()

    sheets_dict = {}
    for r in rows:
        sheet_name = r["sheet_name"]
        if sheet_name not in sheets_dict:
            sheets_dict[sheet_name] = {
                "sheet_name": sheet_name,
                "columns": [],
                "sample_rows": [],
                "description_cells": [],
            }
        sheets_dict[sheet_name]["columns"].append(r["column_name_flat"])

    # Collect description-like cell text per sheet (metadata: table/column descriptions).
    sheet_hashes_by_name: Dict[str, str] = {}
    for r in rows:
        sn = r["sheet_name"]
        if sn not in sheet_hashes_by_name:
            cursor.execute(
                """
                SELECT sheet_hash FROM sheets
                WHERE file_hash = ? AND sheet_name = ?
                """,
                (file_hash, sn),
            )
            sh = cursor.fetchone()
            if sh:
                sheet_hashes_by_name[sn] = sh["sheet_hash"]

    description_snippets: list[str] = []
    for sheet_name, sheet_hash in sheet_hashes_by_name.items():
        cursor.execute(
            """
            SELECT c.column_hash, c.column_name_flat
            FROM columns c
            WHERE c.sheet_hash = ?
            """,
            (sheet_hash,),
        )
        col_rows = cursor.fetchall()
        texts: list[str] = []
        for cr in col_rows:
            if isinstance(cr, dict):
                cname = (cr.get("column_name_flat") or "").strip()
                ch = cr.get("column_hash")
            else:
                try:
                    cname = (cr["column_name_flat"] or "").strip()
                    ch = cr["column_hash"]
                except (KeyError, TypeError):
                    continue
            if not ch:
                continue
            if not _column_looks_like_description(cname):
                continue
            cursor.execute(
                """
                SELECT DISTINCT d.value
                FROM data d
                WHERE d.column_hash = ? AND d.value IS NOT NULL
                  AND TRIM(d.value) != ''
                """,
                (ch,),
            )
            for vr in cursor.fetchall():
                v = (vr["value"] or "").strip()
                if len(v) < 8:
                    continue
                cut = v[:1200]
                texts.append(cut)
                if len(description_snippets) < 400:
                    description_snippets.append(f"[{sheet_name} / {cname}] {cut}")

        sheets_dict[sheet_name]["description_cells"] = texts[:80]

    for sheet_name in sheets_dict:
        cursor.execute("""
            SELECT s.sheet_hash
            FROM sheets s
            WHERE s.file_hash = ? AND s.sheet_name = ?
        """, (file_hash, sheet_name))
        sheet_row = cursor.fetchone()
        if not sheet_row:
            continue
        sheet_hash = sheet_row["sheet_hash"]
        cursor.execute("""
            SELECT d.row_num, GROUP_CONCAT(d.value, ' | ') AS row_values
            FROM data d
            WHERE d.sheet_hash = ?
            GROUP BY d.row_num
            ORDER BY d.row_num
            LIMIT 5
        """, (sheet_hash,))
        sample_rows = cursor.fetchall()
        sheets_dict[sheet_name]["sample_rows"] = [
            {"row_num": r["row_num"], "values": r["row_values"][:500] if r["row_values"] else ""}
            for r in sample_rows
        ]

    important_values = set()
    cursor.execute("""
        SELECT DISTINCT d.value
        FROM data d
        JOIN sheets s ON d.sheet_hash = s.sheet_hash
        WHERE s.file_hash = ? AND d.value IS NOT NULL AND d.value != ''
        LIMIT 500
    """, (file_hash,))
    all_values = cursor.fetchall()
    for val_row in all_values:
        val = str(val_row["value"])
        if re.search(r'(КЮЛ|ТБО|Сбер|ВТБ|IFRS|МСФО|субсид|продуктовый регистр|процентная ставка|гарантия|обеспечение)', val, re.IGNORECASE):
            important_values.add(val[:100])
        if re.search(r'\b[A-Z]{2,5}\b', val):
            important_values.add(val[:100])

    conn.close()
    sheets_list = list(sheets_dict.values())
    return {
        "file_hash": file_hash,
        "filename": filename,
        "raw_sheets": sheets_list,
        "important_values": list(important_values)[:30],
        "source_description_snippets": description_snippets[:120],
        "schema": {},
        "section_summaries": [],
        "final_summary": "",
        "validation_errors": [],
    }

@observe()
def extract_schema(state: Dict[str, Any]) -> Dict[str, Any]:
    sheets = state["raw_sheets"]
    important_vals = state["important_values"]
    snippets = state.get("source_description_snippets") or []
    sheets_columns = []
    sample_data = []
    description_bullets = []
    for sheet in sheets:
        cols = ", ".join(sheet["columns"][:20])
        sheets_columns.append(f"Лист '{sheet['sheet_name']}': {cols}")
        desc_cells = sheet.get("description_cells") or []
        if desc_cells:
            description_bullets.append(f"Лист '{sheet['sheet_name']}' — фрагменты из полей описания:")
            for d in desc_cells[:12]:
                description_bullets.append(f"  • {d[:500]}")
        if sheet["sample_rows"]:
            sample_data.append(f"Лист '{sheet['sheet_name']}':")
            for row in sheet["sample_rows"]:
                sample_data.append(f"  Строка {row['row_num']}: {row['values']}")

    desc_focus = "\n".join(description_bullets) if description_bullets else ""
    snippets_joined = "\n".join(snippets[:35]) if snippets else ""

    prompt = f"""
Ты — эксперт по доменной модели данных. Главный источник — **описания и комментарии в бизнес-полях**
(поля вроде Description, Описание, Attribute Note и т.п.) и цитаты из ячеек ниже.

**Не включай в ответ и не выводи в JSON:** имя файла, Excel, листы, вкладки, «документ», количество таблиц в книге,
тип файла, загрузку, шаблон — это служебный контекст для тебя, не для пользователя.

1) Тексты из полей описания (приоритет):
{desc_focus if desc_focus else "(Явные длинные тексты в полях описания не найдены — опирайся осторожно на имена полей и примеры значений.)"}

2) Дополнительные сниппеты метаданных:
{snippets_joined if snippets_joined else "(нет)"}

3) Имена полей по группам (только чтобы выделить сущности; не описывай это как «структуру документа»):
{chr(10).join(sheets_columns)}

4) Примеры значений (вторично — уточнение домена; не выдумывай организации/проекты без опоры на п.1–2):
{chr(10).join(sample_data[:10])}

5) Эвристические токены (шум возможен):
{chr(10).join(important_vals[:15])}

Извлеки JSON — только бизнес-содержание (домен, сущности, темы таблиц по смыслу полей, не «о файле»):
{{
    "business_domain": "...",
    "bank_hint": "...",
    "project_codes": [],
    "key_entities": [],
    "history_types": [],
    "source_tables": [],
    "target_tables": [],
    "transformation_patterns": [],
    "description_highlights": ["ключевые тезисы только из смысла описаний и полей"]
}}
"""
    answer = call_gigachat(prompt)
    try:
        schema = _schema_json_parser.parse(answer)
        if "key_entities" in schema:
            schema["key_entities"] = [
                (e if isinstance(e, str) else e.get("name") or e.get("entity") or str(e))
                for e in schema["key_entities"]
            ]
        if "description_highlights" not in schema:
            schema["description_highlights"] = []
        state["schema"] = schema
        logger.info(f"Schema extracted: {schema}")
    except Exception as e:
        logger.error(f"Schema extraction failed: {e}")
        state["schema"] = {"business_domain": "не определено", "key_entities": [], "description_highlights": []}
        state["validation_errors"].append(f"Schema error: {e}")
    return state

@observe()
def structural_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    sheets = state["raw_sheets"]
    lines = []
    for sheet in sheets:
        cols = ", ".join(sheet["columns"][:20])
        lines.append(f"Группа полей / тема «{sheet['sheet_name']}»: {cols}")
        desc_cells = sheet.get("description_cells") or []
        if desc_cells:
            lines.append("  Тексты из полей описания и примечаний:")
            for t in desc_cells[:15]:
                lines.append(f"    — {t[:450]}")
    prompt = f"""
Сформируй 1–2 абзаца **только о предметной области и бизнес-объектах** (договоры, клиенты, продукты, события —
что следует из текстов ниже и из смысла имён полей).

**Категорически запрещено** упоминать или обсуждать: Excel, файл, документ, книгу, листы, вкладки, число листов,
загрузку, шаблон, «данный файл», «настоящая спецификация», формат xlsx, маппинг как *тип документа*.
Не описывай артефакт хранения — только домен данных.

Если явных описаний мало, честно скажи об **ограниченности доменных данных**, без перехода к метаописанию книги.

Исходные фрагменты:
{chr(10).join(lines)}
"""
    answer = call_gigachat(prompt)
    state["section_summaries"].append(answer)
    return state

@observe()
def domain_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    schema = state.get("schema", {})
    highlights = schema.get("description_highlights") or []
    schema_trim = {k: v for k, v in schema.items() if k != "description_highlights"}
    schema_str = json.dumps(schema_trim, ensure_ascii=False, indent=2)
    hl_text = "\n".join(f"• {h}" for h in highlights[:20]) if highlights else "(нет выделенных тезисов)"

    prompt = f"""
Используй только **тезисы из описаний полей** и структурированные признаки домена ниже.

Тезисы из полей описания:
{hl_text}

Признаки домена (JSON; не повторяй как «описание файла»):
{schema_str}

Задача: 2 абзаца на русском о **бизнес-домене, объектах и потоках данных** (что означают данные, какие сущности, какая логика).
**Не пиши** про Excel, листы, файлы, спецификацию как документ, загрузку, количество таблиц в книге.

Банк, МСФО, коды проектов — **только** если явно следуют из описаний / highlights / key_entities.
"""
    answer = call_gigachat(prompt)
    state["section_summaries"].append(answer)
    return state

@observe()
def synthesize(state: Dict[str, Any]) -> Dict[str, Any]:
    combined = "\n\n".join(state["section_summaries"])
    prompt = f"""
Собери **один связный абзац** (5–8 предложений) на русском — **исключительно про предметную область и данные**:
сущности, процессы, назначение и смысл полей по **бизнес-описаниям** из материала ниже.

**Запрещено** (не использовать ни в зачине, ни в тексте):
- «данный файл / документ / книга Excel / xlsx», лист, вкладка, число листов, шаблон, «спецификация» как **оформление материала**;
- загрузка, хранение **файла**, «в этом файле», «в книге».

Разрешено: предметная область, объекты данных, атрибуты, правила, перенос или соответствие **полей и сущностей по смыслу метаданных** (если это следует из описаний), без акцента на формате носителя.

Разрешено: домен (например кредитование, регистры договоров), объекты, атрибуты, связи, правила — если они следуют из полей и описаний.

Промежуточные фрагменты (игнорируй любые упоминания про файл/Excel, если они случайно попали — не переноси в итог):
{combined}
"""
    answer = call_gigachat(prompt)
    state["final_summary"] = answer
    return state

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

@observe()
def validate(state: Dict[str, Any]) -> Dict[str, Any]:
    final = state["final_summary"]
    schema = state.get("schema", {})
    entities = schema.get("key_entities", [])
    if entities:
        normalized_summary = normalize_text(final)
        found = False
        for entity in entities:
            if isinstance(entity, dict):
                entity_str = entity.get("name") or entity.get("entity") or str(entity)
            else:
                entity_str = str(entity)
            norm_entity = normalize_text(entity_str)
            if norm_entity in normalized_summary:
                found = True
                break
            for word in norm_entity.split():
                if len(word) > 3 and word in normalized_summary:
                    found = True
                    break
            if found:
                break
        if not found:
            state["validation_errors"].append(f"Summary does not mention any key entity from schema: {entities}")
            logger.warning(state["validation_errors"][-1])
        else:
            logger.info("Validation passed")
    return state

# Build LCEL chain: fetch -> extract_schema -> structural -> domain -> synthesize -> validate
summarizer_chain = (
    RunnableLambda(fetch_file_data)
    | RunnableLambda(extract_schema)
    | RunnableLambda(structural_summary)
    | RunnableLambda(domain_summary)
    | RunnableLambda(synthesize)
    | RunnableLambda(validate)
)

@observe()
def generate_summary(file_hash: str) -> str:
    handler = get_callback_handler()
    if handler:
        # Apply config for Langfuse callbacks
        chain_with_config = summarizer_chain.with_config({"callbacks": [handler], "run_name": f"summarize_{file_hash}"})
        result = chain_with_config.invoke(file_hash)
    else:
        result = summarizer_chain.invoke(file_hash)
    if result["validation_errors"]:
        logger.warning(f"Validation issues: {result['validation_errors']}")
    return result["final_summary"]

def summarize_file(file_hash: str, save: bool = True) -> str:
    summary = generate_summary(file_hash)
    if save:
        update_file_summary(file_hash, summary)
    return summary