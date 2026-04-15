import logging
import json
import re
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from db_storage import get_db_connection, update_file_summary
from agent import giga, get_model_name
from gigachat.models import Chat, Messages, MessagesRole
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

SYSTEM_PROMPT = f"""
{load_skills()}

{load_tools()}

You are a business analyst and technical writer. Use the skills and tools above.
"""

def call_gigachat(user_content: str) -> str:
    """Call GigaChat with system prompt and user content."""
    messages = [
        Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT),
        Messages(role=MessagesRole.USER, content=user_content)
    ]
    response = giga.chat(Chat(messages=messages))
    return response.choices[0].message.content.strip()

def extract_json(text: str) -> str:
    """Extract JSON from markdown code blocks."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()

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
                "sample_rows": []
            }
        sheets_dict[sheet_name]["columns"].append(r["column_name_flat"])

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
        "schema": {},
        "section_summaries": [],
        "final_summary": "",
        "validation_errors": []
    }

@observe()
def extract_schema(state: Dict[str, Any]) -> Dict[str, Any]:
    filename = state["filename"]
    sheets = state["raw_sheets"]
    important_vals = state["important_values"]
    sheets_columns = []
    sample_data = []
    for sheet in sheets:
        cols = ", ".join(sheet["columns"][:20])
        sheets_columns.append(f"Лист '{sheet['sheet_name']}': {cols}")
        if sheet["sample_rows"]:
            sample_data.append(f"Лист '{sheet['sheet_name']}':")
            for row in sheet["sample_rows"]:
                sample_data.append(f"  Строка {row['row_num']}: {row['values']}")

    prompt = f"""
Ты – эксперт по анализу данных. Извлеки структурированную информацию из Excel-файла.

Файл: {filename}

Листы и колонки:
{chr(10).join(sheets_columns)}

Примеры данных (первые строки):
{chr(10).join(sample_data[:10])}

Важные бизнес-термины, найденные в данных:
{chr(10).join(important_vals[:15])}

Извлеки JSON:
{{
    "business_domain": "...",
    "bank_hint": "...",
    "project_codes": [],
    "key_entities": [],
    "history_types": [],
    "source_tables": [],
    "target_tables": [],
    "transformation_patterns": []
}}
"""
    answer = call_gigachat(prompt)
    try:
        json_str = extract_json(answer)
        schema = json.loads(json_str)
        if "key_entities" in schema:
            schema["key_entities"] = [
                (e if isinstance(e, str) else e.get("name") or e.get("entity") or str(e))
                for e in schema["key_entities"]
            ]
        state["schema"] = schema
        logger.info(f"Schema extracted: {schema}")
    except Exception as e:
        logger.error(f"Schema extraction failed: {e}")
        state["schema"] = {"business_domain": "не определено", "key_entities": []}
        state["validation_errors"].append(f"Schema error: {e}")
    return state

@observe()
def structural_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    sheets = state["raw_sheets"]
    sheets_columns = []
    for sheet in sheets:
        cols = ", ".join(sheet["columns"][:20])
        sheets_columns.append(f"Лист '{sheet['sheet_name']}': {cols}")
    prompt = f"Опиши структуру документа (листы и их назначение). 1-2 абзаца на русском.\n\nЛисты и колонки:\n{chr(10).join(sheets_columns)}"
    answer = call_gigachat(prompt)
    state["section_summaries"].append(answer)
    return state

@observe()
def domain_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    schema = state.get("schema", {})
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
    prompt = f"""
На основе извлечённой схемы:
{schema_str}

Опиши бизнес-домен, ключевые сущности, жизненный цикл данных. Обязательно упомяни:
- Банк или организацию (если есть намёк)
- Коды проектов (например, КЮЛ)
- Финансовые стандарты (МСФО, IFRS)
- Конкретные продукты (кредиты, гарантии, субсидии, продуктовые регистры)
Напиши 2 абзаца на русском.
"""
    answer = call_gigachat(prompt)
    state["section_summaries"].append(answer)
    return state

@observe()
def synthesize(state: Dict[str, Any]) -> Dict[str, Any]:
    combined = "\n\n".join(state["section_summaries"])
    prompt = f"""
На основе следующих резюме создай **ОДИН связный абзац** (5-7 предложений) на русском языке.
Абзац должен быть максимально похож на пример ниже по стилю и детализации.

Промежуточные резюме:
{combined}

Пример желаемого стиля:
"Данный файл представляет собой спецификацию маппинга «Источник-Приёмник» (Source-to-Target, S2T) для хранилища данных по корпоративному кредитованию (код проекта «КЮЛ» — Кредиты Юридическим Лицам) в рамках крупной российской банковской среды (предположительно Сбербанк). Документ определяет логику ETL-процессов для трансформации данных из исходных систем в единую аналитическую модель, охватывающую полный жизненный цикл кредитных продуктов: кредитные договоры, договоры банковской гарантии и договоры обеспечения, а также связанные с ними атрибуты, такие как контрагенты, валюты, правила расчёта процентных ставок, субсидии, метрики классификации по МСФО (IFRS 9), продуктовые регистры, а также плановые и фактические финансовые операции."

Напиши абзац, следуя этому примеру: укажи тип документа (S2T, маппинг, ETL), проект, банк, перечисли конкретные сущности и финансовые показатели.
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