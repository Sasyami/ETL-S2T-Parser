import os
import json
import logging
import re
from typing import List, Any, Tuple
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from load_skills_tools import load_skills, load_tools

try:
    from langfuse import observe, get_client
    from langfuse.langchain import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    def observe(*args, **kwargs):
        def decorator(func): return func
        return decorator

load_dotenv()
logger = logging.getLogger(__name__)

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_API_KEY") or os.getenv("GIGACHAT_EMBEDDINGS_CREDENTIALS")
if not GIGACHAT_CREDENTIALS:
    raise ValueError("Missing GigaChat credentials")

GIGACHAT_BASE_URL = os.getenv("GIGACHAT_API_URL", "https://gigachat.devices.sberbank.ru/api/v1")
VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true"
SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
MODEL = os.getenv("MODEL", "GigaChat-Pro")
TIMEOUT = int(os.getenv("GIGACHAT_TIMEOUT", "120"))

giga = GigaChat(
    model=MODEL, credentials=GIGACHAT_CREDENTIALS, base_url=GIGACHAT_BASE_URL,
    verify_ssl_certs=VERIFY_SSL, scope=SCOPE, timeout=TIMEOUT
)

SYSTEM_PROMPT = f"{load_skills()}\n{load_tools()}\nYou are an expert in analyzing messy Excel sheet structures. Use the skills and tools above."

# Helper functions
def is_long_text(value: Any) -> bool:
    if isinstance(value, str):
        return len(value) > 100 or '\n' in value
    return False

def looks_like_data(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        if value.isdigit():
            return True
        if any(k in value.upper() for k in ['SELECT', 'FROM', 'WHERE', 'JOIN']):
            return True
        if len(value) > 50:
            return True
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', value) and not value.isalpha():
            return True
    return False

def call_gigachat_with_retry(system_content: str, user_content: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            messages = [
                Messages(role=MessagesRole.SYSTEM, content=system_content),
                Messages(role=MessagesRole.USER, content=user_content)
            ]
            response = giga.chat(Chat(messages=messages))
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"GigaChat attempt {attempt+1}/{retries} failed: {e}")
            if attempt == retries - 1:
                raise
    raise RuntimeError("GigaChat call failed after retries")

def safe_extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    return match.group(1).strip() if match else text

# Runnable that calls GigaChat
def llm_call(user_prompt: str) -> str:
    return call_gigachat_with_retry(SYSTEM_PROMPT, user_prompt)

llm_runnable = RunnableLambda(llm_call)

# Build analysis prompt
def build_analysis_prompt(sheet_name: str, preview_rows: List[List[Any]], max_preview_rows: int = 10) -> str:
    limited_preview = preview_rows[:max_preview_rows]
    preview_json = json.dumps(limited_preview, ensure_ascii=False, default=str)
    return f"""You are given a preview of the first {len(limited_preview)} rows of a sheet named "{sheet_name}".
Each row is a list; empty cells may appear as None.
Goal: detect header_start_row, header_rows, nested.
Rules:
- header_start_row = 0 if first row has short labels (even with None for merged cells), else 1 if first row is long text.
- header_rows = 1 for single row headers, 2+ for multi‑level.
- Stop at data rows (numbers, dates, SQL, long text).
Output JSON:
{{"header_start_row": <int>, "header_rows": <int>, "nested": <bool>, "explanation": "<string>"}}
Preview:
{preview_json}
"""

# LCEL chain: input dict -> build prompt -> LLM -> extract JSON -> parse
analysis_chain = (
    RunnablePassthrough()
    | (lambda x: build_analysis_prompt(x["sheet_name"], x["preview_rows"]))
    | llm_runnable
    | (lambda text: json.loads(safe_extract_json(text)))
)

def apply_post_processing(start_row: int, header_rows: int, nested: bool, preview_rows: List[List[Any]]) -> Tuple[int, int, bool]:
    """Apply heuristic corrections to the LLM decision."""
    if header_rows == 2 and len(preview_rows) >= 2:
        second_row = [v for v in preview_rows[1] if v is not None]
        if second_row and sum(1 for v in second_row if looks_like_data(v)) / len(second_row) > 0.3:
            header_rows, nested = 1, False

    if header_rows == 1 and len(preview_rows) >= 2 and start_row == 0:
        first_short = any(v is not None and not is_long_text(v) for v in preview_rows[0])
        second_short = any(v is not None and not is_long_text(v) for v in preview_rows[1])
        if first_short and second_short and not any(looks_like_data(v) for v in preview_rows[1]):
            header_rows, nested = 2, True
    return start_row, header_rows, nested

@observe()
def get_header_decision(sheet_name: str, preview_rows: List[List[Any]]) -> Tuple[int, int, bool]:
    """Main function: uses LCEL chain to get header decision."""
    try:
        result = analysis_chain.invoke({"sheet_name": sheet_name, "preview_rows": preview_rows})
        start_row = max(0, result.get("header_start_row", 0))
        header_rows = max(1, min(result.get("header_rows", 1), 5))
        nested = result.get("nested", header_rows >= 2)
        # Apply post‑processing
        start_row, header_rows, nested = apply_post_processing(start_row, header_rows, nested, preview_rows)
        logger.info(f"AI decision for '{sheet_name}': start_row={start_row}, header_rows={header_rows}, nested={nested}")
        return start_row, header_rows, nested
    except Exception as e:
        logger.error(f"GigaChat analysis failed for '{sheet_name}': {e}")
        # Fallback logic
        if len(preview_rows) >= 2:
            f1 = any(v is not None and not is_long_text(v) for v in preview_rows[0])
            f2 = any(v is not None and not is_long_text(v) for v in preview_rows[1])
            if f1 and f2:
                return 0, 2, True
        return 0, 1, False

def get_model_name() -> str:
    return MODEL