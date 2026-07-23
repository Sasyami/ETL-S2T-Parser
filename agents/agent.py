import os
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.json import parse_json_markdown
from langchain_gigachat import GigaChat
from load_skills_tools import (
    format_stored_files_answer,
    get_tool_functions,
    load_skills,
    load_tools,
    try_answer_sheets_for_workbook_query,
    try_answer_target_table_fields_query,
    try_answer_target_table_for_mapping_identifier,
    user_wants_stored_file_inventory,
)

try:
    from langfuse import observe
    from langfuse_setup import get_callback_handler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    def observe(*args, **kwargs):
        def decorator(func): return func
        return decorator

load_dotenv()
logger = logging.getLogger(__name__)

# GigaChat configuration
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_API_KEY") or os.getenv("GIGACHAT_EMBEDDINGS_CREDENTIALS")
if not GIGACHAT_CREDENTIALS:
    raise ValueError("Missing GigaChat credentials")

GIGACHAT_BASE_URL = os.getenv("GIGACHAT_API_URL", "https://gigachat.devices.sberbank.ru/api/v1")
VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true"
SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
MODEL = os.getenv("MODEL", "GigaChat-Pro")
TIMEOUT = int(os.getenv("GIGACHAT_TIMEOUT", "120"))

# LangChain chat model (Runnable); use .invoke(...) instead of the raw SDK .chat(...)
chat_model = GigaChat(
    model=MODEL,
    credentials=GIGACHAT_CREDENTIALS,
    base_url=GIGACHAT_BASE_URL,
    verify_ssl_certs=VERIFY_SSL,
    scope=SCOPE,
    timeout=float(TIMEOUT),
)

chat_model_with_retry = chat_model.with_retry(stop_after_attempt=3)

# Back-compat for imports / patches (same object as chat_model).
giga = chat_model


def _message_content_text(content: Union[str, list, Any]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts) if parts else str(content)
    return str(content)


def _message_text(message: BaseMessage) -> str:
    return _message_content_text(message.content).strip()

# Load skills and tools for system prompt
SKILLS = load_skills()
TOOLS_DESCRIPTION = load_tools()
TOOL_FUNCTIONS = get_tool_functions()

SYSTEM_PROMPT = f"""
{SKILLS}

{TOOLS_DESCRIPTION}

You are an expert in analyzing messy Excel sheet structures. Use the skills and tools above.
"""

ANALYSIS_PROMPT = """You are given a preview of the first {preview_rows_count} rows of a sheet named "{sheet_name}".
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

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
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

def _header_completion_messages(prompt_vars: Dict[str, str]) -> List[BaseMessage]:
    """LCEL-friendly message list for SYSTEM + USER header analysis."""
    return [
        SystemMessage(content=prompt_vars["system_content"]),
        HumanMessage(content=prompt_vars["user_content"]),
    ]


_header_completion_chain = (
    RunnableLambda(_header_completion_messages)
    | chat_model_with_retry
    | StrOutputParser()
)


def call_gigachat_with_retry(system_content: str, user_content: str) -> str:
    return (
        _header_completion_chain.invoke(
            {"system_content": system_content, "user_content": user_content}
        ).strip()
    )


def extract_json_payload(raw_text: str) -> str:
    """
    Pull a JSON object/array out of LLM text (fences or leading/trailing prose).
    Prefers the earliest '{'/'[' and the matching outermost '}'/']'.
    """
    if not raw_text:
        return ""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    elif text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    first_obj = text.find("{")
    first_arr = text.find("[")
    starts = [i for i in (first_obj, first_arr) if i >= 0]
    if not starts:
        return text
    start = min(starts)
    end_obj = text.rfind("}")
    end_arr = text.rfind("]")
    end = max(end_obj, end_arr)
    if end < start:
        return text[start:]
    return text[start : end + 1]


def safe_extract_json(text: str) -> str:
    return extract_json_payload(text)


def _extract_tool_input_dict(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse the JSON object after 'Action Input:' (handles nested objects).
    A non-greedy regex like {.*?} breaks on the first '}' inside nested dicts.
    """
    m = re.search(r"Action Input:\s*", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    tail = text[m.end() :].lstrip()
    if not tail.startswith("{"):
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(tail)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _extract_final_answer(text: str) -> Optional[str]:
    m = re.search(r"Final Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    ans = m.group(1).strip()
    return ans if ans else None

# ------------------------------------------------------------
# Header decision (original)
# ------------------------------------------------------------
@observe()
def get_header_decision(sheet_name: str, preview_rows: List[List[Any]]) -> Tuple[int, int, bool]:
    """
    Call GigaChat to decide header structure.
    Returns (header_start_row, header_rows_count, nested)
    """
    limited_preview = preview_rows[:10]
    preview_json = json.dumps(limited_preview, ensure_ascii=False, default=str)
    user_prompt = ANALYSIS_PROMPT.format(
        sheet_name=sheet_name,
        preview_rows_count=len(limited_preview),
        preview_json=preview_json
    )
    try:
        answer = call_gigachat_with_retry(SYSTEM_PROMPT, user_prompt)
        try:
            result = parse_json_markdown(answer)
        except json.JSONDecodeError:
            result = json.loads(safe_extract_json(answer))
        start_row = max(0, result.get("header_start_row", 0))
        header_rows = max(1, min(result.get("header_rows", 1), 5))
        nested = result.get("nested", header_rows >= 2)
        logger.info(f"AI decision for '{sheet_name}': start_row={start_row}, header_rows={header_rows}, nested={nested}")
        # Post‑processing heuristics
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
    except Exception as e:
        logger.error(f"GigaChat analysis failed for '{sheet_name}': {e}")
        # Fallback heuristics
        if len(preview_rows) >= 2:
            f1 = any(v is not None and not is_long_text(v) for v in preview_rows[0])
            f2 = any(v is not None and not is_long_text(v) for v in preview_rows[1])
            if f1 and f2:
                return 0, 2, True
        return 0, 1, False

def get_model_name() -> str:
    return MODEL

# ------------------------------------------------------------
# AI Agent Chat with Tool Calling
# ------------------------------------------------------------
def react_tool_loop(user_query: str, system_prompt: str, max_steps: int = 5) -> str:
    """
    Shared ReAct loop over GigaChat with tools from TOOL_FUNCTIONS.
    """
    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]
    step = 0
    while step < max_steps:
        try:
            reply = chat_model.invoke(messages)
            answer = _message_text(reply)
        except Exception as e:
            logger.error(f"GigaChat error in agent loop: {e}")
            return f"Error communicating with LLM: {e}"

        action_match = re.search(r"Action:\s*(\w+)", answer)
        tool_input = _extract_tool_input_dict(answer)

        if action_match and tool_input is not None:
            tool_name = action_match.group(1)
            try:
                if tool_name in TOOL_FUNCTIONS:
                    tool_func = TOOL_FUNCTIONS[tool_name]
                    result = tool_func(**tool_input)
                    observation = json.dumps(result, ensure_ascii=False, default=str)
                else:
                    observation = (
                        f"Error: Tool '{tool_name}' not found. "
                        f"Available: {list(TOOL_FUNCTIONS.keys())}"
                    )
            except TypeError as e:
                observation = (
                    f"Error: wrong arguments for tool '{tool_name}': {e}. "
                    "Check tools.md for parameter names and types."
                )
            except Exception as e:
                observation = f"Error executing tool '{tool_name}': {str(e)}"

            messages.append(AIMessage(content=answer))
            messages.append(HumanMessage(content=f"Observation: {observation}"))
            step += 1
        elif action_match and tool_input is None:
            messages.append(AIMessage(content=answer))
            messages.append(
                HumanMessage(
                    content=(
                        "Observation: Error: could not parse JSON after 'Action Input:'. "
                        "Use a single JSON object with double-quoted keys, "
                        'e.g. Action Input: {"query": "SELECT 1"}.'
                    ),
                )
            )
            step += 1
        else:
            final = _extract_final_answer(answer)
            if final:
                return final
            return answer

    return "Max steps reached without final answer."


def agent_chat(user_query: str, max_steps: int = 5) -> str:
    """
    Multi‑step reasoning agent that uses tools defined in tools.md.
    """
    stripped = user_query.strip()
    if user_wants_stored_file_inventory(stripped):
        return format_stored_files_answer()
    sheet_ans = try_answer_sheets_for_workbook_query(stripped)
    if sheet_ans is not None:
        return sheet_ans
    tgt_ans = try_answer_target_table_for_mapping_identifier(stripped)
    if tgt_ans is not None:
        return tgt_ans
    fields_ans = try_answer_target_table_fields_query(stripped)
    if fields_ans is not None:
        return fields_ans
    system_prompt = f"""You are a data intelligence assistant. You have access to the following tools:

{TOOLS_DESCRIPTION}

Use the following format:

Thought: what you need to do next
Action: tool_name
Action Input: {{"param1": "value1", "param2": "value2"}}
Observation: result from tool
... (repeat as needed)
Final Answer: your concise answer to the user

Always use tools to answer questions about data lineage, similarity, files, sheets, columns, or S2T mapping facts.
Logical target names (`t_*`) do not imply a SQLite physical table: use **`list_target_table_columns`** for field lists.
Never tell the user to run PRAGMA or SQL manually — call **`run_sql`** yourself for real SQLite tables (`SELECT`, **`PRAGMA table_info(tab)`**). For **`t_*` logical target names**, use **`list_target_table_columns`**, not `PRAGMA` on that identifier.
Never invent filenames or hashes: for any full list of uploads, call `list_files` with `Action Input: {{}}` and report only Observation data.
If a tool requires a column ID, first use similarity_search or list_columns to find it.
If a tool returns an error, try a different approach or explain the issue to the user.
"""
    return react_tool_loop(user_query, system_prompt, max_steps)