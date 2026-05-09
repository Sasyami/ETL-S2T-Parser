"""
Streamlit UI for mapping/source/target insights: DB snapshot + GigaChat ReAct agent.
Run from repo root: streamlit run streamlit_insights.py
"""

from __future__ import annotations

import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

from db_storage import DB_PATH, init_db
from load_skills_tools import list_files, mapping_overview


def _strip_json_code_fences(text: str) -> str:
    """Remove ```json ... ``` blocks (deterministic answers append these for copy-paste)."""
    if not text:
        return text
    out = re.sub(
        r"\n?```json\s*[\s\S]*?```\s*",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    return out.strip()


def _render_mapping_overview_pretty(snap: dict) -> None:
    """Tables + metrics instead of one large JSON blob."""
    import pandas as pd

    rtot = snap.get("relationships_total")
    if rtot is not None:
        st.metric("Lineage relationships (total edges)", int(rtot))

    for key in sorted(snap.keys()):
        if key == "relationships_total":
            continue
        val = snap[key]
        if key == "relationships_by_type":
            st.subheader("relationships_by_type")
            rows = val if isinstance(val, list) else []
            st.dataframe(
                pd.DataFrame(rows) if rows else pd.DataFrame(),
                use_container_width=True,
                hide_index=True,
            )
            continue
        with st.expander(key, expanded=False):
            if isinstance(val, dict) and "count" in val:
                st.caption(f"**{val.get('count', 0)}** rows in table")
                sample = val.get("sample") or []
                if sample:
                    st.dataframe(
                        pd.DataFrame(sample),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No sample rows in this range.")
            else:
                st.write(val)


def _try_import_insights_chat():
    try:
        from agents.insights_agent import insights_chat

        return insights_chat, None
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


init_db()


st.set_page_config(page_title="S2T insights", layout="wide")
st.title("Source / target / mapping insights")
st.caption(f"SQLite: `{DB_PATH}`")

insights_chat_fn, llm_import_error = _try_import_insights_chat()

with st.sidebar:
    max_steps = st.slider("Agent max steps", min_value=2, max_value=20, value=8)
    file_rows = list_files()
    choices = ["(none)"] + [
        f"{r.get('filename', '?')} · {r.get('file_hash', '')}" for r in file_rows
    ]
    picked = st.selectbox("Prefer file_hash (optional)", choices)
    preferred_hash = None
    if picked and picked != "(none)" and " · " in picked:
        preferred_hash = picked.rsplit(" · ", 1)[-1].strip() or None
    if st.button("Reload DB snapshot"):
        st.cache_data.clear()
    show_raw_json = st.checkbox("Show raw JSON (chat replies + snapshot)", value=False)


@st.cache_data(ttl=5)
def _cached_overview(limit: int):
    return mapping_overview(limit=limit)


with st.expander("DB snapshot (mapping_overview)", expanded=False):
    lim = st.number_input("Sample rows per table", min_value=1, max_value=100, value=15)
    try:
        snap = _cached_overview(int(lim))
        if show_raw_json:
            st.json(snap)
        else:
            _render_mapping_overview_pretty(snap)
    except Exception as e:
        st.error(f"Snapshot failed: {e}")

if llm_import_error:
    st.warning(
        "GigaChat is not configured (`GIGACHAT_API_KEY` or related env). "
        "The chat agent is disabled; you can still use the DB snapshot above."
    )
    st.code(llm_import_error)
else:
    if "insight_messages" not in st.session_state:
        st.session_state.insight_messages = []

    for msg in st.session_state.insight_messages:
        with st.chat_message(msg["role"]):
            text = msg["content"]
            if msg["role"] == "assistant" and not show_raw_json:
                text = _strip_json_code_fences(text)
            st.markdown(text)

    prompt = st.chat_input("Ask about sources, targets, mappings, or lineage…")
    if prompt and insights_chat_fn:
        st.session_state.insight_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Running insights agent…"):
                answer = insights_chat_fn(
                    prompt,
                    max_steps=max_steps,
                    file_hash=preferred_hash,
                )
            display = answer if show_raw_json else _strip_json_code_fences(answer)
            st.markdown(display)
        st.session_state.insight_messages.append({"role": "assistant", "content": answer})
