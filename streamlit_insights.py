"""
Streamlit UI for mapping/source/target insights: DB snapshot + GigaChat ReAct agent.
Run from repo root: streamlit run streamlit_insights.py
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

from db_storage import DB_PATH, init_db
from load_skills_tools import list_files, mapping_overview


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


@st.cache_data(ttl=5)
def _cached_overview(limit: int):
    return mapping_overview(limit=limit)


with st.expander("DB snapshot (mapping_overview)", expanded=False):
    lim = st.number_input("Sample rows per table", min_value=1, max_value=100, value=15)
    try:
        snap = _cached_overview(int(lim))
        st.json(snap)
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
            st.markdown(msg["content"])

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
            st.markdown(answer)
        st.session_state.insight_messages.append({"role": "assistant", "content": answer})
