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

from db_storage import DB_PATH, get_db_connection, init_db
from load_skills_tools import get_aggregated_target_table_catalog, list_files, mapping_overview


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


_DISPLAY_CATALOG_COLUMNS: tuple[tuple[str, str], ...] = (
    ("target_column", "Target column"),
    ("data_type", "Data type"),
    ("column_description", "Description"),
    ("transformation_rule", "Transformation rule"),
    ("source_column", "Source column"),
    ("is_primary_key", "PK"),
    ("mapping_id", "Mapping IDs"),
)


def _catalog_summary_table(df: "pd.DataFrame") -> "pd.DataFrame":
    """Reorder and rename aggregated catalog columns for Streamlit/table export."""
    import pandas as pd

    out = df[[src for src, _ in _DISPLAY_CATALOG_COLUMNS if src in df.columns]].copy()
    rename = {src: label for src, label in _DISPLAY_CATALOG_COLUMNS if src in out.columns}
    return out.rename(columns=rename)


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


@st.cache_data(ttl=5)
def _target_table_name_options() -> list[str]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM target_tables ORDER BY lower(name)")
    names = [r[0] for r in cur.fetchall() if r[0]]
    conn.close()
    return names


with st.expander("Logical target table: columns & rules", expanded=True):
    import pandas as pd

    opts = _target_table_name_options()
    if not opts:
        st.info("No `target_tables` rows yet — upload and parse mappings first.")
    else:
        c1, c2 = st.columns((2, 1))
        with c1:
            pick = st.selectbox("Pick table", opts, index=0, key="streamlit_target_table_pick")
        with c2:
            override = st.text_input(
                "Or paste id/name",
                key="streamlit_target_table_override",
                placeholder="target_tables.id or name",
            )
        tid = override.strip() or pick
        if tid:
            try:
                catalog = get_aggregated_target_table_catalog(tid)
            except Exception as e:
                st.error(str(e))
            else:
                if catalog.get("error"):
                    st.error(catalog["error"])
                    hint = catalog.get("hint") or ""
                    if hint:
                        st.caption(hint)
                else:
                    tinfo = catalog.get("target_tables") or []
                    for tt in tinfo:
                        nm = tt.get("name") or ""
                        tid_show = tt.get("id") or ""
                        desc = (tt.get("description") or "").strip() or "—"
                        st.markdown(f"**`{nm}`** · `{tid_show}`  \n{desc}")
                    if catalog.get("had_duplicate_target_columns"):
                        st.caption(
                            "Several `column_mappings` rows map to the same target column "
                            '(e.g. real source vs "target catalog"). Values are merged in one row.'
                        )
                    agg = catalog.get("aggregated") or []
                    if agg:
                        df_raw = pd.DataFrame(agg)
                        tbl = _catalog_summary_table(df_raw)
                        safe_name = re.sub(r"[^\w\-]+", "_", tid.strip()) or "catalog"
                        st.subheader("Catalog")
                        csv_bytes = tbl.to_csv(index=False).encode("utf-8")
                        dl1, dl2 = st.columns((1, 3))
                        with dl1:
                            st.download_button(
                                "Download CSV table",
                                data=csv_bytes,
                                file_name=f"{safe_name}_columns.csv",
                                mime="text/csv;charset=utf-8",
                                key="download_target_catalog_csv",
                            )
                        st.dataframe(
                            tbl,
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.warning("Target table exists but has no rows in column_mappings.")
                    if show_raw_json:
                        st.json(catalog)

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
