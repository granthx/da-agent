"""Streamlit Web UI for da-agent — Autonomous Data Analysis Agent."""
import glob
import json
import os
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from da_agent.config import GROQ_MODEL, MAX_TOOL_STEPS, OUTPUT_DIR
from da_agent.data import STORE
from da_agent.graph import build_graph, _text
from da_agent.memory import MEMORY

st.set_page_config(
    page_title="da-agent | Autonomous Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex
if "graph" not in st.session_state or st.session_state.get("current_model") != GROQ_MODEL or st.session_state.get("graph_error"):
    try:
        st.session_state.graph = build_graph()
        st.session_state.current_model = GROQ_MODEL
        st.session_state.graph_error = None
    except Exception as e:
        st.session_state.graph = None
        st.session_state.graph_error = str(e)
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False


# -----------------------------------------------------------------------------
# Sidebar: Dataset Loading & Memory Inspector
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 da-agent")
    st.caption("Autonomous, tool-augmented analytical multi-agent system")

    st.divider()
    st.subheader("📁 Dataset Selection")

    data_source = st.radio("Choose source:", ["Sample Datasets", "Upload File"], horizontal=True)

    if data_source == "Sample Datasets":
        sample_choice = st.selectbox(
            "Select sample:",
            ["customers.csv (SaaS Churn/Spend)", "sales.csv (Multi-Region Sales)"]
        )
        sample_file = "sample_data/customers.csv" if "customers" in sample_choice else "sample_data/sales.csv"
        if st.button("Load Sample Dataset", use_container_width=True):
            try:
                STORE.load(sample_file)
                st.session_state.dataset_loaded = True
                st.session_state.thread_id = uuid.uuid4().hex
                st.session_state.messages = []
                st.success(f"Loaded '{STORE.name}' ({len(STORE.df)} rows)")
            except Exception as e:
                st.error(f"Error loading: {e}")
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV, Excel, or Parquet",
            type=["csv", "tsv", "xlsx", "xls", "json", "parquet"]
        )
        if uploaded_file is not None:
            if st.button("Ingest Uploaded File", use_container_width=True):
                temp_dir = Path("outputs") / "uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / uploaded_file.name
                temp_path.write_bytes(uploaded_file.getbuffer())
                try:
                    STORE.load(str(temp_path))
                    st.session_state.dataset_loaded = True
                    st.session_state.thread_id = uuid.uuid4().hex
                    st.session_state.messages = []
                    st.success(f"Loaded '{STORE.name}' ({len(STORE.df)} rows)")
                except Exception as e:
                    st.error(f"Failed to parse file: {e}")

    # Dataset status & schema inspect
    if STORE.df is not None:
        st.divider()
        st.subheader("📋 Active Dataset")
        st.write(f"**Name:** `{STORE.name}`")
        st.write(f"**Shape:** `{STORE.df.shape[0]:,} rows × {STORE.df.shape[1]} cols`")
        with st.expander("Inspect Column Schema"):
            st.text(STORE.schema_summary())

    # Persistent Memory View
    st.divider()
    st.subheader("🧠 Knowledge Memory")
    eligible_items = MEMORY.eligible(STORE.name) if STORE.name else MEMORY.items
    if eligible_items:
        with st.expander(f"Active Memories ({len(eligible_items)})", expanded=True):
            for it in eligible_items:
                st.markdown(f"- **{it['type']}** (`{it['key']}`): `{it['value']}` *[{it['scope']}]*")
        if st.button("Clear Memory Store", type="secondary", use_container_width=True):
            MEMORY.clear()
            st.rerun()
    else:
        st.caption("No persistent rules or named filters active.")

    st.divider()
    st.caption(f"🤖 **Model:** `{GROQ_MODEL}`")


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
col1, col2 = st.columns([4, 1])
with col1:
    st.header("Autonomous Data Analyst")
    st.markdown("Natural language tabular analytics powered by LangGraph, Pandas, and Groq LPUs.")
with col2:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = uuid.uuid4().hex
        try:
            st.session_state.graph = build_graph()
            st.session_state.current_model = GROQ_MODEL
            st.session_state.graph_error = None
        except Exception as e:
            st.session_state.graph_error = str(e)
        st.rerun()

if st.session_state.graph_error:
    st.error(
        f"⚠️ **Configuration Notice:** {st.session_state.graph_error}\n\n"
        "Ensure your `GROQ_API_KEY` is configured in `.env`."
    )
    st.stop()

if STORE.df is None:
    st.info("👈 Please load a sample dataset or upload a file from the sidebar to begin analysis.")
    st.stop()

# -----------------------------------------------------------------------------
# Render Chat History
# -----------------------------------------------------------------------------
# Render Chat History
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("tools"):
            with st.expander(f"🛠️ Executed {len(msg['tools'])} Tool Call(s)"):
                for t in msg["tools"]:
                    st.code(t, language="json")
        if msg.get("evidence"):
            with st.expander(f"🧾 Computational Evidence & Proof ({len(msg['evidence'])} calculation(s))", expanded=False):
                st.caption(
                    "🛡️ **Proof of Non-Hallucination:** Below are the exact raw arithmetic results computed "
                    "directly by Python / Pandas on the real DataFrame. The LLM acted solely as the orchestrator."
                )
                for i, ev in enumerate(msg["evidence"], 1):
                    st.markdown(f"**Step {i}: `{ev['name']}`**")
                    if ev.get("args"):
                        st.caption(f"Arguments: `{json.dumps(ev['args'], default=str)}`")
                    st.code(ev.get("result", ""), language="text")
        if msg.get("memory_log"):
            for m in msg["memory_log"]:
                st.caption(f"🧠 *Memory updated:* `{m}`")
        st.markdown(msg["content"])
        if msg.get("charts"):
            for chart_path in msg["charts"]:
                if os.path.exists(chart_path):
                    st.image(chart_path, caption=os.path.basename(chart_path), use_container_width=True)

# -----------------------------------------------------------------------------
# User Input & Agent Invocation
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about the dataset or teach a domain rule..."):
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Track output files before run to detect newly created charts
    existing_charts = set(glob.glob(str(OUTPUT_DIR / "*.png")))

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        status_placeholder = st.status("Analyzing dataset...", expanded=True)
        tools_executed = []
        pending_calls = {}
        evidence = []
        memory_logs = []
        final_answer = ""

        graph = st.session_state.graph
        config = {
            "configurable": {"thread_id": st.session_state.thread_id},
            "recursion_limit": 25 + MAX_TOOL_STEPS * 2
        }

        try:
            for update in graph.stream({"messages": [HumanMessage(prompt)]}, config, stream_mode="updates"):
                for node, out in (update or {}).items():
                    out = out or {}
                    if node == "analyst":
                        msg = out["messages"][-1]
                        for tc in getattr(msg, "tool_calls", None) or []:
                            sig = f"{tc['name']}({json.dumps(tc['args'], default=str)})"
                            tools_executed.append(sig)
                            if tc.get("id"):
                                pending_calls[tc["id"]] = tc.get("args", {})
                            status_placeholder.write(f"🔧 Calling tool: `{tc['name']}` with {json.dumps(tc['args'])}")
                        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                            final_answer = _text(msg.content)
                    elif node == "tools":
                        for tm in out.get("messages", []):
                            t_name = getattr(tm, "name", "tool")
                            t_call_id = getattr(tm, "tool_call_id", "")
                            t_content = getattr(tm, "content", "")
                            args = pending_calls.get(t_call_id, {})
                            evidence.append({
                                "name": t_name,
                                "args": args,
                                "result": str(t_content),
                            })
                            status_placeholder.write(f"📊 Evaluated deterministic calculation for `{t_name}`")
                    elif node == "curator":
                        for line in out.get("memory_log", []):
                            memory_logs.append(line)
                            status_placeholder.write(f"🧠 Memory curator: `{line}`")

            status_placeholder.update(label="Analysis Complete", state="complete", expanded=False)
            answer_placeholder.markdown(final_answer)

            # Render evidence expander for current answer
            if evidence:
                with st.expander(f"🧾 Computational Evidence & Proof ({len(evidence)} calculation(s))", expanded=False):
                    st.caption(
                        "🛡️ **Proof of Non-Hallucination:** Below are the exact raw arithmetic results computed "
                        "directly by Python / Pandas on the real DataFrame. The LLM acted solely as the orchestrator."
                    )
                    for i, ev in enumerate(evidence, 1):
                        st.markdown(f"**Step {i}: `{ev['name']}`**")
                        if ev.get("args"):
                            st.caption(f"Arguments: `{json.dumps(ev['args'], default=str)}`")
                        st.code(ev.get("result", ""), language="text")

            # Check for newly generated chart images
            new_charts = list(set(glob.glob(str(OUTPUT_DIR / "*.png"))) - existing_charts)
            for chart in new_charts:
                st.image(chart, caption=os.path.basename(chart), use_container_width=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "tools": tools_executed,
                "evidence": evidence,
                "memory_log": memory_logs,
                "charts": new_charts,
            })
            if memory_logs:
                st.rerun()

        except Exception as e:
            status_placeholder.update(label="Execution Failed", state="error", expanded=True)
            st.error(f"Analysis failed: {e}")
