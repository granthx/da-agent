"""Streamlit Web UI for da-agent — Autonomous Data Analysis Agent."""
import streamlit as st

st.set_page_config(
    page_title="da-agent | Autonomous Data Analyst",
    page_icon="📊",
    layout="wide",
)
st.title("📊 da-agent")

import uuid
from da_agent.config import GROQ_MODEL
from da_agent.graph import build_graph

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex

from da_agent.data import STORE

with st.sidebar:
    st.subheader("📁 Dataset Selection")
    data_source = st.radio("Choose source:", ["Sample Datasets", "Upload File"], horizontal=True)

    uploaded_file = st.file_uploader("Upload CSV, Excel, or Parquet", type=["csv", "tsv", "xlsx", "parquet"])

    if STORE.df is not None:
        st.write(f"Active dataset: {STORE.name}")
        with st.expander("Inspect Schema"):
            st.text(STORE.schema_summary())

from da_agent.memory import MEMORY
with st.sidebar:
    st.subheader("🧠 Knowledge Memory")
    if st.button("Clear Memory"):
        MEMORY.clear()

col1, col2 = st.columns([4, 1])
with col2:
    if st.button("Clear Chat"):
        st.session_state.messages = []

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    status = st.status("Analyzing...")
