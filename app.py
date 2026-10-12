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
