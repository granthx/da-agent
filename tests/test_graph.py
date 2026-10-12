"""End-to-end deterministic tests for the LangGraph state machine without network or API tokens."""
import pathlib
import sys
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from da_agent.data import STORE
from da_agent.graph import build_graph
from da_agent.memory import MemoryStore
import da_agent.tools as T
import da_agent.graph as G


class MockLLM:
    """Deterministic offline mock model for testing LangGraph cycles."""
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.invocations = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return AIMessage(content="Default fallback mock answer")


@pytest.fixture(autouse=True)
def setup_test_state(tmp_path, monkeypatch):
    STORE.load(ROOT / "sample_data" / "customers.csv")
    mem = MemoryStore(tmp_path / "test_memory.json")
    monkeypatch.setattr(T, "MEMORY", mem)
    monkeypatch.setattr(G, "MEMORY", mem)
    monkeypatch.setattr(T, "OUTPUT_DIR", tmp_path)
