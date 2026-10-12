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


def test_graph_compilation():
    """Verify graph compiles and contains expected nodes."""
    mock_llm = MockLLM([])
    graph = build_graph(llm=mock_llm)
    assert graph is not None
    node_names = set(graph.nodes.keys())
    assert {"recall", "analyst", "tools", "curator"}.issubset(node_names)


def test_graph_direct_answer_flow():
    """Verify flow when analyst immediately produces a final answer without tools."""
    mock_llm = MockLLM([
        # 1. Analyst node response (no tool calls)
        AIMessage(content="The average monthly spend is $45.20."),
        # 2. Curator node response (no operations)
        AIMessage(content='{"operations": []}'),
    ])
    graph = build_graph(llm=mock_llm)
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    state = graph.invoke({"messages": [HumanMessage("What is the average spend?")]}, config)
    last_msg = state["messages"][-1]
    assert "45.20" in last_msg.content


def test_graph_tool_execution_cycle():
    """Verify ReAct loop: analyst -> tools -> analyst -> curator -> END."""
    tool_call = {
        "id": "call_abc123",
        "name": "aggregate",
        "args": {"metric_column": "monthly_spend", "agg": "mean"}
    }
    mock_llm = MockLLM([
        # 1. First analyst turn: requests tool execution
        AIMessage(content="", tool_calls=[tool_call]),
        # 2. Second analyst turn: receives tool output and synthesizes answer
        AIMessage(content="The verified mean monthly spend across all customers is $48.15."),
        # 3. Curator turn: no durable rules to save
        AIMessage(content='{"operations": []}'),
    ])
    graph = build_graph(llm=mock_llm)
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    state = graph.invoke({"messages": [HumanMessage("Calculate average monthly spend")]}, config)
    # The message history should contain: HumanMessage, AIMessage(tool_call), ToolMessage, AIMessage(final)
    msg_types = [type(m).__name__ for m in state["messages"]]
    assert "ToolMessage" in msg_types
    assert "48.15" in state["messages"][-1].content


def test_graph_curator_saves_memory():
    """Verify curator saves a durable definition to memory during graph execution."""
    mock_llm = MockLLM([
        AIMessage(content="Understood, I will remember that high spend means monthly_spend > 100."),
        # Curator extracts definition
        AIMessage(content='{"operations": [{"op": "upsert", "type": "definition", "key": "high_spend", "value": "monthly_spend > 100", "scope": "dataset"}]}'),
    ])
    graph = build_graph(llm=mock_llm)
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    state = graph.invoke({"messages": [HumanMessage("High spend means monthly spend above 100.")]}, config)
    assert any("added" in entry or "high_spend" in entry for entry in state.get("memory_log", []))
    assert any(it["key"] == "high_spend" for it in G.MEMORY.items)


def test_graph_out_of_domain_refusal():
    """Verify that an out-of-domain refusal short-circuits directly to END without calling curator."""
    refusal_text = (
        "I am a specialized Data Analyst Agent scoped exclusively to your loaded dataset. "
        "I cannot answer general knowledge or off-topic questions. Please ask a question about your data."
    )
    mock_llm = MockLLM([
        # Only 1 model response needed because curator is skipped!
        AIMessage(content=refusal_text)
    ])
    graph = build_graph(llm=mock_llm)
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    state = graph.invoke({"messages": [HumanMessage("What is the capital of France?")]}, config)
    assert "scoped exclusively to your loaded dataset" in state["messages"][-1].content
    # mock_llm was only invoked once by analyst; curator was never invoked
    assert len(mock_llm.invocations) == 1
