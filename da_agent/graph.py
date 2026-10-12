"""LangGraph workflow (two cooperating agents + tools):

  START -> recall -> analyst <-> tools -> curator -> END
                       ^  (loop until no more tool calls)

  recall   : pulls relevant saved memory (definitions, filters, preferences) into the prompt
  analyst  : agent #1, a tool-calling LLM that does the analysis
  tools    : ToolNode executing the 9 analysis tools on the real DataFrame
  curator  : agent #2, a separate LLM call that decides what (if anything) is worth remembering
"""
import json
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import HISTORY_TURNS, MAX_TOOL_STEPS
from .data import STORE
from .llm import get_llm
from .memory import MEMORY, VALID_TYPES
from .tools import get_tools


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    memory_context: str
    memory_log: list[str]


ANALYST_PROMPT = """You are a specialized data analyst agent working strictly on the loaded tabular dataset.

RULES
1. DOMAIN BOUNDARY (STRICT): You are exclusively a data analyst assistant for the loaded dataset.
   You must STRICTLY REFUSE to answer any general knowledge, world facts, geography, trivia, creative writing,
   or off-topic questions (e.g., "what is the capital of India", "who was Einstein", "write a joke", "tell me the weather").
   If the user asks an off-topic or general knowledge question, respond ONLY with:
   "I am a specialized Data Analyst Agent scoped exclusively to your loaded dataset. I cannot answer general knowledge or off-topic questions. Please ask a question about your data or define a dataset filter/rule."
   Do NOT provide the answer first or append general knowledge as an aside.
2. Never state a number from guesswork. Call tools and base every figure on tool output.
3. Column names are exact; use them verbatim. If a tool errors, read the message and retry with fixed arguments.
4. Choose the most specific tool (aggregate for group stats, trend_over_time for time series,
   compare_groups for 'is there a difference', detect_outliers for anomalies, run_python only as a last resort).
5. Apply the user's SAVED MEMORY automatically: use named_filter for saved filters, follow definitions,
   corrections and preferences (e.g. preferred time granularity), and briefly say when you did.
   If the user's current message contradicts a saved item, follow the current message.
6. Final answer: the direct answer first, then the key supporting numbers, then filters used and caveats
   (missing values, small samples, correlation != causation). Keep it concise.

DATASET
{schema}

SAVED MEMORY
{memory}
"""

CURATOR_PROMPT = """You are the memory curator of a data-analysis assistant. Decide whether the user's latest
message contains DURABLE information worth remembering for future sessions.

Save ONLY:
- definition: meaning of a column/metric/term ("churn means status == cancelled")
- named_filter: a reusable subset. value MUST be a valid pandas DataFrame.query() expression using the real
  column names. e.g. "active customers excludes cancelled accounts" -> key active_customers, value status != 'cancelled'
- preference: how the user wants answers ("weekly summaries", "always show percentages")
- correction: the user corrects a fact or a previous answer
Do NOT save: one-off questions, computed results, greetings, raw data, anything the assistant merely said.
To change an existing item, upsert with the SAME type and key. If the user asks to forget something, use delete.
scope: "dataset" if it depends on this dataset's columns/values, "global" if it is a general preference.

DATASET COLUMNS
{schema}

EXISTING MEMORY
{memory}

USER MESSAGE
{user}

Reply with ONLY JSON: {{"operations": [{{"op": "upsert"|"delete", "type": "definition|named_filter|preference|correction",
"key": "snake_case_name", "value": "text or pandas expression", "scope": "dataset|global"}}]}}
Use {{"operations": []}} when nothing should be saved."""


def _text(content) -> str:
    if isinstance(content, list):
        content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return re.sub(r"<think>.*?</think>", "", str(content), flags=re.S).strip()


def _window(messages: list[BaseMessage], turns: int = HISTORY_TURNS) -> list[BaseMessage]:
    """Keep only the last N user turns (with their tool calls intact) to bound token use."""
    idx = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    return messages[idx[-turns]:] if len(idx) > turns else messages


def _last_user(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _text(m.content)
    return ""


def _tool_rounds_this_turn(messages) -> int:
    n = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage):
            n += 1
    return n


def build_graph(llm=None):
    llm = llm or get_llm()
    tools = get_tools()
    llm_tools = llm.bind_tools(tools)

    def recall(state: AgentState):
        items = MEMORY.relevant(STORE.name, _last_user(state["messages"]))
        return {"memory_context": MEMORY.render(items), "memory_log": []}

    def analyst(state: AgentState):
        system = SystemMessage(ANALYST_PROMPT.format(schema=STORE.schema_summary(), memory=state["memory_context"]))
        history = _window(state["messages"])
        try:
            if _tool_rounds_this_turn(state["messages"]) >= MAX_TOOL_STEPS:  # budget guard
                nudge = SystemMessage("Tool budget used up. Answer now using the results you already have.")
                return {"messages": [llm.invoke([system, *history, nudge])]}
            return {"messages": [llm_tools.invoke([system, *history])]}
        except Exception as e:  # e.g. malformed tool call from the model
            return {"messages": [AIMessage(f"The model call failed ({type(e).__name__}: {e}). Please rephrase or retry.")]}

    def route(state: AgentState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        text = _text(getattr(last, "content", ""))
        if "scoped exclusively to your loaded dataset" in text or "specialized exclusively in tabular data" in text:
            return "end"
        return "curator"

    def curator(state: AgentState):
        user = _last_user(state["messages"])
        prompt = CURATOR_PROMPT.format(schema=STORE.schema_summary(), user=user,
                                       memory=MEMORY.render(MEMORY.eligible(STORE.name)))
        log: list[str] = []
        try:
            raw = _text(llm.invoke([HumanMessage(prompt)]).content)
            data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            for op in data.get("operations", []):
                log.append(_apply_op(op))
        except Exception as e:
            log.append(f"(curator skipped: {type(e).__name__})")
        return {"memory_log": [l for l in log if l]}

    def _apply_op(op: dict) -> str:
        kind, typ = op.get("op"), op.get("type")
        key, value = op.get("key", ""), op.get("value", "")
        scope = STORE.name if op.get("scope") == "dataset" else "global"
        if kind == "delete":
            n = MEMORY.delete(key, typ if typ in VALID_TYPES else None)
            return f"forgot '{key}'" if n else ""
        if kind != "upsert" or typ not in VALID_TYPES:
            return ""
        if typ == "named_filter":  # validate against the real data before trusting it
            try:
                STORE.df.query(value, engine="python")
            except Exception as e:
                return f"rejected filter '{key}': `{value}` is not valid here ({type(e).__name__})"
        res = MEMORY.upsert(typ, key, value, scope)
        return "" if res == "unchanged" else f"{res} {typ} '{key}' = {value}"

    g = StateGraph(AgentState)
    g.add_node("recall", recall)
    g.add_node("analyst", analyst)
    g.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    g.add_node("curator", curator)
    g.add_edge(START, "recall")
    g.add_edge("recall", "analyst")
    g.add_conditional_edges("analyst", route, {"tools": "tools", "curator": "curator", "end": END})
    g.add_edge("tools", "analyst")
    g.add_edge("curator", END)
    return g.compile(checkpointer=MemorySaver())
