"""CLI for the data-analysis agent.

  python main.py sample_data/customers.csv
  python main.py sample_data/sales.csv --ask "Which region has the highest revenue?"
"""
import argparse
import json
import uuid

from langchain_core.messages import AIMessage, HumanMessage

from da_agent.config import MAX_TOOL_STEPS
from da_agent.data import STORE
from da_agent.graph import build_graph, _text
from da_agent.memory import MEMORY

HELP = """Commands: /load <file>  /schema  /memory  /forget <id-or-key>  /clear-memory  /help  /quit
Anything else is a question about the data."""


def ask(graph, thread_id: str, question: str):
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25 + MAX_TOOL_STEPS * 2}
    answer = ""
    for update in graph.stream({"messages": [HumanMessage(question)]}, config, stream_mode="updates"):
        for node, out in (update or {}).items():
            out = out or {}
            if node == "analyst":
                msg = out["messages"][-1]
                for tc in getattr(msg, "tool_calls", None) or []:
                    print(f"  [tool] {tc['name']}({json.dumps(tc['args'], default=str)})")
                if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                    answer = _text(msg.content)
            elif node == "curator":
                for line in out.get("memory_log", []):
                    print(f"  [memory] {line}")
    print(f"\n{answer}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="csv / tsv / xlsx / json / parquet")
    ap.add_argument("--ask", help="ask one question and exit")
    args = ap.parse_args()

    graph = build_graph()
    thread = uuid.uuid4().hex
    if args.file:
        STORE.load(args.file)
        print(f"Loaded {STORE.name}: {STORE.df.shape[0]} rows x {STORE.df.shape[1]} cols")
    if args.ask:
        if STORE.df is None:
            ap.error("--ask needs a data file")
        return ask(graph, thread, args.ask)

    print(HELP)
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/help":
            print(HELP)
        elif q.startswith("/load"):
            try:
                STORE.load(q.split(maxsplit=1)[1])
                thread = uuid.uuid4().hex  # fresh conversation for new data; memory persists
                print(f"Loaded {STORE.name}: {STORE.df.shape[0]} rows x {STORE.df.shape[1]} cols")
            except Exception as e:
                print(f"Could not load: {e}")
        elif q == "/schema":
            print(STORE.schema_summary())
        elif q == "/memory":
            print(MEMORY.render(MEMORY.items))
        elif q.startswith("/forget"):
            ref = q.split(maxsplit=1)[1] if " " in q else ""
            print(f"removed {MEMORY.delete(ref)} item(s)")
        elif q == "/clear-memory":
            MEMORY.clear()
            print("memory cleared")
        elif STORE.df is None:
            print("Load a dataset first: /load path/to/file.csv")
        else:
            ask(graph, thread, q)


if __name__ == "__main__":
    main()
