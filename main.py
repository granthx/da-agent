"""CLI for the data-analysis agent.

  python main.py sample_data/customers.csv
  python main.py sample_data/sales.csv --ask "Which region has the highest revenue?"
"""
import argparse
import json
import time
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from da_agent.config import MAX_TOOL_STEPS, OUTPUT_DIR
from da_agent.data import STORE
from da_agent.graph import build_graph, _text
from da_agent.memory import MEMORY

HELP = """Commands: /load <file>  /schema  /memory  /forget <id-or-key>  /clear-memory  /export  /help  /quit
Anything else is a question about the data."""


def ask(graph, thread_id: str, question: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25 + MAX_TOOL_STEPS * 2}
    answer = ""
    tools_used = []
    for update in graph.stream({"messages": [HumanMessage(question)]}, config, stream_mode="updates"):
        for node, out in (update or {}).items():
            out = out or {}
            if node == "analyst":
                msg = out["messages"][-1]
                for tc in getattr(msg, "tool_calls", None) or []:
                    call_sig = f"{tc['name']}({json.dumps(tc['args'], default=str)})"
                    tools_used.append(tc['name'])
                    print(f"  [tool] {call_sig}")
                if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                    answer = _text(msg.content)
            elif node == "curator":
                for line in out.get("memory_log", []):
                    print(f"  [memory] {line}")
    print(f"\n{answer}\n")
    return {"question": question, "tools": tools_used, "answer": answer}


def export_report(history: list[dict]) -> str:
    if not history:
        return "No questions asked yet; nothing to export."
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"report_{stamp}.md"
    lines = [
        f"# Data Analysis Report — {STORE.name or 'Session'}",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset Profile",
        f"- **Name:** `{STORE.name or 'None'}`",
        f"- **Rows:** {len(STORE.df) if STORE.df is not None else 0:,}",
        f"- **Columns:** {STORE.df.shape[1] if STORE.df is not None else 0}",
        "",
        "## Analysis Q&A",
    ]
    for i, item in enumerate(history, 1):
        lines.append(f"### Q{i}: {item['question']}")
        if item.get("tools"):
            lines.append(f"*Tools executed:* " + ", ".join(f"`{t}`" for t in set(item["tools"])))
            lines.append("")
        lines.append(item.get("answer", ""))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return f"Report successfully exported to {path}"


def main():
    ap = argparse.ArgumentParser(description="da-agent: Autonomous Data Analysis Agent")
    ap.add_argument("file", nargs="?", help="csv / tsv / xlsx / json / parquet")
    ap.add_argument("--ask", help="ask one question and exit")
    args = ap.parse_args()

    try:
        graph = build_graph()
    except RuntimeError as e:
        print(f"\n[Configuration Error] {e}\n")
        return

    thread = uuid.uuid4().hex
    history: list[dict] = []

    if args.file:
        try:
            STORE.load(args.file)
            print(f"Loaded {STORE.name}: {STORE.df.shape[0]} rows x {STORE.df.shape[1]} cols")
        except Exception as e:
            print(f"Failed to load '{args.file}': {e}")
            return

    if args.ask:
        if STORE.df is None:
            ap.error("--ask needs a data file")
        ask(graph, thread, args.ask)
        return

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
        elif q in ("/export", "/report"):
            print(export_report(history))
        elif STORE.df is None:
            print("Load a dataset first: /load path/to/file.csv")
        else:
            record = ask(graph, thread, q)
            history.append(record)


if __name__ == "__main__":
    main()
