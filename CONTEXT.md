# CONTEXT.md — Data Analysis Agent (read this first)

Two audiences: (1) **you**, to understand and defend the project; (2) **Antigravity / any coding AI**,
to get full context. Section 9 is a ready-to-paste prompt for Antigravity.

---
## 1. The task in one paragraph
Build an agent that takes any tabular dataset plus natural-language questions and answers **by calling its own
tools** (not by making things up). Requirements: ≥5 genuinely different tools, works on datasets beyond one example,
clearly state what stack you used and what you built. Bonus: **persistent memory** that stores *useful facts*
(definitions, corrections, preferences, reusable filters), retrieves them when relevant, changes future answers,
handles conflicts, and lets the user inspect/control it.

## 2. Key decisions (and why) — be ready to explain these
| Decision | Choice | Reason |
|---|---|---|
| Framework | **LangGraph** (not plain LangChain) | The job is a *loop* (think → call tool → observe → repeat) plus a separate memory step. LangGraph models that as an explicit graph with state; LangChain's classic agent executors hide the loop and are legacy. We still use `langchain-core` (tool decorator, messages) and `langchain-groq` (model wrapper), so knowing both helps. |
| Multi-agent? | **Yes, a light 2-agent design** | *Analyst* agent does the analysis with tools. *Curator* agent (separate LLM call, different prompt, no tools) decides what to remember. Separation of concerns: the analyst is never distracted by memory bookkeeping, and the curator judges the user's message on its own. We deliberately did **not** add more agents (planner, critic…) — more agents = more cost, latency and failure modes for no gain at this scale. |
| LLM | **Groq API**, default `qwen/qwen3-32b` | Groq is a US inference company (fast, free tier). It is *not* China-based; what you probably mean are **Chinese open models served on Groq**: Qwen (Alibaba) or Kimi K2 (Moonshot). Set `GROQ_MODEL` in `.env`. **Verify the exact model ID and tool-calling support at console.groq.com/docs/models** — Groq rotates models. Any tool-calling model works. |
| Dataset-agnostic | Tools take column names as arguments; the schema is injected into the prompt at runtime | Nothing is hard-coded to one dataset. |
| Safety of tools | Tools are fixed, typed functions; `run_python` is filtered + restricted | Predictable behaviour; escape hatch only as last resort. It is **not** a hardened sandbox (say so if asked). |

## 3. Architecture
```
START → recall → analyst ⇄ tools → curator → END
```
- **recall** – loads memory items relevant to the question (scoped to the current dataset or global) and puts them in the prompt.
- **analyst** – tool-calling LLM. Prompt forces "never state a number without a tool result", apply saved memory, answer concisely with caveats.
- **tools** – LangGraph `ToolNode`. Errors are returned to the LLM (`handle_tool_errors=True`) so it self-corrects (e.g. misspelt column → tool suggests the closest names).
- **curator** – second LLM call returning JSON operations (`upsert`/`delete`). Code validates them (a saved named filter must actually run on the current data) before writing.
- Guards: max tool rounds per question (`MAX_TOOL_STEPS`), history windowing (last 4 user turns), tool output truncation, try/except around model calls (Groq sometimes emits malformed tool calls).
- State (`AgentState`): `messages`, `memory_context`, `memory_log`. `MemorySaver` keeps within-session chat; `memory.json` keeps cross-session knowledge.

## 4. The nine tools (seven kinds of analysis)
| Tool | Kind | Example question |
|---|---|---|
| `describe_dataset` | profiling | "What's in this data?" |
| `query_rows` | filter/sort/lookup | "Show the 5 oldest Pro customers" |
| `aggregate` | group-by statistics | "Average spend per plan?" |
| `trend_over_time` | time series + growth | "How did monthly revenue change?" |
| `correlation_analysis` | relationships | "What correlates with support tickets?" |
| `compare_groups` | statistical test (ANOVA, eta²) | "Does satisfaction differ by status?" |
| `detect_outliers` | anomaly detection (IQR/z-score) | "Any weird spend values?" |
| `plot_chart` | visualisation (PNG) | "Plot spend by plan" |
| `run_python` | flexible pandas (restricted) | "Share of customers with >3 tickets AND Pro plan?" |

All return text computed on the real DataFrame and echo the filter used, so answers are auditable.

## 5. Memory design (the brownie-points part)
Stored item: `{id, type, key, value, scope, created, updated, history[]}`.
- **Types:** `definition`, `named_filter` (a pandas `.query()` string), `preference`, `correction`.
- **What is NOT stored:** chat transcripts, results, raw data. The curator prompt says so explicitly.
- **Retrieval:** items are scored by keyword overlap with the question, filtered to `global` + current dataset scope, top 15 injected into the analyst prompt.
- **Changes behaviour:** `named_filter` is a real tool argument, so "how many active customers?" runs `status != 'cancelled'` with no re-explaining.
- **Conflicts / stale facts:** same `(type, key, scope)` → *update*, old value kept in `history`. Current user message beats saved memory.
- **Validation:** a filter that doesn't run on the dataset is rejected and reported.
- **User control:** `/memory` (inspect), `/forget <id or key>`, `/clear-memory`. Plain JSON file, human-readable.
- **Honest limits:** relevance is keyword-based (fine for dozens of items; embeddings would be the next step); the curator is an LLM so it can occasionally miss or over-save — that's why every change is printed as `[memory] ...` and reversible.

## 6. File map
```
main.py                     CLI (interactive or --ask "question")
da_agent/config.py          env config
da_agent/data.py            loader (csv/tsv/xlsx/json/parquet), date auto-parsing, schema summary
da_agent/tools.py           the 9 tools
da_agent/memory.py          persistent memory store
da_agent/llm.py             Groq model factory
da_agent/graph.py           LangGraph workflow + both agents' prompts
sample_data/                customers.csv, sales.csv (+ generator script)
tests/test_core.py          tests for tools + memory (no API key needed)
memory/memory.json          created at runtime
outputs/                    charts land here
```

## 7. Run it + demo script
```bash
pip install -r requirements.txt && cp .env.example .env   # add GROQ_API_KEY
python main.py sample_data/customers.csv
```
Suggested live demo (shows tools, memory, generality):
1. `What columns do I have and any data-quality issues?` → describe_dataset
2. `Average monthly spend by plan` → aggregate; `Any outliers in monthly_spend?` → detect_outliers
3. `Does satisfaction differ by status?` → compare_groups (honest "not significant" result)
4. `Active customers means status is not cancelled. Remember that.` → `[memory] added named_filter 'active_customers'`
5. `How many active customers per region?` → agent uses `named_filter="active_customers"`
6. **Quit and restart**, ask `Average spend of active customers` → still applied (persistence)
7. `Actually, active means status is exactly 'active'.` → `[memory] updated ...` (conflict handling), check `/memory`
8. `/load sample_data/sales.csv` → `Monthly revenue trend` → different dataset, same agent
9. Let evaluators bring their own CSV.

Tests: `python -m pytest -q` (no API key required).

## 8. What was verified vs. what you must verify
- ✅ **Verified in the build environment:** all tools run on both sample datasets; tool results match pandas; memory add/update/history/persistence/delete; safety filter blocks `import os`; every module compiles (7 tests pass).
- ⚠️ **Not run end-to-end:** LangGraph, `langchain-groq` and network were unavailable there, so `graph.py`/`main.py` weren't executed with a live model. Expect to spend a little time on first run fixing version/model-ID issues — that is normal, and worth mentioning honestly as part of your "how I learned" story.
- Likely first-run snags: (a) wrong/retired `GROQ_MODEL` id → pick one from Groq's models page that supports tool use; (b) `pip` version drift in `langgraph` imports; (c) the model occasionally passing a string where a list is expected → the tool error message makes it retry.

## 9. Ready-to-paste prompt for Antigravity
> I have a working LangGraph data-analysis agent in this repo (read CONTEXT.md, then da_agent/graph.py, tools.py, memory.py).
> Goal: run it end to end with Groq and fix anything that breaks. Steps: create a venv, install requirements.txt,
> create .env from .env.example with my GROQ_API_KEY, run `python -m pytest -q`, then run
> `python main.py sample_data/customers.csv --ask "Average monthly_spend by plan"` and iterate until tool calls work.
> If the model ID in GROQ_MODEL is unavailable, check Groq's model list and choose a tool-calling model.
> Constraints: keep the 2-agent design (analyst + curator), keep tools dataset-agnostic (no hard-coded columns),
> keep memory curated (no raw chat/data dumps). Do not add features until the baseline works. Then propose
> (don't implement) improvements.

## 10. "My own contribution" statement (adapt for submission)
"I designed the architecture (LangGraph analyst/tools/curator loop), tool set and memory schema; the code was written
with LLM assistance (Claude), which I reviewed, ran and adapted. Stack: Python, LangGraph, LangChain-core, Groq API with
an open Chinese model (Qwen), pandas/scipy/matplotlib. Sample data is synthetic and generated by my script."
(Only claim what's true — edit this after you've actually run and modified it.)

## 11. Likely interview questions
- *Why LangGraph over LangChain?* Explicit control flow/state for a loop + memory step; LangChain agents are higher-level and now built on LangGraph anyway.
- *How do you stop hallucinated numbers?* Prompt rule + numbers only come from tool outputs + tools echo filters and row counts.
- *Why not let the LLM write all the pandas?* Fixed tools are predictable, testable and safe; `run_python` exists only as a filtered fallback.
- *How does memory avoid junk?* Separate curator with strict save rules, typed items, validation of filters, dedupe/update by key, user can inspect and delete.
- *What would you improve?* Embedding-based retrieval, a critic step that re-checks numbers, evals with a question/answer set, Streamlit UI, real sandbox (subprocess/container) for `run_python`, big-data support via DuckDB.
- *Weaknesses?* Keyword memory retrieval; curator is an LLM (can misjudge); `run_python` is not a true sandbox; date detection heuristic; whole dataset in RAM.

## 12. Learning path (to actually understand it)
1. Read `tools.py` top to bottom — plain pandas, the easiest part.
2. Read `memory.py` — plain Python + JSON.
3. Read `graph.py` — see how `StateGraph`, nodes, conditional edges and `ToolNode` fit.
4. Change one thing per experiment: add a 10th tool (e.g. `pivot_table`), edit the curator prompt, swap `GROQ_MODEL`, and note what changed. Those notes are your "how I learn" evidence.
