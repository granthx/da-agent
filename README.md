# `da-agent` — Autonomous Data Analysis Agent

[![CI Tests](https://github.com/granthx/da-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/granthx/da-agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Groq LPU](https://img.shields.io/badge/Inference-Groq_LPU-green.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An intelligent, tool-augmented multi-agent system designed to explore, query, statistically analyze, and visualize tabular datasets using natural language — **without guessing or hallucinating numbers**.

Powered by **LangGraph**, **Pandas**, **SciPy**, and **Groq LPUs**, with a persistent **curated memory engine** that remembers definitions, reusable filters, and preferences across sessions.

---

## 💡 Key Highlights

- **Anti-Hallucination Guarantee:** The LLM acts purely as a reasoning coordinator. All calculations (means, ANOVA tests, correlations, percentiles) are strictly computed by Python tools on real DataFrames.
- **Dual-Agent Architecture:** Clean separation of concerns:
  - **Analyst Agent:** Focuses exclusively on answering questions using data tools.
  - **Curator Agent:** Silently identifies, validates, and stores durable business facts without polluting analytical reasoning.
- **Dataset Agnostic:** Load CSV, TSV, Excel, JSON, or Parquet files. Tools automatically discover and validate column schemas dynamically.
- **Persistent Knowledge Memory:** Saves domain-specific terms (e.g. *"churn means cancelled status"*) and reusable pandas query filters that persist across sessions in `memory/memory.json`.
- **Fuzzy Self-Correction:** When column names are misspelled, tools automatically suggest the closest available matches via `difflib` so the agent self-heals in flight.
- **Safe Execution:** Sandboxed pandas fallback with strict keyword filters preventing unsafe system-level calls.

---

## 🛠️ System Architecture

```
                    ┌─────────────────────────┐
                    │       User Query        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │      recall Node      │ ───► Pulls relevant knowledge
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
         ┌─────────► │     analyst Node      │ ◄─────────┐
         │           └───────────┬───────────┘           │
         │                       │                       │
         │ Tool Observation      │ Tool Call Request     │ Tool Feedback
         │                       ▼                       │ (Self-healing)
         │           ┌───────────────────────┐           │
         └────────── │      tools Node       │ ──────────┘
                     │  (Pandas/SciPy/Stats) │
                     └───────────┬───────────┘
                                 │ (No more tool calls)
                                 ▼
                     ┌───────────────────────┐
                     │     curator Node      │ ───► Validates & saves durable
                     └───────────┬───────────┘      rules to memory/memory.json
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Final Answer       │
                    └─────────────────────────┘
```

---

## 🧰 The 9 Analytical Tools

| Tool | Category | Capabilities & Example Questions |
|---|---|---|
| `describe_dataset` | Profiling | Dataset shape, null counts, data types, and quartile statistics. *"What columns are here?"* |
| `query_rows` | Retrieval | Filtering, sorting, and row inspections with pagination. *"Show top 5 customers by spend."* |
| `aggregate` | Group Statistics | Aggregations (`mean`, `sum`, `std`, `median`, `nunique`). *"What is the churn rate by plan?"* |
| `trend_over_time` | Time Series | Bucketing (D, W, ME, QE, YE) with period-over-period growth rates. *"Monthly revenue growth."* |
| `correlation_analysis` | Relationships | Pearson / Spearman correlation with sample sanity warnings. *"Does age correlate with spend?"* |
| `compare_groups` | Statistical Tests | One-way ANOVA $F$-test with $p$-value and effect size ($\eta^2$). *"Does spend differ across tiers?"* |
| `detect_outliers` | Anomaly Detection | Interquartile Range (IQR) and Z-score outlier detection. *"Flag unusual transactions."* |
| `plot_chart` | Data Visualization | Headless rendering of bar, line, scatter, box, or hist plots saved to `outputs/*.png`. |
| `run_python` | Dynamic Fallback | Restricted pandas sandbox for complex multi-step queries when no single tool applies. |

---

## 🧠 Curated Memory Types

Unlike naive chat log storage, memory items are structured, validated, and de-duplicated:

- **`definition`**: Business domain rules (e.g., `key="churn"`, `value="status == 'cancelled'"`).
- **`named_filter`**: Validated Pandas `.query()` strings automatically bound into tool arguments.
- **`preference`**: User analysis and display formatting preferences.
- **`correction`**: User adjustments overriding earlier knowledge.

---

## 🚀 Quick Start

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/granthx/da-agent.git
cd da-agent

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and configure your [Groq API Key](https://console.groq.com/keys):
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=qwen/qwen3-32b
```

### 3. Run Tests (Zero API Tokens Required)

Verify that all 9 tools, pandas operations, and memory mechanisms pass deterministic tests:
```bash
python -m pytest -q
```

---

---

## 💻 Usage

### 🌐 Option 1: Interactive Web UI (Recommended for Demos)

Launch the Streamlit web dashboard for browser-based conversational analysis, live tool inspection, and inline chart visualization:

```bash
streamlit run app.py
```

Features:
- **File Ingestion:** Drag-and-drop CSV, TSV, Excel, JSON, or Parquet files, or choose built-in sample datasets.
- **Auditable Execution:** Real-time expanders reveal exact tool signatures and parameters (`aggregate`, `compare_groups`, etc.).
- **Visual Rendering:** Headless matplotlib charts display directly in the chat stream.
- **Memory Inspector:** Sidebar panel displays active definitions, reusable query filters, and conflict histories.

---

### 🖥️ Option 2: Interactive Terminal Shell

Launch the agent in your terminal:

```bash
python main.py sample_data/customers.csv
# Or if installed via pip install -e .:
da-agent sample_data/customers.csv
```

Inside the interactive shell:
- Type natural language questions:
  ```text
  you> What are the most common subscription plans?
  you> Compare monthly spend between active and cancelled accounts.
  you> Active users means status is 'active'. Remember that.
  you> What is the average spend of active users?
  ```
- Use CLI control commands:
  - `/load <path>` — Switch to a different dataset on the fly.
  - `/schema` — Print the current dataset summary and column types.
  - `/memory` — Inspect all remembered definitions and named filters.
  - `/forget <key>` — Delete a specific remembered item.
  - `/clear-memory` — Wipe persisted memory.
  - `/export` — Compile session findings into an executive Markdown report in `outputs/`.
  - `/quit` — Exit the agent session.

---

### ⚡ Option 3: Single-Query Command Line Mode

Run a one-off analytical question directly and exit:

```bash
python main.py sample_data/sales.csv --ask "Which region has the highest total revenue?"
```

---

## 📂 Project Structure

```
da-agent/
├── da_agent/                      # Core agent framework package
│   ├── __init__.py                # Package exports
│   ├── config.py                  # Environment config & runtime constants
│   ├── data.py                    # Multi-format data loader & schema inspector
│   ├── llm.py                     # Groq LLM factory & validation
│   ├── memory.py                  # Persistent curated MemoryStore
│   ├── tools.py                   # 9 statistical, querying, & plotting tools
│   └── graph.py                   # LangGraph dual-agent state machine
├── sample_data/                   # Demo datasets & generators
│   ├── customers.csv              # Synthetic customer SaaS data
│   ├── sales.csv                  # Multi-region sales transactions
│   └── make_sample_data.py        # Reproducible dataset generator script
├── tests/                         # Pytest test suite (100% offline)
│   ├── test_core.py               # 9 analysis tools + persistent memory tests
│   └── test_graph.py              # LangGraph compilation & ReAct loop tests
├── memory/                        # Runtime storage
│   └── .gitkeep                   # Tracks directory (memory.json ignored)
├── outputs/                       # Visualization & report output directory
│   └── .gitkeep                   # Tracks directory (*.png and *.md ignored)
├── app.py                         # Interactive Streamlit Web UI
├── main.py                        # CLI entrypoint
├── pyproject.toml                 # Standard PEP 621 packaging & CLI entrypoint
├── requirements.txt               # Project dependencies
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules
├── ROADMAP.md                     # Production roadmap & priority checklist
├── CONTEXT.md                     # Initial project context
├── KNOWLEDGE_AND_ARCHITECTURE.md  # Deep theoretical & technical architectural guide
└── README.md                      # Project documentation
```

---

## 📖 Deep Dive Documentation

For a comprehensive breakdown of the mathematics, LangGraph state graph internals, ReAct loop mechanics, and interview defense topics, read [KNOWLEDGE_AND_ARCHITECTURE.md](KNOWLEDGE_AND_ARCHITECTURE.md).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
