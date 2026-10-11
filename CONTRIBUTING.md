# Contributing to da-agent

We welcome contributions, bug reports, and suggestions!

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/granthx/da-agent.git
   cd da-agent
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest
   ```

## Running Tests

All core analytical tools and memory operations can be tested without an API key:
```bash
python -m pytest -q
```

## Adding New Tools

When adding new tools to `da_agent/tools.py`:
- Use the `@tool` decorator from `langchain_core.tools`.
- Validate column existence with the `_col` helper for friendly error messages and typo suggestions.
- Ensure the tool handles optional `named_filter` and `filter_expr` via `_subset`.
- Add test coverage in `tests/test_core.py`.
