"""Central configuration (all overridable through environment variables / .env)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
MEMORY_PATH = Path(os.getenv("MEMORY_PATH", ROOT / "memory" / "memory.json"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "8"))     # tool rounds per question
ALLOW_PYTHON = os.getenv("ALLOW_PYTHON_TOOL", "1") == "1"  # the run_python escape hatch
MAX_TOOL_OUTPUT_CHARS = 4000                               # keeps the LLM context small
HISTORY_TURNS = 4                                          # user turns sent to the LLM
