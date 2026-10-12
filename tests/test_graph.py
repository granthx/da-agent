"""End-to-end deterministic tests for the LangGraph state machine without network or API tokens."""
import pathlib
import sys
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
