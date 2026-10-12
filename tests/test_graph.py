"""End-to-end deterministic tests for the LangGraph state machine without network or API tokens."""
import pathlib
import sys
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage
