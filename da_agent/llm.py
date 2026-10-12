"""LLM factory. Groq hosts open models; the model id is configurable so you can swap
between Qwen (Alibaba), Kimi K2 (Moonshot), Llama, etc. without touching code."""
import os

from .config import GROQ_MODEL


def get_llm(model: str | None = None):
    from langchain_groq import ChatGroq
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    model = model or GROQ_MODEL
    kwargs = dict(model=model, temperature=0, max_retries=3, timeout=60)
    if "qwen3" in model.lower():
        kwargs["reasoning_format"] = "hidden"
    return ChatGroq(**kwargs)
