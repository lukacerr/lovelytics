"""Chat-model factories.

Single source of truth for LLM instantiation. Both factories point at Novita's
OpenAI-compatible endpoint and use `temperature=0` for fully deterministic
output — fraud-analyst answers should be stable across re-runs of the same
question, and we lean on tool calls for any "creativity" we need.

`top_p=0.9` is kept as a defensive belt-and-braces against any provider that
silently re-introduces stochasticity at temp 0.
"""

from langchain_openai import ChatOpenAI

from app.config import settings

_TEMPERATURE = 0.0
_TOP_P = 0.9


def main_chat_model() -> ChatOpenAI:
    """Model for the main DeepAgents agent (high-context, function-calling)."""
    return ChatOpenAI(
        model=settings.MAIN_MODEL,
        base_url=settings.NOVITA_BASE_URL,
        api_key=settings.NOVITA_API_KEY,
        temperature=_TEMPERATURE,
        top_p=_TOP_P,
    )


def subagent_chat_model() -> ChatOpenAI:
    """Smaller / cheaper model used inside subagents and the pandas agent."""
    return ChatOpenAI(
        model=settings.SUBAGENT_MODEL,
        base_url=settings.NOVITA_BASE_URL,
        api_key=settings.NOVITA_API_KEY,
        temperature=_TEMPERATURE,
        top_p=_TOP_P,
    )
