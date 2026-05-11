"""DeepAgents subagent definitions.

Each subagent is a `SubAgent` TypedDict — DeepAgents auto-wraps it into a
`task()` tool that the main agent can call by name.
"""

from deepagents import SubAgent

from app.agent.prompts import KB_RESEARCHER_PROMPT
from app.agent.tools.kb import kb_search
from app.llm import subagent_chat_model


def kb_researcher_subagent() -> SubAgent:
    return {
        "name": "kb_researcher",
        "description": (
            "Searches the financial-fraud knowledge base and returns a "
            "synthesised answer with citations. Use for any question about "
            "fraud indicators, AML, KYC, regulations, or domain concepts."
        ),
        "system_prompt": KB_RESEARCHER_PROMPT,
        "tools": [kb_search],
        "model": subagent_chat_model(),
    }
