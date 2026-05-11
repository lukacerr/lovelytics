"""Build the main DeepAgents agent.

Single entry point — used by `scripts/chat.py` today and the FastAPI `/chat`
route. Returns a LangGraph runnable that supports `.invoke`, `.ainvoke`,
`.astream`, and `.astream_events`.

Slim harness: DeepAgents bakes `FilesystemMiddleware` into every graph as
required scaffolding (see `_REQUIRED_MIDDLEWARE` in `deepagents.graph`), so we
can't strip the middleware itself. Instead we register a `HarnessProfile`
keyed on our resolved model spec (`openai:zai-org/glm-5`) with
`excluded_tools` covering every filesystem tool. The exclusion runs after
middleware-injected tools are added, so the visible tool set on the main
agent is just our three tools + `write_todos` + `task`.
"""

from deepagents import HarnessProfile, create_deep_agent, register_harness_profile

from app.agent.prompts import MAIN_PROMPT
from app.agent.subagents import kb_researcher_subagent
from app.agent.tools.dataframe import analyze_dataframe
from app.agent.tools.ml import predict_fraud, predict_purchase
from app.config import settings
from app.llm import main_chat_model

_FILESYSTEM_TOOL_NAMES = frozenset({
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
})

_profile_registered = False


def _ensure_slim_profile():
    """Register the FS-tool exclusion profile once per process.

    Keyed on `openai:<MAIN_MODEL>` because Novita is reached via
    `langchain-openai`'s `ChatOpenAI`, which reports provider `"openai"` and
    identifier `MAIN_MODEL` to DeepAgents' profile resolver.
    """
    global _profile_registered
    if _profile_registered:
        return
    register_harness_profile(
        f"openai:{settings.MAIN_MODEL}",
        HarnessProfile(excluded_tools=_FILESYSTEM_TOOL_NAMES),
    )
    _profile_registered = True


def build_agent():
    _ensure_slim_profile()
    return create_deep_agent(
        model=main_chat_model(),
        tools=[predict_fraud, predict_purchase, analyze_dataframe],
        subagents=[kb_researcher_subagent()],
        system_prompt=MAIN_PROMPT,
    )
