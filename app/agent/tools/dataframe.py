"""Pandas-dataframe analysis tool.

Wraps `create_pandas_dataframe_agent` so the main DeepAgents agent sees a
single tool, `analyze_dataframe`, that takes a natural-language question and
returns the answer. The wrapper exists because the pandas agent is itself an
`AgentExecutor`, not a callable tool — promoting it to a DeepAgents subagent
would add a redundant LLM hop with no planning benefit (the pandas agent has
no surface for a parent to plan over).

The pandas agent is built lazily once via `lru_cache`: the FastAPI process
pays the CSV-load + LLM-bind cost on the first request, not at import time.
"""

from functools import lru_cache

import pandas as pd
from langchain_core.tools import tool
from langchain_experimental.agents import create_pandas_dataframe_agent

from app.agent.prompts import DATA_ANALYST_PROMPT
from app.config import settings
from app.llm import subagent_chat_model

# Hard ceiling on the string returned to the main agent. The inner pandas
# REPL can produce arbitrarily large outputs (e.g. when the caller asks for
# "all columns of N rows"); without a cap a single tool call can balloon
# the run to millions of tokens. ~4k characters is enough for a small table
# or a summary stat block, and forces the inner agent to actually reduce.
_MAX_OUTPUT_CHARS = 8000
_TRUNCATION_NOTE = (
    "\n\n[output truncated by analyze_dataframe — the inner agent returned "
    "more than {limit} characters; re-ask with an aggregation or .head(N) "
    "to get a smaller result]"
)


@lru_cache(maxsize=1)
def _pandas_agent():
    fraud = pd.read_csv(settings.DATASETS_DIR / "fraud_dataset.csv")
    purchase = pd.read_csv(settings.DATASETS_DIR / "product_purchase_dataset.csv")
    return create_pandas_dataframe_agent(
        llm=subagent_chat_model(),
        df=[fraud, purchase],
        agent_type="tool-calling",
        prefix=DATA_ANALYST_PROMPT,
        allow_dangerous_code=True,  # required; documented in README §9
        verbose=False,
        return_intermediate_steps=False,
        max_iterations=8,
    )


@tool
async def analyze_dataframe(question: str) -> str:
    """Answer questions about the fraud and purchase CSV datasets via a
    pandas REPL agent.

    Use for aggregations, filters, statistics, and ad-hoc analytics that the
    ML predictors don't cover. The agent has access to two DataFrames:
    `df1` (fraud transactions, target `fraud`) and `df2` (customer purchases,
    target `purchase_amount`). Ask one focused question at a time.

    The tool's output is hard-capped at ~4000 characters. Do NOT ask for
    "all columns" or "full details of N rows" — request an aggregation,
    a `.head(N)` with the specific columns you need, or summary statistics.
    """
    result = await _pandas_agent().ainvoke({"input": question})
    output = str(result["output"])  # pyright: ignore[reportAny]
    if len(output) > _MAX_OUTPUT_CHARS:
        return output[:_MAX_OUTPUT_CHARS] + _TRUNCATION_NOTE.format(
            limit=_MAX_OUTPUT_CHARS
        )
    return output
