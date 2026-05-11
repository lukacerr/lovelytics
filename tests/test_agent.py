"""Tests for agent assembly and tool wiring.

These tests deliberately avoid live LLM calls — they're flaky, slow, and
expensive, and they don't catch the things that actually break (tool wiring,
schema drift, accidentally exposing kb_search to the main agent).

Live end-to-end validation happens via `scripts/chat.py` during development.
"""

from typing import Any, cast

import pandas as pd
import pytest

from app.agent.builder import build_agent
from app.agent.subagents import kb_researcher_subagent
from app.agent.tools.kb import kb_search
from app.agent.tools.ml import predict_fraud, predict_purchase
from app.ml.schemas import FraudFeatures, PurchaseFeatures


def _fraud_row() -> dict[str, Any]:
    df = pd.read_csv("datasets/fraud_dataset.csv")
    row = df.drop(columns=["transaction_id", "customer_id", "fraud"]).iloc[0]
    return cast(dict[str, Any], row.to_dict())


def _purchase_row() -> dict[str, Any]:
    df = pd.read_csv("datasets/product_purchase_dataset.csv")
    row = df.drop(columns=["customer_id", "purchase_amount"]).iloc[0]
    return cast(dict[str, Any], row.to_dict())


# --- ML tool wiring ---------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_fraud_tool_returns_toon_with_expected_keys():
    features = FraudFeatures(**_fraud_row())  # pyright: ignore[reportAny]
    result: object = await predict_fraud.ainvoke({"features": features.model_dump()})  # pyright: ignore[reportAny]
    # TOON output is a string; we just assert the headline keys appear in it.
    assert isinstance(result, str)
    assert "probability:" in result
    assert "label:" in result
    assert "top_features" in result


@pytest.mark.asyncio
async def test_predict_purchase_tool_returns_toon_with_expected_keys():
    features = PurchaseFeatures(**_purchase_row())  # pyright: ignore[reportAny]
    result: object = await predict_purchase.ainvoke({"features": features.model_dump()})  # pyright: ignore[reportAny]
    assert isinstance(result, str)
    assert "predicted_amount:" in result
    assert "top_features" in result


# --- KB tool wiring ---------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_search_tool_returns_toon_string():
    """Live Pinecone call — fast and deterministic given the static KB."""
    result: object = await kb_search.ainvoke({"query": "credit card fraud", "k": 3})  # pyright: ignore[reportAny]
    assert isinstance(result, str)
    # TOON header for a list of dicts: `[N,]{key1,key2,...}:`
    assert result.startswith("[")
    assert "source" in result
    assert "header_path" in result


# --- Subagent definition ----------------------------------------------------


def test_kb_researcher_subagent_has_kb_search_only():
    sub = kb_researcher_subagent()
    assert sub["name"] == "kb_researcher"
    tools = sub.get("tools", [])
    tool_names = {getattr(t, "name", None) for t in tools}
    assert tool_names == {"kb_search"}, (
        f"kb_researcher should only have kb_search; got {tool_names}"
    )


# --- Main-agent assembly ---------------------------------------------------


def test_build_agent_returns_runnable_with_streaming():
    agent = build_agent()
    assert agent is not None
    # LangGraph runnables expose both invoke and astream_events.
    assert hasattr(agent, "invoke")
    assert hasattr(agent, "astream_events")


def test_main_agent_does_not_expose_kb_search():
    """The boundary in AGENTS.md §7 is enforced by *construction*, not at
    runtime — kb_search is wired into the kb_researcher subagent's tool list,
    never into the main agent's. This test pins that down by inspecting the
    main agent's tool registry.
    """
    agent = build_agent()
    tools_node = agent.get_graph().nodes["tools"].data
    assert tools_node is not None
    tools_by_name = set(tools_node.tools_by_name.keys())  # pyright: ignore[reportAttributeAccessIssue]
    assert "kb_search" not in tools_by_name, (
        "kb_search must not be exposed to the main agent — it belongs to the "
        f"kb_researcher subagent. Got main-agent tools: {sorted(tools_by_name)}"
    )
    # Our three first-class tools must be wired in.
    assert {"predict_fraud", "predict_purchase", "analyze_dataframe"}.issubset(tools_by_name)
    # DeepAgents scaffolding: planning + delegation are always available.
    assert {"write_todos", "task"}.issubset(tools_by_name)


def test_main_agent_filesystem_tools_excluded_from_model():
    """DeepAgents bakes `FilesystemMiddleware` into every graph; we can't
    strip the middleware itself (it's required scaffolding). Instead we
    register a `HarnessProfile` with `excluded_tools` covering all FS tools,
    so `_ToolExclusionMiddleware` filters them before each model call.

    This test pins the registration: it builds the agent (which registers
    the profile) and then asks DeepAgents' own resolver what profile applies
    to our model. If the resolver no longer returns our exclusion set, FS
    tools would silently leak back into the model's tool list — fail fast.
    """
    from deepagents._models import get_model_identifier, get_model_provider  # noqa: PLC0415
    from deepagents.profiles.harness.harness_profiles import (  # noqa: PLC0415
        _harness_profile_for_model,  # pyright: ignore[reportPrivateUsage]
    )

    from app.agent.builder import _FILESYSTEM_TOOL_NAMES  # noqa: PLC0415  # pyright: ignore[reportPrivateUsage]
    from app.llm import main_chat_model

    build_agent()  # ensures profile is registered
    model = main_chat_model()
    spec = f"{get_model_provider(model)}:{get_model_identifier(model)}"
    profile = _harness_profile_for_model(model, spec)
    assert _FILESYSTEM_TOOL_NAMES.issubset(profile.excluded_tools), (
        f"expected FS tools {sorted(_FILESYSTEM_TOOL_NAMES)} in the resolved "
        f"harness profile's excluded_tools; got {sorted(profile.excluded_tools)}"
    )
