"""CLI smoke test for the agent.

Usage:
    uv run python -m scripts.chat "What does the KB say about credit card fraud?"
    uv run python -m scripts.chat --stream "Predict fraud for a $5000 transaction"

`--stream` prints token-level output via `astream_events(version="v2")`,
labelling tool calls and subagent transitions inline. Without it, the script
calls `.invoke` synchronously and prints the final assistant message.

This is a manual rig until the FastAPI `/chat` SSE endpoint lands. Treat the
printed event labels as the spec for the future SSE event mapper.
"""

import argparse
import asyncio
import logging
import sys
from typing import Any, cast

from app.agent.builder import build_agent

logger = logging.getLogger(__name__)


def _print_invoke(prompt: str):
    asyncio.run(_ainvoke(prompt))


async def _ainvoke(prompt: str):
    agent = build_agent()
    result = cast(
        dict[str, Any],
        await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}),
    )
    final = result["messages"][-1]  # pyright: ignore[reportAny]
    print(getattr(final, "content", final))  # pyright: ignore[reportAny]


async def _print_stream(prompt: str):
    agent = build_agent()
    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": prompt}]},
        version="v2",
    ):
        kind: str = event["event"]
        name: str = event.get("name", "")
        data = event.get("data", {})

        if kind == "on_chat_model_stream":
            chunk = data.get("chunk")
            text = getattr(chunk, "content", "")
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()
        elif kind == "on_tool_start":
            sys.stdout.write(f"\n[tool:{name} start]\n")
            sys.stdout.flush()
        elif kind == "on_tool_end":
            output = data.get("output")
            status = getattr(output, "status", None)
            if status == "error":
                sys.stdout.write(
                    f"\n[tool:{name} ERROR]\n{getattr(output, 'content', output)!r}\n"
                )
            else:
                sys.stdout.write(f"\n[tool:{name} end]\n")
            sys.stdout.flush()
        elif kind == "on_chain_error" or kind == "on_tool_error":
            sys.stdout.write(f"\n[{kind}:{name}]\n{data!r}\n")
            sys.stdout.flush()
    sys.stdout.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Run the agent on a single prompt.")
    parser.add_argument("prompt", help="The user message to send to the agent.")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens and tool events via astream_events.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging (HTTP calls, retries, etc.).",
    )
    args = parser.parse_args()

    verbose = bool(args.verbose)  # pyright: ignore[reportAny]
    stream = bool(args.stream)  # pyright: ignore[reportAny]
    prompt = str(args.prompt)  # pyright: ignore[reportAny]

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if stream:
        asyncio.run(_print_stream(prompt))
    else:
        _print_invoke(prompt)


if __name__ == "__main__":
    main()
