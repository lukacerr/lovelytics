"""Map `agent.astream_events(version="v2")` outputs to typed SSE frames.

This module is **pure**: no FastAPI imports, no I/O of its own. It takes an
async iterator of LangChain v2 events and yields `(event_name, payload_dict)`
pairs. The route handler in `app/api/chat.py` wraps each pair with
`format_sse(...)` and writes it to the response.

Keeping the mapper pure makes it easy to test in isolation and means the SSE
event taxonomy lives in one place — see README §5.5 and AGENTS.md §8.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import toon

log = logging.getLogger(__name__)

# Subagent fan-out goes through DeepAgents' `task` tool. We surface those as
# `subagent_*` events rather than generic `tool_*` so the UI can render them
# distinctly. Today there's only one subagent (kb_researcher), but the lookup
# keeps the mapping explicit.
_SUBAGENT_TOOL_NAMES = {"task"}


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Serialise a single SSE frame (`event:` + `data:` + blank line)."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stringify(value: Any) -> str:  # pyright: ignore[reportAny]
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError, ValueError:
        return str(value)  # pyright: ignore[reportAny]


def _truncate(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def _citations_from_kb_search_output(output: Any) -> list[dict[str, str]]:  # pyright: ignore[reportAny]
    """Decode the toon payload returned by `kb_search` into citation dicts.

    `kb_search` returns `toon.encode([{source, header_path, snippet}, ...])`.
    We decode it back so the SSE mapper can emit one `citation` event per
    retrieved chunk. Failures here are non-fatal — citations are best-effort.
    """
    if not isinstance(output, str):
        return []
    try:
        decoded = toon.decode(output)
    except Exception as exc:  # pragma: no cover — defensive
        log.debug("citation decode failed: %s", exc)
        return []
    if not isinstance(decoded, list):
        return []
    citations: list[dict[str, str]] = []
    for item in decoded:  # pyright: ignore[reportAny]
        if not isinstance(item, dict):
            continue
        citations.append(
            {
                "source": str(item.get("source", "")),
                "header_path": str(item.get("header_path", "")),
                "snippet": _truncate(str(item.get("snippet", ""))),
            }
        )
    return citations


async def events_from_stream(
    stream: AsyncIterator[Any],
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Translate v2 stream events into the SSE taxonomy.

    Emits:
      - `token`          on every non-empty chat-model chunk
      - `tool_start`     for any tool except `task`
      - `tool_end`       for any tool except `task` and `kb_search`
      - `subagent_start` for `task` (subagent dispatch)
      - `subagent_end`   for `task`
      - `citation`       one per chunk returned by `kb_search`
      - `final`          when the root chain finishes
      - `error`          if the stream raises
    """
    final_emitted = False
    try:
        async for event in stream:  # pyright: ignore[reportAny]
            event_dict: dict[str, Any] = event if isinstance(event, dict) else {}
            kind = str(event_dict.get("event", ""))  # pyright: ignore[reportAny]
            name = str(event_dict.get("name", ""))  # pyright: ignore[reportAny]
            data: dict[str, Any] = event_dict.get("data") or {}

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                text = getattr(chunk, "content", "")
                if isinstance(text, str) and text:
                    yield "token", {"delta": text}

            elif kind == "on_tool_start":
                tool_input = data.get("input")
                if name in _SUBAGENT_TOOL_NAMES:
                    # `task` input shape: {"description": "...", "subagent_type": "..."}
                    sub_name = "kb_researcher"
                    task_text = ""
                    if isinstance(tool_input, dict):
                        sub_name = str(tool_input.get("subagent_type") or sub_name)
                        task_text = str(tool_input.get("description") or "")
                    yield "subagent_start", {"name": sub_name, "task": task_text}
                else:
                    yield (
                        "tool_start",
                        {
                            "name": name,
                            "args": tool_input if isinstance(tool_input, dict) else {},
                        },
                    )

            elif kind == "on_tool_end":
                output = data.get("output")
                result_text = _stringify(getattr(output, "content", output))
                if name in _SUBAGENT_TOOL_NAMES:
                    yield (
                        "subagent_end",
                        {
                            "name": "kb_researcher",
                            "result": _truncate(result_text, 800),
                        },
                    )
                elif name == "kb_search":
                    # Surface citations rather than the raw toon blob.
                    for citation in _citations_from_kb_search_output(
                        getattr(output, "content", output)
                    ):
                        yield "citation", citation
                else:
                    yield (
                        "tool_end",
                        {"name": name, "result": _truncate(result_text, 800)},
                    )

            elif kind == "on_chain_end" and event_dict.get("parent_ids") == []:
                # Root-level chain completion. The output is the final agent state;
                # we pluck the last assistant message's content.
                output = data.get("output")
                content = _extract_final_content(output)
                if content:
                    yield "final", {"content": content}
                    final_emitted = True
    except Exception as exc:  # pragma: no cover — runtime guard
        log.exception("SSE stream error")
        yield "error", {"message": str(exc), "type": exc.__class__.__name__}
        return

    if not final_emitted:
        # Defensive: emit a final marker so clients can always close cleanly,
        # even if the root `on_chain_end` shape changes in a future LangChain
        # release.
        yield "final", {"content": ""}


def _extract_final_content(output: Any) -> str:  # pyright: ignore[reportAny]
    """Best-effort grab of the final assistant message text from the root output."""
    if output is None:
        return ""
    messages = None
    if isinstance(output, dict):
        messages = output.get("messages")
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Handle multimodal/structured content blocks
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""
