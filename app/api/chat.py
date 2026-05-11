"""`POST /chat` — Server-Sent Events stream of the agent run.

The route is intentionally simple: parse a stateless `{messages: [...]}`
request, hand it to `agent.astream_events(version="v2")`, map each event to
an SSE frame via `app.sse`, and stream the result back to the client.

Conversation history is client-managed — there's no thread or session
identifier (README §9, "Stateless `/chat`").
"""

import logging
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.builder import build_agent
from app.sse import events_from_stream, format_sse

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Building the agent is cheap after the first call (LangChain caches the bound
# tools), but doing it at import time keeps the first request responsive.
_agent = build_agent()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


async def _stream(messages: list[ChatMessage]) -> AsyncIterator[str]:
    """Run the agent and yield SSE-formatted frames.

    Any exception inside the agent run is caught, surfaced as a single
    `error` SSE frame, and the stream closes cleanly. The client always sees
    either a `final` frame or an `error` frame.
    """
    payload = {"messages": [m.model_dump() for m in messages]}
    try:
        async for event_name, data in events_from_stream(
            _agent.astream_events(payload, version="v2")
        ):
            yield format_sse(event_name, data)
    except Exception as exc:
        log.exception("/chat stream failed")
        yield format_sse("error", {"message": str(exc), "type": exc.__class__.__name__})


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream(req.messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Defeat reverse-proxy buffering (nginx, Cloudflare).
            "X-Accel-Buffering": "no",
        },
    )
