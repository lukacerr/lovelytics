"""Knowledge-base search tool.

This tool is **only** exposed inside the `kb_researcher` subagent — never to
the main agent (see AGENTS.md §7). The boundary is deliberate: it forces
multi-hop research patterns to happen inside a sandboxed planner instead of
the main agent doing brittle one-shot retrievals.
"""

import toon
from langchain_core.tools import tool

from app.retrieval.retrieve import search


@tool
async def kb_search(query: str, k: int = 5) -> str:
    """Search the financial-fraud knowledge base.

    Issue focused queries — call this multiple times with different phrasings
    when the question has several facets. Returns a TOON-encoded list of
    `{source, header_path, snippet}` chunks, ordered by relevance.
    """
    docs = await search(query, k=k)
    payload = [
        {
            "source": d.metadata.get("source", ""),
            "header_path": d.metadata.get("header_path", ""),
            "snippet": d.page_content,
        }
        for d in docs
    ]
    return toon.encode(payload)
