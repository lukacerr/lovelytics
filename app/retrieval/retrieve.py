"""Async similarity search against the KB.

This is the function the future `kb_search` tool (owned by the `kb_researcher`
subagent — see AGENTS.md §7) will wrap.

Implementation note: we cache the resolved Pinecone index host (one network
call to look it up) but rebuild the `PineconeVectorStore` on every search.
Caching the vector store across calls runs into a langchain-pinecone bug
where the underlying aiohttp session goes stale after the first request,
yielding `RuntimeError: Session is closed` on every subsequent call. A fresh
vector store per call carries its own short-lived session, which is fine for
our latency budget (the index handle is what's expensive to resolve).
"""

from langchain_core.documents import Document

from app.config import settings
from app.retrieval.vectorstore import build_vector_store, ensure_index_async

_index_host: str | None = None


async def _get_host() -> str:
    global _index_host
    if _index_host is None:
        _index_host = await ensure_index_async()
    return _index_host


async def search(query: str, k: int | None = None) -> list[Document]:
    host = await _get_host()
    store = build_vector_store(host)
    return await store.asimilarity_search(query, k=k or settings.KB_TOP_K)
