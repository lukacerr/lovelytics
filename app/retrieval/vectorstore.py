"""Pinecone index lifecycle + LangChain vector store factory.

`ensure_index_async` is idempotent: it creates the serverless index if missing
and waits until it reports ready. Cloud/region are pinned to AWS us-east-1
(Pinecone's free-tier region) at the call site rather than in `Settings`,
because they are not knobs we expect to flip at runtime.
"""

import asyncio
from typing import cast

from langchain_pinecone import PineconeVectorStore
from pinecone import PineconeAsyncio, ServerlessSpec

from app.config import settings
from app.retrieval.embeddings import build_embeddings

_DEFAULT_CLOUD = "aws"
_DEFAULT_REGION = "us-east-1"
_READY_POLL_SECONDS = 1.0


async def ensure_index_async() -> str:
    """Create the Pinecone serverless index if missing; return the index host.

    Returns the host string that `langchain-pinecone` uses to construct its
    own async client per call (preferred over passing a long-lived `Index`
    handle, which carries an aiohttp session that goes stale across calls).
    """
    pc = PineconeAsyncio(api_key=settings.PINECONE_API_KEY.get_secret_value())
    try:
        if not await pc.has_index(settings.PINECONE_INDEX):
            await pc.create_index(
                name=settings.PINECONE_INDEX,
                dimension=settings.EMBEDDING_DIMENSIONS,
                metric="cosine",
                spec=ServerlessSpec(cloud=_DEFAULT_CLOUD, region=_DEFAULT_REGION),
            )
            while True:
                # `pinecone` SDK: `describe_index().status` is typed as `Any`.
                desc = await pc.describe_index(settings.PINECONE_INDEX)
                if desc.status["ready"]:  # pyright: ignore[reportAny]
                    break
                await asyncio.sleep(_READY_POLL_SECONDS)
        # `pinecone` SDK: `.host` is typed as `Any`.
        host = cast(str, (await pc.describe_index(settings.PINECONE_INDEX)).host)
    finally:
        await pc.close()
    return host


def build_vector_store(host: str) -> PineconeVectorStore:
    return PineconeVectorStore(
        embedding=build_embeddings(),
        namespace=settings.PINECONE_NAMESPACE,
        host=host,
        pinecone_api_key=settings.PINECONE_API_KEY.get_secret_value(),
    )
