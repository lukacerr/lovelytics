"""Embeddings client.

Uses `OpenAIEmbeddings` against Novita's OpenAI-compatible endpoint. We disable
`check_embedding_ctx_length` because the model name (`baai/bge-m3`) is not
recognised by tiktoken; without this flag langchain-openai would either error
or chunk inputs unnecessarily using the wrong tokenizer.
"""

from langchain_openai import OpenAIEmbeddings

from app.config import settings


def build_embeddings():
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.NOVITA_API_KEY,
        base_url=settings.NOVITA_BASE_URL,
        dimensions=settings.EMBEDDING_DIMENSIONS,
        check_embedding_ctx_length=False,
        # Novita's bge-m3 endpoint rejects oversized payloads (HTTP 413).
        # Keep batches small so a 200+ chunk ingest succeeds in one run.
        chunk_size=64,
    )
