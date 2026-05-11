"""Rebuild the Pinecone KB namespace from `financial_documents/`.

Always wipes the namespace before re-upserting so the index reflects the source
files exactly. Designed to be called from a CLI (`scripts/ingest_kb.py`) or
from the future `POST /kb/ingest` FastAPI endpoint — `IngestReport` is the
shared response shape.
"""

import logging
import time
from pathlib import Path

from pydantic import BaseModel

from app.config import settings
from app.retrieval.splitter import split_markdown_dir
from app.retrieval.vectorstore import build_vector_store, ensure_index_async

log = logging.getLogger(__name__)


class IngestReport(BaseModel):
    files: int
    chunks: int
    index: str
    namespace: str
    elapsed_seconds: float


async def rebuild_kb(directory: Path | None = None) -> IngestReport:
    started = time.perf_counter()
    directory = directory or settings.KB_DIR

    log.info("ensuring pinecone index %s exists", settings.PINECONE_INDEX)
    host = await ensure_index_async()

    log.info("wiping namespace %s", settings.PINECONE_NAMESPACE)
    # `delete(delete_all=True)` is a no-op against a missing namespace, so
    # this is safe on a brand-new index.
    from pinecone import PineconeAsyncio  # noqa: PLC0415

    pc = PineconeAsyncio(api_key=settings.PINECONE_API_KEY.get_secret_value())
    try:
        async with pc.IndexAsyncio(host=host) as idx:
            try:
                await idx.delete(delete_all=True, namespace=settings.PINECONE_NAMESPACE)
            except Exception as exc:  # pragma: no cover — Pinecone returns 404 on empty namespace
                log.info("namespace wipe skipped: %s", exc)
    finally:
        await pc.close()

    log.info("splitting markdown files in %s", directory)
    chunks = list(split_markdown_dir(directory))
    files = len({c.metadata["source"] for c in chunks})
    log.info("split %d files into %d chunks", files, len(chunks))

    vector_store = build_vector_store(host)
    log.info("upserting %d chunks into %s", len(chunks), settings.PINECONE_NAMESPACE)
    await vector_store.aadd_documents(
        documents=chunks,
        ids=[c.metadata["chunk_id"] for c in chunks],
    )

    elapsed = time.perf_counter() - started
    report = IngestReport(
        files=files,
        chunks=len(chunks),
        index=settings.PINECONE_INDEX,
        namespace=settings.PINECONE_NAMESPACE,
        elapsed_seconds=round(elapsed, 2),
    )
    log.info("ingest complete: %s", report.model_dump_json())
    return report
