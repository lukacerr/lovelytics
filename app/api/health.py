"""Health endpoint.

Returns 200 even when sub-checks fail; the body reflects which subsystems
are degraded. Kept cheap on purpose — readiness probes shouldn't trigger
expensive work.
"""

import asyncio
import logging

from fastapi import APIRouter
from pinecone import PineconeAsyncio
from pydantic import BaseModel

from app.config import settings

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_PINECONE_TIMEOUT_SECONDS = 2.0


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    kb_indexed: bool


async def _check_kb_indexed() -> bool:
    """Best-effort: does the configured Pinecone index/namespace look populated?

    Wrapped in a short timeout so a misconfigured or slow Pinecone never
    blocks readiness. All failures collapse to `False`.
    """
    try:
        async with asyncio.timeout(_PINECONE_TIMEOUT_SECONDS):
            pc = PineconeAsyncio(api_key=settings.PINECONE_API_KEY.get_secret_value())
            try:
                exists = await pc.has_index(settings.PINECONE_INDEX)
                if not exists:
                    return False
                description = await pc.describe_index(settings.PINECONE_INDEX)
                host: str = description.host  # pyright: ignore[reportAny]
                async with pc.IndexAsyncio(host=host) as idx:
                    stats = await idx.describe_index_stats()
                    namespaces = stats.get("namespaces") or {}
                    ns = namespaces.get(settings.PINECONE_NAMESPACE)
                    if not ns:
                        return False
                    count = int(ns.get("vector_count", 0) or 0)
                    return count > 0
            finally:
                await pc.close()
    except Exception as exc:
        log.debug("kb health check failed: %s", exc)
        return False


def _check_models_loaded() -> bool:
    fraud = settings.MODELS_DIR / "fraud.joblib"
    purchase = settings.MODELS_DIR / "purchase.joblib"
    return fraud.exists() and purchase.exists()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=_check_models_loaded(),
        kb_indexed=await _check_kb_indexed(),
    )
