"""`POST /kb/ingest` — rebuild the Pinecone KB namespace.

Wraps `app.retrieval.ingest.rebuild_kb` and returns its `IngestReport`. The
heavy lifting (split + embed + upsert) lives in the retrieval package; this
file is just the HTTP surface and the auth gate.
"""

from fastapi import APIRouter, Depends

from app.api.deps import require_api_key
from app.retrieval.ingest import IngestReport, rebuild_kb

router = APIRouter(prefix="/kb", tags=["kb"])


@router.post(
    "/ingest", response_model=IngestReport, dependencies=[Depends(require_api_key)]
)
async def ingest() -> IngestReport:
    return await rebuild_kb()
