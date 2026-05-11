"""FastAPI app entry point.

Lives inside the `app/` package rather than at the repo root so all
application code is colocated. `fastapi dev` and `fastapi run` accept the
explicit path (`app/main.py`); the Makefile and Dockerfile pass it.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, health, kb
from app.config import settings

app = FastAPI(
    title="Lovelytics — Financial Fraud Analyst",
    description=(
        "Prototype agent assistant for financial fraud analysts. "
        "Streams a DeepAgents run over SSE on `POST /chat`."
    ),
    version="0.1.0",
)

# CORS — open in development, explicit allow-list in production.
_cors_origins = ["*"] if settings.ENV == "development" else settings.CORS_ALLOW_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    # No cookies / credentials in this design; SSE auth is via header only.
    allow_credentials=False,
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(kb.router)
