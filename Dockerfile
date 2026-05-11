FROM ghcr.io/astral-sh/uv:alpine
WORKDIR /app

COPY pyproject.toml uv.lock* ./
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev

# Application code + data assets needed at runtime.
# - app/        : Python package (config, retrieval, future agent / API code)
# - scripts/    : CLI wrappers (kb ingestion, model training)
# - main.py     : FastAPI entry point at the repo root
# - financial_documents/ : KB source for /kb/ingest
# - datasets/   : CSVs consumed by the data_analyst subagent and ML training
# - models/     : pre-trained ML artifacts (committed; see README §6.3)
COPY app ./app
COPY scripts ./scripts
COPY main.py ./
COPY financial_documents ./financial_documents
COPY datasets ./datasets
COPY models ./models

ENV PATH="/app/.venv/bin:$PATH" ENV=production

EXPOSE 8080
CMD ["fastapi", "run", "--port", "8080"]
