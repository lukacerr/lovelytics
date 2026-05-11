FROM ghcr.io/astral-sh/uv:alpine
WORKDIR /app

COPY pyproject.toml uv.lock* ./
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev

# Application code + data assets needed at runtime.
# - app/        : Python package (config, agent, API, retrieval, ML)
#                 — includes app/main.py, the FastAPI entry point
# - scripts/    : CLI wrappers (kb ingestion, model training)
# - financial_documents/ : KB source for /kb/ingest
# - datasets/   : CSVs consumed by the analyze_dataframe tool and ML training
# - models/     : pre-trained ML artifacts (committed; see README §6.3)
COPY app ./app
COPY scripts ./scripts
COPY financial_documents ./financial_documents
COPY datasets ./datasets
COPY models ./models

ENV PATH="/app/.venv/bin:$PATH" ENV=production

EXPOSE 8080
CMD ["fastapi", "run", "app/main.py", "--port", "8080"]
