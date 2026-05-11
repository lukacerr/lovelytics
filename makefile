.PHONY: install train ingest chat api web web-build test check pre-commit

install:
	uv sync && uv run pre-commit install && cd web && bun install

train:
	uv run python -m scripts.train_all

ingest:
	uv run python -m scripts.ingest_kb

chat:
	uv run python -m scripts.chat "$(PROMPT)"

api:
	uv run fastapi dev app/main.py

web:
	cd web && bun run dev

web-build:
	cd web && bun run build

test:
	uv run --dev pytest -x

check:
	uv run ruff check --fix --unsafe-fixes && uv run basedpyright

pre-commit:
	uv run pre-commit install && uv run pre-commit autoupdate
