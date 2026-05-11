# AGENTS.md

Operational guide for AI coding agents (and humans) working on this repo. Read this before touching code.

---

## 1. What this project is

A prototype financial fraud-analyst assistant. One FastAPI service exposes an SSE `/chat` endpoint backed by a [DeepAgents](https://docs.langchain.com/oss/python/deepagents) main agent that delegates to two specialised subagents and two ML tools. Full architecture and rationale live in `README.md` — read it first.

---

## 2. Non-negotiables

- **Plan before coding.** Even small features get a brief plan. The README's architecture and trade-off sections are the source of truth; if your change contradicts them, update the README in the same PR.
- **Keep the README honest.** After every meaningful change — new module, deleted file, renamed setting, new SSE event, changed schema, new env var, modified repo layout — re-read the relevant README section and bring it back into sync in the same commit. A README that lies is worse than no README. The §11 repo layout in particular drifts the fastest; check it whenever you add or remove a `.py` file.
- **Quality bar.** `make check` (ruff + basedpyright) and `make test` must pass before any commit. Pre-commit enforces this.
- **No new top-level dependencies without a reason.** State the reason in the PR description.
- **No secrets in code.** Everything goes through `app/config.py` (`pydantic-settings`).
- **Don't break the slim DeepAgents harness.** The main agent intentionally ships without the filesystem tools (`builtin_tools=["write_todos"]`). Don't re-enable them unless you have a concrete need.
- **Don't expose `kb_search` to the main agent.** It belongs to `kb_researcher` only — that boundary is deliberate (see README §5.3).
- **Tune LLM params.** Default `temperature=0.1`, `top_p=0.9`. If you change them, justify it in code comments.
- **Don't annotate types when they're inferrable.** Reserve annotations for function parameters, pydantic/`Settings` model fields, and return types that aren't obvious from the body. `def main():` over `def main() -> None:`.
- **Prefer async.** When a library exposes both sync and async APIs (Pinecone, langchain vector stores, FastAPI handlers), use the async one.

---

## 3. Stack at a glance

- Python 3.14, `uv` for env management.
- FastAPI + SSE for serving (entry point: `main.py` at repo root).
- LangChain + DeepAgents + `langchain-openai` (pointed at `https://api.novita.ai/openai`) + `langchain-pinecone` + `langchain-experimental` (pandas dataframe agent).
- scikit-learn `HistGradientBoosting{Classifier,Regressor}` for the ML tools.
- pydantic v2 for all input validation; `pydantic-settings` for config.
- pytest (asyncio mode = auto) for tests, flat under `tests/`.
- Ruff + basedpyright + pre-commit for code quality.
- Web: React 19 + TanStack Router + Vite + Tailwind + shadcn primitives, built as a static SPA.

---

## 4. Models

- **Main agent**: `zai-org/glm-5` (Novita). 200K context, function calling, structured output.
- **Subagents**: `deepseek/deepseek-v4-flash` (Novita). Cheaper, fast, good enough for delegated tasks.
- **Embeddings**: `baai/bge-m3` (Novita, OpenAI-compatible embeddings endpoint), 1024-d, cosine.

All model IDs are env vars (`MAIN_MODEL`, `SUBAGENT_MODEL`, `EMBEDDING_MODEL`). Don't hard-code them.

---

## 5. Repo layout

See README §11. Key conventions:

- `app/` is a package; everything importable lives there.
- The FastAPI app object lives at the repo root (`main.py`) — `fastapi dev` and `fastapi run` work without args.
- `scripts/` contains thin CLI wrappers around `app/` functions. Prefer adding logic to `app/` and calling it from a script, not the other way around.
- `tests/` is **flat**, not split into `unit/` and `integration/`. Coverage is intentionally narrow — enough for safe refactoring, not exhaustive.

---

## 6. How to work

Common targets (see `Makefile`):

```bash
make install     # uv sync + bun install
make train       # train both models, write models/*.joblib + metrics.json
make ingest      # rebuild Pinecone KB namespace
make api         # uv run fastapi dev
make web         # bun run dev (web/)
make web-build   # static SPA build (web/dist)
make test        # pytest
make check       # ruff + basedpyright
make pre-commit  # install + autoupdate pre-commit hooks
```

`make train` is also wired into pre-commit so artifacts in `models/` stay in sync with the CSVs and training code.

After every meaningful edit run `make check` (ruff + basedpyright) and fix what it reports before moving on. Run `make test` after touching anything covered by tests.

---

## 7. Boundaries between agent components

This is the most common place to make a wrong call. Use the table:

| Capability | Owner | Tool name | Notes |
|---|---|---|---|
| Plan / track work | Main agent | `write_todos` | DeepAgents builtin, kept on |
| Delegate | Main agent | `task` | Spawns subagents |
| Predict fraud | Main agent | `predict_fraud` | pydantic-validated input |
| Predict purchase amount | Main agent | `predict_purchase` | pydantic-validated input |
| Query CSVs | `data_analyst` subagent | (pandas REPL) | `allow_dangerous_code=True` — prototype only |
| Search KB | `kb_researcher` subagent | `kb_search` | **Not** exposed to main agent |
| Cite sources | `kb_researcher` subagent | (returned in result) | Mapped to `citation` SSE events |

If you find yourself wanting to give the main agent a fourth tool, ask whether it should be a subagent instead.

---

## 8. SSE event taxonomy

The single source of truth for what the UI consumes. Don't add events without updating `README.md` §5.5 and the web client at the same time.

```
token | tool_start | tool_end | subagent_start | subagent_end | citation | final | error
```

`/chat` builds these by mapping `runnable.astream_events(version="v2")` outputs in `app/sse.py`.

---

## 9. ML conventions

- Pipelines, not raw estimators. `ColumnTransformer` first, model last.
- Fixed `random_state=42` everywhere.
- Metrics → `models/metrics.json` (read by README and tests).
- `permutation_importance` precomputed at training time and stored alongside the model so inference doesn't pay for it.
- Pydantic schemas (`FraudFeatures`, `PurchaseFeatures`) are the contract. If you change a feature column, change the schema and re-train in the same commit.

---

## 10. Knowledge base conventions

- Splitter chain: `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` (`chunk_size=800`, `chunk_overlap=120`).
- Per-chunk metadata: `source`, `header_path`, `chunk_id`, `content_hash`.
- Re-ingestion **rebuilds** the namespace (delete + re-upsert). Don't add incremental upsert logic unless asked.
- `/kb/ingest` is API-key gated whenever `ENV != "development"`.

---

## 11. Web

- TanStack Router but **SPA mode only**. No `@tanstack/react-start` server routes. The `web/dist` build must be deployable to Cloudflare Pages with no Node runtime.
- API base URL via `VITE_API_BASE_URL`.
- One chat route. Don't add navigation cruft — this is a demo.

---

## 12. When in doubt

- Re-read the relevant section of `README.md`.
- Prefer the simpler change. The brief explicitly values "functional over perfect" and "substance over features".
- If a decision changes the architecture or a stated trade-off, update both this file and the README.
