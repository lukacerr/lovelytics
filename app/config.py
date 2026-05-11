"""Application settings.

Single source of truth for runtime configuration. Imported as a module-level
singleton (`from app.config import settings`).

Operator-facing knobs (anything you'd set per-environment) live here as
required `SecretStr`/`Literal` fields and are populated from OS env vars (or a
`.env` file in development). In-code defaults for things we don't expect ops
to touch (model IDs, index name, KB directory) live as plain class attributes
so swapping them is a code change, not a configuration change.
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load `.env` into `os.environ` early so libraries that read env vars at import
# time (notably LangSmith / langchain tracing) pick them up. Pydantic-settings
# below still reads `.env` directly for its own fields; the two are compatible.
_ = load_dotenv()


class Settings(BaseSettings):
    # Operator-facing (required from environment).
    NOVITA_API_KEY: SecretStr
    PINECONE_API_KEY: SecretStr
    ENV: Literal["development", "production"] = "production"

    # API auth — required in production (enforced at request time by the
    # `require_api_key` dependency). Optional here so dev runs don't need it.
    API_KEY: SecretStr | None = None

    # CORS — ignored in development (we open everything). In production this
    # is the explicit allow-list; defaults cover the local dev SPA and the
    # deployed Cloudflare Pages origin. `pydantic-settings` accepts either a
    # JSON array or a comma-separated string in the env var.
    CORS_ALLOW_ORIGINS: list[str] = [
        "http://localhost:3000",
        "https://lovelytics.luka.software",
    ]

    NOVITA_BASE_URL: str = "https://api.novita.ai/openai/v1"

    MAIN_MODEL: str = "zai-org/glm-5"
    SUBAGENT_MODEL: str = "deepseek/deepseek-v4-flash"
    EMBEDDING_MODEL: str = "baai/bge-m3"
    EMBEDDING_DIMENSIONS: int = 1024

    PINECONE_INDEX: str = "lovelytics-kb"
    PINECONE_NAMESPACE: str = "financial-docs"

    KB_DIR: Path = Path("financial_documents")
    KB_TOP_K: int = 10

    DATASETS_DIR: Path = Path("datasets")
    MODELS_DIR: Path = Path("models")

    # `.env` is read in development; in production (Docker / Cloud Run) the
    # file is absent and pydantic-settings silently falls back to OS env vars.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()  # pyright: ignore[reportCallIssue]
