"""FastAPI dependencies shared across routes.

`require_api_key` is the only one for now. It's a no-op in development and a
strict equality check on `X-API-Key` against `settings.API_KEY` otherwise.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from app.config import settings

# `auto_error=False` so we control the error path ourselves — FastAPI's
# default would return 403 with a generic detail, and we want 401 with a
# clearer message.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_ApiKey = Annotated[str | None, Depends(_api_key_header)]


def require_api_key(api_key: _ApiKey) -> None:
    """Gate a route on `X-API-Key` in production.

    - `ENV=development`: returns immediately (no header required).
    - Otherwise: header must exactly match `settings.API_KEY`. If `API_KEY`
      is unset in production this is an operator misconfiguration and we
      raise 500 rather than silently accepting requests.
    """
    if settings.ENV == "development":
        return

    if settings.API_KEY is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY is not configured on the server",
        )

    if api_key is None or api_key != settings.API_KEY.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-API-Key",
        )
