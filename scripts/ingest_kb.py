"""CLI: rebuild the Pinecone KB namespace from `financial_documents/`.

Run via `make ingest` or `uv run python -m scripts.ingest_kb`.
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import cast

from app.retrieval.ingest import rebuild_kb


def main():
    parser = argparse.ArgumentParser(description="Rebuild the Pinecone KB namespace.")
    parser.add_argument(
        "--directory",
        type=Path,
        default=None,
        help="Directory of markdown files (defaults to settings.KB_DIR).",
    )
    args = parser.parse_args()
    # `argparse.Namespace` attributes are `Any`; pin the type locally.
    directory = cast(Path | None, args.directory)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    report = asyncio.run(rebuild_kb(directory))
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
