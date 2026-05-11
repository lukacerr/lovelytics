"""Markdown → chunked `Document`s with citation-friendly metadata.

Two-stage splitter chain:

1. `MarkdownHeaderTextSplitter` preserves section context as metadata. It has
   no library default for `headers_to_split_on`, so we pass the canonical
   three-level set; the KB documents in this repo never go below `###`.
2. `RecursiveCharacterTextSplitter` bounds chunk size so embeddings and
   downstream LLM context stay predictable.

Each output chunk carries the metadata schema:

```
{
  "source":      "01_credit_card_fraud_indicators.md",
  "header_path": "Credit Card Fraud Indicators > Common Fraud Indicators > Geographic Red Flags",
  "chunk_id":    "01_credit_card_fraud_indicators::0007",
}
```
"""

from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 120

_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS_TO_SPLIT_ON)
_recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
)


def _build_header_path(metadata: dict[str, str]) -> str:
    parts = [metadata.get("h1"), metadata.get("h2"), metadata.get("h3")]
    return " > ".join(p for p in parts if p)


def split_markdown_file(path: Path) -> list[Document]:
    """Split a single markdown file into citation-ready chunks."""
    text = path.read_text(encoding="utf-8")
    header_chunks = _header_splitter.split_text(text)
    sized_chunks = _recursive_splitter.split_documents(header_chunks)

    out: list[Document] = []
    for idx, chunk in enumerate(sized_chunks):
        header_path = _build_header_path(chunk.metadata)
        chunk_id = f"{path.stem}::{idx:04d}"
        out.append(
            Document(
                page_content=chunk.page_content,
                metadata={
                    "source": path.name,
                    "header_path": header_path,
                    "chunk_id": chunk_id,
                },
            )
        )
    return out


def split_markdown_dir(directory: Path) -> Iterator[Document]:
    """Yield chunks from every `*.md` file in `directory`, sorted by name."""
    for path in sorted(directory.glob("*.md")):
        yield from split_markdown_file(path)
