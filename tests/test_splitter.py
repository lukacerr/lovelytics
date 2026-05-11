"""Tests for the markdown splitter."""

from pathlib import Path

from app.retrieval.splitter import split_markdown_dir, split_markdown_file

KB_DIR = Path("financial_documents")
SAMPLE_FILE = KB_DIR / "01_credit_card_fraud_indicators.md"
_CHUNK_SIZE_LIMIT = 800
_CHUNK_SIZE_SLACK = 50


def test_split_markdown_file_produces_chunks_with_required_metadata():
    chunks = split_markdown_file(SAMPLE_FILE)
    assert chunks, "expected at least one chunk"

    for chunk in chunks:
        assert chunk.metadata["source"] == SAMPLE_FILE.name
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"].startswith(f"{SAMPLE_FILE.stem}::")
        assert "header_path" in chunk.metadata

    populated = sum(1 for c in chunks if c.metadata["header_path"])
    assert populated / len(chunks) >= 0.8, (
        "header_path should be populated on the vast majority of chunks"
    )


def test_chunk_ids_are_unique():
    chunks = list(split_markdown_dir(KB_DIR))
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id collisions detected"


def test_chunk_sizes_are_bounded():
    chunks = list(split_markdown_dir(KB_DIR))
    longest = max(len(c.page_content) for c in chunks)
    assert longest <= _CHUNK_SIZE_LIMIT + _CHUNK_SIZE_SLACK, (
        f"chunk size {longest} exceeds limit {_CHUNK_SIZE_LIMIT} + slack"
    )
