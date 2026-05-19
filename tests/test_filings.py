from __future__ import annotations

import pytest

from thandv import filings, rag


# --- filing_source -------------------------------------------------------

def test_filing_source_includes_type_ticker_filename(tmp_path):
    src = filings.filing_source("10-K", "AAPL", tmp_path / "aapl-10k.txt")
    assert src == "filing::10-K::AAPL::aapl-10k.txt"


def test_filing_source_uppercases_ticker():
    assert filings.filing_source("10-Q", "aapl", "x.txt") == "filing::10-Q::AAPL::x.txt"


def test_filing_source_handles_no_ticker():
    assert filings.filing_source("8-K", None, "x.txt") == "filing::8-K::UNKNOWN::x.txt"


# --- ingest_filing -------------------------------------------------------

def test_ingest_filing_writes_chunks(thandv_home, monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "embed_model_available", lambda: True)
    monkeypatch.setattr(rag, "embed", lambda text: [0.1] * rag.EMBED_DIM)

    p = tmp_path / "aapl-10k.txt"
    p.write_text(
        "Item 1. Business\n\nThe company designs phones.\n\n" * 5
    )
    n = filings.ingest_filing(p, filing_type="10-K", ticker="AAPL")
    assert n >= 1

    chunks = rag.load_chunks("finance")
    assert chunks
    assert all(c.source.startswith("filing::10-K::AAPL::") for c in chunks)


def test_ingest_filing_unknown_type_raises(thandv_home, tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("content")
    with pytest.raises(ValueError, match="unknown filing_type"):
        filings.ingest_filing(p, filing_type="form-fake")


def test_ingest_filing_missing_path_raises(thandv_home, tmp_path):
    with pytest.raises(FileNotFoundError):
        filings.ingest_filing(tmp_path / "nope.txt")


def test_ingest_filing_directory_raises(thandv_home, tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        filings.ingest_filing(tmp_path)


def test_ingest_filing_no_embed_model(thandv_home, monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "embed_model_available", lambda: False)
    p = tmp_path / "x.txt"
    p.write_text("content")
    with pytest.raises(RuntimeError, match="embed model"):
        filings.ingest_filing(p)


# --- list_filings --------------------------------------------------------

def test_list_filings_empty(thandv_home):
    assert filings.list_filings() == []


def test_list_filings_groups_by_source(thandv_home, monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "embed_model_available", lambda: True)
    monkeypatch.setattr(rag, "embed", lambda text: [0.1] * rag.EMBED_DIM)

    a = tmp_path / "aapl-10k.txt"
    a.write_text("Apple business description.\n\nIPhone revenue grew.\n\n" * 4)
    b = tmp_path / "msft-10k.txt"
    b.write_text("Microsoft cloud growth.\n\nAzure dominates.\n\n" * 4)
    filings.ingest_filing(a, filing_type="10-K", ticker="AAPL")
    filings.ingest_filing(b, filing_type="10-K", ticker="MSFT")

    rows = filings.list_filings()
    assert len(rows) == 2
    tickers = {r["ticker"] for r in rows}
    assert tickers == {"AAPL", "MSFT"}
    for r in rows:
        assert r["filing_type"] == "10-K"
        assert r["chunks"] >= 1


def test_list_filings_ignores_non_filing_sources(thandv_home, monkeypatch):
    """Chunks ingested via the generic ingest path (no `filing::` prefix)
    must not pollute the filings listing."""
    monkeypatch.setattr(rag, "embed_model_available", lambda: True)
    monkeypatch.setattr(rag, "embed", lambda text: [0.1] * rag.EMBED_DIM)

    rag.ingest_text("Some user-ingested note.", source="manual::note", persona="finance")
    assert filings.list_filings() == []
