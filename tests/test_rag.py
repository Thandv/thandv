from __future__ import annotations

import json

import pytest

from thandv import rag
from thandv.rag import (
    Chunk,
    chunk_text,
    clear_corpus,
    corpus_path,
    corpus_stats,
    cosine_sim,
    ingest_path,
    ingest_text,
    list_personas_with_corpora,
    load_chunks,
    retrieve,
)


# --- Chunking -------------------------------------------------------------

def test_chunk_text_empty():
    assert list(chunk_text("")) == []
    assert list(chunk_text("   \n\n   ")) == []


def test_chunk_text_single_paragraph_fits():
    out = list(chunk_text("Hello world", max_chars=100))
    assert out == ["Hello world"]


def test_chunk_text_splits_on_double_newline():
    text = "para one\n\npara two\n\npara three"
    out = list(chunk_text(text, max_chars=20, overlap=0))
    # All three paragraphs are short enough to stay separate when the limit
    # is 20 chars and we don't want overlap.
    assert all("para" in c for c in out)
    # No chunk should massively exceed max_chars.
    assert all(len(c) <= 50 for c in out)


def test_chunk_text_overlap_carries_tail():
    text = "a" * 50 + "\n\n" + "b" * 50
    out = list(chunk_text(text, max_chars=60, overlap=10))
    # We expect at least two chunks; the second should start with part of the
    # first chunk's tail.
    assert len(out) >= 2
    assert "a" in out[1][:15]


def test_chunk_text_hard_splits_oversized_paragraph():
    big = "x" * 5000
    out = list(chunk_text(big, max_chars=1000, overlap=100))
    assert len(out) >= 5
    assert all(len(c) <= 1000 for c in out)


# --- Cosine ---------------------------------------------------------------

def test_cosine_identical():
    v = [1.0, 2.0, 3.0]
    assert cosine_sim(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert cosine_sim([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_vector():
    assert cosine_sim([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_sim([1.0, 1.0], [0.0, 0.0]) == 0.0


def test_cosine_opposite():
    assert cosine_sim([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


# --- Paths / persona discovery -------------------------------------------

def test_list_personas_with_corpora_empty(thandv_home):
    assert list_personas_with_corpora() == []


def test_corpus_path_under_corpora_dir(thandv_home):
    p = corpus_path("code")
    assert p.parent.name == "code"
    assert p.name == "chunks.jsonl"


# --- Ingest --------------------------------------------------------------

def test_ingest_text_writes_jsonl(thandv_home, fake_embed):
    n = ingest_text("hello world", source="<inline>", persona="code")
    assert n == 1
    path = corpus_path("code")
    assert path.exists()
    line = path.read_text().strip()
    data = json.loads(line)
    assert data["text"] == "hello world"
    assert data["source"] == "<inline>"
    assert len(data["embedding"]) == rag.EMBED_DIM


def test_ingest_text_multi_chunk(thandv_home, fake_embed):
    text = ("a" * 1000 + "\n\n") * 3
    n = ingest_text(text, source="<big>", persona="code")
    assert n >= 2
    chunks = load_chunks("code")
    assert len(chunks) == n
    assert all(isinstance(c, Chunk) for c in chunks)


def test_ingest_text_propagates_embed_error(thandv_home, monkeypatch):
    def boom(text):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(rag, "embed", boom)
    with pytest.raises(RuntimeError, match="embedding failed"):
        ingest_text("anything", source="<x>", persona="code")


def test_ingest_path_file(thandv_home, fake_embed, tmp_path):
    src = tmp_path / "doc.md"
    src.write_text("# Title\n\nbody paragraph one\n\nbody paragraph two")
    files, chunks = ingest_path(src, persona="writer")
    assert files == 1
    assert chunks >= 1


def test_ingest_path_directory_recurses(thandv_home, fake_embed, tmp_path):
    (tmp_path / "a.md").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.rst").write_text("gamma")
    (tmp_path / "ignore.bin").write_bytes(b"\x00\x01")
    files, chunks = ingest_path(tmp_path, persona="code")
    assert files == 3  # a.md, b.txt, c.rst — not ignore.bin
    assert chunks >= 3


# --- Retrieve ------------------------------------------------------------

def test_retrieve_empty_corpus(thandv_home, fake_embed):
    assert retrieve("anything", persona="code") == []


def test_retrieve_orders_by_cosine(thandv_home, fake_embed):
    # Stage three texts with orthogonal embeddings, then query with a vector
    # that matches "apple" exactly.
    fake_embed["apple text"] = [1.0, 0.0, 0.0] + [0.0] * (rag.EMBED_DIM - 3)
    fake_embed["banana text"] = [0.0, 1.0, 0.0] + [0.0] * (rag.EMBED_DIM - 3)
    fake_embed["cherry text"] = [0.0, 0.0, 1.0] + [0.0] * (rag.EMBED_DIM - 3)
    fake_embed["apple"] = fake_embed["apple text"]

    ingest_text("apple text", source="a.md", persona="code")
    ingest_text("banana text", source="b.md", persona="code")
    ingest_text("cherry text", source="c.md", persona="code")

    results = retrieve("apple", persona="code", k=3)
    assert results[0]["text"] == "apple text"
    assert results[0]["score"] == pytest.approx(1.0, rel=1e-3)
    assert {r["text"] for r in results} == {"apple text", "banana text", "cherry text"}


def test_retrieve_respects_k(thandv_home, fake_embed):
    for i in range(10):
        ingest_text(f"doc {i}", source=f"d{i}.md", persona="code")
    results = retrieve("query", persona="code", k=3)
    assert len(results) == 3


def test_retrieve_persona_scoped(thandv_home, fake_embed):
    ingest_text("code doc", source="c.md", persona="code")
    ingest_text("writer doc", source="w.md", persona="writer")
    code_results = retrieve("anything", persona="code", k=5)
    assert {r["text"] for r in code_results} == {"code doc"}
    writer_results = retrieve("anything", persona="writer", k=5)
    assert {r["text"] for r in writer_results} == {"writer doc"}


def test_retrieve_all_aggregates_across_personas(thandv_home, fake_embed):
    ingest_text("code doc", source="c.md", persona="code")
    ingest_text("writer doc", source="w.md", persona="writer")
    results = retrieve("anything", persona="all", k=5)
    assert {r["text"] for r in results} == {"code doc", "writer doc"}
    personas = {r["persona"] for r in results}
    assert personas == {"code", "writer"}


def test_retrieve_min_score_cutoff(thandv_home, fake_embed):
    fake_embed["a"] = [1.0, 0.0, 0.0] + [0.0] * (rag.EMBED_DIM - 3)
    fake_embed["b"] = [0.0, 1.0, 0.0] + [0.0] * (rag.EMBED_DIM - 3)
    fake_embed["query"] = fake_embed["a"]
    ingest_text("a", source="a.md", persona="code")
    ingest_text("b", source="b.md", persona="code")
    results = retrieve("query", persona="code", k=5, min_score=0.5)
    assert len(results) == 1
    assert results[0]["text"] == "a"


# --- Stats / clear -------------------------------------------------------

def test_corpus_stats_specific(thandv_home, fake_embed):
    ingest_text("one", source="s1.md", persona="code")
    ingest_text("two", source="s2.md", persona="code")
    stats = corpus_stats("code")
    assert stats["chunks"] == 2
    assert stats["sources"] == 2
    assert stats["size_bytes"] > 0


def test_corpus_stats_all(thandv_home, fake_embed):
    ingest_text("x", source="x.md", persona="code")
    ingest_text("y", source="y.md", persona="writer")
    stats = corpus_stats(None)
    assert stats["total_chunks"] == 2
    names = {p["persona"] for p in stats["personas"]}
    assert names == {"code", "writer"}


def test_clear_corpus(thandv_home, fake_embed):
    ingest_text("hi", source="s.md", persona="code")
    assert corpus_path("code").exists()
    assert clear_corpus("code") is True
    assert not corpus_path("code").exists()
    assert clear_corpus("code") is False  # already gone
