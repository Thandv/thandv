from __future__ import annotations

import pytest

from thandv import rag, style_corpus


# --- Registry sanity ----------------------------------------------------

def test_corpus_has_snippets():
    assert len(style_corpus.STYLE_CORPUS) >= 5


def test_every_snippet_has_provenance():
    for s in style_corpus.STYLE_CORPUS:
        assert s.title
        assert s.source
        assert s.license
        assert s.text.strip()


def test_only_pd_or_inhouse_licenses():
    """No copyrighted material slips in. Acceptable: public-domain, or our
    own MIT-licensed in-house material."""
    allowed = {"public-domain", "MIT (per repo LICENSE)"}
    for s in style_corpus.STYLE_CORPUS:
        assert s.license in allowed, f"{s.title!r} has license {s.license!r}"


def test_strunk_snippets_present():
    """We should ship at least three Strunk excerpts for style-guide signal."""
    n_strunk = sum(1 for s in style_corpus.STYLE_CORPUS if "Strunk" in s.title)
    assert n_strunk >= 3


# --- bundle_writer_corpus -----------------------------------------------

def test_bundle_writer_corpus_writes_chunks(thandv_home, monkeypatch):
    """Happy path: ingest is called for every snippet under 'writer'."""
    monkeypatch.setattr(rag, "embed_model_available", lambda: True)
    # Deterministic fake embedding so ingest_text can run.
    monkeypatch.setattr(rag, "embed", lambda text: [0.1] * rag.EMBED_DIM)

    n = style_corpus.bundle_writer_corpus()
    assert n >= len(style_corpus.STYLE_CORPUS)  # at least one chunk per snippet

    chunks = rag.load_chunks("writer")
    assert chunks
    sources = {c.source for c in chunks}
    # Every snippet's source should appear, prefixed with style_corpus::
    for s in style_corpus.STYLE_CORPUS:
        assert any(c.source == f"{style_corpus.SOURCE_PREFIX}{s.title}" for c in chunks), (
            f"missing snippet: {s.title}"
        )
    assert all(src.startswith(style_corpus.SOURCE_PREFIX) for src in sources)


def test_bundle_writer_corpus_refuses_without_embed_model(thandv_home, monkeypatch):
    monkeypatch.setattr(rag, "embed_model_available", lambda: False)
    with pytest.raises(RuntimeError, match="embed model"):
        style_corpus.bundle_writer_corpus()


def test_bundle_writer_corpus_clear_wipes_first(thandv_home, monkeypatch):
    """With clear=True, an existing corpus is wiped before ingest -- so the
    bundle is idempotent rather than doubling on repeat calls."""
    monkeypatch.setattr(rag, "embed_model_available", lambda: True)
    monkeypatch.setattr(rag, "embed", lambda text: [0.1] * rag.EMBED_DIM)

    first = style_corpus.bundle_writer_corpus()
    second = style_corpus.bundle_writer_corpus(clear=True)
    # Second call wipes and re-ingests -- final chunk count matches one run.
    assert second == first
    chunks = rag.load_chunks("writer")
    assert len(chunks) == first


def test_bundle_writer_corpus_no_clear_appends(thandv_home, monkeypatch):
    """Without clear=True we follow `thandv ingest` semantics: appends."""
    monkeypatch.setattr(rag, "embed_model_available", lambda: True)
    monkeypatch.setattr(rag, "embed", lambda text: [0.1] * rag.EMBED_DIM)

    first = style_corpus.bundle_writer_corpus()
    style_corpus.bundle_writer_corpus()  # no clear
    chunks = rag.load_chunks("writer")
    assert len(chunks) == 2 * first
