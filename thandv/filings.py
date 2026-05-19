"""10-K / earnings-call ingestion into the finance-persona RAG corpus.

This is a thin wrapper over `rag.ingest_text` that stamps each chunk
with structured provenance (filing type, ticker, source filename) so
the finance persona can answer "what did APPL say about supply chain
in their 10-K?" by retrieving the matching chunks.

We don't parse XBRL or HTML; the user is responsible for handing us
plain text. SEC filings download as HTML by default -- a recommended
pre-processing step (documented below) is `lynx -dump` or `pandoc -t
plain`. Out of scope for this module to bundle a parser.
"""

from __future__ import annotations

from pathlib import Path

from thandv import rag

FINANCE_PERSONA = "finance"

FILING_TYPES = ("10-K", "10-Q", "8-K", "earnings-call", "annual-report", "other")


def _validate_filing_type(filing_type: str) -> str:
    if filing_type not in FILING_TYPES:
        raise ValueError(
            f"unknown filing_type {filing_type!r}. "
            f"Known: {list(FILING_TYPES)}"
        )
    return filing_type


def filing_source(
    filing_type: str, ticker: str | None, path: str | Path
) -> str:
    """Build the canonical source string. Greppable: starts with
    `filing::` and embeds type + ticker, so a single substring search
    finds every chunk from a given filing without parsing JSON.
    """
    name = Path(path).name
    ticker_part = (ticker or "UNKNOWN").upper()
    return f"filing::{filing_type}::{ticker_part}::{name}"


def ingest_filing(
    path: str | Path,
    *,
    filing_type: str = "10-K",
    ticker: str | None = None,
    persona: str = FINANCE_PERSONA,
) -> int:
    """Ingest a plain-text filing into the persona's RAG corpus.

    Returns the chunk count written. Raises FileNotFoundError if the
    path doesn't exist and ValueError on an unknown filing_type.
    """
    _validate_filing_type(filing_type)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if not p.is_file():
        raise ValueError(f"{p}: not a file (directories not supported)")
    if not rag.embed_model_available():
        raise RuntimeError(
            f"embed model {rag.EMBED_MODEL!r} not available on Ollama. "
            "Run `ollama pull nomic-embed-text` first."
        )
    text = p.read_text(errors="replace")
    source = filing_source(filing_type, ticker, p)
    return rag.ingest_text(text, source=source, persona=persona)


def list_filings(persona: str = FINANCE_PERSONA) -> list[dict]:
    """List ingested filings grouped by source. Useful for `thandv finance
    filings` to remind the user what's already in the corpus.

    Returns one dict per unique source: {source, filing_type, ticker,
    filename, chunks}. Sources that don't match the `filing::` prefix
    are skipped (they came from `thandv ingest`, not this module).
    """
    chunks = rag.load_chunks(persona)
    by_source: dict[str, int] = {}
    for c in chunks:
        if c.source.startswith("filing::"):
            by_source[c.source] = by_source.get(c.source, 0) + 1
    out: list[dict] = []
    for source, n in sorted(by_source.items()):
        parts = source.split("::", 3)
        # filing::TYPE::TICKER::filename
        if len(parts) == 4:
            _, ftype, ticker, fname = parts
        else:
            ftype, ticker, fname = "?", "?", source
        out.append(
            {
                "source": source,
                "filing_type": ftype,
                "ticker": ticker,
                "filename": fname,
                "chunks": n,
            }
        )
    return out
