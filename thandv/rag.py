"""Local RAG: chunking, embedding, per-persona JSONL stores, retrieve.

v0.3.0 ships a pure-Python pipeline with no new dependencies. Each persona
has its own append-only JSONL store under
`~/.thandv/corpora/<persona>/chunks.jsonl`. Embeddings come from Ollama's
`nomic-embed-text` (local). Cosine similarity is computed in pure Python.

This is fast enough up to ~50k chunks per persona (~100 ms / retrieve at
that scale). When we outgrow it, the same `ingest_text` / `retrieve` API
can swap in numpy or a real vector DB without touching callers.

Design notes:
- Append-only. Re-ingesting the same source duplicates by default; clear
  the persona's corpus with `thandv ingest --clear --persona <name>`.
- Paragraph-aware chunking with character-level overlap. Code is handled
  by the same chunker — most code has natural blank-line separation; a
  code-aware splitter is a later enhancement.
- `persona="all"` aggregates across every persona's corpus.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests

from thandv.config import OLLAMA_HOST, THANDV_HOME

CORPORA_DIR = THANDV_HOME / "corpora"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768  # nomic-embed-text output dimension

DEFAULT_CHUNK_CHARS = 1200
DEFAULT_OVERLAP = 200


# --- Data model -----------------------------------------------------------

@dataclass
class Chunk:
    text: str
    source: str
    embedding: list[float]


# --- Paths ---------------------------------------------------------------

def corpus_path(persona: str) -> Path:
    return CORPORA_DIR / persona / "chunks.jsonl"


def list_personas_with_corpora() -> list[str]:
    if not CORPORA_DIR.exists():
        return []
    return sorted(
        p.name for p in CORPORA_DIR.iterdir()
        if p.is_dir() and (p / "chunks.jsonl").exists()
    )


# --- Chunking ------------------------------------------------------------

def chunk_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> Iterator[str]:
    """Paragraph-aware splitter with character overlap.

    Splits on blank lines, greedily concatenates paragraphs up to `max_chars`,
    then yields with the last `overlap` characters carried into the next
    chunk so context isn't lost at boundaries.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive; got {max_chars}")
    # Clamp overlap to half a chunk: the hard-split step (max_chars -
    # overlap) must stay positive — overlap >= max_chars used to raise
    # (step 0) or silently yield nothing (negative step) for oversized
    # paragraphs — and an overlap near max_chars would re-embed almost
    # every character many times.
    overlap = max(0, min(overlap, max_chars // 2))
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return

    buf: list[str] = []
    buf_len = 0

    for para in paragraphs:
        # Oversized single paragraph — emit current buffer, then hard-split.
        if len(para) > max_chars:
            if buf:
                yield "\n\n".join(buf)
                buf = []
                buf_len = 0
            for i in range(0, len(para), max_chars - overlap):
                yield para[i : i + max_chars]
                if i + max_chars >= len(para):
                    break  # the rest is already inside this piece
            continue

        if buf_len + len(para) + 2 > max_chars and buf:
            chunk = "\n\n".join(buf)
            yield chunk
            # Carry at most `overlap` chars, and never so many that tail +
            # separator + paragraph overflows max_chars (chunks used to grow
            # to max_chars + overlap + 2).
            tail_len = min(overlap, max_chars - len(para) - 2)
            tail = chunk[-tail_len:] if tail_len > 0 else ""
            buf = [tail, para] if tail else [para]
            buf_len = sum(len(x) for x in buf) + 2 * (len(buf) - 1)
        else:
            buf.append(para)
            buf_len += len(para) + 2

    if buf:
        yield "\n\n".join(buf)


# --- Embedding ----------------------------------------------------------

def embed(text: str) -> list[float]:
    """Embed via Ollama's /api/embeddings. Raises on network/HTTP error."""
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


# --- Similarity --------------------------------------------------------

def cosine_sim(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine. Returns 0 if either vector is zero."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# --- Storage -----------------------------------------------------------

def load_chunks(persona: str) -> list[Chunk]:
    path = corpus_path(persona)
    if not path.exists():
        return []
    out: list[Chunk] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip lines we can't use rather than failing every retrieve:
            # an ingest interrupted mid-write (Ctrl-C, disk full) leaves a
            # truncated final line in this append-only file.
            try:
                data = json.loads(line)
                out.append(
                    Chunk(text=data["text"], source=data["source"], embedding=data["embedding"])
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return out


def _load_all() -> list[Chunk]:
    out: list[Chunk] = []
    for persona in list_personas_with_corpora():
        out.extend(load_chunks(persona))
    return out


def clear_corpus(persona: str) -> bool:
    path = corpus_path(persona)
    if not path.exists():
        return False
    path.unlink()
    return True


# --- Ingest ------------------------------------------------------------

INGESTABLE_SUFFIXES = (".md", ".markdown", ".txt", ".rst", ".py", ".pyi")

# Directory names skipped when walking a directory: VCS metadata, virtual
# envs, caches, and vendored dependencies. `thandv ingest .` in a project
# checkout used to embed every .py file under .venv/ and node_modules/ —
# thousands of third-party files that drown out the user's own code.
# Any directory whose name starts with "." is skipped as well.
SKIPPED_DIR_NAMES = frozenset(
    {"node_modules", "__pycache__", "site-packages", "venv", "build", "dist"}
)


def _iter_ingestable_files(root: Path) -> Iterator[Path]:
    """Ingestable files under `root`, in a stable order, pruning skipped
    dirs. Suffix match is case-insensitive (README.MD counts)."""
    for f in sorted(root.rglob("*")):
        rel_dirs = f.relative_to(root).parts[:-1]
        if any(d.startswith(".") or d in SKIPPED_DIR_NAMES for d in rel_dirs):
            continue
        if f.name.startswith("."):
            continue
        if f.suffix.lower() in INGESTABLE_SUFFIXES and f.is_file():
            yield f


def ingest_text(text: str, source: str, persona: str = "all") -> int:
    """Chunk → embed → append. Returns number of chunks written."""
    path = corpus_path(persona)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a") as f:
        for chunk in chunk_text(text):
            try:
                emb = embed(chunk)
            except Exception as e:
                raise RuntimeError(f"embedding failed: {e}") from e
            f.write(
                json.dumps(
                    {
                        "text": chunk,
                        "source": source,
                        "embedding": emb,
                        "ingested_at": time.time_ns(),
                    }
                )
                + "\n"
            )
            written += 1
    return written


def ingest_path(p: Path, persona: str = "all") -> tuple[int, int]:
    """Ingest a file or directory.

    Returns (files_ingested, chunks_written). Directories are walked
    recursively for `INGESTABLE_SUFFIXES`.
    """
    if p.is_file():
        n = ingest_text(p.read_text(errors="replace"), source=str(p), persona=persona)
        return (1, n)

    files = 0
    chunks = 0
    for f in _iter_ingestable_files(p):
        chunks += ingest_text(f.read_text(errors="replace"), source=str(f), persona=persona)
        files += 1
    return (files, chunks)


# --- Retrieve ---------------------------------------------------------

def retrieve(
    query: str,
    persona: str = "all",
    k: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    """Top-k chunks by cosine similarity, scoped to a persona's corpus.

    `persona="all"` searches across every persona's corpus.
    Returns dicts: {source, score, text, persona?}.
    """
    if persona == "all":
        chunks: list[tuple[Chunk, str]] = [
            (c, name) for name in list_personas_with_corpora() for c in load_chunks(name)
        ]
    else:
        chunks = [(c, persona) for c in load_chunks(persona)]

    if not chunks:
        return []

    q_emb = embed(query)
    scored = [(cosine_sim(q_emb, c.embedding), c, name) for c, name in chunks]
    scored.sort(key=lambda t: t[0], reverse=True)

    out: list[dict] = []
    for score, c, name in scored[:k]:
        if score < min_score:
            break
        out.append(
            {
                "source": c.source,
                "score": round(score, 4),
                "text": c.text,
                "persona": name,
            }
        )
    return out


# --- Stats ------------------------------------------------------------

def corpus_stats(persona: str | None = None) -> dict:
    if persona is None or persona == "all":
        per_persona = [corpus_stats(p) for p in list_personas_with_corpora()]
        return {
            "personas": per_persona,
            "total_chunks": sum(s["chunks"] for s in per_persona),
        }
    chunks = load_chunks(persona)
    sources = sorted({c.source for c in chunks})
    path = corpus_path(persona)
    return {
        "persona": persona,
        "chunks": len(chunks),
        "sources": len(sources),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "path": str(path),
    }


# --- Health ----------------------------------------------------------

def embed_model_available() -> bool:
    """Check Ollama has the embedding model pulled."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        tags = [m["name"] for m in r.json().get("models", [])]
        return any(t.startswith(EMBED_MODEL) for t in tags)
    except requests.RequestException:
        return False
