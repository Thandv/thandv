"""Persistent memory and skills.

This is intentionally simple in v0: skills are markdown files concatenated
into the system prompt; sessions are append-only JSONL transcripts. The
self-improvement track lives here — see research/SELF_IMPROVEMENT.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from thandv.config import MEMORY_DIR, SESSIONS_DIR, SKILLS_DIR


def load_skills(only: tuple[str, ...] | None = None) -> str:
    """Concatenate skill markdown files into one block.

    If `only` is given, load only those skill stems (e.g. ("coding-style",
    "honesty")). If `only` is None, load every markdown file in the skills
    directory.
    """
    if not SKILLS_DIR.exists():
        return ""
    chunks: list[str] = []
    for md in sorted(SKILLS_DIR.glob("*.md")):
        if only is not None and md.stem not in only:
            continue
        chunks.append(f"# Skill: {md.stem}\n{md.read_text()}")
    return "\n\n".join(chunks)


def load_memory() -> str:
    """Load all memory facts (markdown files in ~/.thandv/memory/)."""
    if not MEMORY_DIR.exists():
        return ""
    chunks: list[str] = []
    for md in sorted(MEMORY_DIR.glob("*.md")):
        chunks.append(md.read_text().strip())
    return "\n".join(chunks)


def new_session_path() -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR / f"{int(time.time())}.jsonl"


def append_event(path: Path, event: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(event) + "\n")
