"""Persistent memory and skills.

This is intentionally simple in v0: skills are markdown files concatenated
into the system prompt; sessions are append-only JSONL transcripts. The
self-improvement track lives here — see research/SELF_IMPROVEMENT.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from thandv.config import MEMORY_DIR, SESSIONS_DIR, SKILLS_DIR


def _bundled_skills_dir() -> Path:
    """Locate the bundled `skills/` directory across install modes.

    - dev / pip install: lives next to memory.py (shipped via
      `[tool.setuptools.package-data]` in pyproject.toml).
    - PyInstaller frozen exe: extracted to `sys._MEIPASS/thandv/skills`
      (declared via `--add-data` in build.sh).
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "thandv" / "skills"
    return Path(__file__).parent / "skills"


BUNDLED_SKILLS_DIR = _bundled_skills_dir()


def load_skills(only: tuple[str, ...] | None = None) -> str:
    """Concatenate skill markdown files into one block.

    Resolution order: every stem in `only` is looked up in the user's
    `~/.thandv/skills/` first (so users can override), then falling
    back to the bundled `thandv/skills/`. If `only` is None, every
    bundled skill plus every user skill is loaded; user files with the
    same stem as a bundled file override.

    This is what makes the persona `skills=(...)` declarations work on
    a fresh install: prior to this, only user-side files were read, so
    fresh installs silently dropped every skill.
    """
    by_stem: dict[str, Path] = {}
    if BUNDLED_SKILLS_DIR.exists():
        for md in BUNDLED_SKILLS_DIR.glob("*.md"):
            by_stem[md.stem] = md
    if SKILLS_DIR.exists():
        for md in SKILLS_DIR.glob("*.md"):
            by_stem[md.stem] = md  # user overrides bundled
    chunks: list[str] = []
    for stem in sorted(by_stem):
        if only is not None and stem not in only:
            continue
        chunks.append(f"# Skill: {stem}\n{by_stem[stem].read_text()}")
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
    return SESSIONS_DIR / f"{time.time_ns()}.jsonl"


def append_event(path: Path, event: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(event) + "\n")
