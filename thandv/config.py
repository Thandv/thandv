"""User-level configuration and paths."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

THANDV_HOME = Path(os.environ.get("THANDV_HOME", Path.home() / ".thandv"))
SESSIONS_DIR = THANDV_HOME / "sessions"
MEMORY_DIR = THANDV_HOME / "memory"
SKILLS_DIR = THANDV_HOME / "skills"
PERSONAS_DIR = THANDV_HOME / "personas"
CONFIG_PATH = THANDV_HOME / "config.json"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


@dataclass
class Config:
    model: str = ""  # auto-picked by runtime if empty
    backend: str = "ollama"  # ollama | llamacpp (future)
    temperature: float = 0.2
    max_tokens: int = 4096
    auto_tools: bool = True
    persona: str = "code"  # code | writer | finance
    finance_paper_trading_enabled: bool = False  # opt-in gate; see paper_trading.py
    image_generation_enabled: bool = False  # opt-in gate; see image_backend.py

    @classmethod
    def load(cls) -> "Config":
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # A broken config used to crash every command, including
                # `thandv config --set` (the way to fix it). Defaults keep
                # every opt-in gate off, so falling back is the safe choice.
                print(
                    f"warning: ignoring unreadable config {CONFIG_PATH} ({e}); "
                    "using defaults",
                    file=sys.stderr,
                )
                return cls()
            if not isinstance(data, dict):
                print(
                    f"warning: ignoring config {CONFIG_PATH} (not a JSON object); "
                    "using defaults",
                    file=sys.stderr,
                )
                return cls()
            return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
        return cls()

    def save(self) -> None:
        THANDV_HOME.mkdir(parents=True, exist_ok=True)
        atomic_write_text(CONFIG_PATH, json.dumps(asdict(self), indent=2))


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` so readers see the old or the new content,
    never a truncated file (write to a sibling temp file, then rename).

    Plain `write_text` truncates first; a crash, Ctrl-C or full disk in
    between leaves an empty or half-written JSON file behind, which the
    next `json.loads` rejects.
    """
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def ensure_dirs() -> None:
    for d in (THANDV_HOME, SESSIONS_DIR, MEMORY_DIR, SKILLS_DIR, PERSONAS_DIR):
        d.mkdir(parents=True, exist_ok=True)
