"""User-level configuration and paths."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

THANDV_HOME = Path(os.environ.get("THANDV_HOME", Path.home() / ".thandv"))
SESSIONS_DIR = THANDV_HOME / "sessions"
MEMORY_DIR = THANDV_HOME / "memory"
SKILLS_DIR = THANDV_HOME / "skills"
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

    @classmethod
    def load(cls) -> "Config":
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
        return cls()

    def save(self) -> None:
        THANDV_HOME.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))


def ensure_dirs() -> None:
    for d in (THANDV_HOME, SESSIONS_DIR, MEMORY_DIR, SKILLS_DIR):
        d.mkdir(parents=True, exist_ok=True)
