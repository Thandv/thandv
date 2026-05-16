"""Shared fixtures.

We don't want tests to read or write the user's real ~/.thandv. Every test
gets a fresh tmp_path home, monkeypatched into both `thandv.config` and
`thandv.memory` (the latter binds the paths by name at import time).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thandv import config, memory


@pytest.fixture
def thandv_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "thandv_home"
    sessions = home / "sessions"
    skills = home / "skills"
    mem = home / "memory"
    for d in (home, sessions, skills, mem):
        d.mkdir(parents=True, exist_ok=True)
    cfg_path = home / "config.json"

    monkeypatch.setattr(config, "THANDV_HOME", home)
    monkeypatch.setattr(config, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(config, "SKILLS_DIR", skills)
    monkeypatch.setattr(config, "MEMORY_DIR", mem)
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_path)

    monkeypatch.setattr(memory, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(memory, "SKILLS_DIR", skills)
    monkeypatch.setattr(memory, "MEMORY_DIR", mem)

    return home
