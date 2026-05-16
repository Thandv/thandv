"""Shared fixtures.

We don't want tests to read or write the user's real ~/.thandv. Every test
gets a fresh tmp_path home, monkeypatched into both `thandv.config` and
`thandv.memory` (the latter binds the paths by name at import time).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thandv import config, memory, trainer


@pytest.fixture
def thandv_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "thandv_home"
    sessions = home / "sessions"
    skills = home / "skills"
    mem = home / "memory"
    training = home / "training"
    tq = training / "queue"
    tp = training / "processed"
    ta = training / "adapters"
    tl = training / "logs"
    for d in (home, sessions, skills, mem, training, tq, tp, ta, tl):
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

    monkeypatch.setattr(trainer, "TRAINING_DIR", training)
    monkeypatch.setattr(trainer, "QUEUE_DIR", tq)
    monkeypatch.setattr(trainer, "PROCESSED_DIR", tp)
    monkeypatch.setattr(trainer, "ADAPTERS_DIR", ta)
    monkeypatch.setattr(trainer, "LOGS_DIR", tl)
    monkeypatch.setattr(trainer, "STATE_PATH", training / "state.json")
    monkeypatch.setattr(trainer, "PID_PATH", training / "trainer.pid")

    return home
