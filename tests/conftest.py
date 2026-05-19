"""Shared fixtures.

We don't want tests to read or write the user's real ~/.thandv. Every test
gets a fresh tmp_path home, monkeypatched into both `thandv.config` and
`thandv.memory` (the latter binds the paths by name at import time).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thandv import config, evals as eval_mod, memory, rag, session_promotion, trainer, training_backend


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

    # Isolate the bundled-skills dir too so existing tests can write only
    # the skills they assert on without inheriting the real bundled set.
    bundled_skills = home / "bundled_skills"
    bundled_skills.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(memory, "BUNDLED_SKILLS_DIR", bundled_skills)

    monkeypatch.setattr(session_promotion, "SESSIONS_DIR", sessions)

    monkeypatch.setattr(trainer, "TRAINING_DIR", training)
    monkeypatch.setattr(trainer, "QUEUE_DIR", tq)
    monkeypatch.setattr(trainer, "PROCESSED_DIR", tp)
    monkeypatch.setattr(trainer, "ADAPTERS_DIR", ta)
    monkeypatch.setattr(trainer, "LOGS_DIR", tl)
    monkeypatch.setattr(trainer, "STATE_PATH", training / "state.json")
    monkeypatch.setattr(trainer, "PID_PATH", training / "trainer.pid")

    corpora = home / "corpora"
    corpora.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "CORPORA_DIR", corpora)

    evals_dir = home / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(eval_mod, "BEST_RECORDS_PATH", evals_dir / "best.json")

    base_models = training / "base"
    base_models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(training_backend, "TRAINING_DIR", training)
    monkeypatch.setattr(training_backend, "BASE_MODELS_DIR", base_models)
    monkeypatch.setattr(training_backend, "LLAMA_CPP_DIR", training / "llama.cpp")

    return home


@pytest.fixture
def fake_embed(monkeypatch: pytest.MonkeyPatch):
    """Replace `rag.embed` (and the rag.embed alias inside `tools`) with a
    deterministic fake. Mapping tests use this to control the score order.
    """
    mapping: dict[str, list[float]] = {}

    def _embed(text: str) -> list[float]:
        if text in mapping:
            return mapping[text]
        # Default fallback: 1 in the slot derived from the first 4 chars.
        v = [0.0] * rag.EMBED_DIM
        idx = sum(ord(c) for c in text[:4]) % rag.EMBED_DIM
        v[idx] = 1.0
        return v

    monkeypatch.setattr(rag, "embed", _embed)
    return mapping
