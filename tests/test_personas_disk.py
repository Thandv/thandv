"""User-defined personas loaded from ~/.thandv/personas/*.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thandv import personas


def _write(pdir: Path, name: str, **extra) -> None:
    data = {"name": name, "description": f"{name} desc",
            "system_prompt": f"You are {name}.", "skills": ["tool-use", "honesty"]}
    data.update(extra)
    (pdir / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def test_disk_persona_is_loaded(isolate_personas_dir: Path):
    _write(isolate_personas_dir, "security-auditor")
    p = personas.get_persona("security-auditor")
    assert p.system_prompt == "You are security-auditor."
    assert p.skills == ("tool-use", "honesty")
    assert "security-auditor" in personas.list_personas()


def test_builtins_win_over_disk(isolate_personas_dir: Path):
    # A disk persona named like a built-in must NOT shadow it.
    _write(isolate_personas_dir, "code", system_prompt="HIJACKED")
    assert personas.get_persona("code") is personas.CODE


def test_builtins_intact_with_no_disk_personas():
    assert set(personas.PERSONAS) == {"code", "writer", "finance", "image"}


def test_malformed_persona_is_skipped(isolate_personas_dir: Path):
    (isolate_personas_dir / "broken.json").write_text("{ not json", encoding="utf-8")
    _write(isolate_personas_dir, "good")
    # broken file ignored; good one still loads; CLI not broken
    assert "good" in personas.list_personas()
    with pytest.raises(ValueError, match="unknown persona"):
        personas.get_persona("broken")


def test_persona_without_prompt_is_skipped(isolate_personas_dir: Path):
    (isolate_personas_dir / "empty.json").write_text(
        json.dumps({"name": "empty", "system_prompt": ""}), encoding="utf-8")
    assert "empty" not in personas.list_personas()
