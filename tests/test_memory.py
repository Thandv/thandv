import json

from thandv import memory
from thandv.memory import (
    BUNDLED_SKILLS_DIR,
    append_event,
    load_memory,
    load_skills,
    new_session_path,
)


def test_load_skills_empty(thandv_home):
    assert load_skills() == ""


def test_load_skills_concatenates_in_order(thandv_home):
    skills_dir = thandv_home / "skills"
    (skills_dir / "b.md").write_text("beta")
    (skills_dir / "a.md").write_text("alpha")
    out = load_skills()
    # Sorted alphabetically => a before b
    assert out.index("alpha") < out.index("beta")
    assert "# Skill: a" in out and "# Skill: b" in out


def test_load_skills_ignores_non_md(thandv_home):
    skills_dir = thandv_home / "skills"
    (skills_dir / "x.txt").write_text("ignore me")
    (skills_dir / "y.md").write_text("keep me")
    out = load_skills()
    assert "keep me" in out
    assert "ignore me" not in out


def test_load_skills_filters_by_only(thandv_home):
    skills_dir = thandv_home / "skills"
    (skills_dir / "alpha.md").write_text("alpha-body")
    (skills_dir / "beta.md").write_text("beta-body")
    (skills_dir / "gamma.md").write_text("gamma-body")
    out = load_skills(only=("alpha", "gamma"))
    assert "alpha-body" in out
    assert "gamma-body" in out
    assert "beta-body" not in out


def test_load_skills_only_empty_loads_nothing(thandv_home):
    (thandv_home / "skills" / "x.md").write_text("anything")
    assert load_skills(only=()) == ""


def test_load_memory(thandv_home):
    mem_dir = thandv_home / "memory"
    (mem_dir / "fact.md").write_text("the sky is blue\n")
    out = load_memory()
    assert "the sky is blue" in out


def test_new_session_path_is_under_sessions_dir(thandv_home):
    p = new_session_path()
    assert p.parent == thandv_home / "sessions"
    assert p.suffix == ".jsonl"


def test_append_event_writes_jsonl(thandv_home):
    p = new_session_path()
    append_event(p, {"role": "user", "content": "hi"})
    append_event(p, {"role": "assistant", "content": "yo"})
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["content"] == "hi"
    assert json.loads(lines[1])["content"] == "yo"


# --- Bundled skill loading -----------------------------------------------

def test_bundled_skills_load_by_default(thandv_home):
    """Skills in BUNDLED_SKILLS_DIR are loaded even when the user has
    nothing under ~/.thandv/skills/. This is the fix that makes
    persona.skills declarations actually work on a fresh install."""
    (memory.BUNDLED_SKILLS_DIR / "honesty.md").write_text("bundled honesty body")
    out = load_skills()
    assert "bundled honesty body" in out
    assert "# Skill: honesty" in out


def test_user_skill_overrides_bundled(thandv_home):
    """If both a bundled and user file share a stem, the user wins."""
    (memory.BUNDLED_SKILLS_DIR / "tool-use.md").write_text("BUNDLED tool-use")
    (thandv_home / "skills" / "tool-use.md").write_text("USER tool-use")
    out = load_skills()
    assert "USER tool-use" in out
    assert "BUNDLED tool-use" not in out


def test_bundled_only_when_user_dir_missing(thandv_home, monkeypatch):
    """If the user skills dir doesn't exist at all, bundled still loads."""
    import shutil
    shutil.rmtree(thandv_home / "skills")
    (memory.BUNDLED_SKILLS_DIR / "honesty.md").write_text("just bundled")
    out = load_skills()
    assert "just bundled" in out


def test_only_filter_applies_across_both_dirs(thandv_home):
    (memory.BUNDLED_SKILLS_DIR / "alpha.md").write_text("bundled-alpha")
    (memory.BUNDLED_SKILLS_DIR / "beta.md").write_text("bundled-beta")
    (thandv_home / "skills" / "gamma.md").write_text("user-gamma")
    out = load_skills(only=("alpha", "gamma"))
    assert "bundled-alpha" in out
    assert "user-gamma" in out
    assert "bundled-beta" not in out


def test_real_bundled_skills_dir_ships_strategy_critique():
    """The bundled `thandv/skills/strategy-critique.md` exists on disk
    in the source tree (this is what the conftest isolates away). It's
    also what `[tool.setuptools.package-data]` ships with the wheel."""
    # _bundled_skills_dir() resolves to thandv/skills in the source tree
    real = memory._bundled_skills_dir()
    assert (real / "strategy-critique.md").exists()
    assert (real / "finance-discipline.md").exists()
    # Just a smoke check: file is non-empty.
    assert (real / "strategy-critique.md").read_text().strip()


# Silence unused-import lint -- BUNDLED_SKILLS_DIR is imported to document
# the module contract even though most tests use memory.BUNDLED_SKILLS_DIR
# after the conftest monkeypatch.
assert BUNDLED_SKILLS_DIR is not None
