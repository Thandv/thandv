import json

from thandv.memory import append_event, load_memory, load_skills, new_session_path


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
