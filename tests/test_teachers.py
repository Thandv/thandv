from __future__ import annotations

import json

import pytest

from thandv import teachers as tch


# --- Registry + refusal ---------------------------------------------------

def test_three_teachers_registered():
    names = {t.name for t in tch.list_teachers()}
    assert {
        "groq-llama-3.3-70b",
        "together-llama-3.3-70b",
        "openrouter-llama-3.3-70b",
    } == names


def test_every_teacher_has_required_fields():
    for t in tch.list_teachers():
        assert t.name
        assert t.provider in {"groq", "together", "openrouter"}
        assert t.base_url.startswith("https://")
        assert t.api_key_env
        assert t.model_id
        assert t.persona_hint
        assert t.tos_url.startswith("https://")
        assert t.free_tier_note  # honest about rate limits


def test_get_teacher_unknown_raises():
    with pytest.raises(ValueError, match="unknown teacher"):
        tch.get_teacher("totally-fake-teacher")


def test_get_teacher_refuses_anthropic():
    with pytest.raises(ValueError, match=r"refused|anthropic|Claude") as ei:
        tch.get_teacher("anthropic-claude-3-opus")
    msg = str(ei.value).lower()
    assert "anthropic" in msg
    # Should include the ToS reference so future debuggers see why.
    assert "anthropic.com/legal" in str(ei.value)


def test_get_teacher_refuses_openai():
    with pytest.raises(ValueError, match=r"refused"):
        tch.get_teacher("openai-gpt-4")
    with pytest.raises(ValueError):
        tch.get_teacher("gpt-3.5-turbo")  # via "gpt" alias


def test_get_teacher_refuses_claude_alias():
    with pytest.raises(ValueError, match=r"refused"):
        tch.get_teacher("claude-3-haiku")


def test_get_teacher_refusal_is_case_insensitive():
    with pytest.raises(ValueError, match=r"refused"):
        tch.get_teacher("Anthropic-Claude")
    with pytest.raises(ValueError, match=r"refused"):
        tch.get_teacher("OPENAI-anything")


def test_refused_providers_list_has_tos_urls():
    """Every refusal reason embeds its source ToS URL so the ban is
    auditable rather than vibes-based."""
    for provider, reason in tch.REFUSED_PROVIDERS.items():
        if "alias" in reason:
            continue
        assert "https://" in reason, f"{provider} refusal missing ToS URL"


# --- HTTP call ------------------------------------------------------------

class _FakeResponse:
    def __init__(self, content: str, status: int = 200):
        self._content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def test_call_teacher_missing_key_raises(monkeypatch):
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.delenv(teacher.api_key_env, raising=False)
    with pytest.raises(RuntimeError, match=r"missing API key"):
        tch.call_teacher(teacher, "hi")


def test_call_teacher_posts_to_correct_url_with_bearer(monkeypatch):
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "test-token-123")

    seen: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        seen["body"] = json
        return _FakeResponse("HELLO BACK")

    monkeypatch.setattr(tch.requests, "post", fake_post)
    out = tch.call_teacher(teacher, "Hi", system="be terse")
    assert out == "HELLO BACK"
    assert seen["url"].endswith("/chat/completions")
    assert seen["headers"]["Authorization"] == "Bearer test-token-123"
    assert seen["body"]["model"] == teacher.model_id
    # The system message should come before the user one.
    assert seen["body"]["messages"][0] == {"role": "system", "content": "be terse"}
    assert seen["body"]["messages"][1] == {"role": "user", "content": "Hi"}


def test_call_teacher_omits_system_when_none(monkeypatch):
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "x")
    seen: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["body"] = json
        return _FakeResponse("ok")

    monkeypatch.setattr(tch.requests, "post", fake_post)
    tch.call_teacher(teacher, "Hi")
    assert seen["body"]["messages"] == [{"role": "user", "content": "Hi"}]


# --- Distill orchestration -----------------------------------------------

def test_distill_writes_queue_jsonl_and_provenance(thandv_home, monkeypatch):
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "x")

    completions = iter(["AA", "BB", "CC"])

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(next(completions))

    monkeypatch.setattr(tch.requests, "post", fake_post)

    prompts = ["one", "two", "three"]
    queue_path, prov_path = tch.distill(teacher.name, prompts)

    assert queue_path.exists()
    assert prov_path.exists()

    lines = queue_path.read_text().splitlines()
    assert len(lines) == 3
    pairs = [json.loads(line) for line in lines]
    assert [p["prompt"] for p in pairs] == prompts
    assert [p["completion"] for p in pairs] == ["AA", "BB", "CC"]

    prov = json.loads(prov_path.read_text())
    assert prov["teacher"]["name"] == teacher.name
    assert prov["teacher"]["tos_url"] == teacher.tos_url
    assert prov["n_prompts"] == 3
    assert prov["n_successes"] == 3
    assert prov["n_failures"] == 0
    assert "started_at" in prov and "finished_at" in prov


def test_distill_records_failures_keeps_successes(thandv_home, monkeypatch):
    """If the teacher errors mid-stream, finish the rest and record the
    failure rather than aborting — partial output is more useful than no
    output."""
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "x")

    counter = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        counter["n"] += 1
        if counter["n"] == 2:
            return _FakeResponse("", status=500)
        return _FakeResponse(f"reply{counter['n']}")

    monkeypatch.setattr(tch.requests, "post", fake_post)
    queue_path, prov_path = tch.distill(teacher.name, ["a", "b", "c"])

    # Two successes (prompts 1 + 3); one recorded failure (prompt 2).
    lines = queue_path.read_text().splitlines()
    assert len(lines) == 2

    prov = json.loads(prov_path.read_text())
    assert prov["n_successes"] == 2
    assert prov["n_failures"] == 1
    assert prov["failures"][0]["index"] == 2


def test_distill_refuses_anthropic_before_calling_anything(thandv_home, monkeypatch):
    """A refused teacher must surface the ToS reason without ever
    touching the network."""
    def boom(*a, **kw):
        raise AssertionError("requests.post must NOT be called for refused teachers")

    monkeypatch.setattr(tch.requests, "post", boom)
    with pytest.raises(ValueError, match=r"refused"):
        tch.distill("anthropic-claude-3", ["hi"])


def test_distill_calls_progress_callback(thandv_home, monkeypatch):
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "x")
    monkeypatch.setattr(
        tch.requests, "post",
        lambda *a, **kw: _FakeResponse("ok"),
    )

    progress: list[tuple[int, int, str]] = []
    tch.distill(
        teacher.name, ["a", "b"],
        on_progress=lambda i, n, status: progress.append((i, n, status)),
    )
    assert progress == [(1, 2, "ok"), (2, 2, "ok")]


def test_distill_progress_callback_sees_failures(thandv_home, monkeypatch):
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "x")

    def fake_post(*a, **kw):
        return _FakeResponse("", status=500)

    monkeypatch.setattr(tch.requests, "post", fake_post)

    progress: list[tuple[int, int, str]] = []
    tch.distill(
        teacher.name, ["only-prompt"],
        on_progress=lambda i, n, status: progress.append((i, n, status)),
    )
    assert progress == [(1, 1, "fail: RuntimeError")]
