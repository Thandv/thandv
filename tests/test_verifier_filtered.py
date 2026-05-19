from __future__ import annotations

import json

import pytest

from thandv import teachers as tch
from thandv import verifier_filtered as vf


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def _post_returning(completions: list[str]):
    """Build a fake `requests.post` that returns the given completions in order."""
    it = iter(completions)

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(next(it))

    return fake_post


# --- Basic shape ---------------------------------------------------------

def test_distill_and_filter_keeps_only_passing_completions(thandv_home, monkeypatch):
    """The smoke arithmetic task accepts replies containing '56'.
    Mix 3 completions: one passes, two fail. Only the passing one is queued.
    """
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "x")
    monkeypatch.setattr(
        tch.requests, "post",
        _post_returning(["nope", "56", "still nope"]),
    )

    queue_path, stats = vf.distill_and_filter(
        teacher.name, "smoke",
        n_per_task=3,
        limit=1,  # only the first smoke task
    )
    lines = queue_path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["completion"] == "56"
    assert entry["task_id"] == "arithmetic"

    assert stats["n_attempted"] == 3
    assert stats["n_passed"] == 1
    assert stats["yield"] == pytest.approx(1 / 3)


def test_distill_and_filter_records_zero_pass_task(thandv_home, monkeypatch):
    """If a task gets 0 passing completions, it's recorded in per_task."""
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "x")
    monkeypatch.setattr(
        tch.requests, "post",
        _post_returning(["wrong", "wrong", "wrong"]),
    )

    _, stats = vf.distill_and_filter(teacher.name, "smoke", n_per_task=3, limit=1)
    assert stats["n_passed"] == 0
    assert stats["per_task"][0]["attempted"] == 3
    assert stats["per_task"][0]["passed"] == 0


def test_distill_and_filter_runs_all_tasks_when_no_limit(thandv_home, monkeypatch):
    """Without --limit, all smoke tasks (3) get visited."""
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "x")
    # 3 tasks * 1 completion each = 3 calls, all wrong.
    monkeypatch.setattr(
        tch.requests, "post", _post_returning(["x", "x", "x"]),
    )

    _, stats = vf.distill_and_filter(teacher.name, "smoke", n_per_task=1)
    assert stats["n_tasks"] == 3
    assert stats["n_attempted"] == 3
    assert len(stats["per_task"]) == 3


def test_distill_and_filter_provenance_sidecar_written(thandv_home, monkeypatch):
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "x")
    monkeypatch.setattr(tch.requests, "post", _post_returning(["56"]))

    queue_path, _ = vf.distill_and_filter(teacher.name, "smoke", n_per_task=1, limit=1)
    prov_path = queue_path.with_suffix(queue_path.suffix + ".provenance.json")
    assert prov_path.exists()
    prov = json.loads(prov_path.read_text())
    assert prov["suite"] == "smoke"
    assert prov["teacher"]["name"] == teacher.name
    assert prov["teacher"]["tos_url"] == teacher.tos_url
    assert "started_at" in prov and "finished_at" in prov


# --- Failure modes -------------------------------------------------------

def test_distill_and_filter_refuses_anthropic(thandv_home, monkeypatch):
    """Refused teacher errors before any HTTP call."""
    def boom(*a, **kw):
        raise AssertionError("HTTP must not be called for refused teachers")

    monkeypatch.setattr(tch.requests, "post", boom)
    with pytest.raises(ValueError, match="refused"):
        vf.distill_and_filter("anthropic-claude-3", "smoke")


def test_distill_and_filter_unknown_suite(thandv_home, monkeypatch):
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "x")
    with pytest.raises(ValueError, match="unknown suite"):
        vf.distill_and_filter(teacher.name, "no-such-suite")


def test_distill_and_filter_teacher_call_failure_continues(thandv_home, monkeypatch):
    """Teacher errors on one prompt → recorded in teacher_failures, others continue."""
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "x")

    counter = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        counter["n"] += 1
        if counter["n"] == 2:
            raise RuntimeError("rate limited")
        return _FakeResponse("56")

    monkeypatch.setattr(tch.requests, "post", fake_post)
    queue_path, stats = vf.distill_and_filter(
        teacher.name, "smoke", n_per_task=3, limit=1,
    )
    # 1st and 3rd calls succeed and pass; 2nd raises.
    lines = queue_path.read_text().splitlines()
    assert len(lines) == 2
    assert stats["n_passed"] == 2
    assert len(stats["teacher_failures"]) == 1
    assert "rate limited" in stats["teacher_failures"][0]["error"]


def test_distill_and_filter_verifier_exception_treated_as_reject(thandv_home, monkeypatch):
    """If a task's verifier itself raises, treat it as 'reject' rather than
    abort the whole run. Honest about what we know: we couldn't verify it."""
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "x")

    from thandv import evals as _ev
    # Inject a suite whose verifier always raises.
    bad_suite = _ev.EvalSuite(
        name="badverify",
        tasks=[_ev.EvalTask(
            id="explodes",
            prompt="anything",
            verify=lambda r: (_ for _ in ()).throw(RuntimeError("verifier broken")),
        )],
    )
    monkeypatch.setitem(_ev.SUITES, "badverify", bad_suite)
    monkeypatch.setattr(tch.requests, "post", _post_returning(["ok"]))

    queue_path, stats = vf.distill_and_filter(
        teacher.name, "badverify", n_per_task=1,
    )
    # Verifier raised → not added to queue, but the run completed.
    assert queue_path.read_text() == ""
    assert stats["n_passed"] == 0
    assert stats["n_attempted"] == 1


# --- Progress callback ---------------------------------------------------

def test_progress_callback_distinguishes_pass_reject_teacher_fail(
    thandv_home, monkeypatch,
):
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, "x")

    calls = {"n": 0}

    def fake_post(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse("56")           # pass
        if calls["n"] == 2:
            raise RuntimeError("network blip")    # teacher-fail
        return _FakeResponse("wrong")             # reject

    monkeypatch.setattr(tch.requests, "post", fake_post)

    events: list[str] = []
    vf.distill_and_filter(
        teacher.name, "smoke", n_per_task=3, limit=1,
        on_progress=lambda i, n, status: events.append(status.split(":")[0]),
    )
    assert events == ["pass", "teacher-fail", "reject"]
