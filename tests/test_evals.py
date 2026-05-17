from __future__ import annotations

import json

import pytest

from thandv import evals as eval_mod
from thandv.evals import (
    EvalResult,
    EvalSuite,
    EvalTask,
    SUITES,
    get_suite,
    list_suites,
    run_suite,
    summarise,
)


# --- Suite registry --------------------------------------------------------

def test_list_suites_contains_smoke():
    assert "smoke" in list_suites()


def test_get_suite_unknown_raises():
    with pytest.raises(ValueError, match="unknown suite"):
        get_suite("nonexistent")


def test_smoke_suite_has_tasks():
    suite = get_suite("smoke")
    assert len(suite.tasks) >= 3
    assert suite.default_persona == "code"
    for t in suite.tasks:
        assert t.id and t.prompt and callable(t.verify)


# --- Verifier behaviour ----------------------------------------------------

def test_smoke_arithmetic_verifier():
    task = next(t for t in get_suite("smoke").tasks if t.id == "arithmetic")
    assert task.verify("56") is True
    assert task.verify("The answer is 56") is True
    assert task.verify("42") is False


def test_smoke_is_prime_verifier_accepts_correct():
    task = next(t for t in get_suite("smoke").tasks if t.id == "is-prime")
    reply = (
        "```python\n"
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    for i in range(2, n):\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "    return True\n"
        "```"
    )
    assert task.verify(reply) is True


def test_smoke_is_prime_verifier_rejects_wrong():
    task = next(t for t in get_suite("smoke").tasks if t.id == "is-prime")
    bad = "```python\ndef is_prime(n):\n    return True\n```"
    assert task.verify(bad) is False


def test_smoke_is_prime_verifier_rejects_garbage():
    task = next(t for t in get_suite("smoke").tasks if t.id == "is-prime")
    assert task.verify("here is some prose without code") is False


# --- Runner with mocked agent ---------------------------------------------

class _FakeAgent:
    """Stands in for thandv.agent.Agent. Returns canned replies per task."""

    _replies: dict[str, str] = {}

    def __init__(self, config, persona=None):
        self.config = config
        self.persona = persona
        self.session_path = type("P", (), {"name": "fake-session"})()

    def turn(self, prompt: str):
        for needle, reply in self._replies.items():
            if needle in prompt:
                yield reply
                return
        yield "<no canned reply>"


def _install_fake_agent(monkeypatch, replies: dict[str, str]) -> None:
    _FakeAgent._replies = replies
    monkeypatch.setattr(eval_mod, "Agent", _FakeAgent)


def test_run_suite_all_pass(thandv_home, monkeypatch):
    _install_fake_agent(monkeypatch, {
        "7 * 8": "56",
        "Reverse": "olleh",
        "is_prime": (
            "```python\n"
            "def is_prime(n):\n"
            "    if n < 2: return False\n"
            "    for i in range(2, n):\n"
            "        if n % i == 0: return False\n"
            "    return True\n"
            "```"
        ),
    })
    results = run_suite("smoke", model="fake")
    assert len(results) == 3
    assert all(r.passed for r in results)
    assert summarise(results) == "passed 3/3 (100.0%)"


def test_run_suite_mixed_results(thandv_home, monkeypatch):
    _install_fake_agent(monkeypatch, {
        "7 * 8": "I don't know",
        "Reverse": "olleh",
        "is_prime": "```python\ndef is_prime(n): return True\n```",
    })
    results = run_suite("smoke", model="fake")
    by_id = {r.task_id: r for r in results}
    assert by_id["arithmetic"].passed is False
    assert by_id["reverse-string"].passed is True
    assert by_id["is-prime"].passed is False


def test_run_suite_persists_json(thandv_home, monkeypatch):
    _install_fake_agent(monkeypatch, {"7 * 8": "56", "Reverse": "olleh"})
    out = thandv_home / "smoke-out.json"
    run_suite("smoke", model="fake", limit=2, out_path=out)
    data = json.loads(out.read_text())
    assert len(data) == 2
    assert {d["task_id"] for d in data} == {"arithmetic", "reverse-string"}
    assert all("reply" in d and "secs" in d for d in data)


def test_run_suite_limit_respected(thandv_home, monkeypatch):
    _install_fake_agent(monkeypatch, {"7 * 8": "56"})
    results = run_suite("smoke", model="fake", limit=1)
    assert len(results) == 1
    assert results[0].task_id == "arithmetic"


def test_run_suite_persona_override(thandv_home, monkeypatch):
    _install_fake_agent(monkeypatch, {"7 * 8": "56"})
    results = run_suite("smoke", model="fake", persona_name="writer", limit=1)
    assert results[0].persona == "writer"


def test_run_suite_progress_callback(thandv_home, monkeypatch):
    _install_fake_agent(monkeypatch, {"7 * 8": "56"})
    progress: list[tuple[int, int, str]] = []

    def cb(i, total, result):
        progress.append((i, total, result.task_id))

    run_suite("smoke", model="fake", limit=1, on_progress=cb)
    assert progress == [(1, 1, "arithmetic")]


def test_run_suite_handles_agent_exception(thandv_home, monkeypatch):
    class BoomAgent(_FakeAgent):
        def turn(self, prompt):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(eval_mod, "Agent", BoomAgent)
    results = run_suite("smoke", model="fake", limit=1)
    assert results[0].passed is False
    assert "kaboom" in results[0].reply


# --- Suite definition object ----------------------------------------------

def test_eval_dataclasses():
    task = EvalTask(id="x", prompt="p", verify=lambda r: True)
    suite = EvalSuite(name="t", tasks=[task])
    assert suite.default_persona == "code"
    result = EvalResult(
        task_id="x",
        suite="t",
        model="m",
        persona="code",
        reply="r",
        passed=True,
        secs=0.1,
    )
    assert result.passed
    # Smoke suite is registered with the expected name
    assert SUITES["smoke"].name == "smoke"


# --- HumanEval --------------------------------------------------------------

def test_humaneval_suite_registered():
    suite = get_suite("humaneval")
    assert suite.name == "humaneval"
    assert suite.default_persona == "code"
    # Tasks load lazily via `loader`; not populated until first run.
    assert suite.loader is not None


def test_extract_python_code_from_fenced_block():
    reply = "Sure, here it is:\n```python\ndef f():\n    return 1\n```\nthanks"
    assert eval_mod._extract_python_code(reply) == "def f():\n    return 1"


def test_extract_python_code_from_unfenced_reply():
    reply = "def f():\n    return 1"
    assert eval_mod._extract_python_code(reply) == reply


def test_extract_python_code_accepts_generic_fence():
    reply = "```\ndef f():\n    return 1\n```"
    assert eval_mod._extract_python_code(reply) == "def f():\n    return 1"


def test_humaneval_verifier_passes_canonical_solution():
    test_code = (
        "def check(candidate):\n"
        "    assert candidate(2) == 4\n"
        "    assert candidate(3) == 6\n"
    )
    verify = eval_mod._humaneval_verifier(test_code, "f")
    reply = "```python\ndef f(n):\n    return n * 2\n```"
    assert verify(reply) is True


def test_humaneval_verifier_rejects_wrong_solution():
    test_code = (
        "def check(candidate):\n"
        "    assert candidate(2) == 4\n"
    )
    verify = eval_mod._humaneval_verifier(test_code, "f")
    reply = "```python\ndef f(n):\n    return n + 1\n```"
    assert verify(reply) is False


def test_humaneval_verifier_rejects_syntax_error():
    test_code = "def check(candidate):\n    pass\n"
    verify = eval_mod._humaneval_verifier(test_code, "f")
    reply = "```python\ndef f(:\n    return\n```"
    assert verify(reply) is False


def test_humaneval_verifier_rejects_missing_function():
    test_code = "def check(candidate):\n    candidate(1)\n"
    verify = eval_mod._humaneval_verifier(test_code, "f")
    reply = "```python\ndef g(n):\n    return n\n```"  # wrong name
    assert verify(reply) is False


def test_humaneval_verifier_handles_timeout(monkeypatch):
    monkeypatch.setattr(eval_mod, "HUMANEVAL_TIMEOUT_S", 1)
    test_code = "def check(candidate):\n    candidate()\n"
    verify = eval_mod._humaneval_verifier(test_code, "f")
    reply = "```python\ndef f():\n    while True:\n        pass\n```"
    assert verify(reply) is False


def test_humaneval_loader_clear_error_without_datasets(monkeypatch):
    """If `datasets` isn't installed, the loader raises a clear message."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match=r"thandv\[eval\]"):
        eval_mod._load_humaneval_tasks()


# --- MBPP -----------------------------------------------------------------

def test_mbpp_suite_registered():
    suite = get_suite("mbpp")
    assert suite.name == "mbpp"
    assert suite.default_persona == "code"
    assert suite.loader is not None


def test_mbpp_verifier_passes_canonical():
    test_list = [
        "assert is_even(2) == True",
        "assert is_even(3) == False",
    ]
    verify = eval_mod._mbpp_verifier(test_list)
    reply = "```python\ndef is_even(n):\n    return n % 2 == 0\n```"
    assert verify(reply) is True


def test_mbpp_verifier_rejects_wrong_answer():
    test_list = ["assert add(2, 3) == 5"]
    verify = eval_mod._mbpp_verifier(test_list)
    reply = "```python\ndef add(a, b):\n    return a - b\n```"
    assert verify(reply) is False


def test_mbpp_verifier_rejects_syntax_error():
    test_list = ["assert f(1) == 1"]
    verify = eval_mod._mbpp_verifier(test_list)
    reply = "```python\ndef f(:\n    return 1\n```"
    assert verify(reply) is False


def test_mbpp_verifier_rejects_missing_function():
    test_list = ["assert thing(1) == 1"]
    verify = eval_mod._mbpp_verifier(test_list)
    reply = "```python\ndef other(n):\n    return n\n```"
    assert verify(reply) is False


def test_mbpp_verifier_runs_all_assertions():
    """A solution that satisfies the first assertion but not the second
    must still fail."""
    test_list = [
        "assert is_positive(1) == True",
        "assert is_positive(-1) == False",
    ]
    verify = eval_mod._mbpp_verifier(test_list)
    reply = "```python\ndef is_positive(n):\n    return True\n```"  # wrong on n=-1
    assert verify(reply) is False


def test_mbpp_verifier_handles_timeout(monkeypatch):
    monkeypatch.setattr(eval_mod, "MBPP_TIMEOUT_S", 1)
    test_list = ["loop()"]
    verify = eval_mod._mbpp_verifier(test_list)
    reply = "```python\ndef loop():\n    while True:\n        pass\n```"
    assert verify(reply) is False


def test_mbpp_loader_clear_error_without_datasets(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match=r"thandv\[eval\]"):
        eval_mod._load_mbpp_tasks()


def test_run_suite_invokes_lazy_loader(thandv_home, monkeypatch):
    """A suite with a loader should populate tasks on first run."""
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return [
            EvalTask(id="lazy", prompt="hi", verify=lambda r: r == "ok"),
        ]

    lazy = EvalSuite(name="lazyfake", loader=loader)
    monkeypatch.setitem(eval_mod.SUITES, "lazyfake", lazy)

    class FA:
        def __init__(self, *a, **kw):
            self.session_path = type("P", (), {"name": "fake"})()

        def turn(self, _):
            yield "ok"

    monkeypatch.setattr(eval_mod, "Agent", FA)

    results = eval_mod.run_suite("lazyfake", model="m")
    assert calls["n"] == 1
    assert len(results) == 1
    assert results[0].passed
