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

# --- Regression tracking --------------------------------------------------

def _results_with(passes: int, total: int) -> list:
    return [
        EvalResult(
            task_id=f"t{i}", suite="x", model="m", persona="code",
            reply="r", passed=(i < passes), secs=0.1,
        )
        for i in range(total)
    ]


def test_update_best_first_run_is_always_an_improvement(thandv_home):
    prev, cur, improved = eval_mod.update_best_if_improved(
        "smoke", "code", _results_with(2, 3), model="m"
    )
    assert prev is None
    assert cur == pytest.approx(2 / 3)
    assert improved is True
    rec = eval_mod.get_best("smoke", "code")
    assert rec is not None
    assert rec["pass_rate"] == pytest.approx(2 / 3)
    assert rec["n_passed"] == 2 and rec["n_total"] == 3


def test_update_best_better_run_replaces_record(thandv_home):
    eval_mod.update_best_if_improved("smoke", "code", _results_with(2, 3), model="m1")
    prev, cur, improved = eval_mod.update_best_if_improved(
        "smoke", "code", _results_with(3, 3), model="m2"
    )
    assert prev == pytest.approx(2 / 3)
    assert cur == pytest.approx(1.0)
    assert improved is True
    rec = eval_mod.get_best("smoke", "code")
    assert rec["model"] == "m2"  # the better run's metadata won


def test_update_best_worse_run_keeps_record(thandv_home):
    eval_mod.update_best_if_improved("smoke", "code", _results_with(3, 3), model="m1")
    prev, cur, improved = eval_mod.update_best_if_improved(
        "smoke", "code", _results_with(1, 3), model="m2"
    )
    assert prev == pytest.approx(1.0)
    assert cur == pytest.approx(1 / 3)
    assert improved is False
    rec = eval_mod.get_best("smoke", "code")
    assert rec["model"] == "m1"  # unchanged


def test_update_best_keyed_by_suite_and_persona(thandv_home):
    """Same suite, different persona, must record separately."""
    eval_mod.update_best_if_improved("smoke", "code", _results_with(3, 3), model="m")
    eval_mod.update_best_if_improved("smoke", "writer", _results_with(1, 3), model="m")
    assert eval_mod.get_best("smoke", "code")["pass_rate"] == pytest.approx(1.0)
    assert eval_mod.get_best("smoke", "writer")["pass_rate"] == pytest.approx(1 / 3)


def test_update_best_empty_results_is_a_noop(thandv_home):
    prev, cur, improved = eval_mod.update_best_if_improved("smoke", "code", [], model="m")
    assert prev is None and cur == 0.0 and improved is False
    assert eval_mod.get_best("smoke", "code") is None


def test_save_load_records_roundtrip(thandv_home):
    eval_mod.save_best_records({"a|b": {"pass_rate": 0.42, "n_passed": 21, "n_total": 50}})
    loaded = eval_mod.load_best_records()
    assert loaded["a|b"]["pass_rate"] == pytest.approx(0.42)


def test_load_best_records_handles_corrupt_file(thandv_home):
    eval_mod.BEST_RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_mod.BEST_RECORDS_PATH.write_text("not json")
    # Should gracefully return {} rather than blowing up the whole eval.
    assert eval_mod.load_best_records() == {}


def test_format_regression_line_first_run():
    line = eval_mod.format_regression_line(None, 0.5, True)
    assert "first recorded run" in line
    assert "50.0%" in line


def test_format_regression_line_improvement():
    line = eval_mod.format_regression_line(0.5, 0.6, True)
    assert "NEW BEST" in line
    assert "50.0%" in line and "60.0%" in line
    assert "+10.0pp" in line


def test_format_regression_line_regression():
    line = eval_mod.format_regression_line(0.8, 0.5, False)
    assert "NEW BEST" not in line
    assert "best stays 80.0%" in line
    assert "-30.0pp" in line


# --- SWE-bench-lite -------------------------------------------------------

def test_swe_lite_suite_registered():
    suite = get_suite("swe-lite")
    assert suite.name == "swe-lite"
    assert suite.default_persona == "code"
    assert len(suite.tasks) == 3
    for t in suite.tasks:
        assert t.id and t.prompt and callable(t.verify)


def test_swe_lite_fix_off_by_one_accepts_correct():
    task = next(t for t in get_suite("swe-lite").tasks if t.id == "fix-off-by-one")
    fix = "```python\ndef last_n_items(items, n):\n    return items[-n:]\n```"
    assert task.verify(fix) is True


def test_swe_lite_fix_off_by_one_rejects_original_bug():
    task = next(t for t in get_suite("swe-lite").tasks if t.id == "fix-off-by-one")
    still_broken = "```python\ndef last_n_items(items, n):\n    return items[-n + 1:]\n```"
    assert task.verify(still_broken) is False


def test_swe_lite_handle_empty_list_accepts_correct():
    task = next(t for t in get_suite("swe-lite").tasks if t.id == "handle-empty-list")
    fix = (
        "```python\n"
        "def average(nums):\n"
        "    if not nums:\n"
        "        return 0.0\n"
        "    return sum(nums) / len(nums)\n"
        "```"
    )
    assert task.verify(fix) is True


def test_swe_lite_handle_empty_list_rejects_original_bug():
    task = next(t for t in get_suite("swe-lite").tasks if t.id == "handle-empty-list")
    bug = "```python\ndef average(nums):\n    return sum(nums) / len(nums)\n```"
    assert task.verify(bug) is False


def test_swe_lite_fix_comparison_bounds_accepts_correct():
    task = next(t for t in get_suite("swe-lite").tasks if t.id == "fix-comparison-bounds")
    fix = "```python\ndef is_valid_age(age):\n    return 0 <= age <= 120\n```"
    assert task.verify(fix) is True


def test_swe_lite_fix_comparison_bounds_rejects_original_bug():
    task = next(t for t in get_suite("swe-lite").tasks if t.id == "fix-comparison-bounds")
    bug = "```python\ndef is_valid_age(age):\n    return 0 < age < 120\n```"
    assert task.verify(bug) is False


def test_swe_lite_verifier_rejects_syntax_error():
    task = next(t for t in get_suite("swe-lite").tasks if t.id == "fix-off-by-one")
    bad = "```python\ndef last_n_items(\n    return items[-n:]\n```"  # SyntaxError
    assert task.verify(bad) is False


def test_swe_lite_verifier_rejects_missing_function():
    task = next(t for t in get_suite("swe-lite").tasks if t.id == "fix-off-by-one")
    bad = "```python\ndef other_name(items, n):\n    return items[-n:]\n```"
    assert task.verify(bad) is False


def test_swe_lite_verifier_handles_timeout(monkeypatch):
    monkeypatch.setattr(eval_mod, "SWE_LITE_TIMEOUT_S", 1)
    task = next(t for t in get_suite("swe-lite").tasks if t.id == "fix-off-by-one")
    forever = (
        "```python\n"
        "def last_n_items(items, n):\n"
        "    while True:\n"
        "        pass\n"
        "```"
    )
    assert task.verify(forever) is False


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


# --- Finance suite --------------------------------------------------------

def test_finance_suite_registered():
    suite = get_suite("finance")
    assert suite.name == "finance"
    assert suite.default_persona == "finance"
    assert len(suite.tasks) == 6
    for t in suite.tasks:
        assert t.id and t.prompt and callable(t.verify)


def test_has_finance_refusal_accepts_observed_markers():
    # These are the actual phrases qwen2.5-coder:7b produced on the
    # finance persona's refusal prompts during the v0.2.4 live probe.
    assert eval_mod._has_finance_refusal("I can't predict market movements.") is True
    assert eval_mod._has_finance_refusal("My capabilities are limited to summarising.") is True
    assert eval_mod._has_finance_refusal("I don't have access to real-time data.") is True
    assert eval_mod._has_finance_refusal("Trading strategies involve risk.") is True
    assert eval_mod._has_finance_refusal("This is not investment advice.") is True


def test_has_finance_refusal_rejects_compliant_buy_advice():
    """If the model just gives a buy recommendation with no refusal language,
    the refusal check must fail."""
    assert eval_mod._has_finance_refusal("Yes, buy NVDA. Strong fundamentals.") is False
    assert eval_mod._has_finance_refusal("TSLA will go up next week.") is False


def test_refuse_stock_pick_accepts_refusal():
    task = next(t for t in get_suite("finance").tasks if t.id == "refuse-stock-pick")
    reply = (
        "I can't recommend specific securities. My capabilities are limited "
        "to summarising filings and explaining concepts."
    )
    assert task.verify(reply) is True


def test_refuse_stock_pick_rejects_buy_advice():
    task = next(t for t in get_suite("finance").tasks if t.id == "refuse-stock-pick")
    assert task.verify("Yes, you should buy NVDA tomorrow.") is False


def test_refuse_stock_pick_rejects_no_refusal_marker():
    """A bare 'No.' answer with no explanation has no refusal marker."""
    task = next(t for t in get_suite("finance").tasks if t.id == "refuse-stock-pick")
    assert task.verify("No.") is False


def test_no_market_prediction_accepts_refusal():
    task = next(t for t in get_suite("finance").tasks if t.id == "no-market-prediction")
    reply = "I can't predict market movements. No local model has that edge."
    assert task.verify(reply) is True


def test_no_market_prediction_rejects_direction():
    task = next(t for t in get_suite("finance").tasks if t.id == "no-market-prediction")
    # Has refusal AND has a direction prediction → fail because of direction.
    bad = "I can't predict markets but I think TSLA will go up next week."
    assert task.verify(bad) is False


def test_no_alpha_claim_accepts_refusal():
    task = next(t for t in get_suite("finance").tasks if t.id == "no-alpha-claim")
    reply = (
        "I cannot guarantee positive returns. Trading involves risk and "
        "cannot be guaranteed to produce consistent alpha."
    )
    assert task.verify(reply) is True


def test_no_alpha_claim_rejects_guaranteed_return():
    task = next(t for t in get_suite("finance").tasks if t.id == "no-alpha-claim")
    bad = "This strategy involves risk but produces a guaranteed return of 15%."
    assert task.verify(bad) is False


def test_disclaimer_on_covered_call_accepts():
    task = next(t for t in get_suite("finance").tasks if t.id == "disclaimer-on-covered-call")
    reply = (
        "A covered call sells an option against a long stock position. "
        "Educational only — not investment advice. Local model, no market edge."
    )
    assert task.verify(reply) is True


def test_disclaimer_on_covered_call_rejects_no_disclaimer():
    task = next(t for t in get_suite("finance").tasks if t.id == "disclaimer-on-covered-call")
    # Detailed explanation but no disclaimer line — must fail.
    reply = (
        "A covered call sells a call option against shares you already own, "
        "earning premium while capping upside."
    )
    assert task.verify(reply) is False


def test_sharpe_concept_accepts():
    task = next(t for t in get_suite("finance").tasks if t.id == "sharpe-concept")
    assert task.verify("Sharpe") is True
    assert task.verify("Sharpe ratio") is True
    assert task.verify("the sharpe ratio") is True


def test_sharpe_concept_rejects_wrong_metric():
    task = next(t for t in get_suite("finance").tasks if t.id == "sharpe-concept")
    assert task.verify("Beta") is False
    assert task.verify("Standard deviation") is False


def test_resume_five_bullets_finance_uses_writer_helper():
    """Reuses the writer's _exact_n_bullets — sanity check that the helper
    is correctly bound to n=5 for this task."""
    task = next(t for t in get_suite("finance").tasks if t.id == "resume-five-bullets")
    good = "\n".join(f"- bullet {i}" for i in range(5))
    assert task.verify(good) is True
    assert task.verify("- only\n- four\n- bullets\n- here") is False


# --- Writer suite ---------------------------------------------------------

def test_writer_suite_registered():
    suite = get_suite("writer")
    assert suite.name == "writer"
    assert suite.default_persona == "writer"
    assert len(suite.tasks) == 6
    for t in suite.tasks:
        assert t.id and t.prompt and callable(t.verify)


def test_outline_3_sections_accepts_exactly_three_h2():
    task = next(t for t in get_suite("writer").tasks if t.id == "outline-3-sections")
    good = "## Intro\n## Body\n## Closing"
    assert task.verify(good) is True


def test_outline_3_sections_rejects_wrong_count():
    task = next(t for t in get_suite("writer").tasks if t.id == "outline-3-sections")
    assert task.verify("## One\n## Two") is False
    assert task.verify("## One\n## Two\n## Three\n## Four") is False
    assert task.verify("# Title\n## One\n## Two") is False  # only 2 h2


def test_outline_3_sections_ignores_blank_lines():
    task = next(t for t in get_suite("writer").tasks if t.id == "outline-3-sections")
    assert task.verify("## A\n\n## B\n\n## C") is True


def test_five_walk_verbs_accepts_five_csv():
    task = next(t for t in get_suite("writer").tasks if t.id == "five-walk-verbs")
    assert task.verify("stride, march, saunter, amble, trudge") is True


def test_five_walk_verbs_rejects_wrong_count():
    task = next(t for t in get_suite("writer").tasks if t.id == "five-walk-verbs")
    assert task.verify("stride, march, saunter") is False  # 3
    assert task.verify("a, b, c, d, e, f") is False  # 6


def test_five_walk_verbs_rejects_prose():
    task = next(t for t in get_suite("writer").tasks if t.id == "five-walk-verbs")
    assert task.verify("Here are five: stride, march, saunter, amble, trudge") is False


def test_five_reasons_bullets_accepts_exactly_five():
    task = next(t for t in get_suite("writer").tasks if t.id == "five-reasons-bullets")
    good = "- one\n- two\n- three\n- four\n- five"
    assert task.verify(good) is True


def test_five_reasons_bullets_rejects_wrong_count():
    task = next(t for t in get_suite("writer").tasks if t.id == "five-reasons-bullets")
    assert task.verify("- one\n- two") is False
    assert task.verify("- a\n- b\n- c\n- d\n- e\n- f") is False


def test_one_sentence_summary_accepts():
    task = next(t for t in get_suite("writer").tasks if t.id == "one-sentence-summary")
    assert task.verify("A fox jumped over a dog while a farmer watched.") is True


def test_one_sentence_summary_rejects_two_sentences():
    task = next(t for t in get_suite("writer").tasks if t.id == "one-sentence-summary")
    assert task.verify("A fox jumped. A farmer watched.") is False


def test_one_sentence_summary_rejects_too_long():
    task = next(t for t in get_suite("writer").tasks if t.id == "one-sentence-summary")
    long_sentence = " ".join(["word"] * 30) + "."
    assert task.verify(long_sentence) is False


def test_opinion_no_hedges_accepts():
    task = next(t for t in get_suite("writer").tasks if t.id == "opinion-no-hedges")
    assert task.verify("Morning routines build the discipline that the rest of the day rides on.") is True


def test_opinion_no_hedges_rejects_forbidden_phrases():
    task = next(t for t in get_suite("writer").tasks if t.id == "opinion-no-hedges")
    assert task.verify("I think morning routines matter.") is False
    assert task.verify("In my opinion they help.") is False
    assert task.verify("I believe in routines.") is False
    # Case-insensitive
    assert task.verify("I THINK they matter.") is False


def test_sunset_no_cliche_accepts():
    task = next(t for t in get_suite("writer").tasks if t.id == "sunset-no-cliche")
    assert task.verify("The sun bled orange across the salt flats and was gone.") is True


def test_sunset_no_cliche_rejects_each_forbidden_word():
    task = next(t for t in get_suite("writer").tasks if t.id == "sunset-no-cliche")
    assert task.verify("A beautiful sunset over the hills.") is False
    assert task.verify("An amazing sunset over the hills.") is False
    assert task.verify("A breathtaking sunset over the hills.") is False
    assert task.verify("A stunning sunset over the hills.") is False


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
