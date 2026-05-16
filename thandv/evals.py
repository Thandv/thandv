"""Eval harness.

v0.2 ships a tiny `smoke` suite — three hand-coded problems that exercise
the agent end-to-end without needing any dataset download. Real HumanEval
and MBPP integrations arrive in v0.2.x once we wire in `datasets`.

The harness is persona-aware: every suite declares a default persona, and
the runner can override it via `--persona`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from thandv.agent import Agent
from thandv.config import Config, ensure_dirs
from thandv.personas import get_persona


@dataclass
class EvalTask:
    id: str
    prompt: str
    verify: Callable[[str], bool]


@dataclass
class EvalSuite:
    name: str
    tasks: list[EvalTask]
    default_persona: str = "code"


@dataclass
class EvalResult:
    task_id: str
    suite: str
    model: str
    persona: str
    reply: str
    passed: bool
    secs: float


# --- Verifiers --------------------------------------------------------------

def _contains(needle: str) -> Callable[[str], bool]:
    n = needle.lower()
    return lambda reply: n in reply.lower()


def _is_prime_function(reply: str) -> bool:
    """Exec the reply and check `is_prime` on a handful of cases."""
    m = re.search(r"```(?:python)?\n(.+?)\n```", reply, re.DOTALL)
    code = m.group(1) if m else reply
    namespace: dict = {}
    try:
        exec(code, namespace)
    except Exception:
        return False
    fn = namespace.get("is_prime")
    if not callable(fn):
        return False
    cases = [(2, True), (3, True), (4, False), (5, True), (9, False), (11, True), (1, False)]
    try:
        return all(fn(n) is expected for n, expected in cases)
    except Exception:
        return False


# --- Suites -----------------------------------------------------------------

SMOKE = EvalSuite(
    name="smoke",
    default_persona="code",
    tasks=[
        EvalTask(
            id="arithmetic",
            prompt="What is 7 * 8? Reply with only the number, nothing else.",
            verify=_contains("56"),
        ),
        EvalTask(
            id="reverse-string",
            prompt="Reverse the string 'hello' and reply with only the result.",
            verify=_contains("olleh"),
        ),
        EvalTask(
            id="is-prime",
            prompt=(
                "Write a Python function `is_prime(n)` that returns True for "
                "primes and False otherwise. Reply with only the function "
                "definition inside a single ```python``` code block."
            ),
            verify=_is_prime_function,
        ),
    ],
)


SUITES: dict[str, EvalSuite] = {SMOKE.name: SMOKE}


def list_suites() -> list[str]:
    return sorted(SUITES)


def get_suite(name: str) -> EvalSuite:
    if name not in SUITES:
        raise ValueError(f"unknown suite: {name}. Known: {list_suites()}")
    return SUITES[name]


# --- Runner -----------------------------------------------------------------

def run_suite(
    suite_name: str,
    model: str,
    persona_name: str | None = None,
    limit: int | None = None,
    out_path: Path | None = None,
    on_progress: Callable[[int, int, EvalResult], None] | None = None,
) -> list[EvalResult]:
    suite = get_suite(suite_name)
    persona = get_persona(persona_name or suite.default_persona)
    tasks = suite.tasks if limit is None else suite.tasks[:limit]

    ensure_dirs()
    results: list[EvalResult] = []

    for i, task in enumerate(tasks, 1):
        cfg = Config(model=model, persona=persona.name)
        agent = Agent(config=cfg, persona=persona)
        t0 = time.time()
        try:
            reply = "".join(agent.turn(task.prompt))
        except Exception as e:
            reply = f"<error: {type(e).__name__}: {e}>"
        secs = time.time() - t0
        try:
            passed = task.verify(reply)
        except Exception:
            passed = False
        result = EvalResult(
            task_id=task.id,
            suite=suite.name,
            model=model,
            persona=persona.name,
            reply=reply,
            passed=passed,
            secs=secs,
        )
        results.append(result)
        if on_progress:
            on_progress(i, len(tasks), result)

    _persist(suite.name, results, out_path)
    return results


def _persist(suite_name: str, results: list[EvalResult], out_path: Path | None) -> Path:
    from thandv.config import THANDV_HOME

    if out_path is None:
        out_dir = THANDV_HOME / "evals"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{suite_name}-{time.time_ns()}.json"
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    return out_path


def summarise(results: list[EvalResult]) -> str:
    n_passed = sum(1 for r in results if r.passed)
    pct = 100 * n_passed / len(results) if results else 0.0
    return f"passed {n_passed}/{len(results)} ({pct:.1f}%)"
