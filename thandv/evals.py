"""Eval harness.

Two suites ship in-tree:

- `smoke` — three hand-coded problems, no dependencies, runs in seconds.
- `humaneval` — the canonical OpenAI HumanEval benchmark, 164 Python
  function-completion problems verified by the dataset's unit tests in a
  subprocess sandbox. Requires `pip install thandv[eval]` (pulls
  `datasets`) and a one-time ~300 KB download cached under
  `~/.thandv/datasets/`.

The harness is persona-aware: every suite declares a default persona, and
the runner can override it via `--persona`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from thandv.agent import Agent
from thandv.config import THANDV_HOME, Config, ensure_dirs
from thandv.personas import get_persona


@dataclass
class EvalTask:
    id: str
    prompt: str
    verify: Callable[[str], bool]


@dataclass
class EvalSuite:
    name: str
    tasks: list[EvalTask] = field(default_factory=list)
    default_persona: str = "code"
    # Optional lazy-load hook. If set, the runner calls this *once* before
    # iterating, so big datasets (HumanEval) only download when actually
    # requested. The callable returns the populated task list.
    loader: Callable[[], list[EvalTask]] | None = None


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


# --- HumanEval ------------------------------------------------------------

HUMANEVAL_DATASET_DIR = THANDV_HOME / "datasets" / "humaneval"
HUMANEVAL_TIMEOUT_S = 10  # per-task subprocess timeout when running tests


def _extract_python_code(reply: str) -> str:
    """Strip optional markdown fences. Falls back to the raw reply."""
    m = re.search(r"```(?:python)?\s*\n(.*?)\n```", reply, re.DOTALL)
    return m.group(1) if m else reply


def _humaneval_verifier(test_code: str, entry_point: str) -> Callable[[str], bool]:
    """Build a verifier that exec's the model's completion + the task's
    canonical test code in a *subprocess* and checks the exit code.

    Sandboxing here is best-effort: timeout, separate process, no network
    isolation. Acceptable for research; do not run this on untrusted models
    or in production multi-tenant contexts.
    """

    def verify(reply: str) -> bool:
        code = _extract_python_code(reply)
        program = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(program)
            path = f.name
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=HUMANEVAL_TIMEOUT_S,
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            try:
                Path(path).unlink()
            except OSError:
                pass

    return verify


def _load_humaneval_tasks() -> list[EvalTask]:
    """Download (cache) and convert the HumanEval dataset to EvalTasks.

    Lazy `datasets` import: users without the optional extra get a clear
    error instead of a heavy dep at every `thandv eval --list`.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "humaneval suite needs `datasets`. Install: pip install thandv[eval]"
        ) from e

    HUMANEVAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "openai_humaneval",
        split="test",
        cache_dir=str(HUMANEVAL_DATASET_DIR),
    )

    tasks: list[EvalTask] = []
    for row in ds:
        prompt = (
            "Complete the following Python function. Return ONLY the full "
            "function definition inside a single ```python``` code block — "
            "no prose, no examples, no test cases.\n\n"
            f"{row['prompt']}"
        )
        tasks.append(
            EvalTask(
                id=row["task_id"],
                prompt=prompt,
                verify=_humaneval_verifier(row["test"], row["entry_point"]),
            )
        )
    return tasks


HUMANEVAL = EvalSuite(
    name="humaneval",
    default_persona="code",
    loader=_load_humaneval_tasks,
)
SUITES[HUMANEVAL.name] = HUMANEVAL


# --- MBPP -----------------------------------------------------------------

MBPP_DATASET_DIR = THANDV_HOME / "datasets" / "mbpp"
MBPP_TIMEOUT_S = 10


def _mbpp_verifier(test_list: list[str]) -> Callable[[str], bool]:
    """Build a verifier that exec's the model's completion alongside every
    assertion in `test_list` in a subprocess. Same sandbox caveat as
    HumanEval: timeout + separate process, no network isolation. Acceptable
    for research; not for untrusted models in multi-tenant contexts.

    MBPP differs from HumanEval in shape only: the dataset gives us a list
    of bare assertion strings rather than a `check(candidate)` harness, so
    we concatenate them directly after the model's code.
    """
    tests = "\n".join(test_list)

    def verify(reply: str) -> bool:
        code = _extract_python_code(reply)
        program = code + "\n\n" + tests + "\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(program)
            path = f.name
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=MBPP_TIMEOUT_S,
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            try:
                Path(path).unlink()
            except OSError:
                pass

    return verify


def _load_mbpp_tasks() -> list[EvalTask]:
    """Download (cache) and convert the MBPP `sanitized` test split.

    The sanitized config is the cleaner subset (~427 problems on test split
    in current snapshots); the full config has more but is messier.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "mbpp suite needs `datasets`. Install: pip install thandv[eval]"
        ) from e

    MBPP_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "mbpp",
        "sanitized",
        split="test",
        cache_dir=str(MBPP_DATASET_DIR),
    )

    tasks: list[EvalTask] = []
    for row in ds:
        prompt = (
            "Write the Python function described below. Return ONLY the full "
            "function definition inside a single ```python``` code block — "
            "no prose, no examples, no test cases.\n\n"
            f"{row['prompt']}"
        )
        tasks.append(
            EvalTask(
                id=f"MBPP/{row['task_id']}",
                prompt=prompt,
                verify=_mbpp_verifier(row["test_list"]),
            )
        )
    return tasks


MBPP = EvalSuite(
    name="mbpp",
    default_persona="code",
    loader=_load_mbpp_tasks,
)
SUITES[MBPP.name] = MBPP


def list_suites() -> list[str]:
    return sorted(SUITES)


def get_suite(name: str) -> EvalSuite:
    if name not in SUITES:
        raise ValueError(f"unknown suite: {name}. Known: {list_suites()}")
    return SUITES[name]


def _ensure_tasks(suite: EvalSuite) -> list[EvalTask]:
    """Run the suite's lazy loader once and memoise the result on the suite."""
    if not suite.tasks and suite.loader is not None:
        suite.tasks = suite.loader()
    return suite.tasks


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
    all_tasks = _ensure_tasks(suite)
    tasks = all_tasks if limit is None else all_tasks[:limit]

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
