"""Verifier-filtered synthetic data.

The compounding piece of the training loop. Idea:

1. Take an eval suite — every task in it is already a `(prompt, verifier)`
   pair. We trust the verifier.
2. For each task, ask a teacher LLM for `N` completions (oversample).
3. Run the verifier on each completion.
4. Keep only the completions that **pass**. Write them to the training
   queue.

What we end up with is training data whose ground truth is the verifier
itself — not the teacher's opinion. The teacher is just a candidate
generator; the verifier is the filter that decides what's actually right.

Why this matters: pure teacher distillation (v0.5) trains the local
model toward the teacher's outputs, mistakes and all. Verifier-filtered
distillation trains it toward outputs that *pass an automatic check*,
which is the strongest signal we can extract locally. It's the only
honest path to "self-improvement" — improvements are bounded by what the
verifier accepts.

See [`../research/SELF_IMPROVEMENT.md`](../research/SELF_IMPROVEMENT.md)
section "Verifiable-reward fine-tuning" for the underlying argument.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from thandv import evals as _evals
from thandv import teachers as _teachers
from thandv import trainer


def distill_and_filter(
    teacher_name: str,
    suite_name: str,
    *,
    n_per_task: int = 3,
    limit: int | None = None,
    persona: str | None = None,
    out_path: Path | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> tuple[Path, dict]:
    """For each task in the suite, distill `n_per_task` completions from
    the teacher; keep only the ones the task's verifier accepts.

    Returns ``(queue_path, stats_dict)``. The stats dict carries the
    teacher provenance and the per-task yield breakdown (how many
    completions were attempted vs how many passed). It also lands on
    disk as `<queue>.provenance.json`.

    Higher `temperature` by default (0.7) than plain `distill` (0.2):
    we *want* diversity here so the verifier has something to choose
    from. If every completion is identical, the filter is useless.
    """
    teacher = _teachers.get_teacher(teacher_name)
    suite = _evals.get_suite(suite_name)
    tasks = _evals._ensure_tasks(suite)
    if limit is not None:
        tasks = tasks[:limit]

    trainer._ensure_dirs()
    if out_path is None:
        out_path = (
            trainer.QUEUE_DIR
            / f"{int(time.time())}-vfilt-{teacher.name}-{suite_name}.jsonl"
        )

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_attempted = 0
    n_passed = 0
    per_task: list[dict] = []
    teacher_failures: list[dict] = []

    total_calls = len(tasks) * n_per_task
    call_idx = 0

    with out_path.open("w") as f:
        for task in tasks:
            attempted = 0
            kept = 0
            for _ in range(n_per_task):
                call_idx += 1
                attempted += 1
                n_attempted += 1
                try:
                    completion = _teachers.call_teacher(
                        teacher,
                        task.prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as e:
                    teacher_failures.append(
                        {"task_id": task.id, "error": f"{type(e).__name__}: {e}"}
                    )
                    if on_progress:
                        on_progress(call_idx, total_calls, f"teacher-fail: {task.id}")
                    continue

                try:
                    accepted = task.verify(completion)
                except Exception:
                    accepted = False

                if accepted:
                    f.write(
                        json.dumps(
                            {
                                "prompt": task.prompt,
                                "completion": completion,
                                "task_id": task.id,
                            }
                        )
                        + "\n"
                    )
                    kept += 1
                    n_passed += 1
                    if on_progress:
                        on_progress(call_idx, total_calls, f"pass: {task.id}")
                else:
                    if on_progress:
                        on_progress(call_idx, total_calls, f"reject: {task.id}")

            per_task.append(
                {"task_id": task.id, "attempted": attempted, "passed": kept}
            )

    provenance = {
        "teacher": asdict(teacher),
        "suite": suite_name,
        "persona": persona,
        "n_tasks": len(tasks),
        "n_per_task": n_per_task,
        "n_attempted": n_attempted,
        "n_passed": n_passed,
        "yield": (n_passed / n_attempted) if n_attempted else 0.0,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "queue_file": out_path.name,
        "per_task": per_task,
        "teacher_failures": teacher_failures[:20],
    }
    provenance_path = out_path.with_suffix(out_path.suffix + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2))
    return out_path, provenance
