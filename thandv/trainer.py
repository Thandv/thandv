"""Background training daemon.

The trainer reads training examples from a local queue, runs LoRA fine-tuning
on the local base model, evaluates against a chosen suite, and promotes the
new adapter **only if the eval pass rate improves**. Everything beyond data
ingestion is fully local; the binary stays self-sufficient at runtime.

v0.2.x ships the daemon scaffolding with a *stub* training step (no real
weight updates yet). The wiring — queue mgmt, eval gate, adapter promotion,
state, logs, pause/resume, idempotent restart — is real and exercised by
tests. The actual LoRA training step plugs in at v0.4 once the trainer
dependency (Unsloth on CUDA / MLX-LM on Apple Silicon) lands.

The daemon is paranoid by design:
- Eval failure (e.g. Ollama down) skips the tick rather than discarding work.
- A baseline eval is measured once and stored; subsequent adapters must
  *beat* it to be promoted.
- Adapters that don't beat the baseline are deleted; their source queue file
  is moved to `processed/` regardless so we don't reprocess it forever.
"""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from thandv.config import THANDV_HOME
from thandv.evals import run_suite

TRAINING_DIR = THANDV_HOME / "training"
QUEUE_DIR = TRAINING_DIR / "queue"
PROCESSED_DIR = TRAINING_DIR / "processed"
ADAPTERS_DIR = TRAINING_DIR / "adapters"
LOGS_DIR = TRAINING_DIR / "logs"
STATE_PATH = TRAINING_DIR / "state.json"
PID_PATH = TRAINING_DIR / "trainer.pid"


@dataclass
class TrainerState:
    started_at: str = ""
    last_tick_at: str = ""
    ticks_completed: int = 0
    last_eval_pass_rate: float = 0.0
    best_eval_pass_rate: float = 0.0
    baseline_measured: bool = False
    active_adapter: str = ""
    paused: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_dirs() -> None:
    for d in (TRAINING_DIR, QUEUE_DIR, PROCESSED_DIR, ADAPTERS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --- State persistence -----------------------------------------------------

def load_state() -> TrainerState:
    _ensure_dirs()
    if STATE_PATH.exists():
        data = json.loads(STATE_PATH.read_text())
        return TrainerState(
            **{k: v for k, v in data.items() if k in TrainerState.__annotations__}
        )
    return TrainerState()


def save_state(state: TrainerState) -> None:
    _ensure_dirs()
    STATE_PATH.write_text(json.dumps(asdict(state), indent=2))


# --- Queue operations -----------------------------------------------------

def enqueue_examples(name: str, examples: list[dict]) -> Path:
    """Append a JSONL file of `{"prompt": ..., "completion": ...}` to the queue."""
    _ensure_dirs()
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = QUEUE_DIR / f"{int(time.time())}-{safe_name}.jsonl"
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return path


def enqueue_path(src: Path) -> Path:
    """Copy an existing JSONL file into the queue."""
    _ensure_dirs()
    dst = QUEUE_DIR / f"{int(time.time())}-{src.name}"
    dst.write_bytes(src.read_bytes())
    return dst


def queue_size() -> int:
    _ensure_dirs()
    return sum(1 for _ in QUEUE_DIR.glob("*.jsonl"))


def _pop_queue() -> Path | None:
    _ensure_dirs()
    files = sorted(QUEUE_DIR.glob("*.jsonl"))
    return files[0] if files else None


# --- Training (currently stubbed) -----------------------------------------

def _stub_train(queue_file: Path) -> str:
    """Pretend to train a LoRA on the queue file. Returns an adapter id.

    v0.4 replaces this with a real Unsloth / MLX-LM call. The interface
    stays the same: receive a queue file, return an adapter id, with the
    adapter persisted under ADAPTERS_DIR / f"{adapter_id}.json" (today as a
    marker, later as the actual safetensors plus a manifest).
    """
    _ensure_dirs()
    adapter_id = f"adapter-{time.time_ns()}"
    (ADAPTERS_DIR / f"{adapter_id}.json").write_text(
        json.dumps(
            {
                "adapter_id": adapter_id,
                "source_queue_file": queue_file.name,
                "trained_at": _now(),
                "stub": True,
            },
            indent=2,
        )
    )
    return adapter_id


# --- Tick: one full train → eval → promote cycle --------------------------

def _log_outcome(outcome: dict) -> dict:
    _ensure_dirs()
    log_path = LOGS_DIR / f"tick-{time.time_ns()}.json"
    log_path.write_text(json.dumps(outcome, indent=2))
    return outcome


def tick(model: str, *, dry_run: bool = False) -> dict:
    """Run one training iteration.

    Returns a dict describing what happened: action ∈ {baseline, skip,
    promote, discard}, plus context.
    """
    _ensure_dirs()
    state = load_state()
    state.last_tick_at = _now()
    state.ticks_completed += 1

    # Establish the no-adapter baseline once.
    if not state.baseline_measured:
        try:
            baseline = _run_eval(model)
        except Exception as e:
            save_state(state)
            return _log_outcome(
                {"action": "skip", "reason": f"baseline eval failed: {e}", "at": _now()}
            )
        state.best_eval_pass_rate = baseline
        state.last_eval_pass_rate = baseline
        state.baseline_measured = True
        save_state(state)
        return _log_outcome(
            {"action": "baseline", "pass_rate": baseline, "at": _now()}
        )

    queue_file = _pop_queue()
    if queue_file is None:
        save_state(state)
        return _log_outcome({"action": "skip", "reason": "queue empty", "at": _now()})

    if dry_run:
        save_state(state)
        return _log_outcome(
            {
                "action": "skip",
                "reason": "dry run",
                "queue_file": queue_file.name,
                "at": _now(),
            }
        )

    adapter_id = _stub_train(queue_file)

    try:
        new_rate = _run_eval(model)
    except Exception as e:
        # Eval unavailable → can't gate. Discard adapter, keep queue file.
        (ADAPTERS_DIR / f"{adapter_id}.json").unlink(missing_ok=True)
        save_state(state)
        return _log_outcome(
            {
                "action": "skip",
                "reason": f"eval failed: {e}",
                "queue_file": queue_file.name,
                "at": _now(),
            }
        )

    state.last_eval_pass_rate = new_rate
    improved = new_rate > state.best_eval_pass_rate
    if improved:
        state.best_eval_pass_rate = new_rate
        state.active_adapter = adapter_id
        action = "promote"
    else:
        (ADAPTERS_DIR / f"{adapter_id}.json").unlink(missing_ok=True)
        action = "discard"

    # Move queue file out regardless so we don't loop on the same file.
    queue_file.rename(PROCESSED_DIR / queue_file.name)
    save_state(state)

    return _log_outcome(
        {
            "action": action,
            "adapter_id": adapter_id,
            "queue_file": queue_file.name,
            "pass_rate": new_rate,
            "best_pass_rate": state.best_eval_pass_rate,
            "at": _now(),
        }
    )


def _run_eval(model: str) -> float:
    """Return the pass rate (0.0–1.0) of the smoke suite for `model`."""
    results = run_suite("smoke", model=model)
    if not results:
        return 0.0
    return sum(1 for r in results if r.passed) / len(results)


# --- Daemon control -------------------------------------------------------

def run_forever(model: str, interval_s: int = 300) -> None:
    """Foreground daemon loop. Designed to be invoked by launchd / systemd
    or with `thandv train run` for manual debug.
    """
    _ensure_dirs()
    state = load_state()
    state.started_at = _now()
    state.paused = False
    save_state(state)
    PID_PATH.write_text(str(os.getpid()))

    def _term(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)

    try:
        while True:
            if not load_state().paused:
                tick(model)
            time.sleep(interval_s)
    except KeyboardInterrupt:
        pass
    finally:
        PID_PATH.unlink(missing_ok=True)


def is_running() -> bool:
    if not PID_PATH.exists():
        return False
    try:
        pid = int(PID_PATH.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def stop() -> bool:
    if not PID_PATH.exists():
        return False
    try:
        pid = int(PID_PATH.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        return True
    except (ValueError, ProcessLookupError):
        PID_PATH.unlink(missing_ok=True)
        return False


def pause() -> None:
    state = load_state()
    state.paused = True
    save_state(state)


def resume() -> None:
    state = load_state()
    state.paused = False
    save_state(state)
