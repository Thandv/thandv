"""Background training daemon.

The trainer reads training examples from a local queue, runs real LoRA
fine-tuning on the local base model via `thandv.training_backend`,
evaluates the resulting adapter, and promotes it **only if the eval pass
rate improves**. Everything beyond data ingestion is fully local.

The daemon is paranoid by design:
- Eval failure (e.g. Ollama down) skips the tick rather than discarding work.
- Train failure (e.g. MLX-LM crashes mid-run) leaves the queue file in
  place so the next tick can retry.
- A baseline eval is measured once against the current `active_ollama_model`
  and stored; subsequent adapters must *beat* it to be promoted.
- Adapters that don't beat the baseline are unregistered from Ollama
  (`ollama rm`) and their on-disk artifacts removed. Their source queue
  file moves to `processed/` regardless to prevent loops.

State migration: TrainerState gained `active_ollama_model` (what to eval
against) and `hf_base_model` (HF repo id for training) in v0.4.1. Old
state files load without these fields; the trainer falls back to
`config.model` and a default HF mapping when they're empty.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from thandv import training_backend as _tb
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
    # The Ollama model the trainer currently considers best; starts as the
    # untouched base (e.g. "qwen2.5-coder:7b") and gets replaced with
    # "thandv-<adapter_id>" when an adapter is promoted. Empty means
    # "fall back to whatever caller passed as base_model".
    active_ollama_model: str = ""
    # HF repo id of the training base, e.g. "Qwen/Qwen2.5-Coder-7B". Set by
    # `thandv train enable`; the trainer reads it to find the local HF
    # safetensors for LoRA. Empty means "infer from the Ollama tag via the
    # default mapping table in training_backend".
    hf_base_model: str = ""
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


# --- Training -------------------------------------------------------------

def _train_lora(
    queue_file: Path,
    base_ollama_model: str,
    *,
    hf_base_repo: str | None = None,
) -> tuple[str, str]:
    """Run real LoRA training + GGUF export + Ollama registration.

    Returns ``(adapter_id, new_ollama_model_name)``. Raises on any
    sub-step failure — caller decides whether that's "skip this tick"
    (transient) or worse.

    Side effects:
    - Picks the strongest available `TrainerBackend`.
    - Trains a LoRA in `ADAPTERS_DIR / <adapter_id> / adapter/`.
    - Fuses + exports `ADAPTERS_DIR / <adapter_id> / merged.gguf`.
    - Writes an Ollama Modelfile and runs `ollama create thandv-<id>`.
    """
    _ensure_dirs()
    backend = _tb.pick_backend()
    if backend is None:
        raise RuntimeError(
            "no training backend available. Run: "
            "pip install thandv[train-mlx] && thandv train enable"
        )

    # Resolve the HF base model dir.
    hf_repo = hf_base_repo or _tb.ollama_to_hf(base_ollama_model)
    base_hf_dir = _tb.base_model_dir(hf_repo)
    if not base_hf_dir.exists() or not any(base_hf_dir.iterdir()):
        raise FileNotFoundError(
            f"HF base model not downloaded: {base_hf_dir}. "
            "Run: thandv train enable"
        )

    adapter_id = f"adapter-{time.time_ns()}"
    out_dir = ADAPTERS_DIR / adapter_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: LoRA train. Adapter weights land under out_dir/adapter/.
    backend.train(base_hf_dir, queue_file, out_dir)

    # Step 2: Merge + export GGUF. Single GGUF file at out_dir/merged.gguf.
    gguf_path = out_dir / "merged.gguf"
    backend.merge_to_gguf(base_hf_dir, out_dir / "adapter", gguf_path)

    # Step 3: Modelfile + ollama create.
    modelfile = out_dir / "Modelfile"
    modelfile.write_text(f"FROM {gguf_path.resolve()}\n")
    new_model_name = f"thandv-{adapter_id}"
    subprocess.run(
        ["ollama", "create", new_model_name, "-f", str(modelfile)],
        check=True,
    )

    return adapter_id, new_model_name


def _discard_adapter(adapter_id: str, ollama_model: str) -> None:
    """Clean up everything created by a discarded adapter run.

    Best-effort: `ollama rm` may fail if the model wasn't actually
    registered (e.g. we crashed mid-create); ignore.
    """
    subprocess.run(["ollama", "rm", ollama_model], check=False, capture_output=True)
    shutil.rmtree(ADAPTERS_DIR / adapter_id, ignore_errors=True)


# --- Tick: one full train → eval → promote cycle --------------------------

def _log_outcome(outcome: dict) -> dict:
    _ensure_dirs()
    log_path = LOGS_DIR / f"tick-{time.time_ns()}.json"
    log_path.write_text(json.dumps(outcome, indent=2))
    return outcome


def tick(base_model: str, *, dry_run: bool = False) -> dict:
    """Run one training iteration.

    `base_model` is the Ollama tag of the untouched base (e.g.
    "qwen2.5-coder:7b"). The trainer evaluates against
    `state.active_ollama_model` if a promoted adapter exists, else against
    `base_model`. Returns a dict describing what happened:
    action ∈ {baseline, skip, promote, discard}.
    """
    _ensure_dirs()
    state = load_state()
    state.last_tick_at = _now()
    state.ticks_completed += 1

    # What model are we currently treating as best?
    current_best_model = state.active_ollama_model or base_model

    # Baseline eval: measured once at the start, against the active model.
    if not state.baseline_measured:
        try:
            baseline = _run_eval(current_best_model)
        except Exception as e:
            save_state(state)
            return _log_outcome(
                {"action": "skip", "reason": f"baseline eval failed: {e}", "at": _now()}
            )
        state.best_eval_pass_rate = baseline
        state.last_eval_pass_rate = baseline
        state.baseline_measured = True
        # Lock in the active_ollama_model so we keep evaluating consistently.
        if not state.active_ollama_model:
            state.active_ollama_model = base_model
        save_state(state)
        return _log_outcome(
            {"action": "baseline", "pass_rate": baseline, "model": current_best_model, "at": _now()}
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

    # Step 1: Train. Failures here leave the queue file in place so the
    # next tick can retry — could be transient (mlx OOM, disk full, etc.).
    try:
        adapter_id, new_ollama_model = _train_lora(
            queue_file,
            base_model,
            hf_base_repo=state.hf_base_model or None,
        )
    except Exception as e:
        save_state(state)
        return _log_outcome(
            {
                "action": "skip",
                "reason": f"train failed: {type(e).__name__}: {e}",
                "queue_file": queue_file.name,
                "at": _now(),
            }
        )

    # Step 2: Eval the new model against the same suite as baseline.
    try:
        new_rate = _run_eval(new_ollama_model)
    except Exception as e:
        # Couldn't gate. Clean up the new model + adapter dir, keep the
        # queue file in place for a retry.
        _discard_adapter(adapter_id, new_ollama_model)
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
        # Promote: remember the new model + adapter; rotate the old active
        # adapter out (but don't `ollama rm` the base — only thandv-<id>
        # adapters get cleaned up here).
        old_active_adapter = state.active_adapter
        old_active_model = state.active_ollama_model
        state.best_eval_pass_rate = new_rate
        state.active_adapter = adapter_id
        state.active_ollama_model = new_ollama_model
        action = "promote"
        if old_active_adapter and old_active_model and old_active_model.startswith("thandv-"):
            _discard_adapter(old_active_adapter, old_active_model)
    else:
        _discard_adapter(adapter_id, new_ollama_model)
        action = "discard"

    # Consume the queue file regardless.
    queue_file.rename(PROCESSED_DIR / queue_file.name)
    save_state(state)

    return _log_outcome(
        {
            "action": action,
            "adapter_id": adapter_id,
            "ollama_model": new_ollama_model,
            "queue_file": queue_file.name,
            "pass_rate": new_rate,
            "best_pass_rate": state.best_eval_pass_rate,
            "at": _now(),
        }
    )


def _run_eval(model: str, suite: str = "smoke") -> float:
    """Return the pass rate (0.0–1.0) of `suite` for `model`."""
    results = run_suite(suite, model=model)
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
