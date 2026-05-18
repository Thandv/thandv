"""Training backends.

A `TrainerBackend` is the seam between the trainer daemon
([`thandv/trainer.py`](trainer.py)) and the underlying LoRA training
implementation. Currently shipped:

- **MLXBackend** — Apple Silicon via `mlx-lm`. Primary on this codebase
  because the dev hardware is M-series.
- **HFPEFTBackend** — universal, slow, used as a portability fallback. Stub
  in v0.4.0; first real impl in v0.4.x.

A new backend implements three things: `is_available()` to gate selection,
`train()` to produce a LoRA adapter from a JSONL queue file, and
`merge_adapter()` to fold the adapter into the base for serving. The
trainer never touches the underlying library directly — it goes through
this seam, so backends can be added without touching the daemon.

v0.4.0 ships the abstraction + MLX impl + setup command (`thandv train
enable`). v0.4.1 wires `trainer._stub_train` to call into the backend.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from thandv.config import THANDV_HOME

TRAINING_DIR = THANDV_HOME / "training"
BASE_MODELS_DIR = TRAINING_DIR / "base"


@dataclass
class TrainResult:
    """What a backend hands back from `train()`."""
    adapter_dir: Path        # where the adapter weights live
    n_examples: int          # how many examples were trained on
    n_iters: int             # how many training steps were taken
    backend: str             # the backend's name


@runtime_checkable
class TrainerBackend(Protocol):
    """The contract every training backend must satisfy."""

    name: str

    def is_available(self) -> bool:
        """True iff this backend can actually run on the current host
        (dependencies installed, GPU/Metal accessible, etc.)."""
        ...

    def train(
        self,
        base_hf_dir: Path,
        queue_file: Path,
        out_dir: Path,
        *,
        iters: int = 100,
        batch_size: int = 1,
    ) -> TrainResult:
        """Train a LoRA on `queue_file` (JSONL of {prompt, completion}).

        - `base_hf_dir` — the HF-format base model directory.
        - `out_dir` — where the adapter artifacts get written.
        - `iters` and `batch_size` — backend-specific knobs with sensible
          defaults appropriate for "small adapter, fast iteration".
        """
        ...

    def merge_adapter(
        self,
        base_hf_dir: Path,
        adapter_dir: Path,
        out_hf_dir: Path,
    ) -> Path:
        """Fold the adapter into the base, producing a new HF-format model
        directory at `out_hf_dir`. Returns the same path on success.

        The merged model is what we later convert to GGUF for Ollama
        serving.
        """
        ...


# --- MLX-LM backend (Apple Silicon) ---------------------------------------

class MLXBackend:
    """Apple Silicon backend via the `mlx-lm` CLI.

    Invokes `mlx_lm.lora` and `mlx_lm.fuse` as subprocesses. We use the
    subprocess interface (not the Python API) because mlx-lm's CLI is the
    stable surface and isolates training memory from the calling process —
    if training OOMs, our process survives.
    """

    name = "mlx-lm"

    def is_available(self) -> bool:
        if sys.platform != "darwin":
            return False
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            return False
        return True

    def train(
        self,
        base_hf_dir: Path,
        queue_file: Path,
        out_dir: Path,
        *,
        iters: int = 100,
        batch_size: int = 1,
    ) -> TrainResult:
        if not base_hf_dir.exists():
            raise FileNotFoundError(f"base model not found: {base_hf_dir}")
        if not queue_file.exists():
            raise FileNotFoundError(f"queue file not found: {queue_file}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # mlx_lm.lora expects a data dir containing train.jsonl / valid.jsonl
        # / test.jsonl. We stage a minimal one.
        data_dir = out_dir / "data"
        data_dir.mkdir(exist_ok=True)
        shutil.copy(queue_file, data_dir / "train.jsonl")
        # mlx-lm requires a valid set; reuse train for tiny smoke runs.
        shutil.copy(queue_file, data_dir / "valid.jsonl")

        n_examples = sum(1 for _ in queue_file.open())

        adapter_path = out_dir / "adapter"
        adapter_path.mkdir(exist_ok=True)
        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", str(base_hf_dir),
            "--train",
            "--data", str(data_dir),
            "--iters", str(iters),
            "--batch-size", str(batch_size),
            "--adapter-path", str(adapter_path),
        ]
        subprocess.run(cmd, check=True)

        return TrainResult(
            adapter_dir=adapter_path,
            n_examples=n_examples,
            n_iters=iters,
            backend=self.name,
        )

    def merge_adapter(
        self,
        base_hf_dir: Path,
        adapter_dir: Path,
        out_hf_dir: Path,
    ) -> Path:
        out_hf_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "mlx_lm.fuse",
            "--model", str(base_hf_dir),
            "--adapter-path", str(adapter_dir),
            "--save-path", str(out_hf_dir),
        ]
        subprocess.run(cmd, check=True)
        return out_hf_dir


# --- HF Transformers + PEFT backend (universal fallback) ------------------

class HFPEFTBackend:
    """Universal LoRA via Hugging Face Transformers + PEFT.

    Slow on CPU, slow on Apple Silicon (MPS), usable on CUDA. We ship the
    *shape* in v0.4.0 so the abstraction is honest; the real impl lands
    after MLX is proven end-to-end. `is_available()` returns False until
    then so `pick_backend()` doesn't pick it accidentally.
    """

    name = "hf-peft"

    def is_available(self) -> bool:
        return False  # v0.4.x implements this

    def train(self, *args, **kwargs) -> TrainResult:
        raise NotImplementedError("hf-peft backend lands in a later milestone")

    def merge_adapter(self, *args, **kwargs) -> Path:
        raise NotImplementedError("hf-peft backend lands in a later milestone")


# --- Registry + picker ----------------------------------------------------

BACKENDS: list[TrainerBackend] = [MLXBackend(), HFPEFTBackend()]


def pick_backend() -> TrainerBackend | None:
    """Return the first available backend, or None if none can run.

    Order matters: MLX before HF+PEFT because we'd rather use the
    optimised Apple Silicon path on this hardware than fall through.
    """
    for backend in BACKENDS:
        if backend.is_available():
            return backend
    return None


# --- HF base-model fetch + verification ----------------------------------

def base_model_dir(model_id: str) -> Path:
    """Where we cache an HF-format base model on disk."""
    safe = model_id.replace("/", "_")
    return BASE_MODELS_DIR / safe


def fetch_hf_base_model(model_id: str, *, force: bool = False) -> Path:
    """Download an HF base model to `~/.thandv/training/base/<model>/`.

    `model_id` is a HuggingFace Hub repo id like `"Qwen/Qwen2.5-Coder-7B"`.
    Returns the local directory. Skips download if already present unless
    `force=True`.

    Lazy import of `huggingface_hub` so users without `[train-mlx]` can
    still import this module (e.g. the trainer reads
    `pick_backend()` to know if anything is available).
    """
    target = base_model_dir(model_id)
    if target.exists() and any(target.iterdir()) and not force:
        return target
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "huggingface-hub is not installed. Install: "
            "pip install thandv[train-mlx]"
        ) from e
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(target))
    return target


def verify_training_setup(
    backend: TrainerBackend,
    base_hf_dir: Path,
    *,
    out_dir: Path | None = None,
) -> dict:
    """Sanity-train a 2-example, 1-iter run as proof the pipeline works.

    Returns a dict suitable for printing to the user. Does *not* leave a
    real adapter behind — the output dir is removed unless the caller
    keeps it.
    """
    out_dir = out_dir or (TRAINING_DIR / "verify")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiny_queue = out_dir / "tiny.jsonl"
    tiny_queue.write_text(
        '{"prompt": "Reply with the word PONG only.", "completion": "PONG"}\n'
        '{"prompt": "Reply with the word PING only.", "completion": "PING"}\n'
    )
    result = backend.train(
        base_hf_dir=base_hf_dir,
        queue_file=tiny_queue,
        out_dir=out_dir,
        iters=1,
        batch_size=1,
    )
    return {
        "backend": result.backend,
        "n_examples": result.n_examples,
        "n_iters": result.n_iters,
        "adapter_dir": str(result.adapter_dir),
    }
