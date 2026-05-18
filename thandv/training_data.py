"""Public-dataset → training-queue ingestion.

Each registered dataset has:
- a HuggingFace repo id and split,
- a function mapping a row to our queue's `{"prompt", "completion"}` shape,
- a license string and a "best fits which persona" hint.

`thandv train sample-public NAME [--n N] [--seed S]` samples N rows
(reproducibly with `--seed`), formats them, and appends a single JSONL
file to the trainer's queue. Honest about license/scope per dataset.

Lazy `datasets` import — users who don't run training don't pay for it.
Adding a new dataset is one `PublicDataset(...)` entry below.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from thandv import trainer


@dataclass(frozen=True)
class PublicDataset:
    name: str
    hf_repo: str
    split: str
    description: str
    license: str
    persona_hint: str  # which persona this most naturally fits
    row_to_example: Callable[[dict], dict]


# --- Row mappers ---------------------------------------------------------

def _codealpaca_row_to_example(row: dict) -> dict:
    """CodeAlpaca-20k schema: instruction + optional input + output."""
    prompt = row["instruction"]
    if row.get("input"):
        prompt = f"{prompt}\n\n{row['input']}"
    return {"prompt": prompt, "completion": row["output"]}


def _dolly_row_to_example(row: dict) -> dict:
    """Dolly-15k schema: instruction + optional context + response."""
    prompt = row["instruction"]
    if row.get("context"):
        prompt = f"{prompt}\n\nContext:\n{row['context']}"
    return {"prompt": prompt, "completion": row["response"]}


def _alpaca_row_to_example(row: dict) -> dict:
    """Original Stanford Alpaca schema (same as CodeAlpaca's)."""
    return _codealpaca_row_to_example(row)


# --- Registry -----------------------------------------------------------

CODEALPACA = PublicDataset(
    name="codealpaca",
    hf_repo="sahil2801/CodeAlpaca-20k",
    split="train",
    description="20k instruction-following examples scoped to code tasks.",
    license="cc-by-4.0",
    persona_hint="code",
    row_to_example=_codealpaca_row_to_example,
)

DOLLY = PublicDataset(
    name="dolly",
    hf_repo="databricks/databricks-dolly-15k",
    split="train",
    description="15k human-written instructions, mixed categories.",
    license="cc-by-sa-3.0",
    persona_hint="writer",
    row_to_example=_dolly_row_to_example,
)

ALPACA = PublicDataset(
    name="alpaca",
    hf_repo="tatsu-lab/alpaca",
    split="train",
    description="52k instruction-following examples (Stanford Alpaca). "
                "Generated with text-davinci-003 — check OpenAI ToS before "
                "training a model you'll distribute commercially.",
    license="cc-by-nc-4.0",
    persona_hint="code",
    row_to_example=_alpaca_row_to_example,
)


PUBLIC_DATASETS: dict[str, PublicDataset] = {
    CODEALPACA.name: CODEALPACA,
    DOLLY.name: DOLLY,
    ALPACA.name: ALPACA,
}


def list_public_datasets() -> list[PublicDataset]:
    return list(PUBLIC_DATASETS.values())


def get_public_dataset(name: str) -> PublicDataset:
    if name not in PUBLIC_DATASETS:
        raise ValueError(
            f"unknown dataset: {name!r}. Known: {sorted(PUBLIC_DATASETS)}"
        )
    return PUBLIC_DATASETS[name]


# --- Sampling + queueing ------------------------------------------------

def sample_public_dataset(name: str, n: int, *, seed: int = 0) -> list[dict]:
    """Pull N reproducibly-sampled examples in {prompt, completion} shape.

    Lazy `datasets` import with a clear install hint if missing. Cache dir
    follows HF's default (~/.cache/huggingface) — we don't override here so
    multiple training projects share the same cache.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "datasets not installed. Run: pip install thandv[eval]"
        ) from e

    dataset = get_public_dataset(name)
    ds = load_dataset(dataset.hf_repo, split=dataset.split)
    total = len(ds)
    k = min(n, total)
    indices = random.Random(seed).sample(range(total), k)
    return [dataset.row_to_example(ds[i]) for i in indices]


def queue_public_dataset(name: str, n: int, *, seed: int = 0) -> Path:
    """Sample, format, and write a JSONL file directly into the queue.

    Returns the queue file path. Filename includes the dataset name and
    count for greppable provenance.
    """
    examples = sample_public_dataset(name, n, seed=seed)
    trainer._ensure_dirs()
    path = trainer.QUEUE_DIR / f"{int(time.time())}-{name}-n{n}.jsonl"
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return path
