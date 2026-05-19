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
    hf_config: str | None = None  # set for HF datasets that require a config name


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


# Prose-continuation framing for free-text rows. Each example becomes a
# (head, tail) pair under a continuation prompt so the trainer's queue
# shape ({prompt, completion}) is preserved. The cap keeps any single
# example small enough to not dwarf training batches.
PROSE_MAX_CHARS = 4000
PROSE_HEAD_FRACTION = 0.6


def _prose_row_to_example(row: dict) -> dict:
    """Continuation-style training pair from a free-form prose row.

    Takes up to `PROSE_MAX_CHARS` chars, splits at ~60% on a whitespace
    boundary so we don't slice inside a word, and frames the head under
    a continuation prompt. Very short / empty rows produce minimal pairs
    -- the trainer is responsible for skipping degenerate examples.
    """
    text = (row.get("text") or "").strip()
    text = text[:PROSE_MAX_CHARS]
    if not text:
        return {"prompt": "", "completion": ""}

    target = int(len(text) * PROSE_HEAD_FRACTION)
    cut = text.rfind(" ", 0, target + 1)
    # If we couldn't find whitespace anywhere reasonable, fall back to
    # the raw character index rather than refusing to split.
    if cut == -1 or cut < target // 2:
        cut = target
    head = text[:cut].rstrip()
    tail = text[cut:].lstrip()
    return {
        "prompt": "Continue the following passage in the same style:\n\n" + head,
        "completion": tail,
    }


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


WIKITEXT = PublicDataset(
    name="wikitext",
    hf_repo="Salesforce/wikitext",
    hf_config="wikitext-2-raw-v1",
    split="train",
    description="Wikitext-2 raw train split (~12 MB, ~36k rows). Wikipedia "
                "prose; section-heading rows produce trivial pairs that the "
                "trainer is expected to skip.",
    license="cc-by-sa-3.0",
    persona_hint="writer",
    row_to_example=_prose_row_to_example,
)

TINYSTORIES = PublicDataset(
    name="tinystories",
    hf_repo="roneneldan/TinyStories",
    split="train",
    description="Short synthetic stories (Eldan & Li, 2023; ~150 MB train). "
                "Useful as low-vocabulary narrative prose. Generated with "
                "GPT-3.5/4 -- review the upstream license card before training "
                "a model you intend to distribute commercially.",
    license="cdla-sharing-1.0",
    persona_hint="writer",
    row_to_example=_prose_row_to_example,
)


PUBLIC_DATASETS: dict[str, PublicDataset] = {
    CODEALPACA.name: CODEALPACA,
    DOLLY.name: DOLLY,
    ALPACA.name: ALPACA,
    WIKITEXT.name: WIKITEXT,
    TINYSTORIES.name: TINYSTORIES,
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

    Validate the dataset name *first* (cheap, no deps) — that way an
    unknown name surfaces the right error even when `datasets` isn't
    installed. Then lazy-import `datasets` with a clear install hint if
    missing. Cache dir follows HF's default (~/.cache/huggingface) so
    multiple training projects share the same cache.
    """
    dataset = get_public_dataset(name)  # raises ValueError on unknown
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "datasets not installed. Run: pip install thandv[eval]"
        ) from e

    if dataset.hf_config:
        ds = load_dataset(dataset.hf_repo, dataset.hf_config, split=dataset.split)
    else:
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
