"""Distillation stub.

Goal: read prior Thandv sessions (JSONL under ~/.thandv/sessions/), filter
for successful traces, and emit a JSONL training set for LoRA fine-tuning.

v0 is intentionally a no-op — it just prints the schema and counts. Real
filtering, formatting, and teacher-aided augmentation arrive in v0.3.
"""

from __future__ import annotations

import json
from pathlib import Path

from thandv.config import SESSIONS_DIR


def main() -> int:
    if not SESSIONS_DIR.exists():
        print(f"no sessions dir at {SESSIONS_DIR}")
        return 1
    sessions = sorted(SESSIONS_DIR.glob("*.jsonl"))
    print(f"found {len(sessions)} sessions")
    total_events = 0
    for s in sessions:
        with s.open() as f:
            total_events += sum(1 for _ in f)
    print(f"total events: {total_events}")
    print(
        json.dumps(
            {
                "next": [
                    "label sessions success/failure",
                    "extract (prompt, completion) pairs from labelled-success sessions",
                    "format for Qwen LoRA training (axolotl / unsloth)",
                ]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
