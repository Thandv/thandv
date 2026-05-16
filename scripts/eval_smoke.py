"""Tiny smoke eval: ask the agent a handful of fixed prompts and dump results.

This is *not* a real eval — it just sanity-checks the loop. Real evals
(HumanEval, MBPP, SWE-Bench-lite) land in v0.2.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from thandv.agent import Agent
from thandv.config import Config, ensure_dirs
from thandv.runtime import detect_host, pick_model

PROMPTS = [
    "What is 2 + 2? One word answer.",
    "Write a python function `is_prime(n)` that returns True for primes. Code only.",
    "List the files in the current directory.",
]


def main() -> int:
    ensure_dirs()
    cfg = Config.load()
    if not cfg.model:
        cfg.model = pick_model(detect_host())
    out_dir = Path.home() / ".thandv" / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"smoke-{int(time.time())}.jsonl"

    for prompt in PROMPTS:
        agent = Agent(config=cfg)
        t0 = time.time()
        reply = "".join(agent.turn(prompt))
        with out_path.open("a") as f:
            f.write(json.dumps({"prompt": prompt, "reply": reply, "secs": time.time() - t0}) + "\n")
        print(f"[{time.time() - t0:5.1f}s] {prompt[:60]}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
