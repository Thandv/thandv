"""Teacher distillation.

A "teacher" is a hosted LLM we call to generate `(prompt, completion)`
pairs that get written straight into the training queue. The local
trainer then optionally fine-tunes a LoRA on the result — gated by the
eval suite as always.

**ToS hard-exclusion.** Two providers are refused by name:

- **Anthropic** (Claude): the Commercial Terms forbid using outputs to
  "develop or train any AI model or service that competes with Claude
  or any Anthropic product."
- **OpenAI**: the Terms of Use forbid using outputs "to develop models
  that compete with OpenAI."

Both refusals are enforced at the registry layer — there's no `--teacher
claude` codepath to forget about. The whitelist is OpenAI-compatible
free-tier endpoints from providers whose terms do NOT carry that
restriction (Groq, Together AI, OpenRouter).

Provenance: every distilled queue file gets a sidecar
`<file>.provenance.json` recording the teacher, model id, ToS URL, and
timestamp. So if you ever need to audit *which* model generated *which*
training data, the answer is on disk.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from thandv import trainer


@dataclass(frozen=True)
class Teacher:
    """A hosted LLM endpoint we'll ask to generate completions."""
    name: str             # short id, e.g. "groq-llama-3.3-70b"
    provider: str         # e.g. "groq" / "together" / "openrouter"
    base_url: str         # OpenAI-compatible /v1 base
    api_key_env: str      # env var holding the API key
    model_id: str         # provider's name for the model
    persona_hint: str     # which thandv persona this best fits
    description: str
    free_tier_note: str   # current rate-limit / quota snapshot
    tos_url: str


# --- Hard-refused providers ----------------------------------------------
# These ship with NotImplementedError at the registry level, with the
# exact ToS clause and URL so future-you doesn't accidentally enable them.
REFUSED_PROVIDERS: dict[str, str] = {
    "anthropic": (
        "Anthropic's Commercial Terms forbid using outputs to develop or "
        "train any AI model that competes with Claude. Refused. "
        "https://www.anthropic.com/legal/commercial-terms"
    ),
    "openai": (
        "OpenAI's Terms of Use forbid using outputs to develop models that "
        "compete with OpenAI's services. Refused. "
        "https://openai.com/policies/terms-of-use"
    ),
    "claude": "(alias for anthropic)",
    "gpt": "(alias for openai)",
}


# --- Registered teachers --------------------------------------------------

GROQ_LLAMA_3_3_70B = Teacher(
    name="groq-llama-3.3-70b",
    provider="groq",
    base_url="https://api.groq.com/openai/v1",
    api_key_env="GROQ_API_KEY",
    model_id="llama-3.3-70b-versatile",
    persona_hint="code",
    description="Llama 3.3 70B Instruct served by Groq. Fast inference.",
    free_tier_note=(
        "Groq free tier: roughly 30 RPM / ~14400 tokens/min as of 2026-Q1. "
        "Check https://console.groq.com/settings/billing for current limits."
    ),
    tos_url="https://groq.com/terms-of-use/",
)

TOGETHER_LLAMA_3_3_70B = Teacher(
    name="together-llama-3.3-70b",
    provider="together",
    base_url="https://api.together.xyz/v1",
    api_key_env="TOGETHER_API_KEY",
    model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    persona_hint="code",
    description="Llama 3.3 70B Instruct (Turbo, free variant) on Together AI.",
    free_tier_note="Together free tier: 60 RPM as of 2026-Q1; check https://www.together.ai/pricing",
    tos_url="https://www.together.ai/terms-of-service",
)

OPENROUTER_LLAMA_3_3_70B = Teacher(
    name="openrouter-llama-3.3-70b",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY",
    model_id="meta-llama/llama-3.3-70b-instruct:free",
    persona_hint="code",
    description="Llama 3.3 70B Instruct (free variant) routed via OpenRouter.",
    free_tier_note="OpenRouter free: 20 RPM, 200 req/day as of 2026-Q1; see https://openrouter.ai/docs/limits",
    tos_url="https://openrouter.ai/terms",
)


TEACHERS: dict[str, Teacher] = {
    t.name: t for t in (GROQ_LLAMA_3_3_70B, TOGETHER_LLAMA_3_3_70B, OPENROUTER_LLAMA_3_3_70B)
}


def list_teachers() -> list[Teacher]:
    return list(TEACHERS.values())


def get_teacher(name: str) -> Teacher:
    """Resolve a teacher by name; raise with ToS context on refused providers."""
    lowered = name.lower()
    for refused, reason in REFUSED_PROVIDERS.items():
        if refused in lowered:
            raise ValueError(f"refused teacher {name!r}: {reason}")
    if name not in TEACHERS:
        raise ValueError(
            f"unknown teacher: {name!r}. Known: {sorted(TEACHERS)}"
        )
    return TEACHERS[name]


# --- HTTP call ------------------------------------------------------------

def call_teacher(
    teacher: Teacher,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: float = 120.0,
) -> str:
    """Send one prompt to the teacher; return the completion content.

    All teachers in `TEACHERS` are OpenAI-compatible — same request shape,
    same response path. The only thing that varies is the URL, the
    bearer key, and the model id.
    """
    key = os.environ.get(teacher.api_key_env)
    if not key:
        raise RuntimeError(
            f"missing API key for {teacher.name}: set ${teacher.api_key_env} "
            f"in the environment. ToS: {teacher.tos_url}"
        )
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    r = requests.post(
        f"{teacher.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": teacher.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# --- Distillation orchestration -----------------------------------------

def distill(
    teacher_name: str,
    prompts: list[str],
    *,
    system: str | None = None,
    out_path: Path | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> tuple[Path, Path]:
    """Generate completions for every prompt; return (queue_jsonl, provenance_json).

    The queue file lands in `trainer.QUEUE_DIR`, ready for `train tick`.
    A sidecar `<file>.provenance.json` records exactly which teacher
    generated which file, with ToS URL and timestamp. If the teacher
    errors on any prompt, that prompt is *skipped* and recorded in the
    provenance — partial output is more useful than no output.
    """
    teacher = get_teacher(teacher_name)
    trainer._ensure_dirs()
    if out_path is None:
        out_path = trainer.QUEUE_DIR / f"{int(time.time())}-distill-{teacher.name}.jsonl"

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    successes = 0
    failures: list[dict] = []
    with out_path.open("w") as f:
        for i, prompt in enumerate(prompts, 1):
            try:
                completion = call_teacher(
                    teacher,
                    prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
                successes += 1
                if on_progress:
                    on_progress(i, len(prompts), "ok")
            except Exception as e:
                failures.append({"index": i, "error": f"{type(e).__name__}: {e}"})
                if on_progress:
                    on_progress(i, len(prompts), f"fail: {type(e).__name__}")

    provenance = {
        "teacher": asdict(teacher),
        "n_prompts": len(prompts),
        "n_successes": successes,
        "n_failures": len(failures),
        "failures": failures[:20],  # truncate for readability
        "system": system,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "queue_file": out_path.name,
    }
    provenance_path = out_path.with_suffix(out_path.suffix + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2))
    return out_path, provenance_path
