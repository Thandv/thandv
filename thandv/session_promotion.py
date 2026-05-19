"""Session promotion: chat history → training queue.

The agent already records every chat session as JSONL under
`~/.thandv/sessions/<unix-ns>.jsonl`. Each event is one of:

  {"role": "user",      "content": "..."}              # user typed
  {"role": "assistant", "content": "...streamed..."}   # model's full reply
  {"role": "tool",      "name": "...", "result": {...}}# tool dispatch result

Session promotion walks one of these files and turns it into training
data: every adjacent (user → assistant) pair becomes a `(prompt,
completion)` example, with the assistant's tool-block bookkeeping
stripped out (we want to train on the *answer*, not the reasoning trail).

Two filters available:
- `session_is_clean(path)` — True iff no tool result has an `error` key.
  This is the auto-promotion gate: only sessions that worked end-to-end
  contribute to the training set.
- Manual override — the user can promote any session explicitly via
  `thandv train promote-session <path>`.

Why this matters: the model is being trained on **how the user actually
uses it**. Distillation (v0.5) trains toward a hosted teacher's style;
verifier-filtered (v0.6) trains toward an automated check; session
promotion (this) trains toward the user's own accepted outcomes. The
three sources are complementary.

Honest scope: we strip tool blocks from the assistant content, so the
training pair becomes (user prompt) → (clean final answer). This trains
direct-answer behaviour. If the user wants the model to learn
*agentic* multi-turn tool use, a richer format (multi-message training)
is a later milestone.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from thandv import trainer
from thandv.config import SESSIONS_DIR


TOOL_BLOCK_RE = re.compile(r"```tool\s*\n.*?\n```", re.DOTALL)


@dataclass
class PromotionStats:
    session_path: str
    n_events: int
    n_user_turns: int
    n_assistant_turns: int
    n_pairs_extracted: int
    n_tool_calls: int
    n_tool_errors: int
    is_clean: bool


def _events(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _strip_tool_blocks(text: str) -> str:
    """Remove ```tool ... ``` fences. The training data should be the
    answer the user actually sees, not the agent's bookkeeping."""
    return TOOL_BLOCK_RE.sub("", text).strip()


def extract_pairs(session_path: Path) -> list[dict]:
    """Return a list of `{"prompt", "completion"}` extracted from one
    session. Each pair is one direct (user → assistant) exchange.

    Rules:
    - Tool events are skipped entirely.
    - Assistant turns whose entire content is tool-block bookkeeping are
      dropped (those are "let me look that up" beats, not final answers).
    - A user turn is consumed by the *next* assistant turn; the same user
      message doesn't pair to multiple assistant turns.
    - The very-first event being assistant is ignored.
    """
    events = _events(session_path)
    pairs: list[dict] = []
    last_user: str | None = None

    for ev in events:
        role = ev.get("role")
        if role == "user":
            content = ev.get("content")
            # The agent sometimes appends a "tool-result" pseudo-user event
            # (older protocol) or `role=tool` event (current). Either way,
            # treat actual user content (no tool-result wrapper) as input.
            if isinstance(content, str) and not content.startswith("```tool-result"):
                last_user = content
        elif role == "assistant":
            if last_user is None:
                continue
            cleaned = _strip_tool_blocks(ev.get("content", ""))
            if cleaned:
                pairs.append({"prompt": last_user, "completion": cleaned})
            last_user = None  # consumed
        # role == "tool" → skip; intermediate state only.

    return pairs


def session_is_clean(session_path: Path) -> bool:
    """A session is clean iff no recorded tool result has an `error` key.

    This is conservative: a single tool error anywhere in the session
    means we don't auto-promote it. The user can still promote it
    manually with `thandv train promote-session <path>`.
    """
    for ev in _events(session_path):
        if ev.get("role") == "tool":
            result = ev.get("result")
            if isinstance(result, dict) and "error" in result:
                return False
    return True


def stats(session_path: Path) -> PromotionStats:
    """Per-session metrics used for the promotion CLI's table output."""
    events = _events(session_path)
    n_user = sum(1 for e in events if e.get("role") == "user")
    n_assistant = sum(1 for e in events if e.get("role") == "assistant")
    n_tool = sum(1 for e in events if e.get("role") == "tool")
    n_tool_err = sum(
        1 for e in events
        if e.get("role") == "tool"
        and isinstance(e.get("result"), dict)
        and "error" in e["result"]
    )
    return PromotionStats(
        session_path=str(session_path),
        n_events=len(events),
        n_user_turns=n_user,
        n_assistant_turns=n_assistant,
        n_pairs_extracted=len(extract_pairs(session_path)),
        n_tool_calls=n_tool,
        n_tool_errors=n_tool_err,
        is_clean=n_tool_err == 0,
    )


def promote_session(
    session_path: Path,
    *,
    out_path: Path | None = None,
) -> tuple[Path, dict] | None:
    """Extract pairs from one session and write them to the queue.

    Returns ``(queue_path, provenance_dict)`` on success, or None if the
    session produced zero pairs (nothing to promote).
    """
    pairs = extract_pairs(session_path)
    if not pairs:
        return None

    trainer._ensure_dirs()
    if out_path is None:
        out_path = (
            trainer.QUEUE_DIR
            / f"{int(time.time())}-session-{session_path.stem}.jsonl"
        )

    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    provenance = {
        "source": "session_promotion",
        "session_path": str(session_path),
        "n_pairs": len(pairs),
        "stats": asdict(stats(session_path)),
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "queue_file": out_path.name,
    }
    prov_path = out_path.with_suffix(out_path.suffix + ".provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2))
    return out_path, provenance


def list_sessions(sessions_dir: Path | None = None) -> list[Path]:
    """Sessions ordered newest-first by mtime."""
    sd = sessions_dir or SESSIONS_DIR
    if not sd.exists():
        return []
    return sorted(sd.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def promote_clean_sessions(
    *,
    sessions_dir: Path | None = None,
    max_sessions: int | None = None,
) -> list[tuple[Path, dict]]:
    """Promote every clean session that hasn't been promoted yet.

    "Hasn't been promoted yet" is tracked by the existence of a
    `<session>.promoted` marker file in the sessions dir. This avoids
    double-promotion on subsequent runs.
    """
    sd = sessions_dir or SESSIONS_DIR
    promoted: list[tuple[Path, dict]] = []
    for session_path in list_sessions(sd):
        marker = session_path.with_suffix(session_path.suffix + ".promoted")
        if marker.exists():
            continue
        if not session_is_clean(session_path):
            continue
        result = promote_session(session_path)
        if result is None:
            # Nothing to promote (empty session); still mark to avoid retrying.
            marker.write_text("empty\n")
            continue
        queue_path, provenance = result
        marker.write_text(f"{queue_path.name}\n")
        promoted.append((queue_path, provenance))
        if max_sessions is not None and len(promoted) >= max_sessions:
            break
    return promoted
