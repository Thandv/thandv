"""Agent loop: stream chat from the local model, parse tool calls, feed results back."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import requests

from thandv.config import OLLAMA_HOST, Config
from thandv.memory import append_event, load_memory, load_skills, new_session_path
from thandv.personas import Persona, get_persona
from thandv.tools import dispatch

TOOL_BLOCK_RE = re.compile(r"```tool\s*\n(.*?)\n```", re.DOTALL)
TOOL_OPEN_MARKER = "```tool\n"
MAX_TOOL_HOPS = 8


@dataclass
class Agent:
    config: Config
    persona: Persona | None = None
    messages: list[dict] = field(default_factory=list)
    session_path: Path = field(default_factory=new_session_path)

    def __post_init__(self) -> None:
        if self.persona is None:
            self.persona = get_persona(self.config.persona)
        if not self.messages:
            system = self.persona.system_prompt
            skills = load_skills(only=self.persona.skills)
            memory = load_memory()
            if skills:
                system += "\n\n## Skills\n" + skills
            if memory:
                system += "\n\n## Memory\n" + memory
            self.messages.append({"role": "system", "content": system})

    # --- Ollama streaming I/O ----------------------------------------------

    def _raw_stream(self) -> Iterator[str]:
        """Yield content chunks from Ollama's /api/chat NDJSON stream."""
        with requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": self.config.model,
                "messages": self.messages,
                "stream": True,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
            stream=True,
            timeout=600,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = obj.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if obj.get("done"):
                    return

    # --- Tool parsing ------------------------------------------------------

    @staticmethod
    def _extract_tool_call(text: str) -> dict | None:
        m = TOOL_BLOCK_RE.search(text)
        if not m:
            return None
        try:
            call = json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
        if not isinstance(call, dict) or "name" not in call:
            return None
        call.setdefault("args", {})
        return call

    # --- Public turn API ---------------------------------------------------

    def turn(self, user_input: str) -> Iterator[str]:
        """Run one user turn; yields output chunks incrementally.

        Tool-block JSON is hidden from the stream — the user sees a one-line
        `[tool] <name>(<args>)` summary instead. A small tail buffer ensures
        the `TOOL_OPEN_MARKER` is detected even when it straddles two chunks.
        """
        self.messages.append({"role": "user", "content": user_input})
        append_event(self.session_path, {"role": "user", "content": user_input})

        hold = len(TOOL_OPEN_MARKER) - 1  # bytes to keep buffered for marker detection

        for _ in range(MAX_TOOL_HOPS):
            full = ""
            pending = ""
            in_tool_block = False

            for chunk in self._raw_stream():
                full += chunk
                if in_tool_block:
                    continue
                pending += chunk
                idx = pending.find(TOOL_OPEN_MARKER)
                if idx >= 0:
                    if idx > 0:
                        yield pending[:idx]
                    pending = ""
                    in_tool_block = True
                    continue
                if len(pending) > hold:
                    yield pending[:-hold]
                    pending = pending[-hold:]

            if not in_tool_block and pending:
                yield pending

            self.messages.append({"role": "assistant", "content": full})
            append_event(self.session_path, {"role": "assistant", "content": full})

            call = self._extract_tool_call(full)
            if call is None:
                return

            yield f"\n[tool] {call['name']}({json.dumps(call['args'])[:120]})\n"
            result = dispatch(call["name"], call["args"])
            append_event(self.session_path, {"role": "tool", "name": call["name"], "result": result})
            self.messages.append(
                {"role": "user", "content": f"```tool-result\n{json.dumps(result)[:8000]}\n```"}
            )

        yield "[stopped: tool-call budget exhausted]\n"
