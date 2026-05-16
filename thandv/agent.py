"""Agent loop: chat with the local model, parse tool calls, feed results back."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import requests

from thandv.config import OLLAMA_HOST, Config
from thandv.memory import append_event, load_memory, load_skills, new_session_path
from thandv.prompts import SYSTEM_PROMPT
from thandv.tools import dispatch

TOOL_BLOCK_RE = re.compile(r"```tool\s*\n(.*?)\n```", re.DOTALL)
MAX_TOOL_HOPS = 8


@dataclass
class Agent:
    config: Config
    messages: list[dict] = field(default_factory=list)
    session_path: Path = field(default_factory=new_session_path)

    def __post_init__(self) -> None:
        if not self.messages:
            system = SYSTEM_PROMPT
            skills = load_skills()
            memory = load_memory()
            if skills:
                system += "\n\n## Skills\n" + skills
            if memory:
                system += "\n\n## Memory\n" + memory
            self.messages.append({"role": "system", "content": system})

    # --- Ollama I/O --------------------------------------------------------

    def _chat_once(self) -> str:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": self.config.model,
                "messages": self.messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

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
        """Run one user turn; yields strings to print incrementally."""
        self.messages.append({"role": "user", "content": user_input})
        append_event(self.session_path, {"role": "user", "content": user_input})

        for _ in range(MAX_TOOL_HOPS):
            reply = self._chat_once()
            self.messages.append({"role": "assistant", "content": reply})
            append_event(self.session_path, {"role": "assistant", "content": reply})

            call = self._extract_tool_call(reply)
            if call is None:
                yield reply
                return

            visible = TOOL_BLOCK_RE.sub("", reply).strip()
            if visible:
                yield visible + "\n"
            yield f"[tool] {call['name']}({json.dumps(call['args'])[:120]})\n"

            result = dispatch(call["name"], call["args"])
            append_event(self.session_path, {"role": "tool", "name": call["name"], "result": result})
            self.messages.append(
                {"role": "user", "content": f"```tool-result\n{json.dumps(result)[:8000]}\n```"}
            )

        yield "[stopped: tool-call budget exhausted]\n"
