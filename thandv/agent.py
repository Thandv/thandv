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
from thandv.tools import TOOL_SCHEMAS, TOOLS, dispatch

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
        # Side-channel populated by _raw_stream when the model emits
        # native (OpenAI-style) tool_calls. Drained and reset per hop.
        self._pending_tool_calls: list[dict] = []
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
        """Yield content chunks from Ollama's /api/chat NDJSON stream.

        Side effect: populates `self._pending_tool_calls` if the model
        emits native function calls. Ollama may stream tool_calls
        incrementally or in a single chunk; we always take the *last*
        non-empty value we see (it's the canonical aggregated form).
        """
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": self.messages,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if self.config.auto_tools:
            payload["tools"] = TOOL_SCHEMAS
        with requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
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
                msg = obj.get("message", {})
                tcs = msg.get("tool_calls")
                if tcs:
                    self._pending_tool_calls = tcs
                chunk = msg.get("content", "")
                if chunk:
                    yield chunk
                if obj.get("done"):
                    return

    # --- Tool parsing ------------------------------------------------------

    def _inject_context(self, call: dict) -> None:
        """Fill in tool args that depend on agent context.

        Stateless tool dispatch can't know which persona's corpus to search,
        so we inject it for the `retrieve` tool. The model can still override
        by passing `persona="all"` or a specific name explicitly.
        """
        if call["name"] == "retrieve":
            call["args"].setdefault("persona", self.persona.name if self.persona else "all")

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

    @staticmethod
    def _extract_inline_json_tool_call(text: str) -> dict | None:
        """Fallback for models (e.g. some qwen2.5 builds) that emit tool
        calls as raw JSON in `message.content` instead of using the native
        `message.tool_calls` field.

        Conservative: the trimmed content must be a single JSON object whose
        `name` is a registered tool. Otherwise treat as ordinary output.
        """
        if not text:
            return None
        candidate = text.strip()
        # Strip an optional ```json / ``` fence.
        fence = re.match(r"```(?:json)?\s*\n(.*)\n```\s*$", candidate, re.DOTALL)
        if fence:
            candidate = fence.group(1).strip()
        if not (candidate.startswith("{") and candidate.endswith("}")):
            return None
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        name = obj.get("name")
        if not isinstance(name, str) or name not in TOOLS:
            return None
        args = obj.get("arguments", obj.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return None
        if not isinstance(args, dict):
            return None
        return {"name": name, "args": args}

    @staticmethod
    def _normalise_native_call(raw: dict) -> dict | None:
        """Coerce an Ollama `tool_calls[i]` entry into our {name, args} shape.

        Different versions emit `arguments` either as a JSON-encoded string
        or as an already-decoded object. Reject malformed entries (missing
        name, unparseable args) by returning None — caller falls back.
        """
        fn = raw.get("function") if isinstance(raw, dict) else None
        if not isinstance(fn, dict):
            return None
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            return None
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return None
        if not isinstance(args, dict):
            return None
        return {"name": name, "args": args}

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

        last_call: dict | None = None

        for _ in range(MAX_TOOL_HOPS):
            full = ""
            pending = ""
            in_tool_block = False
            json_buffer_mode = False
            mode_decided = False
            self._pending_tool_calls = []

            for chunk in self._raw_stream():
                full += chunk
                if in_tool_block or json_buffer_mode:
                    continue

                if not mode_decided:
                    stripped = full.lstrip()
                    if not stripped:
                        continue  # only whitespace so far
                    if stripped.startswith("{"):
                        # Model is producing raw-JSON content. Buffer the
                        # whole reply; we'll decide at end-of-stream whether
                        # it's an inline tool call (suppress) or legitimate
                        # JSON output (yield once).
                        json_buffer_mode = True
                        continue
                    mode_decided = True
                    pending = full  # flush whatever leading whitespace existed
                else:
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

            if json_buffer_mode:
                # If the buffered content is a real inline tool call, suppress
                # it entirely (the [tool] line will appear below). If not,
                # deliver it as a single chunk so the user still sees it.
                if self._extract_inline_json_tool_call(full) is None:
                    yield full
            elif not in_tool_block and pending:
                yield pending

            self.messages.append({"role": "assistant", "content": full})
            append_event(self.session_path, {"role": "assistant", "content": full})

            # Prefer native tool_calls; then the explicit text protocol;
            # then inline-JSON-in-content (for smaller models that emit the
            # call as raw JSON without using the native field).
            call: dict | None = None
            for raw in self._pending_tool_calls:
                call = self._normalise_native_call(raw)
                if call is not None:
                    break
            if call is None:
                call = self._extract_tool_call(full)
            if call is None:
                call = self._extract_inline_json_tool_call(full)
            if call is None:
                return

            self._inject_context(call)

            # Guard: if the model emits the *same* tool call twice in a row
            # it's stuck in a loop (common on smaller models that don't
            # synthesise tool results into a final answer). Break out
            # rather than chew through the hop budget silently.
            if last_call is not None and call == last_call:
                yield "\n[stopped: model is repeating the same tool call]\n"
                return
            last_call = call

            yield f"\n[tool] {call['name']}({json.dumps(call['args'])[:120]})\n"
            result = dispatch(call["name"], call["args"])
            append_event(self.session_path, {"role": "tool", "name": call["name"], "result": result})
            # Native function-calling expects role="tool" with the result
            # as plain content. This is the OpenAI / Ollama contract and
            # makes the model treat the next turn as a continuation rather
            # than a fresh user request.
            self.messages.append(
                {"role": "tool", "name": call["name"], "content": json.dumps(result)[:8000]}
            )

        yield "[stopped: tool-call budget exhausted]\n"
