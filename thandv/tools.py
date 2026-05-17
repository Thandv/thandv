"""Minimal local tools the agent can invoke.

Schema is JSON-in / JSON-out; the agent loop is responsible for parsing the
model's tool-call block and dispatching here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from thandv import rag

MAX_READ_BYTES = 200_000
MAX_BASH_SECONDS = 30
MAX_RETRIEVE_K = 20


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def read_file(path: str) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        return {"error": f"not found: {p}"}
    if p.is_dir():
        return {"error": f"is a directory: {p}"}
    data = p.read_bytes()[:MAX_READ_BYTES]
    try:
        return {"path": str(p), "content": data.decode("utf-8")}
    except UnicodeDecodeError:
        return {"path": str(p), "content": f"<binary, {len(data)} bytes>"}


def write_file(path: str, content: str) -> dict[str, Any]:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"path": str(p), "bytes": len(content.encode("utf-8"))}


def edit_file(path: str, old: str, new: str) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        return {"error": f"not found: {p}"}
    text = p.read_text()
    if old not in text:
        return {"error": "old string not found"}
    if text.count(old) > 1:
        return {"error": "old string is not unique; include more context"}
    p.write_text(text.replace(old, new, 1))
    return {"path": str(p), "ok": True}


def list_dir(path: str = ".") -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        return {"error": f"not found: {p}"}
    if not p.is_dir():
        return {"error": f"not a directory: {p}"}
    entries = []
    for child in sorted(p.iterdir()):
        entries.append({"name": child.name, "type": "dir" if child.is_dir() else "file"})
    return {"path": str(p), "entries": entries}


_DANGEROUS = ("rm -rf", "mkfs", "dd if=", ":(){:|", "shutdown", "reboot")


def run_bash(command: str, confirm: bool = False) -> dict[str, Any]:
    lowered = command.lower()
    if any(token in lowered for token in _DANGEROUS) and not confirm:
        return {"error": "refused: command looks destructive; ask the user to confirm"}
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=MAX_BASH_SECONDS,
        )
        return {
            "exit": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {MAX_BASH_SECONDS}s"}


def retrieve(query: str, persona: str = "all", k: int = 5) -> dict[str, Any]:
    """Retrieve top-k chunks from the persona's corpus by cosine similarity."""
    if not isinstance(query, str) or not query.strip():
        return {"error": "query must be a non-empty string"}
    try:
        k = int(k)
    except (TypeError, ValueError):
        return {"error": "k must be an integer"}
    if k < 1 or k > MAX_RETRIEVE_K:
        return {"error": f"k must be in [1, {MAX_RETRIEVE_K}]"}
    try:
        results = rag.retrieve(query, persona=persona, k=k)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"persona": persona, "k": k, "results": results}


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_dir": list_dir,
    "run_bash": run_bash,
    "retrieve": retrieve,
}


# OpenAI / Ollama native function-calling schemas. Sent to the model via
# `/api/chat` `tools` parameter so it can emit structured tool_calls
# instead of the older text-based ```tool``` blocks. Names and arguments
# mirror the TOOLS registry above; keep the two in sync.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file from disk. Content is truncated to 200 KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Overwrites if it exists; creates parent dirs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace a single occurrence of `old` with `new` in the file at `path`. "
                "Errors if `old` is missing or appears more than once — include enough "
                "surrounding context to make it unique."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List entries of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a shell command with a 30-second timeout. Captures stdout and "
                "stderr (tail-truncated). Refuses destructive commands (rm -rf, "
                "mkfs, shutdown, etc.) unless `confirm=true`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "confirm": {
                        "type": "boolean",
                        "description": "Set true to allow destructive commands. Default false.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Look up the top-k most relevant chunks from your active persona's "
                "local corpus by cosine similarity. The runtime fills in `persona` "
                "with the active persona name; you can override with `\"all\"` to "
                "search across every corpus, or a specific persona name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "description": "1 to 20. Default 5."},
                    "persona": {
                        "type": "string",
                        "description": "Defaults to the active persona; set to \"all\" or another persona name to override.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def _validate_schemas() -> None:
    """Tripwire: every tool in TOOLS has a schema, and vice versa."""
    impl_names = set(TOOLS)
    schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    missing_schema = impl_names - schema_names
    missing_impl = schema_names - impl_names
    if missing_schema:
        raise RuntimeError(f"tools missing schemas: {sorted(missing_schema)}")
    if missing_impl:
        raise RuntimeError(f"schemas missing implementations: {sorted(missing_impl)}")


_validate_schemas()


def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"bad args for {name}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
