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
