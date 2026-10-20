"""Minimal local tools the agent can invoke.

Schema is JSON-in / JSON-out; the agent loop is responsible for parsing the
model's tool-call block and dispatching here.

Security model (see docs/ARCHITECTURE.md, "Human gate"):

- `run_bash`, `write_file` and `edit_file` have side effects. The agent asks
  the person before dispatching them (`GATED_TOOLS`); nothing in a tool's
  arguments can skip that question.
- `write_file` and `edit_file` only touch regular files whose resolved path
  (symlinks followed) lies under the session's working-directory root.
- `dispatch` rejects arguments that are not in the tool's schema, so a model
  cannot pass hidden keywords such as `root`.
"""

from __future__ import annotations

import difflib
import os
import signal
import stat
import subprocess
from pathlib import Path
from typing import Any, Callable

from thandv import rag

MAX_READ_BYTES = 200_000
MAX_BASH_SECONDS = 30
MAX_RETRIEVE_K = 20
# Commands longer than this are refused: the person must be able to read
# the whole thing in the approval prompt.
MAX_COMMAND_CHARS = 4000
MAX_COMMAND_LINES = 40
# Larger file changes are refused rather than approved half-read.
MAX_PREVIEW_DIFF_LINES = 2000

# Tools that run without asking the person. Everything else in TOOLS has a
# side effect and must pass the approval gate in the agent loop.
UNGATED_TOOLS = frozenset({"read_file", "list_dir", "retrieve"})
GATED_TOOLS = frozenset({"run_bash", "write_file", "edit_file"})


class Refused(Exception):
    """A tool refused to act; the message is returned to the model."""


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def confine(path: str, root: Path) -> Path:
    """Resolve `path` (relative paths are taken from `root`) and require the
    result to lie under `root`. Symlinks, `..` and `~` are resolved first, so
    a link inside the root that points out of it is refused."""
    if not isinstance(path, str) or not path:
        raise Refused("path must be a non-empty string")
    if path.endswith(("/", os.sep)):
        raise Refused(f"refused: {path!r} names a directory, not a file")
    try:
        root = Path(root).resolve()
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = root / p
        resolved = p.resolve()
    except (ValueError, OSError, RuntimeError) as e:
        raise Refused(f"refused: bad path {path!r}: {e}") from e
    if not resolved.is_relative_to(root):
        raise Refused(f"refused: {resolved} is outside the working directory {root}")
    return resolved


def _open_regular(p: Path, flags: int, root: Path) -> int:
    """Open the confined path `p` without following symlinks and require a
    regular file with a single link (a hard link could alias a file outside
    the root). Never truncates: callers do that after these checks pass.

    Where the OS has O_NOFOLLOW_ANY (macOS) no path component may be a
    symlink. Elsewhere only the last component is checked by the open, so
    the path is re-confined afterwards and must still name the opened file;
    a parent swapped for a symlink in between is refused before any write.
    """
    # O_NOFOLLOW_ANY implies O_NOFOLLOW; macOS rejects the two together.
    flags |= os.O_NONBLOCK | (getattr(os, "O_NOFOLLOW_ANY", 0) or os.O_NOFOLLOW)
    try:
        fd = os.open(p, flags, 0o666)
    except OSError as e:
        raise Refused(f"refused: cannot open {p} safely: {e.strerror}") from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise Refused(f"refused: {p} is not a regular file")
        if st.st_nlink > 1:
            raise Refused(f"refused: {p} has more than one hard link")
        if confine(str(p), root) != p or not os.path.samestat(st, os.stat(p)):
            raise Refused(f"refused: {p} changed while it was being opened")
    except BaseException:
        os.close(fd)
        raise
    return fd


def read_file(path: str) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        return {"error": f"not found: {p}"}
    if p.is_dir():
        return {"error": f"is a directory: {p}"}
    # Non-blocking open + regular-file check: a FIFO or /dev/tty would
    # otherwise hang the turn or read the person's next typed line.
    try:
        fd = os.open(p, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        return {"error": f"cannot open {p}: {e.strerror}"}
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        return {"error": f"not a regular file: {p}"}
    with os.fdopen(fd, "rb") as f:
        # Read one byte past the cap so we know whether we truncated
        # without loading arbitrarily large files into memory.
        data = f.read(MAX_READ_BYTES + 1)
    truncated = len(data) > MAX_READ_BYTES
    data = data[:MAX_READ_BYTES]
    try:
        content = _decode_utf8_prefix(data, truncated)
    except UnicodeDecodeError:
        return {"path": str(p), "content": f"<binary, {len(data)} bytes>"}
    out: dict[str, Any] = {"path": str(p), "content": content}
    if truncated:
        out["truncated"] = True
    return out


def _decode_utf8_prefix(data: bytes, truncated: bool) -> str:
    """Decode `data` as UTF-8, tolerating a multi-byte character split by
    the truncation cap.

    A UTF-8 character is at most 4 bytes, so when the cap cut the file we
    retry without up to 3 trailing bytes before concluding the file is
    binary. Untruncated input must decode cleanly.
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        if not truncated:
            raise
        for cut in range(1, 4):
            try:
                return data[:-cut].decode("utf-8")
            except UnicodeDecodeError:
                continue
        raise


def write_file(path: str, content: str, *, root: Path) -> dict[str, Any]:
    try:
        p = confine(path, root)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Re-check after mkdir: the parent chain now exists and resolves.
        p = confine(str(p), root)
        fd = _open_regular(p, os.O_WRONLY | os.O_CREAT, root)
    except Refused as e:
        return {"error": str(e)}
    data = content.encode("utf-8")
    with os.fdopen(fd, "wb") as f:
        f.truncate(0)
        f.write(data)
    return {"path": str(p), "bytes": len(data)}


def _apply_edit(text: str, old: str, new: str) -> str:
    """Return `text` with the single occurrence of `old` replaced by `new`,
    or raise Refused with the reason it cannot be applied."""
    if not old:
        raise Refused("old string must be non-empty; use write_file to create or overwrite")
    if old not in text and "\r\n" in text:
        # The model almost always sends LF-only snippets. Match them
        # against a CRLF file by converting the snippet's endings.
        old_crlf = _to_crlf(old)
        if old_crlf in text:
            old, new = old_crlf, _to_crlf(new)
    if old not in text:
        raise Refused("old string not found")
    if text.count(old) > 1:
        raise Refused("old string is not unique; include more context")
    return text.replace(old, new, 1)


def edit_file(path: str, old: str, new: str, *, root: Path) -> dict[str, Any]:
    try:
        p = confine(path, root)
        if not p.exists():
            return {"error": f"not found: {p}"}
        if p.is_dir():
            return {"error": f"is a directory: {p}"}
        fd = _open_regular(p, os.O_RDWR, root)
    except Refused as e:
        return {"error": str(e)}
    # newline="" disables universal-newline translation in both directions,
    # so a CRLF file stays CRLF after the edit instead of being silently
    # rewritten with LF endings on every line.
    with os.fdopen(fd, "r+", encoding="utf-8", newline="") as f:
        text = f.read()
        try:
            updated = _apply_edit(text, old, new)
        except Refused as e:
            return {"error": str(e)}
        f.seek(0)
        f.truncate(0)
        f.write(updated)
    return {"path": str(p), "ok": True}


def _to_crlf(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\n", "\r\n")


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


def run_with_timeout(
    args: str | list[str],
    *,
    timeout: float,
    shell: bool = False,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess:
    """`subprocess.run` with a timeout that actually bounds the whole job.

    Plain `subprocess.run(timeout=...)` only kills the direct child. With
    `shell=True` (or any program that spawns helpers) the grandchildren
    survive the timeout, keep running unattended, and can hold the output
    pipes open so collecting output blocks. Here the child leads its own
    process group (POSIX) and the entire group is killed on timeout.

    stdin is /dev/null: the command must never read from the user's
    terminal (an interactive `python`, `cat`, or a commit editor would
    otherwise steal the REPL's input).

    Raises `subprocess.TimeoutExpired` on timeout, like `subprocess.run`.
    """
    posix = os.name == "posix"
    proc = subprocess.Popen(
        args,
        shell=shell,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=posix,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc, posix)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise
    except BaseException:
        # Ctrl-C: the job leads its own session, so the terminal's SIGINT
        # never reached it or its helpers. Kill the group here.
        _kill_group(proc, posix)
        proc.wait()
        raise
    return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)


def _kill_group(proc: subprocess.Popen, posix: bool) -> None:
    if posix:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    proc.kill()


# Second layer only. This substring denylist is NOT a security boundary:
# it is trivially bypassed (quoting, variables, `find -delete`, ...). The
# boundary is the human approval gate in the agent loop; this only catches
# an obviously destructive command the person approved by mistake, and
# nothing the model sends can switch it off.
_DANGEROUS = ("rm -rf", "mkfs", "dd if=", ":(){:|", "shutdown", "reboot")


def run_bash(command: str, *, root: Path) -> dict[str, Any]:
    err = _check_command(command)
    if err:
        return {"error": err}
    lowered = command.lower()
    if any(token in lowered for token in _DANGEROUS):
        return {
            "error": "refused: command matches the destructive-command denylist; "
            "the user can run it themselves"
        }
    try:
        # shell=True is the point of this tool: the person approved this
        # exact string, shown in full, before it reaches /bin/sh.
        proc = run_with_timeout(command, timeout=MAX_BASH_SECONDS, shell=True, cwd=Path(root))
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
            "description": (
                "Read a UTF-8 file from disk. Content is truncated to 200 KB; "
                "the result carries `truncated: true` when that happens."
            ),
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
            "description": (
                "Write content to a file under the working directory. Overwrites if "
                "it exists; creates parent dirs. The user is asked to approve it."
            ),
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
                "Replace a single occurrence of `old` with `new` in the file at `path` "
                "(under the working directory). Errors if `old` is missing or appears "
                "more than once — include enough surrounding context to make it "
                "unique. The user is asked to approve it."
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
                "Run a shell command in the working directory with a 30-second "
                "timeout. The user is asked to approve every command. Captures "
                "stdout and stderr (tail-truncated)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
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


_SCHEMA_PROPS: dict[str, dict[str, Any]] = {
    sch["function"]["name"]: sch["function"]["parameters"]["properties"] for sch in TOOL_SCHEMAS
}


def _validate_schemas() -> None:
    """Tripwire: every tool in TOOLS has a schema, and vice versa; every tool
    is classified as gated or ungated."""
    impl_names = set(TOOLS)
    schema_names = set(_SCHEMA_PROPS)
    missing_schema = impl_names - schema_names
    missing_impl = schema_names - impl_names
    if missing_schema:
        raise RuntimeError(f"tools missing schemas: {sorted(missing_schema)}")
    if missing_impl:
        raise RuntimeError(f"schemas missing implementations: {sorted(missing_impl)}")
    if GATED_TOOLS & UNGATED_TOOLS or (GATED_TOOLS | UNGATED_TOOLS) != impl_names:
        raise RuntimeError("every tool must be in exactly one of GATED_TOOLS / UNGATED_TOOLS")


_validate_schemas()


def check_args(name: str, args: Any) -> str | None:
    """Return an error message if `args` is not acceptable for tool `name`.

    Only schema properties are allowed (so hidden keywords such as `root` or
    `force` are bad arguments), and string properties must be strings.
    """
    props = _SCHEMA_PROPS.get(name)
    if props is None:
        return f"unknown tool: {name}"
    if not isinstance(args, dict):
        return f"bad args for {name}: arguments must be an object"
    unknown = sorted(str(k) for k in args if k not in props)
    if unknown:
        return f"bad args for {name}: unexpected argument(s) {unknown}"
    for key, spec in props.items():
        if spec.get("type") == "string" and key in args and not isinstance(args[key], str):
            return f"bad args for {name}: {key} must be a string"
    return None


def printable(text: str, *, ascii_only: bool = False) -> str:
    """Escape everything that is not a plainly printable character (newline
    and tab excepted) so what the person sees is what will run: C0/C1
    controls and ANSI escapes, format characters such as bidi overrides,
    lone surrogates (which a UTF-8-mode stdout would emit as raw bytes),
    unassigned code points and non-ASCII spaces. With `ascii_only`, every
    non-ASCII character is escaped too (used for shell commands, where a
    look-alike character can change what runs)."""
    out = []
    for ch in text:
        if ch in "\n\t" or (ch.isprintable() and (ch.isascii() or not ascii_only)):
            out.append(ch)
        else:
            out.append(ch.encode("unicode_escape", "backslashreplace").decode("ascii"))
    return "".join(out)


def _check_command(command: Any) -> str | None:
    if not isinstance(command, str) or not command.strip():
        return "command must be a non-empty string"
    if len(command) > MAX_COMMAND_CHARS or command.count("\n") >= MAX_COMMAND_LINES:
        return (
            f"refused: command is longer than {MAX_COMMAND_CHARS} characters or "
            f"{MAX_COMMAND_LINES} lines; write a script with write_file and run that"
        )
    return None


# Files that make something run later (hooks, build/test config, shell
# start-up). Flagged in the approval prompt; not refused.
# Compared case-insensitively (macOS file systems usually are).
_AUTORUN_NAMES = frozenset({
    "conftest.py", "setup.py", "setup.cfg", "noxfile.py", "tox.ini", "makefile",
    "gnumakefile", "justfile", ".envrc", ".bashrc", ".bash_profile", ".zshrc",
    ".zshenv", ".zprofile", ".profile", "package.json", "pyproject.toml",
    "tasks.json", "launch.json", ".pre-commit-config.yaml", "sitecustomize.py",
    "usercustomize.py", ".mcp.json", "build.rs",
})
# .claude/.cursor hold agent hooks and MCP server commands.
_AUTORUN_DIRS = frozenset({".git", ".github", ".vscode", ".idea", ".husky", ".claude", ".cursor"})


def _diff_summary(path: Path, root: Path, before: str, after: str) -> str:
    lines = list(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(), "before", "after", lineterm=""
        )
    )
    if len(lines) > MAX_PREVIEW_DIFF_LINES:
        raise Refused(
            f"refused: change is larger than {MAX_PREVIEW_DIFF_LINES} diff lines; "
            "split it into smaller edits"
        )
    added = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
    head = [f"{path}  (+{added} -{removed} lines)"]
    rel_parts = {part.casefold() for part in path.relative_to(Path(root).resolve()).parts}
    name = path.name.casefold()
    if rel_parts & _AUTORUN_DIRS or name in _AUTORUN_NAMES or name.endswith(".pth") or (
        path.exists() and os.access(path, os.X_OK)
    ):
        head.append("WARNING: this file can make code run later (hook, build/test config, "
                    "executable or start-up file)")
    return printable("\n".join([*head, *lines]))


def _read_for_preview(p: Path) -> str:
    if not p.is_file():
        raise Refused(f"refused: {p} is not a regular file")
    with p.open(encoding="utf-8", errors="replace", newline="") as f:
        return f.read()


def describe(name: str, args: dict[str, Any], root: Path) -> dict[str, Any]:
    """Build the text the person approves for a gated call.

    Returns {"summary": str}, or {"error": str} when the call would be
    refused anyway (bad args, path outside root) and there is nothing to ask.
    """
    err = check_args(name, args)
    if err:
        return {"error": err}
    try:
        if name == "run_bash":
            command = args.get("command")
            err = _check_command(command)
            if err:
                return {"error": err}
            escaped = printable(command, ascii_only=True)
            shown = "\n".join(f"  | {ln}" for ln in escaped.split("\n"))
            return {"summary": f"run_bash in {Path(root).resolve()}:\n{shown}"}
        if name == "write_file":
            content = args.get("content")
            if not isinstance(content, str):
                return {"error": "bad args for write_file: content is required"}
            p = confine(args.get("path", ""), root)
            if p.exists():
                before = _read_for_preview(p)
                return {"summary": "write_file (overwrite):\n" + _diff_summary(p, root, before, content)}
            return {"summary": "write_file (new file):\n" + _diff_summary(p, root, "", content)}
        if name == "edit_file":
            old, new = args.get("old"), args.get("new")
            if not isinstance(old, str) or not isinstance(new, str):
                return {"error": "bad args for edit_file: old and new are required"}
            p = confine(args.get("path", ""), root)
            if not p.exists():
                return {"error": f"not found: {p}"}
            before = _read_for_preview(p)
            after = _apply_edit(before, old, new)
            return {"summary": "edit_file:\n" + _diff_summary(p, root, before, after)}
    except Refused as e:
        return {"error": str(e)}
    except OSError as e:
        # An unreadable target (permissions, a vanished file) is refused
        # here; raising would end the turn and crash the REPL.
        return {"error": f"refused: cannot preview {name}: {e.strerror or e}"}
    return {"error": f"{name} is not a gated tool"}


def dispatch(name: str, args: dict[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    """Run tool `name`. Does NOT ask the person: callers must gate
    GATED_TOOLS first (see Agent._gate). Gated tools need the session root
    and fail closed without it."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    err = check_args(name, args)
    if err:
        return {"error": err}
    kwargs = dict(args)
    if name in GATED_TOOLS:
        if root is None:
            return {"error": f"{name} needs a working-directory root"}
        kwargs["root"] = root
    try:
        return fn(**kwargs)
    except TypeError as e:
        return {"error": f"bad args for {name}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
