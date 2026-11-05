"""Exit codes of `python -m thandv`, and a session log that cannot be written.

No test here reaches a model: Ollama is pointed at a closed port, or the
Agent's `_raw_stream` is replaced.
"""

import errno
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_python_m_thandv_keeps_the_exit_code(tmp_path):
    """`python -m thandv` (and the PyInstaller binary built from __main__.py)
    must exit with main()'s status, not always 0."""
    env = dict(os.environ)
    env.update(
        OLLAMA_HOST="http://127.0.0.1:9",  # nothing listens on the discard port
        THANDV_HOME=str(tmp_path / "home"),
        HOME=str(tmp_path),
        PYTHONPATH=str(REPO),
    )
    env.pop("THANDV_DEBUG", None)
    r = subprocess.run(
        [sys.executable, "-m", "thandv", "chat", "hi"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert "ollama is not running" in r.stderr
    assert r.returncode == 2


def _no_space(path, event):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_full_disk_session_log_does_not_end_the_repl(thandv_home, monkeypatch, capsys):
    from tests.test_cli import _chat_ready, _Tty
    from thandv import agent as agent_mod
    from thandv.cli import main

    _chat_ready(monkeypatch, agent_mod.Agent.turn)
    replies = iter([["first reply"], ["second reply"]])
    monkeypatch.setattr(agent_mod.Agent, "_raw_stream", lambda self: iter(next(replies)))
    monkeypatch.setattr(agent_mod, "append_event", _no_space)
    monkeypatch.setattr("sys.stdin", _Tty())
    lines = iter(["one", "two", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))

    assert main(["chat", "--persona", "code"]) == 0
    captured = capsys.readouterr()
    assert "first reply" in captured.out and "second reply" in captured.out
    assert captured.err.count("could not write the session log") == 1
    assert "Traceback" not in captured.out + captured.err


def test_full_disk_session_log_one_shot_still_answers(thandv_home, monkeypatch, capsys):
    from tests.test_cli import _chat_ready
    from thandv import agent as agent_mod
    from thandv.cli import main

    _chat_ready(monkeypatch, agent_mod.Agent.turn)
    monkeypatch.setattr(agent_mod.Agent, "_raw_stream", lambda self: iter(["the answer"]))
    monkeypatch.setattr(agent_mod, "append_event", _no_space)

    assert main(["chat", "--persona", "code", "hi"]) == 0
    captured = capsys.readouterr()
    assert "the answer" in captured.out
    assert captured.err.count("could not write the session log") == 1
