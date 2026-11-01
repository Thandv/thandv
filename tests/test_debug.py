"""THANDV_DEBUG=1 prints debug lines to stderr; off prints nothing extra;
secrets are redacted. The teacher API is mocked; no network."""

from __future__ import annotations

import logging

import pytest
import requests

from thandv import debug, runtime, teachers as tch
from thandv.tools import run_with_timeout

SENTINEL = "PROMPT-SENTINEL-7f3a"
KEY = "sk-test-not-a-real-key-123"


@pytest.fixture(autouse=True)
def _switch(monkeypatch):
    """Start every test with the switch unset (even if the shell exports it)
    and drop the handler afterwards."""
    monkeypatch.delenv("THANDV_DEBUG", raising=False)
    debug.setup_debug()
    yield
    monkeypatch.setenv("THANDV_DEBUG", "0")
    debug.setup_debug()


class _Resp:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "reply"}}]}


def _call_teacher(monkeypatch, post):
    teacher = tch.GROQ_LLAMA_3_3_70B
    monkeypatch.setenv(teacher.api_key_env, KEY)
    monkeypatch.setattr(tch.requests, "post", post)
    return tch.call_teacher(teacher, SENTINEL)


def test_off_prints_nothing(monkeypatch, capsys):
    monkeypatch.setenv("THANDV_DEBUG", "0")
    debug.setup_debug()
    assert _call_teacher(monkeypatch, lambda *a, **k: _Resp()) == "reply"
    run_with_timeout(["true"], timeout=10)
    assert capsys.readouterr().err == ""


def test_on_logs_teacher_call_without_prompt_or_key(monkeypatch, capsys):
    monkeypatch.setenv("THANDV_DEBUG", "1")
    debug.setup_debug()
    _call_teacher(monkeypatch, lambda *a, **k: _Resp())
    err = capsys.readouterr().err
    assert "[thandv debug] thandv.teachers teacher call start teacher=" in err
    assert "teacher call end status=ok" in err
    assert SENTINEL not in err and KEY not in err


def test_failed_call_logs_end_line(monkeypatch, capsys):
    monkeypatch.setenv("THANDV_DEBUG", "1")
    debug.setup_debug()

    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    with pytest.raises(requests.ConnectionError):
        _call_teacher(monkeypatch, boom)
    err = capsys.readouterr().err
    assert "teacher call end status=error error=ConnectionError" in err
    assert SENTINEL not in err


def test_subprocess_logs_exit_not_output(monkeypatch, capsys):
    monkeypatch.setenv("THANDV_DEBUG", "1")
    debug.setup_debug()
    proc = run_with_timeout(["echo", SENTINEL], timeout=10)
    assert SENTINEL in proc.stdout
    err = capsys.readouterr().err
    assert "subprocess start argv0=echo args=1" in err
    assert "subprocess end status=exit rc=0 argv0=echo args=1" in err
    assert SENTINEL not in err


def test_swallowed_error_is_logged(monkeypatch, capsys):
    monkeypatch.setenv("THANDV_DEBUG", "1")
    debug.setup_debug()

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(runtime.subprocess, "check_output", missing)
    assert runtime._nvidia_vram_gb() == 0
    assert "nvidia-smi unavailable error=FileNotFoundError" in capsys.readouterr().err


def test_secrets_are_redacted(monkeypatch, capsys):
    monkeypatch.setenv("THANDV_DEBUG", "1")
    debug.setup_debug()
    for shape in ("key=v1secret", '{"api_key": "v2secret"}',
                  "Authorization: Bearer v3secret", "sk-v4secret"):
        logging.getLogger("thandv.test").debug("x %s", shape)
    err = capsys.readouterr().err
    assert err.count("[REDACTED]") == 4
    for s in ("v1secret", "v2secret", "v3secret", "v4secret"):
        assert s not in err
