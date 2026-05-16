import pytest

from thandv import __version__
from thandv.cli import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_help_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "thandv" in out.lower()
    assert "chat" in out
    assert "doctor" in out


def test_config_show_default(thandv_home, capsys):
    rc = main(["config"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "model=" in out
    assert "backend=ollama" in out


def test_config_set_persists(thandv_home, capsys):
    rc = main(["config", "--set", "model=qwen3:8b", "--set", "temperature=0.7"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "model=qwen3:8b" in out
    assert "temperature=0.7" in out

    capsys.readouterr()
    main(["config"])
    out2 = capsys.readouterr().out
    assert "model=qwen3:8b" in out2


def test_config_set_bad_key(thandv_home, capsys):
    rc = main(["config", "--set", "nonsense=1"])
    assert rc == 2


def test_config_set_bad_format(thandv_home, capsys):
    rc = main(["config", "--set", "no-equals-sign"])
    assert rc == 2


def test_chat_errors_when_ollama_down(thandv_home, monkeypatch, capsys):
    from thandv import cli

    monkeypatch.setattr(cli, "_check_ollama", lambda: False)
    rc = main(["chat", "hi"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "ollama" in err.lower()
