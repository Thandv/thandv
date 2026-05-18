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
    assert "eval" in out


def test_eval_list(thandv_home, capsys):
    rc = main(["eval", "--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "smoke" in out
    assert "code" in out
    assert "writer" in out
    assert "finance" in out


def test_chat_rejects_unknown_persona(thandv_home, monkeypatch, capsys):
    from thandv import cli

    monkeypatch.setattr(cli, "_check_ollama", lambda: True)
    monkeypatch.setattr(cli, "_ensure_model", lambda m: True)
    rc = main(["chat", "--persona", "astrology", "hi"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown persona" in err.lower()


def test_eval_rejects_unknown_persona(thandv_home, monkeypatch, capsys):
    from thandv import cli

    monkeypatch.setattr(cli, "_check_ollama", lambda: True)
    monkeypatch.setattr(cli, "_ensure_model", lambda m: True)
    rc = main(["eval", "smoke", "--persona", "astrology"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown persona" in err.lower()


def test_eval_uses_suite_default_persona_not_config(thandv_home, monkeypatch, capsys):
    """Regression: even when config.persona is "code", `thandv eval finance`
    must run against the finance persona — that's the suite's default and
    the whole point of having a finance suite."""
    from thandv import cli, evals as eval_mod
    from thandv.config import Config

    # Set config persona to "code" — this MUST NOT leak into the eval.
    Config(model="fake-model", persona="code").save()
    monkeypatch.setattr(cli, "_check_ollama", lambda: True)
    monkeypatch.setattr(cli, "_ensure_model", lambda m: True)

    captured: dict = {}

    def fake_run_suite(suite_name, model, persona_name=None, **kwargs):
        captured["persona_name"] = persona_name
        return []

    monkeypatch.setattr(eval_mod, "run_suite", fake_run_suite)
    # The CLI imports run_suite directly; patch there too.
    monkeypatch.setattr(cli, "run_suite", fake_run_suite)

    rc = main(["eval", "finance"])
    assert rc == 0
    # persona_name must be None so run_suite falls through to the suite's
    # default_persona ("finance").
    assert captured["persona_name"] is None
    # And the status line printed should show "finance", not "code".
    assert "persona=finance" in capsys.readouterr().out


def test_eval_show_best_empty(thandv_home, capsys):
    rc = main(["eval", "--show-best"])
    assert rc == 0
    assert "no best records yet" in capsys.readouterr().out


def test_eval_show_best_with_records(thandv_home, capsys):
    from thandv import evals as eval_mod
    eval_mod.save_best_records({
        "smoke|code": {
            "pass_rate": 0.85, "n_passed": 85, "n_total": 100,
            "at": "2026-05-18T00:00:00+00:00", "model": "fake-7b", "persona": "code",
        },
    })
    rc = main(["eval", "--show-best"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "smoke|code" in out
    assert "85.0%" in out
    assert "fake-7b" in out


def test_eval_full_run_updates_best(thandv_home, monkeypatch, capsys):
    """A full (no --limit) eval run records its result as a new best."""
    from thandv import cli, evals as eval_mod
    from thandv.config import Config

    Config(model="fake-model").save()
    monkeypatch.setattr(cli, "_check_ollama", lambda: True)
    monkeypatch.setattr(cli, "_ensure_model", lambda m: True)

    fake_results = [
        eval_mod.EvalResult(task_id="t", suite="smoke", model="fake-model",
                            persona="code", reply="ok", passed=True, secs=0.1),
    ]
    monkeypatch.setattr(cli, "run_suite", lambda *a, **kw: fake_results)
    monkeypatch.setattr(eval_mod, "run_suite", lambda *a, **kw: fake_results)

    rc = main(["eval", "smoke"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "first recorded run" in out
    assert eval_mod.get_best("smoke", "code") is not None


def test_eval_limit_run_does_not_update_best(thandv_home, monkeypatch, capsys):
    """A --limit run is a sample and must NOT touch the best record."""
    from thandv import cli, evals as eval_mod
    from thandv.config import Config

    Config(model="fake-model").save()
    monkeypatch.setattr(cli, "_check_ollama", lambda: True)
    monkeypatch.setattr(cli, "_ensure_model", lambda m: True)

    fake_results = [
        eval_mod.EvalResult(task_id="t", suite="smoke", model="fake-model",
                            persona="code", reply="ok", passed=True, secs=0.1),
    ]
    monkeypatch.setattr(cli, "run_suite", lambda *a, **kw: fake_results)
    monkeypatch.setattr(eval_mod, "run_suite", lambda *a, **kw: fake_results)

    rc = main(["eval", "smoke", "--limit", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "first recorded run" not in out
    assert "NEW BEST" not in out
    assert eval_mod.get_best("smoke", "code") is None


def test_eval_no_update_best_flag_disables_recording(thandv_home, monkeypatch, capsys):
    from thandv import cli, evals as eval_mod
    from thandv.config import Config

    Config(model="fake-model").save()
    monkeypatch.setattr(cli, "_check_ollama", lambda: True)
    monkeypatch.setattr(cli, "_ensure_model", lambda m: True)

    fake_results = [
        eval_mod.EvalResult(task_id="t", suite="smoke", model="fake-model",
                            persona="code", reply="ok", passed=True, secs=0.1),
    ]
    monkeypatch.setattr(cli, "run_suite", lambda *a, **kw: fake_results)
    monkeypatch.setattr(eval_mod, "run_suite", lambda *a, **kw: fake_results)

    rc = main(["eval", "smoke", "--no-update-best"])
    assert rc == 0
    assert eval_mod.get_best("smoke", "code") is None


def test_eval_explicit_persona_still_overrides(thandv_home, monkeypatch, capsys):
    """`thandv eval finance --persona code` should still pass `code` through."""
    from thandv import cli, evals as eval_mod
    from thandv.config import Config

    Config(model="fake-model").save()
    monkeypatch.setattr(cli, "_check_ollama", lambda: True)
    monkeypatch.setattr(cli, "_ensure_model", lambda m: True)

    captured: dict = {}

    def fake_run_suite(suite_name, model, persona_name=None, **kwargs):
        captured["persona_name"] = persona_name
        return []

    monkeypatch.setattr(eval_mod, "run_suite", fake_run_suite)
    monkeypatch.setattr(cli, "run_suite", fake_run_suite)

    rc = main(["eval", "finance", "--persona", "code"])
    assert rc == 0
    assert captured["persona_name"] == "code"


# --- ingest --------------------------------------------------------------

def test_ingest_stats_empty(thandv_home, capsys):
    rc = main(["ingest", "--stats"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "total_chunks" in out


def test_ingest_clear_requires_persona(thandv_home, capsys):
    rc = main(["ingest", "--clear"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--clear requires --persona" in err


def test_ingest_no_args_errors(thandv_home, capsys):
    rc = main(["ingest"])
    assert rc == 2


def test_ingest_unknown_path(thandv_home, capsys):
    rc = main(["ingest", "/definitely/not/here"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err


def test_ingest_unknown_persona(thandv_home, capsys):
    rc = main(["ingest", "/tmp", "--persona", "astrology"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown persona" in err.lower()


def test_ingest_requires_embed_model(thandv_home, monkeypatch, tmp_path, capsys):
    from thandv import cli, rag

    src = tmp_path / "doc.md"
    src.write_text("hello")
    monkeypatch.setattr(cli, "_check_ollama", lambda: True)
    monkeypatch.setattr(rag, "embed_model_available", lambda: False)
    rc = main(["ingest", str(src), "--persona", "code"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "embedding model" in err.lower()


def test_ingest_happy_path(thandv_home, monkeypatch, tmp_path, capsys, fake_embed):
    from thandv import cli, rag

    src = tmp_path / "doc.md"
    src.write_text("hello world\n\nsecond para")
    monkeypatch.setattr(cli, "_check_ollama", lambda: True)
    monkeypatch.setattr(rag, "embed_model_available", lambda: True)
    rc = main(["ingest", str(src), "--persona", "code"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "files=1" in out
    assert "chunks=" in out


def test_config_show_default(thandv_home, capsys):
    rc = main(["config"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "model=" in out
    assert "backend=ollama" in out
    assert "persona=code" in out


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
