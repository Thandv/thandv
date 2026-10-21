import pytest

from thandv import __version__
from thandv.cli import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_version_matches_pyproject_and_is_current(capsys):
    """pyproject.toml and thandv.__version__ carry the version in two places
    (a plain string, not package metadata); they must agree."""
    import re
    from pathlib import Path

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE).group(1)
    assert declared == __version__ == "0.10.0"
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == "thandv 0.10.0"


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


# --- train backends / enable ----------------------------------------------

def test_train_backends_lists_all(thandv_home, capsys):
    rc = main(["train", "backends"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mlx-lm" in out
    assert "hf-peft" in out
    assert "active:" in out


def test_train_enable_refuses_when_no_backend(thandv_home, monkeypatch, capsys):
    from thandv import training_backend as tb

    monkeypatch.setattr(tb, "pick_backend", lambda: None)
    rc = main(["train", "enable", "--hf-model", "x/y"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no training backend" in err.lower()
    assert "thandv[train-mlx]" in err


def test_train_enable_uses_default_hf_mapping(thandv_home, monkeypatch, capsys):
    """`train enable` without --hf-model maps config.model → HF repo id."""
    from thandv import training_backend as tb
    from thandv.config import Config

    Config(model="qwen2.5-coder:7b").save()

    captured = {}

    class FB:
        name = "fake"

        def is_available(self):
            return True

    monkeypatch.setattr(tb, "pick_backend", lambda: FB())

    def fake_fetch(model_id):
        captured["model_id"] = model_id
        d = tb.BASE_MODELS_DIR / model_id.replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        (d / "marker").write_text("x")
        return d

    monkeypatch.setattr(tb, "fetch_hf_base_model", fake_fetch)
    monkeypatch.setattr(tb, "verify_training_setup", lambda b, p: {"ok": True})

    rc = main(["train", "enable"])
    assert rc == 0
    assert captured["model_id"] == "Qwen/Qwen2.5-Coder-7B"


def test_train_enable_explicit_hf_model_wins(thandv_home, monkeypatch, capsys):
    from thandv import training_backend as tb

    captured = {}

    class FB:
        name = "fake"

        def is_available(self):
            return True

    monkeypatch.setattr(tb, "pick_backend", lambda: FB())

    def fake_fetch(model_id):
        captured["model_id"] = model_id
        d = tb.BASE_MODELS_DIR / model_id.replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        (d / "marker").write_text("x")
        return d

    monkeypatch.setattr(tb, "fetch_hf_base_model", fake_fetch)
    monkeypatch.setattr(tb, "verify_training_setup", lambda b, p: {"ok": True})

    rc = main(["train", "enable", "--hf-model", "my-org/my-model"])
    assert rc == 0
    assert captured["model_id"] == "my-org/my-model"


def test_train_enable_skip_verify_does_not_call_verifier(thandv_home, monkeypatch, capsys):
    from thandv import training_backend as tb

    class FB:
        name = "fake"

        def is_available(self):
            return True

    monkeypatch.setattr(tb, "pick_backend", lambda: FB())

    def fake_fetch(model_id):
        d = tb.BASE_MODELS_DIR / model_id.replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        (d / "marker").write_text("x")
        return d

    monkeypatch.setattr(tb, "fetch_hf_base_model", fake_fetch)

    def boom(*a, **kw):
        raise AssertionError("verify must NOT be called with --skip-verify")

    monkeypatch.setattr(tb, "verify_training_setup", boom)

    rc = main(["train", "enable", "--hf-model", "x/y", "--skip-verify"])
    assert rc == 0


def test_train_datasets_lists_registered(thandv_home, capsys):
    rc = main(["train", "datasets"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "codealpaca" in out
    assert "dolly" in out
    assert "persona=" in out


def test_train_sample_public_unknown_dataset(thandv_home, capsys):
    rc = main(["train", "sample-public", "no-such-dataset", "--n", "5"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown dataset" in err.lower()


def test_train_sample_public_writes_queue(thandv_home, monkeypatch, capsys):
    from thandv import training_data as td

    def fake_queue(name, n, *, seed=0):
        # Pretend we wrote a queue file.
        p = td.trainer.QUEUE_DIR / f"fake-{name}-n{n}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"prompt": "x", "completion": "y"}\n')
        return p

    monkeypatch.setattr(td, "queue_public_dataset", fake_queue)
    rc = main(["train", "sample-public", "codealpaca", "--n", "7"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "queued:" in out
    assert "codealpaca" in out
    assert "7 examples" in out


def test_distill_list_shows_teachers(thandv_home, capsys):
    rc = main(["distill", "--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "groq-llama-3.3-70b" in out
    assert "together-llama-3.3-70b" in out
    assert "GROQ_API_KEY" in out
    assert "ToS:" in out


def test_distill_refuses_anthropic(thandv_home, capsys):
    rc = main([
        "distill",
        "--teacher", "anthropic-claude",
        "--prompts-file", "/tmp/nope",  # never reached
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "refused" in err.lower()
    assert "anthropic" in err.lower()


def test_distill_needs_prompts_source(thandv_home, capsys):
    rc = main(["distill", "--teacher", "groq-llama-3.3-70b"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--prompts-file" in err or "--prompts-from" in err


def test_distill_prompts_file_unknown_path(thandv_home, capsys):
    rc = main([
        "distill",
        "--teacher", "groq-llama-3.3-70b",
        "--prompts-file", "/definitely/not/here.jsonl",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower()


def test_distill_happy_path_with_prompts_file(thandv_home, monkeypatch, tmp_path, capsys):
    from thandv import teachers as tch

    monkeypatch.setenv("GROQ_API_KEY", "x")

    pf = tmp_path / "prompts.jsonl"
    pf.write_text(
        '{"prompt": "p1"}\n'
        '{"prompt": "p2"}\n'
        'not json line\n'
        '{"no_prompt_field": true}\n'
    )

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "C"}}]}

    monkeypatch.setattr(tch.requests, "post", lambda *a, **kw: FakeResp())

    rc = main([
        "distill",
        "--teacher", "groq-llama-3.3-70b",
        "--prompts-file", str(pf),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # Only 2 valid prompts in the file (non-JSON and no-prompt-field skipped).
    assert "2 prompts attempted" in out
    assert "queued:" in out
    assert "provenance:" in out


def test_train_sessions_empty(thandv_home, capsys):
    rc = main(["train", "sessions"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no sessions" in out.lower()


def test_train_sessions_with_data(thandv_home, capsys):
    import json
    sp = thandv_home / "sessions" / "test1.jsonl"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(
        json.dumps({"role": "user", "content": "q"}) + "\n"
        + json.dumps({"role": "assistant", "content": "a"}) + "\n"
    )
    rc = main(["train", "sessions"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "test1.jsonl" in out
    assert "clean" in out.lower()


def test_train_promote_session_unknown_path(thandv_home, capsys):
    rc = main(["train", "promote-session", "/definitely/not/here.jsonl"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower()


def test_train_promote_session_empty_session_returns_1(thandv_home, capsys):
    import json
    sp = thandv_home / "sessions" / "empty.jsonl"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({"role": "user", "content": "q"}) + "\n")
    rc = main(["train", "promote-session", str(sp)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "nothing to promote" in err.lower()


def test_train_promote_session_happy_path(thandv_home, capsys):
    import json
    sp = thandv_home / "sessions" / "good.jsonl"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(
        json.dumps({"role": "user", "content": "q"}) + "\n"
        + json.dumps({"role": "assistant", "content": "a"}) + "\n"
    )
    rc = main(["train", "promote-session", str(sp)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "queued:" in out
    assert "1 pairs" in out


def test_train_promote_sessions_bulk(thandv_home, capsys):
    import json
    for name in ("a", "b"):
        sp = thandv_home / "sessions" / f"{name}.jsonl"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(
            json.dumps({"role": "user", "content": "q"}) + "\n"
            + json.dumps({"role": "assistant", "content": "a"}) + "\n"
        )
    rc = main(["train", "promote-sessions"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "promoted 2 sessions" in out
    assert "2 pairs total" in out


def test_distill_filtered_refuses_anthropic(thandv_home, capsys):
    rc = main([
        "distill-filtered",
        "--teacher", "anthropic-claude",
        "--suite", "smoke",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "refused" in err.lower()


def test_distill_filtered_unknown_suite(thandv_home, monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    rc = main([
        "distill-filtered",
        "--teacher", "groq-llama-3.3-70b",
        "--suite", "no-such-suite",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown suite" in err.lower()


def test_distill_filtered_happy_path(thandv_home, monkeypatch, capsys):
    from thandv import teachers as tch

    monkeypatch.setenv("GROQ_API_KEY", "x")

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "56"}}]}

    monkeypatch.setattr(tch.requests, "post", lambda *a, **kw: FakeResp())

    rc = main([
        "distill-filtered",
        "--teacher", "groq-llama-3.3-70b",
        "--suite", "smoke",
        "--n", "1",
        "--limit", "1",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "queued:" in out
    assert "yield:" in out
    assert "1/1" in out  # one attempt, one pass
    assert "100.0%" in out


def test_distill_happy_path_with_prompts_from_dataset(thandv_home, monkeypatch, capsys):
    from thandv import teachers as tch, training_data as td

    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(
        td, "sample_public_dataset",
        lambda name, n, seed=0: [
            {"prompt": f"prompt-{i}", "completion": f"ignored-{i}"}
            for i in range(n)
        ],
    )

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr(tch.requests, "post", lambda *a, **kw: FakeResp())

    rc = main([
        "distill",
        "--teacher", "groq-llama-3.3-70b",
        "--prompts-from", "codealpaca",
        "--n", "3",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "3 prompts attempted" in out


def test_train_sample_public_clear_error_without_datasets(thandv_home, monkeypatch, capsys):
    from thandv import training_data as td

    def boom(name, n, *, seed=0):
        raise RuntimeError("datasets not installed. Run: pip install thandv[eval]")

    monkeypatch.setattr(td, "queue_public_dataset", boom)
    rc = main(["train", "sample-public", "codealpaca", "--n", "5"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "thandv[eval]" in err


def test_train_enable_reports_fetch_failure_cleanly(thandv_home, monkeypatch, capsys):
    from thandv import training_backend as tb

    class FB:
        name = "fake"

        def is_available(self):
            return True

    monkeypatch.setattr(tb, "pick_backend", lambda: FB())

    def boom_fetch(model_id):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(tb, "fetch_hf_base_model", boom_fetch)
    rc = main(["train", "enable", "--hf-model", "x/y"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "download failed" in err.lower()
    assert "network unreachable" in err.lower()


# --- thandv writer --------------------------------------------------------

def test_writer_no_action_errors(thandv_home, capsys):
    rc = main(["writer"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "bundle-style-corpus" in err


def test_writer_bundle_refuses_without_embed_model(thandv_home, monkeypatch, capsys):
    from thandv import rag

    monkeypatch.setattr(rag, "embed_model_available", lambda: False)
    rc = main(["writer", "bundle-style-corpus"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "embed model" in err.lower()


def test_writer_bundle_happy_path(thandv_home, monkeypatch, capsys):
    from thandv import rag, style_corpus

    monkeypatch.setattr(rag, "embed_model_available", lambda: True)
    monkeypatch.setattr(rag, "embed", lambda text: [0.1] * rag.EMBED_DIM)

    rc = main(["writer", "bundle-style-corpus"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "writer corpus" in out
    chunks = rag.load_chunks("writer")
    assert len(chunks) >= len(style_corpus.STYLE_CORPUS)


def test_writer_bundle_clear_flag_keeps_count_stable(thandv_home, monkeypatch, capsys):
    """Two `--clear` runs should not double the corpus."""
    from thandv import rag

    monkeypatch.setattr(rag, "embed_model_available", lambda: True)
    monkeypatch.setattr(rag, "embed", lambda text: [0.1] * rag.EMBED_DIM)

    main(["writer", "bundle-style-corpus"])
    first = len(rag.load_chunks("writer"))
    main(["writer", "bundle-style-corpus", "--clear"])
    second = len(rag.load_chunks("writer"))
    assert second == first


# --- thandv finance ------------------------------------------------------

def test_finance_no_action_errors(thandv_home, capsys):
    rc = main(["finance"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "metrics" in err and "paper-trade" in err


def test_finance_metrics_happy_path(thandv_home, capsys, tmp_path):
    p = tmp_path / "px.csv"
    p.write_text("price\n100\n105\n102\n108\n110\n107\n")
    rc = main(["finance", "metrics", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sharpe" in out and "sortino" in out
    assert "max_drawdown" in out
    assert "not investment advice" in out.lower()


def test_finance_metrics_missing_column(thandv_home, capsys, tmp_path):
    p = tmp_path / "px.csv"
    p.write_text("close\n100\n105\n")
    rc = main(["finance", "metrics", str(p)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "missing column" in err


def test_finance_exposure_happy_path(thandv_home, capsys, tmp_path):
    p = tmp_path / "pos.csv"
    p.write_text(
        "symbol,qty,avg_price,asset_class\n"
        "AAPL,10,150,equity\n"
        "TLT,-5,90,bond\n"
    )
    rc = main(["finance", "exposure", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gross:" in out
    assert "AAPL" in out and "TLT" in out


def test_finance_backtest_happy_path(thandv_home, capsys, tmp_path):
    px = tmp_path / "px.csv"
    px.write_text("price\n100\n110\n121\n133.1\n")
    sig = tmp_path / "sig.csv"
    sig.write_text("signal\n1\n1\n1\n1\n")
    rc = main(["finance", "backtest", "--prices", str(px), "--signals", str(sig)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "total_return" in out and "n_trades" in out
    assert "strategy-critique" in out
    assert "not investment advice" in out.lower()


def test_finance_backtest_length_mismatch(thandv_home, capsys, tmp_path):
    px = tmp_path / "px.csv"
    px.write_text("price\n100\n110\n120\n")
    sig = tmp_path / "sig.csv"
    sig.write_text("signal\n1\n1\n")
    rc = main(["finance", "backtest", "--prices", str(px), "--signals", str(sig)])
    assert rc == 2


def test_finance_ingest_filing_happy_path(thandv_home, monkeypatch, capsys, tmp_path):
    from thandv import rag

    monkeypatch.setattr(rag, "embed_model_available", lambda: True)
    monkeypatch.setattr(rag, "embed", lambda text: [0.1] * rag.EMBED_DIM)

    f = tmp_path / "aapl-10k.txt"
    f.write_text("Item 1. Business\n\nApple designs phones.\n\n" * 3)
    rc = main([
        "finance", "ingest-filing", str(f),
        "--type", "10-K", "--ticker", "AAPL",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "AAPL" in out and "10-K" in out
    chunks = rag.load_chunks("finance")
    assert all(c.source.startswith("filing::10-K::AAPL::") for c in chunks)


def test_finance_ingest_filing_unknown_type(thandv_home, capsys, tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("content")
    rc = main(["finance", "ingest-filing", str(f), "--type", "fake-form"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown filing_type" in err


def test_finance_filings_lists_ingested(thandv_home, monkeypatch, capsys, tmp_path):
    from thandv import filings, rag

    monkeypatch.setattr(rag, "embed_model_available", lambda: True)
    monkeypatch.setattr(rag, "embed", lambda text: [0.1] * rag.EMBED_DIM)

    f = tmp_path / "msft.txt"
    f.write_text("Microsoft 10-K excerpt.\n\nAzure growth continues.\n\n" * 3)
    filings.ingest_filing(f, filing_type="10-K", ticker="MSFT")

    rc = main(["finance", "filings"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MSFT" in out and "10-K" in out


def test_finance_filings_empty_prints_marker(thandv_home, capsys):
    rc = main(["finance", "filings"])
    assert rc == 0
    assert "no filings" in capsys.readouterr().out.lower()


def test_finance_paper_trade_status_disabled(thandv_home, monkeypatch, capsys):
    from thandv import paper_trading as pt

    monkeypatch.setattr(pt, "ADAPTERS", {})
    rc = main(["finance", "paper-trade"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DISABLED" in out
    assert "(none)" in out
    assert "NOT_FINANCIAL_ADVICE" in out
    assert "register your own" in out.lower()


def _chat_ready(monkeypatch, turn_impl):
    from thandv import agent as agent_mod
    from thandv import cli

    monkeypatch.setattr(cli, "_check_ollama", lambda: True)
    monkeypatch.setattr(cli, "_ensure_model", lambda m: True)
    monkeypatch.setattr(agent_mod.Agent, "turn", turn_impl)


def test_chat_one_shot_reports_connection_error(thandv_home, monkeypatch, capsys):
    import requests

    def turn(self, prompt):
        yield "partial "
        raise requests.ConnectionError("connection refused")

    _chat_ready(monkeypatch, turn)
    rc = main(["chat", "--persona", "code", "hi"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "partial" in captured.out
    assert "connection refused" in captured.err
    assert "ollama" in captured.err.lower()


def test_chat_repl_survives_failed_turn(thandv_home, monkeypatch, capsys):
    calls = []

    def turn(self, prompt):
        calls.append(prompt)
        if prompt == "first":
            raise RuntimeError("ollama error: model runner has unexpectedly stopped")
        yield "second reply"

    _chat_ready(monkeypatch, turn)
    inputs = iter(["first", "second", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    rc = main(["chat", "--persona", "code"])
    assert rc == 0
    assert calls == ["first", "second"]
    captured = capsys.readouterr()
    assert "unexpectedly stopped" in captured.err
    assert "second reply" in captured.out


def test_chat_ctrl_c_interrupts_reply_not_session(thandv_home, monkeypatch, capsys):
    def turn(self, prompt):
        if prompt == "long":
            yield "start..."
            raise KeyboardInterrupt
        yield "ok"

    _chat_ready(monkeypatch, turn)
    inputs = iter(["long", "short", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    rc = main(["chat", "--persona", "code"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[interrupted]" in out
    assert "ok" in out


# --- Ctrl-C during a streaming turn -------------------------------------------
#
# No test here reaches a model or a shell: the turn is a fake generator, or a
# real Agent whose `_raw_stream` is replaced and whose tools are recorders.


def test_chat_one_shot_ctrl_c_exits_130_without_traceback(thandv_home, monkeypatch, capsys):
    def turn(self, prompt):
        yield "start..."
        raise KeyboardInterrupt

    _chat_ready(monkeypatch, turn)
    rc = main(["chat", "--persona", "code", "hi"])
    assert rc == 130
    captured = capsys.readouterr()
    assert "[interrupted]" in captured.out
    assert "Traceback" not in captured.out + captured.err


def test_stream_turn_closes_the_turn_when_ctrl_c_lands_between_chunks(monkeypatch, capsys):
    """Ctrl-C while a chunk is being printed leaves the generator suspended at
    a yield. It must be closed right away (running its cleanup), not whenever
    the last reference to it happens to go."""
    from thandv import cli

    cleaned = []
    made = []  # keeps a reference, as a caller holding the generator would

    def turn(prompt):
        try:
            yield "one"
            yield "two"
        finally:
            cleaned.append(prompt)

    class FakeAgent:
        def turn(self, prompt):
            gen = turn(prompt)
            made.append(gen)
            return gen

    def interrupt_on_print(text):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_for_terminal", interrupt_on_print)
    assert cli._stream_turn(FakeAgent(), "p") == 130
    assert cleaned == ["p"]
    assert "[interrupted]" in capsys.readouterr().out


def _streaming_agent(monkeypatch, tmp_path, replies, approve=None):
    """A real Agent whose model replies are scripted, one list of chunks per
    hop. `closed` records every model stream that was closed."""
    import subprocess

    from thandv import tools
    from thandv.agent import Agent
    from thandv.config import Config

    spawned = []

    def fake_run_with_timeout(args, **kwargs):
        spawned.append(args)
        return subprocess.CompletedProcess(args, 0, "fake-out\n", "")

    monkeypatch.setattr(tools, "run_with_timeout", fake_run_with_timeout)
    agent = Agent(config=Config(model="fake-model"), approve=approve, root=tmp_path.resolve())
    hops = iter(replies)
    closed = []
    streams = []  # held, so only an explicit close() closes a stream

    def chunks():
        try:
            yield from next(hops)
        finally:
            closed.append(True)

    def raw_stream():
        stream = chunks()
        streams.append(stream)
        return stream

    monkeypatch.setattr(agent, "_raw_stream", raw_stream)
    return agent, spawned, closed


def _session_events(agent):
    import json

    return [json.loads(line) for line in agent.session_path.read_text().splitlines()]


def _tool_block(name, args):
    import json

    return "```tool\n" + json.dumps({"name": name, "args": args}) + "\n```"


def test_ctrl_c_mid_reply_records_partial_reply_and_keeps_session_usable(
    thandv_home, monkeypatch, tmp_path, capsys
):
    from thandv import cli

    agent, _, closed = _streaming_agent(
        monkeypatch, tmp_path,
        [["Hello there, ", "this is a long ", "reply that never ends"], ["fine"]],
    )
    real = cli._for_terminal
    seen = []

    def interrupt_on_second_chunk(text):
        seen.append(text)
        if len(seen) == 2:
            raise KeyboardInterrupt
        return real(text)

    monkeypatch.setattr(cli, "_for_terminal", interrupt_on_second_chunk)
    assert cli._stream_turn(agent, "first") == 130
    assert closed == [True]  # the model stream (HTTP response) was closed
    # The partial reply is kept in the context so roles still alternate...
    assert agent.messages[-2:] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "Hello there, this is a long "},
    ]
    # ...but logged as an interruption, not as an assistant answer that the
    # session-promotion step would turn into training data.
    events = _session_events(agent)
    assert events[-1] == {"role": "interrupted", "partial": "Hello there, this is a long "}
    assert [e["role"] for e in events] == ["user", "interrupted"]

    monkeypatch.setattr(cli, "_for_terminal", real)
    assert cli._stream_turn(agent, "second") == 0
    assert agent.messages[-1] == {"role": "assistant", "content": "fine"}
    assert "fine" in capsys.readouterr().out


def test_interrupted_reply_is_not_promoted_to_training_data(thandv_home, monkeypatch, tmp_path):
    from thandv import cli, session_promotion

    agent, _, _ = _streaming_agent(monkeypatch, tmp_path, [["partial answer ", "more"]])

    def interrupt(text):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_for_terminal", interrupt)
    cli._stream_turn(agent, "question")
    events = _session_events(agent)
    assert events[-1] == {"role": "interrupted", "partial": "partial answer "}
    assert session_promotion.extract_pairs(agent.session_path) == []


class _Tty:
    def isatty(self):
        return True

    def fileno(self):
        raise OSError("not a real fd")


@pytest.mark.parametrize("where", ["preview", "flush", "input", "full-view"])
def test_ctrl_c_anywhere_in_the_approval_prompt_denies_and_ends_the_reply(
    thandv_home, monkeypatch, tmp_path, capsys, where
):
    from thandv import cli
    from thandv.agent import INTERRUPTED_NOT_RUN

    agent, spawned, _ = _streaming_agent(
        monkeypatch, tmp_path,
        [[_tool_block("run_bash", {"command": "echo hi"})], ["after"]],
        approve=cli._repl_approver,
    )
    monkeypatch.setattr("sys.stdin", _Tty())

    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    answers = iter(["f"])

    def full_view_then_ctrl_c(prompt=""):
        for answer in answers:
            return answer
        raise KeyboardInterrupt

    if where == "preview":
        monkeypatch.setattr(cli, "_approval_preview", boom)
    elif where == "flush":
        monkeypatch.setattr(cli, "_flush_typeahead", boom)
    elif where == "input":
        monkeypatch.setattr("builtins.input", boom)
    else:  # Ctrl-C at the second question, after asking for the full view
        monkeypatch.setattr(cli, "_approval_preview", lambda s, cols: (s.splitlines(), 5))
        monkeypatch.setattr("builtins.input", full_view_then_ctrl_c)
    assert cli._stream_turn(agent, "say hi") == 130
    assert spawned == []
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert tool_msgs == [
        {"role": "tool", "name": "run_bash", "content": '{"error": "%s"}' % INTERRUPTED_NOT_RUN}
    ]
    # The tool-call reply is in the history once, followed by its result.
    assert [m["role"] for m in agent.messages[-3:]] == ["user", "assistant", "tool"]
    events = _session_events(agent)
    assert events[-2:] == [
        {"role": "tool", "name": "run_bash", "result": {"error": INTERRUPTED_NOT_RUN}},
        {"role": "interrupted", "partial": ""},
    ]
    out = capsys.readouterr().out
    assert "[interrupted]" in out and "after" not in out


def test_ctrl_c_while_the_tool_line_prints_does_not_run_the_tool(
    thandv_home, monkeypatch, tmp_path
):
    from thandv import cli
    from thandv.agent import INTERRUPTED_NOT_RUN

    asked = []
    agent, spawned, _ = _streaming_agent(
        monkeypatch, tmp_path,
        [[_tool_block("run_bash", {"command": "echo hi"})], ["after"]],
        approve=lambda name, summary: asked.append(name) or True,
    )
    real = cli._for_terminal

    def interrupt_on_tool_line(text):
        if text.lstrip().startswith("[tool]"):
            raise KeyboardInterrupt
        return real(text)

    monkeypatch.setattr(cli, "_for_terminal", interrupt_on_tool_line)
    assert cli._stream_turn(agent, "say hi") == 130
    assert asked == [] and spawned == []
    assert agent.messages[-1]["content"] == '{"error": "%s"}' % INTERRUPTED_NOT_RUN


def test_ctrl_c_while_a_tool_runs_is_recorded_as_maybe_partly_run(
    thandv_home, monkeypatch, tmp_path
):
    from thandv import cli, tools
    from thandv.agent import INTERRUPTED_RUNNING

    agent, _, _ = _streaming_agent(
        monkeypatch, tmp_path,
        [[_tool_block("run_bash", {"command": "make"})], ["after"]],
        approve=lambda name, summary: True,
    )

    def interrupted_run(args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(tools, "run_with_timeout", interrupted_run)
    assert cli._stream_turn(agent, "build") == 130
    assert agent.messages[-1] == {
        "role": "tool", "name": "run_bash", "content": '{"error": "%s"}' % INTERRUPTED_RUNNING
    }


def test_ctrl_c_after_a_finished_tool_adds_no_second_result(thandv_home, monkeypatch, tmp_path):
    """The tool ran and its result is recorded; Ctrl-C in the next hop's
    stream must not add a bogus 'interrupted' result for it."""
    from thandv import cli

    agent, spawned, _ = _streaming_agent(
        monkeypatch, tmp_path,
        [[_tool_block("run_bash", {"command": "echo hi"})], ["Now the long ", "answer follows"]],
        approve=lambda name, summary: True,
    )
    real = cli._for_terminal

    def interrupt_on_answer(text):
        if "Now" in text:
            raise KeyboardInterrupt
        return real(text)

    monkeypatch.setattr(cli, "_for_terminal", interrupt_on_answer)
    assert cli._stream_turn(agent, "say hi") == 130
    assert spawned == ["echo hi"]
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1 and '"exit": 0' in tool_msgs[0]["content"]
    assert agent.messages[-1] == {"role": "assistant", "content": "Now the long "}


def test_record_interrupt_skips_a_result_already_in_history(thandv_home, tmp_path):
    from thandv.agent import Agent, _TurnProgress
    from thandv.config import Config

    agent = Agent(config=Config(model="fake-model"), root=tmp_path.resolve())
    agent.messages.append({"role": "tool", "name": "run_bash", "content": '{"exit": 0}'})
    agent._record_interrupt(_TurnProgress(call={"name": "run_bash", "args": {}}, dispatching=True))
    assert [m["role"] for m in agent.messages].count("tool") == 1
    assert _session_events(agent) == [{"role": "interrupted", "partial": ""}]


def test_ctrl_c_while_the_latch_goes_up_still_restores_the_handler(monkeypatch, capsys):
    import signal

    from thandv import cli

    real_signal = signal.signal

    def ctrl_c_right_after_install(signum, handler):
        previous = real_signal(signum, handler)
        if handler is not signal.default_int_handler:
            handler(signum, None)  # Ctrl-C before the turn starts
        return previous

    started = []

    class FakeAgent:
        def turn(self, prompt):
            started.append(prompt)
            yield "x"

    monkeypatch.setattr(cli.signal, "signal", ctrl_c_right_after_install)
    assert cli._stream_turn(FakeAgent(), "p") == 130
    assert started == []
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert "[interrupted]" in capsys.readouterr().out


def test_stream_turn_works_off_the_main_thread():
    """Signal handlers can only be set in the main thread; elsewhere the
    turn runs without the latch instead of failing."""
    import threading

    from thandv import cli

    class FakeAgent:
        def turn(self, prompt):
            yield "x"

    result = []
    worker = threading.Thread(target=lambda: result.append(cli._stream_turn(FakeAgent(), "p")))
    worker.start()
    worker.join()
    assert result == [0]


def test_ctrl_c_while_the_latch_comes_down_still_restores_the_handler(monkeypatch):
    """A Ctrl-C landing as a finished turn removes the latch is ignored. If it
    raised before the default handler was back, a fired latch would stay
    installed and Ctrl-C would be dead for the rest of the session."""
    import signal

    from thandv import cli

    real_getsignal = signal.getsignal
    finished = []

    def ctrl_c_during_removal(signum):
        current = real_getsignal(signum)
        if finished and callable(current) and current is not signal.default_int_handler:
            current(signum, None)  # Ctrl-C as the latch is checked for removal
        return current

    class FakeAgent:
        def turn(self, prompt):
            yield "x"
            finished.append(True)

    monkeypatch.setattr(cli.signal, "getsignal", ctrl_c_during_removal)
    try:
        assert cli._stream_turn(FakeAgent(), "p") == 0
        assert real_getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)


def test_latch_removal_leaves_a_handler_set_during_the_turn_alone():
    """The latch only removes its own handler, never one installed after it."""
    import signal

    from thandv import cli

    def own(signum, frame):
        pass

    class FakeAgent:
        def turn(self, prompt):
            signal.signal(signal.SIGINT, own)
            yield "x"

    try:
        assert cli._stream_turn(FakeAgent(), "p") == 0
        assert signal.getsignal(signal.SIGINT) is own
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)


def test_ctrl_c_while_the_user_message_is_logged_marks_the_turn_interrupted(
    thandv_home, monkeypatch, tmp_path
):
    """The user message is recorded inside the turn's try, so a Ctrl-C while
    it is written still leaves the interrupted marker in the session log."""
    from thandv import agent as agent_mod
    from thandv import cli

    agent, spawned, _ = _streaming_agent(monkeypatch, tmp_path, [["never streamed"]])
    real = agent_mod.append_event
    calls = []

    def interrupt_first(path, event):
        calls.append(event)
        if len(calls) == 1:
            raise KeyboardInterrupt
        real(path, event)

    monkeypatch.setattr(agent_mod, "append_event", interrupt_first)
    assert cli._stream_turn(agent, "question") == 130
    assert calls[0]["role"] == "user"
    assert _session_events(agent) == [{"role": "interrupted", "partial": ""}]
    assert spawned == []


def test_ctrl_c_at_approval_in_the_repl_returns_to_the_prompt(thandv_home, monkeypatch, capsys):
    from thandv import agent as agent_mod

    _chat_ready(monkeypatch, agent_mod.Agent.turn)
    hops = iter([[_tool_block("run_bash", {"command": "echo hi"})], ["second reply"]])
    monkeypatch.setattr(agent_mod.Agent, "_raw_stream", lambda self: iter(next(hops)))
    monkeypatch.setattr("sys.stdin", _Tty())
    lines = iter(["first", None, "second", "/exit"])  # None: Ctrl-C at y/N

    def fake_input(prompt=""):
        line = next(lines)
        if line is None:
            raise KeyboardInterrupt
        return line

    monkeypatch.setattr("builtins.input", fake_input)
    assert main(["chat", "--persona", "code"]) == 0
    out = capsys.readouterr().out
    assert "[interrupted]" in out and "second reply" in out


def test_second_ctrl_c_during_cleanup_is_ignored(monkeypatch, capsys):
    """A second Ctrl-C while the interrupted turn cleans up must not cut the
    cleanup short or escape as a traceback; the handler is restored after."""
    import os
    import signal

    from thandv import cli

    cleaned = []

    class FakeAgent:
        def turn(self, prompt):
            try:
                yield "partial"
                os.kill(os.getpid(), signal.SIGINT)  # first Ctrl-C
                yield "never"
            finally:
                os.kill(os.getpid(), signal.SIGINT)  # second Ctrl-C, mid-cleanup
                cleaned.append(True)

    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert cli._stream_turn(FakeAgent(), "p") == 130
    assert cleaned == [True]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert "[interrupted]" in capsys.readouterr().out


def test_stream_turn_leaves_an_ignored_sigint_alone(monkeypatch):
    """If Ctrl-C is ignored (e.g. started in the background), keep it that way."""
    import signal

    from thandv import cli

    seen = []

    class FakeAgent:
        def turn(self, prompt):
            seen.append(signal.getsignal(signal.SIGINT))
            yield "x"

    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        assert cli._stream_turn(FakeAgent(), "p") == 0
    finally:
        signal.signal(signal.SIGINT, previous)
    assert seen == [signal.SIG_IGN]


@pytest.mark.parametrize(
    "kv",
    [
        "temperature=hot",
        "max_tokens=lots",
        "temperature=nan",
        "auto_tools=ture",
        "finance_paper_trading_enabled=maybe",
        "persona=no-such-persona",
    ],
)
def test_config_set_rejects_bad_values_without_saving(thandv_home, capsys, kv):
    from thandv import config

    rc = main(["config", "--set", kv])
    assert rc == 2
    assert "bad value" in capsys.readouterr().err
    assert not config.CONFIG_PATH.exists()


def test_config_set_bool_spellings(thandv_home, capsys):
    from thandv.config import Config

    assert main(["config", "--set", "auto_tools=off"]) == 0
    assert Config.load().auto_tools is False
    assert main(["config", "--set", "auto_tools=YES"]) == 0
    assert Config.load().auto_tools is True


def test_train_enqueue_rejects_malformed_file(thandv_home, tmp_path, capsys):
    src = tmp_path / "bad.jsonl"
    src.write_text("[1, 2, 3]\n")
    rc = main(["train", "enqueue", str(src)])
    assert rc == 2
    assert "not queued" in capsys.readouterr().err


def test_ingest_reports_embedding_failure(thandv_home, monkeypatch, tmp_path, capsys):
    from thandv import cli, rag

    src = tmp_path / "doc.md"
    src.write_text("hello")
    monkeypatch.setattr(cli, "_check_ollama", lambda: True)
    monkeypatch.setattr(rag, "embed_model_available", lambda: True)

    def down(_text):
        raise ConnectionError("refused")

    monkeypatch.setattr(rag, "embed", down)
    rc = main(["ingest", str(src), "--persona", "code"])
    assert rc == 2
    assert "ingest failed" in capsys.readouterr().err
