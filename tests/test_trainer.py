from __future__ import annotations

import json


from thandv import trainer
from thandv.trainer import (
    TrainerState,
    enqueue_examples,
    enqueue_path,
    load_state,
    pause,
    queue_size,
    resume,
    save_state,
    tick,
)


# --- State persistence ----------------------------------------------------

def test_load_state_default(thandv_home):
    s = load_state()
    assert isinstance(s, TrainerState)
    assert s.ticks_completed == 0
    assert s.paused is False
    assert s.baseline_measured is False


def test_save_load_roundtrip(thandv_home):
    s = TrainerState(ticks_completed=7, paused=True, best_eval_pass_rate=0.42)
    save_state(s)
    loaded = load_state()
    assert loaded.ticks_completed == 7
    assert loaded.paused is True
    assert loaded.best_eval_pass_rate == 0.42


def test_pause_resume(thandv_home):
    pause()
    assert load_state().paused is True
    resume()
    assert load_state().paused is False


# --- Queue operations -----------------------------------------------------

def test_enqueue_examples_writes_jsonl(thandv_home):
    p = enqueue_examples(
        "demo",
        [
            {"prompt": "hi", "completion": "hello"},
            {"prompt": "bye", "completion": "goodbye"},
        ],
    )
    assert p.exists() and p.suffix == ".jsonl"
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["prompt"] == "hi"
    assert queue_size() == 1


def test_enqueue_sanitises_name(thandv_home):
    p = enqueue_examples("../weird name!?", [{"prompt": "a", "completion": "b"}])
    assert "../" not in p.name
    assert "!" not in p.name


def test_enqueue_path(thandv_home, tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text('{"prompt": "x", "completion": "y"}\n')
    p = enqueue_path(src)
    assert p.exists()
    assert p.read_text() == '{"prompt": "x", "completion": "y"}\n'
    assert queue_size() == 1


def test_queue_size_after_multiple(thandv_home):
    enqueue_examples("a", [{"prompt": "p", "completion": "c"}])
    enqueue_examples("b", [{"prompt": "p", "completion": "c"}])
    assert queue_size() == 2


# --- Tick cycle ------------------------------------------------------------

def test_tick_first_call_measures_baseline(thandv_home, monkeypatch):
    monkeypatch.setattr(trainer, "_run_eval", lambda model, suite=None: 0.5)
    out = tick("fake-model")
    assert out["action"] == "baseline"
    assert out["pass_rate"] == 0.5
    assert out["persona"] == "code"
    s = load_state()
    assert s.baseline_measured is True
    assert s.best_by_persona["code"] == 0.5
    assert s.active_models_by_persona["code"] == "fake-model"


def test_tick_skip_when_baseline_eval_fails(thandv_home, monkeypatch):
    def boom(model, suite=None):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(trainer, "_run_eval", boom)
    out = tick("fake-model")
    assert out["action"] == "skip"
    assert "baseline eval failed" in out["reason"]
    # Persona's best slot must NOT exist after a failed baseline so the
    # next tick re-attempts it.
    assert "code" not in load_state().best_by_persona


def test_tick_skip_when_queue_empty(thandv_home, monkeypatch):
    # First tick establishes baseline.
    monkeypatch.setattr(trainer, "_run_eval", lambda m, suite=None: 0.5)
    tick("fake-model")
    # Second tick has nothing to do.
    out = tick("fake-model")
    assert out["action"] == "skip"
    assert out["reason"] == "queue empty"


def _fake_train_lora(adapter_id_seq=("adapter-1", "adapter-2")):
    """Build a fake _train_lora that returns canned ids and creates the
    adapter dir on disk (so cleanup-on-discard has something to remove).
    """
    iterator = iter(adapter_id_seq)

    def fake(queue_file, base_ollama_model, *, hf_base_repo=None):
        aid = next(iterator)
        d = trainer.ADAPTERS_DIR / aid
        d.mkdir(parents=True, exist_ok=True)
        (d / "merged.gguf").write_text("FAKE_GGUF")
        return aid, f"thandv-{aid}"

    return fake


def test_tick_promotes_when_improved(thandv_home, monkeypatch):
    enqueue_examples("a", [{"prompt": "p", "completion": "c"}])
    rates = iter([0.5, 0.9])  # baseline then post-train
    monkeypatch.setattr(trainer, "_run_eval", lambda m, suite=None: next(rates))
    monkeypatch.setattr(trainer, "_train_lora", _fake_train_lora())
    # Stub the ollama-rm in _discard_adapter — promote path may rm a prior
    # active model, which is None here.
    monkeypatch.setattr(trainer, "_discard_adapter", lambda *a, **kw: None)

    tick("fake-model")  # baseline
    out = tick("fake-model")
    assert out["action"] == "promote"
    assert out["pass_rate"] == 0.9
    assert out["ollama_model"] == "thandv-adapter-1"
    assert out["persona"] == "code"
    s = load_state()
    assert s.best_by_persona["code"] == 0.9
    assert s.active_adapter == "adapter-1"
    assert s.active_models_by_persona["code"] == "thandv-adapter-1"
    # Adapter dir still on disk (we didn't discard).
    assert (trainer.ADAPTERS_DIR / "adapter-1").exists()
    assert queue_size() == 0
    assert any(trainer.PROCESSED_DIR.iterdir())


def test_tick_discards_when_not_improved(thandv_home, monkeypatch):
    enqueue_examples("a", [{"prompt": "p", "completion": "c"}])
    rates = iter([0.9, 0.5])
    monkeypatch.setattr(trainer, "_run_eval", lambda m, suite=None: next(rates))
    monkeypatch.setattr(trainer, "_train_lora", _fake_train_lora())

    discarded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        trainer, "_discard_adapter",
        lambda aid, ollama_model: discarded.append((aid, ollama_model)),
    )

    tick("fake-model")  # baseline at 0.9
    out = tick("fake-model")
    assert out["action"] == "discard"
    s = load_state()
    assert s.best_by_persona["code"] == 0.9  # unchanged
    assert s.active_adapter == ""
    # Baseline locked the persona's active model to the base.
    assert s.active_models_by_persona["code"] == "fake-model"
    # The failed adapter was cleaned up via _discard_adapter.
    assert ("adapter-1", "thandv-adapter-1") in discarded
    # Queue file is still consumed (moved to processed) so we don't loop.
    assert queue_size() == 0


def test_tick_skip_when_post_train_eval_fails(thandv_home, monkeypatch):
    enqueue_examples("a", [{"prompt": "p", "completion": "c"}])
    calls = {"n": 0}

    def flaky(model, suite=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return 0.5  # baseline ok
        raise RuntimeError("ollama died mid-tick")

    monkeypatch.setattr(trainer, "_run_eval", flaky)
    monkeypatch.setattr(trainer, "_train_lora", _fake_train_lora())

    discarded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        trainer, "_discard_adapter",
        lambda aid, ollama_model: discarded.append((aid, ollama_model)),
    )

    tick("fake-model")  # baseline
    out = tick("fake-model")
    assert out["action"] == "skip"
    assert "eval failed" in out["reason"]
    # Queue file should NOT have been consumed (so we'll retry).
    assert queue_size() == 1
    # The trained adapter was cleaned up via _discard_adapter.
    assert ("adapter-1", "thandv-adapter-1") in discarded


def test_tick_skip_when_train_fails(thandv_home, monkeypatch):
    """If _train_lora itself raises (mlx OOM, disk full, etc.), the tick
    skips, the queue file stays in place, no adapter artifacts left."""
    enqueue_examples("a", [{"prompt": "p", "completion": "c"}])
    monkeypatch.setattr(trainer, "_run_eval", lambda m, suite=None: 0.5)

    def boom_train(*args, **kwargs):
        raise RuntimeError("MLX OOM mid-train")

    monkeypatch.setattr(trainer, "_train_lora", boom_train)

    tick("fake-model")  # baseline
    out = tick("fake-model")
    assert out["action"] == "skip"
    assert "train failed" in out["reason"]
    assert "MLX OOM" in out["reason"]
    # Queue file untouched.
    assert queue_size() == 1
    # No adapter dirs left.
    assert not any(trainer.ADAPTERS_DIR.iterdir())


def test_tick_dry_run_doesnt_train(thandv_home, monkeypatch):
    enqueue_examples("a", [{"prompt": "p", "completion": "c"}])
    monkeypatch.setattr(trainer, "_run_eval", lambda m, suite=None: 0.5)

    def must_not_train(*a, **kw):
        raise AssertionError("dry-run must not invoke _train_lora")

    monkeypatch.setattr(trainer, "_train_lora", must_not_train)
    tick("fake-model")  # baseline
    out = tick("fake-model", dry_run=True)
    assert out["action"] == "skip"
    assert out["reason"] == "dry run"
    # Queue file still queued.
    assert queue_size() == 1
    assert not any(trainer.ADAPTERS_DIR.iterdir())


def test_tick_baseline_locks_active_model(thandv_home, monkeypatch):
    """After baseline, the persona's active_models_by_persona slot gets
    set to base_model so subsequent ticks evaluate consistently."""
    monkeypatch.setattr(trainer, "_run_eval", lambda m, suite=None: 0.5)
    tick("fake-model")
    s = load_state()
    assert s.active_models_by_persona["code"] == "fake-model"


def test_tick_per_persona_isolation(thandv_home, monkeypatch):
    """Training the writer persona must not touch the code persona's slot
    (and vice versa). This is the whole point of v0.7.1."""
    monkeypatch.setattr(trainer, "_run_eval", lambda m, suite=None: 0.5)
    tick("fake-base", "code")
    tick("fake-base", "writer")
    s = load_state()
    assert s.active_models_by_persona["code"] == "fake-base"
    assert s.active_models_by_persona["writer"] == "fake-base"
    assert s.best_by_persona["code"] == 0.5
    assert s.best_by_persona["writer"] == 0.5
    # Promoting one persona doesn't disturb the other.
    rates = iter([0.9, 0.5])  # writer baseline, code post-train
    monkeypatch.setattr(trainer, "_run_eval", lambda m, suite=None: next(rates))
    monkeypatch.setattr(trainer, "_train_lora", _fake_train_lora())
    monkeypatch.setattr(trainer, "_discard_adapter", lambda *a, **kw: None)
    # writer's baseline was 0.5; this tick produces a 0.9 result → promote writer
    enqueue_examples("w", [{"prompt": "p", "completion": "c"}])
    out = tick("fake-base", "writer")
    assert out["action"] == "promote"
    s = load_state()
    assert s.active_models_by_persona["writer"] == "thandv-adapter-1"
    assert s.active_models_by_persona["code"] == "fake-base"  # untouched


def test_tick_uses_persona_eval_suite(thandv_home, monkeypatch):
    """The eval suite used to gate promotion comes from the persona, not
    a hardcoded 'smoke'."""
    seen_suites: list[str] = []

    def fake_run_eval(model, suite=None):
        seen_suites.append(suite)
        return 0.5

    monkeypatch.setattr(trainer, "_run_eval", fake_run_eval)
    tick("fake-base", "writer")
    tick("fake-base", "finance")
    assert "writer" in seen_suites
    assert "finance" in seen_suites


def test_load_state_migrates_legacy_active_ollama_model(thandv_home):
    """An old state.json that has `active_ollama_model` set (pre-v0.7.1)
    should auto-migrate the value into the `code` persona slot."""
    import json
    legacy = {
        "active_ollama_model": "qwen2.5-coder:7b",
        "best_eval_pass_rate": 0.848,
        "baseline_measured": True,
    }
    trainer.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    trainer.STATE_PATH.write_text(json.dumps(legacy))

    s = trainer.load_state()
    assert s.active_models_by_persona["code"] == "qwen2.5-coder:7b"
    assert s.best_by_persona["code"] == 0.848


def test_load_state_migration_idempotent(thandv_home):
    """If the new dict already has 'code', the legacy field doesn't overwrite."""
    import json
    state_data = {
        "active_ollama_model": "old-base",
        "active_models_by_persona": {"code": "thandv-adapter-newer"},
        "best_by_persona": {"code": 0.95},
        "best_eval_pass_rate": 0.5,  # stale; migration should NOT overwrite per-persona
    }
    trainer.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    trainer.STATE_PATH.write_text(json.dumps(state_data))

    s = trainer.load_state()
    assert s.active_models_by_persona["code"] == "thandv-adapter-newer"
    assert s.best_by_persona["code"] == 0.95


# --- Logs -----------------------------------------------------------------

def test_each_tick_writes_a_log(thandv_home, monkeypatch):
    monkeypatch.setattr(trainer, "_run_eval", lambda m, suite=None: 0.5)
    tick("fake-model")
    tick("fake-model")
    logs = list(trainer.LOGS_DIR.glob("tick-*.json"))
    assert len(logs) == 2
    sample = json.loads(logs[0].read_text())
    assert "action" in sample
    assert "at" in sample


# --- Daemon-control surface (without actually spawning a daemon) ----------

def test_is_running_when_no_pid(thandv_home):
    assert trainer.is_running() is False


def test_is_running_when_dead_pid(thandv_home, monkeypatch):
    trainer.PID_PATH.write_text("999999999")  # pid that almost certainly doesn't exist
    assert trainer.is_running() is False


def test_stop_when_not_running(thandv_home):
    assert trainer.stop() is False
