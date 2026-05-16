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
    monkeypatch.setattr(trainer, "_run_eval", lambda model: 0.5)
    out = tick("fake-model")
    assert out["action"] == "baseline"
    assert out["pass_rate"] == 0.5
    s = load_state()
    assert s.baseline_measured is True
    assert s.best_eval_pass_rate == 0.5


def test_tick_skip_when_baseline_eval_fails(thandv_home, monkeypatch):
    def boom(model):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(trainer, "_run_eval", boom)
    out = tick("fake-model")
    assert out["action"] == "skip"
    assert "baseline eval failed" in out["reason"]
    assert load_state().baseline_measured is False


def test_tick_skip_when_queue_empty(thandv_home, monkeypatch):
    # First tick establishes baseline.
    monkeypatch.setattr(trainer, "_run_eval", lambda m: 0.5)
    tick("fake-model")
    # Second tick has nothing to do.
    out = tick("fake-model")
    assert out["action"] == "skip"
    assert out["reason"] == "queue empty"


def test_tick_promotes_when_improved(thandv_home, monkeypatch):
    enqueue_examples("a", [{"prompt": "p", "completion": "c"}])
    rates = iter([0.5, 0.9])  # baseline then post-train
    monkeypatch.setattr(trainer, "_run_eval", lambda m: next(rates))
    tick("fake-model")  # baseline
    out = tick("fake-model")
    assert out["action"] == "promote"
    assert out["pass_rate"] == 0.9
    s = load_state()
    assert s.best_eval_pass_rate == 0.9
    assert s.active_adapter == out["adapter_id"]
    # The adapter marker should still exist.
    assert (trainer.ADAPTERS_DIR / f"{out['adapter_id']}.json").exists()
    # The queue file should have moved to processed/.
    assert queue_size() == 0
    assert any(trainer.PROCESSED_DIR.iterdir())


def test_tick_discards_when_not_improved(thandv_home, monkeypatch):
    enqueue_examples("a", [{"prompt": "p", "completion": "c"}])
    rates = iter([0.9, 0.5])
    monkeypatch.setattr(trainer, "_run_eval", lambda m: next(rates))
    tick("fake-model")  # baseline at 0.9
    out = tick("fake-model")
    assert out["action"] == "discard"
    s = load_state()
    assert s.best_eval_pass_rate == 0.9  # unchanged
    assert s.active_adapter == ""
    # Adapter file should have been deleted.
    assert not (trainer.ADAPTERS_DIR / f"{out['adapter_id']}.json").exists()
    # Queue file is still consumed (moved to processed) so we don't loop.
    assert queue_size() == 0


def test_tick_skip_when_post_train_eval_fails(thandv_home, monkeypatch):
    enqueue_examples("a", [{"prompt": "p", "completion": "c"}])
    calls = {"n": 0}

    def flaky(model):
        calls["n"] += 1
        if calls["n"] == 1:
            return 0.5  # baseline ok
        raise RuntimeError("ollama died mid-tick")

    monkeypatch.setattr(trainer, "_run_eval", flaky)
    tick("fake-model")  # baseline
    out = tick("fake-model")
    assert out["action"] == "skip"
    assert "eval failed" in out["reason"]
    # Queue file should NOT have been consumed (so we'll retry).
    assert queue_size() == 1
    # No adapter marker should be left behind.
    assert not any(trainer.ADAPTERS_DIR.iterdir())


def test_tick_dry_run_doesnt_train(thandv_home, monkeypatch):
    enqueue_examples("a", [{"prompt": "p", "completion": "c"}])
    monkeypatch.setattr(trainer, "_run_eval", lambda m: 0.5)
    tick("fake-model")  # baseline
    out = tick("fake-model", dry_run=True)
    assert out["action"] == "skip"
    assert out["reason"] == "dry run"
    # Queue file still queued.
    assert queue_size() == 1
    assert not any(trainer.ADAPTERS_DIR.iterdir())


# --- Logs -----------------------------------------------------------------

def test_each_tick_writes_a_log(thandv_home, monkeypatch):
    monkeypatch.setattr(trainer, "_run_eval", lambda m: 0.5)
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
