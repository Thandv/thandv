from __future__ import annotations

from pathlib import Path

import pytest

from thandv import training_backend as tb


# --- Backend registry / picker --------------------------------------------

def test_backends_list_contains_mlx_and_hfpeft():
    names = {b.name for b in tb.BACKENDS}
    assert "mlx-lm" in names
    assert "hf-peft" in names


def test_hfpeft_not_available_yet():
    """The HF+PEFT stub returns False so pick_backend doesn't pick it."""
    hf = next(b for b in tb.BACKENDS if b.name == "hf-peft")
    assert hf.is_available() is False


def test_pick_backend_returns_none_when_nothing_available(monkeypatch):
    """If both backends report unavailable, pick_backend returns None."""
    monkeypatch.setattr(tb.MLXBackend, "is_available", lambda self: False)
    monkeypatch.setattr(tb.HFPEFTBackend, "is_available", lambda self: False)
    assert tb.pick_backend() is None


def test_pick_backend_prefers_mlx(monkeypatch):
    monkeypatch.setattr(tb.MLXBackend, "is_available", lambda self: True)
    monkeypatch.setattr(tb.HFPEFTBackend, "is_available", lambda self: True)
    picked = tb.pick_backend()
    assert picked is not None
    assert picked.name == "mlx-lm"


# --- MLX backend ----------------------------------------------------------

def test_mlx_is_available_requires_darwin_and_mlx_lm(monkeypatch):
    """Linux hosts can't run MLX even if the package is installed."""
    monkeypatch.setattr(tb.sys, "platform", "linux")
    assert tb.MLXBackend().is_available() is False


def test_mlx_train_errors_on_missing_base(thandv_home, tmp_path):
    q = tmp_path / "q.jsonl"
    q.write_text('{"prompt": "a", "completion": "b"}\n')
    with pytest.raises(FileNotFoundError, match="base model not found"):
        tb.MLXBackend().train(
            base_hf_dir=tmp_path / "no-such-dir",
            queue_file=q,
            out_dir=tmp_path / "out",
        )


def test_mlx_train_errors_on_missing_queue(thandv_home, tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(FileNotFoundError, match="queue file not found"):
        tb.MLXBackend().train(
            base_hf_dir=base,
            queue_file=tmp_path / "no-such.jsonl",
            out_dir=tmp_path / "out",
        )


def test_mlx_train_invokes_subprocess_with_expected_args(thandv_home, tmp_path, monkeypatch):
    """Verifies the CLI invocation shape without actually running MLX."""
    base = tmp_path / "base"
    base.mkdir()
    q = tmp_path / "q.jsonl"
    q.write_text(
        '{"prompt": "a", "completion": "b"}\n'
        '{"prompt": "c", "completion": "d"}\n'
    )
    out = tmp_path / "out"

    captured: dict = {}

    def fake_run(cmd, check=True):
        captured["cmd"] = cmd
        return type("CP", (), {"returncode": 0})()

    monkeypatch.setattr(tb.subprocess, "run", fake_run)
    result = tb.MLXBackend().train(base, q, out, iters=3, batch_size=1)

    cmd = captured["cmd"]
    assert "mlx_lm.lora" in cmd
    assert "--train" in cmd
    assert "--iters" in cmd and "3" in cmd
    assert "--batch-size" in cmd and "1" in cmd
    assert "--model" in cmd
    assert str(base) in cmd

    # Staged data dir present.
    data = out / "data"
    assert (data / "train.jsonl").exists()
    assert (data / "valid.jsonl").exists()
    assert result.n_examples == 2
    assert result.n_iters == 3
    assert result.backend == "mlx-lm"


def test_mlx_merge_adapter_invokes_fuse(thandv_home, tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    out = tmp_path / "merged"

    captured: dict = {}

    def fake_run(cmd, check=True):
        captured["cmd"] = cmd
        return type("CP", (), {"returncode": 0})()

    monkeypatch.setattr(tb.subprocess, "run", fake_run)
    returned = tb.MLXBackend().merge_adapter(base, adapter, out)

    cmd = captured["cmd"]
    assert "mlx_lm.fuse" in cmd
    assert "--model" in cmd and str(base) in cmd
    assert "--adapter-path" in cmd and str(adapter) in cmd
    assert "--save-path" in cmd and str(out) in cmd
    assert returned == out


# --- HF base-model fetch --------------------------------------------------

def test_base_model_dir_is_path_safe():
    """Slashes in the HF repo id are escaped, not allowed to escape THANDV_HOME."""
    p = tb.base_model_dir("Qwen/Qwen2.5-Coder-7B")
    assert p.name == "Qwen_Qwen2.5-Coder-7B"
    assert ".." not in str(p)


def test_fetch_hf_base_model_skip_if_already_present(thandv_home, monkeypatch):
    """Re-fetch is a no-op if the cached dir is non-empty."""
    target = tb.BASE_MODELS_DIR / "fake_model"
    target.mkdir(parents=True)
    (target / "marker.bin").write_text("x")

    called = {"n": 0}

    def fake_snapshot(*a, **kw):
        called["n"] += 1
        return str(target)

    # huggingface_hub may not be installed in CI; mock the import path.
    import sys
    class _FakeHubModule:
        snapshot_download = staticmethod(fake_snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHubModule)
    monkeypatch.setattr(tb, "base_model_dir", lambda mid: target)

    result = tb.fetch_hf_base_model("fake_model")
    assert result == target
    assert called["n"] == 0  # never invoked because cached


def test_fetch_hf_base_model_calls_snapshot(thandv_home, monkeypatch):
    """Fresh fetch invokes huggingface_hub.snapshot_download."""
    target = tb.BASE_MODELS_DIR / "fresh_model"

    called = {"n": 0, "repo_id": None}

    def fake_snapshot(*, repo_id, local_dir):
        called["n"] += 1
        called["repo_id"] = repo_id
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "tokenizer.json").write_text("{}")
        return local_dir

    import sys
    class _FakeHubModule:
        snapshot_download = staticmethod(fake_snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHubModule)
    monkeypatch.setattr(tb, "base_model_dir", lambda mid: target)

    result = tb.fetch_hf_base_model("fresh_model")
    assert result == target
    assert called["n"] == 1
    assert called["repo_id"] == "fresh_model"


def test_fetch_hf_base_model_clear_error_without_hub(thandv_home, monkeypatch):
    """If huggingface_hub isn't installed, raise with an install hint."""
    import sys
    monkeypatch.setattr(tb, "base_model_dir", lambda mid: tb.BASE_MODELS_DIR / "x")

    # Simulate import failure.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *a, **kw):
        if name == "huggingface_hub":
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", fake_import)
    # Also evict any cached module so the lazy import re-runs.
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)

    with pytest.raises(RuntimeError, match=r"thandv\[train-mlx\]"):
        tb.fetch_hf_base_model("anything")


# --- Verification harness -------------------------------------------------

def test_verify_training_setup_writes_tiny_queue_and_calls_train(thandv_home, tmp_path, monkeypatch):
    """`verify_training_setup` should stage two examples and run 1 iter."""
    base = tmp_path / "base"
    base.mkdir()

    captured: dict = {}

    class FakeBackend:
        name = "fake"

        def is_available(self) -> bool:
            return True

        def train(self, base_hf_dir, queue_file, out_dir, *, iters=100, batch_size=1):
            captured["queue_lines"] = sum(1 for _ in queue_file.open())
            captured["iters"] = iters
            return tb.TrainResult(
                adapter_dir=out_dir / "adapter",
                n_examples=captured["queue_lines"],
                n_iters=iters,
                backend=self.name,
            )

        def merge_adapter(self, *a, **kw):
            raise AssertionError("merge should not be called in verification")

    info = tb.verify_training_setup(FakeBackend(), base, out_dir=tmp_path / "verify-out")
    assert captured["queue_lines"] == 2
    assert captured["iters"] == 1
    assert info["backend"] == "fake"
    assert info["n_iters"] == 1
