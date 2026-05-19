from __future__ import annotations

import json
import sys

import pytest

from thandv import training_data as td


# --- Registry + lookup ----------------------------------------------------

def test_registry_has_at_least_three_datasets():
    names = {d.name for d in td.list_public_datasets()}
    assert {"codealpaca", "dolly", "alpaca"}.issubset(names)


def test_get_public_dataset_unknown_raises():
    with pytest.raises(ValueError, match="unknown dataset"):
        td.get_public_dataset("not-real")


def test_every_dataset_has_persona_hint_and_license():
    for d in td.list_public_datasets():
        assert d.persona_hint
        assert d.license
        assert d.description
        assert d.hf_repo and "/" in d.hf_repo


# --- Row mappers ---------------------------------------------------------

def test_codealpaca_row_basic():
    row = {"instruction": "Write a function.", "input": "x = 1", "output": "def f(): pass"}
    out = td._codealpaca_row_to_example(row)
    assert "Write a function" in out["prompt"]
    assert "x = 1" in out["prompt"]
    assert out["completion"] == "def f(): pass"


def test_codealpaca_row_no_input():
    row = {"instruction": "Explain.", "input": "", "output": "OK."}
    out = td._codealpaca_row_to_example(row)
    assert out["prompt"] == "Explain."
    assert out["completion"] == "OK."


def test_dolly_row_with_context():
    row = {
        "instruction": "Summarise.",
        "context": "The fox jumped.",
        "response": "A fox jumped.",
        "category": "summarization",
    }
    out = td._dolly_row_to_example(row)
    assert "Summarise" in out["prompt"]
    assert "fox jumped" in out["prompt"]
    assert out["completion"] == "A fox jumped."


def test_dolly_row_no_context():
    row = {"instruction": "Hello?", "context": "", "response": "Hi!"}
    out = td._dolly_row_to_example(row)
    assert out["prompt"] == "Hello?"
    assert "Context" not in out["prompt"]


# --- Sampling (mocked datasets) ------------------------------------------

class _FakeDS:
    """Stands in for a HuggingFace `datasets.Dataset`."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int) -> dict:
        return self._rows[i]


def _install_fake_load_dataset(monkeypatch, rows: list[dict]) -> dict:
    """Mock `datasets.load_dataset` to return _FakeDS(rows).

    Accepts either signature `(repo, split=...)` (no config) or
    `(repo, config, split=...)` (with config) so the same helper covers
    both old and new dataset registrations.
    """
    seen: dict = {}

    def fake_load(repo, config=None, *, split=None):
        seen["repo"] = repo
        seen["config"] = config
        seen["split"] = split
        return _FakeDS(rows)

    class _Mod:
        load_dataset = staticmethod(fake_load)

    monkeypatch.setitem(sys.modules, "datasets", _Mod)
    return seen


def test_sample_public_dataset_returns_correct_shape(monkeypatch):
    rows = [
        {"instruction": f"q{i}", "input": "", "output": f"a{i}"}
        for i in range(20)
    ]
    _install_fake_load_dataset(monkeypatch, rows)
    out = td.sample_public_dataset("codealpaca", 5, seed=42)
    assert len(out) == 5
    for ex in out:
        assert set(ex.keys()) == {"prompt", "completion"}


def test_sample_public_dataset_n_larger_than_dataset(monkeypatch):
    """If N > len(dataset), return everything."""
    rows = [{"instruction": "q", "input": "", "output": "a"}]
    _install_fake_load_dataset(monkeypatch, rows)
    out = td.sample_public_dataset("codealpaca", 50)
    assert len(out) == 1


def test_sample_public_dataset_seed_is_deterministic(monkeypatch):
    rows = [{"instruction": f"q{i}", "input": "", "output": f"a{i}"} for i in range(50)]
    _install_fake_load_dataset(monkeypatch, rows)
    a = td.sample_public_dataset("codealpaca", 10, seed=0)
    b = td.sample_public_dataset("codealpaca", 10, seed=0)
    c = td.sample_public_dataset("codealpaca", 10, seed=1)
    assert a == b
    assert a != c


def test_sample_public_dataset_uses_correct_hf_repo(monkeypatch):
    """Pass dataset-shaped rows so each mapper succeeds, and verify the
    HF repo id reached load_dataset for each."""
    # Dolly format.
    seen = _install_fake_load_dataset(
        monkeypatch,
        [{"instruction": "q", "context": "", "response": "a", "category": "x"}],
    )
    td.sample_public_dataset("dolly", 1)
    assert seen["repo"] == "databricks/databricks-dolly-15k"

    # CodeAlpaca format.
    seen2 = _install_fake_load_dataset(
        monkeypatch, [{"instruction": "q", "input": "", "output": "a"}]
    )
    td.sample_public_dataset("codealpaca", 1)
    assert seen2["repo"] == "sahil2801/CodeAlpaca-20k"


def test_sample_public_dataset_unknown_raises(monkeypatch):
    """Unknown name fails fast, before importing datasets."""
    with pytest.raises(ValueError, match="unknown dataset"):
        td.sample_public_dataset("not-real", 10)


def test_sample_public_dataset_clear_error_without_datasets(monkeypatch):
    """Missing `datasets` package → clear install hint, not ImportError."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    # Need to make sure we don't have a mocked datasets module either.
    with pytest.raises(RuntimeError, match=r"thandv\[eval\]"):
        td.sample_public_dataset("codealpaca", 10)


# --- Queue writing -------------------------------------------------------

def test_queue_public_dataset_writes_jsonl(thandv_home, monkeypatch):
    rows = [
        {"instruction": f"q{i}", "input": "", "output": f"a{i}"}
        for i in range(20)
    ]
    _install_fake_load_dataset(monkeypatch, rows)
    path = td.queue_public_dataset("codealpaca", 5)
    assert path.exists()
    assert path.name.startswith(str(int(__import__("time").time()))[:5])
    assert "codealpaca" in path.name
    assert "n5" in path.name

    lines = path.read_text().splitlines()
    assert len(lines) == 5
    for line in lines:
        ex = json.loads(line)
        assert "prompt" in ex and "completion" in ex


def test_queue_public_dataset_unknown_dataset(thandv_home):
    with pytest.raises(ValueError, match="unknown dataset"):
        td.queue_public_dataset("nope", 5)


# --- Prose datasets (writer persona) ------------------------------------

def test_prose_datasets_registered():
    names = {d.name for d in td.list_public_datasets()}
    assert {"wikitext", "tinystories"}.issubset(names)


def test_prose_datasets_target_writer_persona():
    for name in ("wikitext", "tinystories"):
        d = td.get_public_dataset(name)
        assert d.persona_hint == "writer"
        assert d.license


def test_wikitext_has_hf_config():
    """Wikitext requires a config arg on HF; tinystories doesn't."""
    assert td.get_public_dataset("wikitext").hf_config == "wikitext-2-raw-v1"
    assert td.get_public_dataset("tinystories").hf_config is None


def test_prose_row_to_example_splits_at_word_boundary():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    out = td._prose_row_to_example({"text": text})
    assert out["prompt"].startswith("Continue the following passage")
    # No word should be split across the boundary.
    head = out["prompt"].split("\n\n", 1)[1]
    tail = out["completion"]
    assert " " not in head[-1:]
    assert not tail.startswith(" ")
    # Round-trips: head + " " + tail covers all the original words.
    assert head.split() + tail.split() == text.split()


def test_prose_row_to_example_empty_text():
    out = td._prose_row_to_example({"text": ""})
    assert out == {"prompt": "", "completion": ""}


def test_prose_row_to_example_missing_text_key():
    out = td._prose_row_to_example({})
    assert out == {"prompt": "", "completion": ""}


def test_prose_row_to_example_caps_long_text():
    """Long rows get truncated to PROSE_MAX_CHARS before splitting so we
    don't generate giant single examples."""
    long_text = "word " * 10_000  # ~50k chars
    out = td._prose_row_to_example({"text": long_text})
    combined_len = len(out["prompt"]) + len(out["completion"])
    # Prompt prefix adds ~50 chars; we should still be near PROSE_MAX_CHARS.
    assert combined_len < td.PROSE_MAX_CHARS + 200


def test_sample_public_dataset_wikitext_passes_config(monkeypatch):
    seen = _install_fake_load_dataset(
        monkeypatch, [{"text": "alpha beta gamma delta " * 5}]
    )
    td.sample_public_dataset("wikitext", 1)
    assert seen["repo"] == "Salesforce/wikitext"
    assert seen["config"] == "wikitext-2-raw-v1"


def test_sample_public_dataset_tinystories_no_config(monkeypatch):
    seen = _install_fake_load_dataset(
        monkeypatch, [{"text": "Once upon a time " * 10}]
    )
    td.sample_public_dataset("tinystories", 1)
    assert seen["repo"] == "roneneldan/TinyStories"
    assert seen["config"] is None
