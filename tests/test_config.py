from thandv import config
from thandv.config import Config, ensure_dirs


def test_default_values():
    c = Config()
    assert c.model == ""
    assert c.backend == "ollama"
    assert 0.0 <= c.temperature <= 1.0
    assert c.max_tokens > 0
    assert c.auto_tools is True
    assert c.persona == "code"


def test_save_load_roundtrip(thandv_home):
    c = Config(
        model="qwen3:8b",
        temperature=0.5,
        max_tokens=512,
        auto_tools=False,
        persona="writer",
    )
    c.save()
    loaded = Config.load()
    assert loaded.model == "qwen3:8b"
    assert loaded.temperature == 0.5
    assert loaded.max_tokens == 512
    assert loaded.auto_tools is False
    assert loaded.persona == "writer"


def test_load_returns_default_when_no_file(thandv_home):
    assert not config.CONFIG_PATH.exists()
    loaded = Config.load()
    assert loaded == Config()


def test_load_ignores_unknown_keys(thandv_home):
    config.CONFIG_PATH.write_text('{"model": "x", "spaceship": "enterprise"}')
    loaded = Config.load()
    assert loaded.model == "x"
    assert not hasattr(loaded, "spaceship")


def test_ensure_dirs_idempotent(thandv_home):
    ensure_dirs()
    ensure_dirs()
    assert config.SESSIONS_DIR.is_dir()
    assert config.MEMORY_DIR.is_dir()
    assert config.SKILLS_DIR.is_dir()


def test_load_tolerates_corrupt_config(thandv_home, capsys):
    from thandv import config

    config.CONFIG_PATH.write_text('{"model": "qwen", ')  # torn write
    cfg = Config.load()
    assert cfg == Config()
    assert "unreadable config" in capsys.readouterr().err


def test_load_tolerates_non_object_config(thandv_home, capsys):
    from thandv import config

    config.CONFIG_PATH.write_text("[1, 2]")
    assert Config.load() == Config()
    assert "not a JSON object" in capsys.readouterr().err


def test_save_is_atomic_and_leaves_no_temp_files(thandv_home):
    from pathlib import Path

    import pytest

    from thandv import config

    Config(model="old").save()

    def boom(self, *a, **k):
        raise OSError("disk full")

    # A failed write must leave the previous config intact.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "write_text", boom)
        with pytest.raises(OSError):
            Config(model="new").save()
    assert Config.load().model == "old"
    assert [p.name for p in config.CONFIG_PATH.parent.iterdir() if ".tmp-" in p.name] == []
