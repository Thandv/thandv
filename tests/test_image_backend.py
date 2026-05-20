from __future__ import annotations

import pytest

from thandv import image_backend as ib
from thandv.config import Config


@pytest.fixture(autouse=True)
def _empty_backend_registry(monkeypatch):
    """Every test gets a clean registry; tests that need a backend
    register one themselves."""
    monkeypatch.setattr(ib, "BACKENDS", {})


class _StubBackend:
    name = "stub"

    def generate(self, request):
        request.out_path.parent.mkdir(parents=True, exist_ok=True)
        request.out_path.write_bytes(b"stub-bytes")
        return ib.GeneratedImage(
            path=request.out_path,
            width=request.width,
            height=request.height,
            format="bin",
            backend=self.name,
        )


# --- registration --------------------------------------------------------

def test_register_and_unregister():
    ib.register_backend("stub", _StubBackend())
    assert "stub" in ib.list_backends()
    assert ib.unregister_backend("stub") is True
    assert ib.list_backends() == []
    assert ib.unregister_backend("stub") is False


def test_register_replaces_existing():
    a, b = _StubBackend(), _StubBackend()
    ib.register_backend("stub", a)
    ib.register_backend("stub", b)
    assert ib.BACKENDS["stub"] is b
    assert len(ib.list_backends()) == 1


# --- is_enabled / status -------------------------------------------------

def test_is_enabled_default_false():
    assert ib.is_enabled(Config()) is False


def test_is_enabled_respects_config_flag():
    assert ib.is_enabled(Config(image_generation_enabled=True)) is True


def test_status_reports_both_halves():
    ib.register_backend("stub", _StubBackend())
    assert ib.status(Config(image_generation_enabled=True)) == {
        "opted_in": True,
        "backends": ["stub"],
    }


# --- require_enabled gate ------------------------------------------------

def test_require_enabled_blocks_when_opt_in_missing():
    ib.register_backend("stub", _StubBackend())
    with pytest.raises(ib.ImageGenerationDisabled, match="disabled"):
        ib.require_enabled(cfg=Config(image_generation_enabled=False))


def test_require_enabled_blocks_when_no_backend():
    with pytest.raises(ib.ImageGenerationDisabled, match="no image backend"):
        ib.require_enabled(cfg=Config(image_generation_enabled=True))


def test_require_enabled_blocks_when_multiple_no_pick():
    ib.register_backend("a", _StubBackend())
    ib.register_backend("b", _StubBackend())
    with pytest.raises(ib.ImageGenerationDisabled, match="multiple backends"):
        ib.require_enabled(cfg=Config(image_generation_enabled=True))


def test_require_enabled_with_name_picks_correctly():
    ib.register_backend("a", _StubBackend())
    ib.register_backend("b", _StubBackend())
    chosen = ib.require_enabled("a", cfg=Config(image_generation_enabled=True))
    assert chosen is ib.BACKENDS["a"]


def test_require_enabled_with_unknown_name_raises():
    ib.register_backend("a", _StubBackend())
    with pytest.raises(ib.ImageGenerationDisabled, match="not registered"):
        ib.require_enabled("nope", cfg=Config(image_generation_enabled=True))


def test_require_enabled_single_no_name_returns_only_backend():
    b = _StubBackend()
    ib.register_backend("only", b)
    assert ib.require_enabled(cfg=Config(image_generation_enabled=True)) is b


# --- PlaceholderBackend --------------------------------------------------

def test_placeholder_writes_valid_ppm(tmp_path):
    backend = ib.PlaceholderBackend()
    out = tmp_path / "img.ppm"
    req = ib.GenerationRequest(prompt="a sunset", out_path=out, seed=42)
    result = backend.generate(req)
    assert result.path == out
    assert result.format == "ppm"
    assert result.width == 1 and result.height == 1
    assert result.backend == "placeholder"
    # PPM P6 header followed by 3 RGB bytes for the 1x1 pixel.
    data = out.read_bytes()
    assert data.startswith(b"P6\n1 1\n255\n")
    assert len(data) == len(b"P6\n1 1\n255\n") + 3


def test_placeholder_is_deterministic_for_same_prompt_and_seed(tmp_path):
    backend = ib.PlaceholderBackend()
    req1 = ib.GenerationRequest("hello", tmp_path / "a.ppm", seed=1)
    req2 = ib.GenerationRequest("hello", tmp_path / "b.ppm", seed=1)
    backend.generate(req1)
    backend.generate(req2)
    assert (tmp_path / "a.ppm").read_bytes() == (tmp_path / "b.ppm").read_bytes()


def test_placeholder_varies_with_prompt(tmp_path):
    backend = ib.PlaceholderBackend()
    backend.generate(ib.GenerationRequest("alpha", tmp_path / "a.ppm", seed=1))
    backend.generate(ib.GenerationRequest("beta", tmp_path / "b.ppm", seed=1))
    assert (tmp_path / "a.ppm").read_bytes() != (tmp_path / "b.ppm").read_bytes()


def test_placeholder_creates_parent_dir(tmp_path):
    backend = ib.PlaceholderBackend()
    nested = tmp_path / "deep" / "tree" / "img.ppm"
    req = ib.GenerationRequest("x", nested)
    backend.generate(req)
    assert nested.exists()


# --- register_placeholder_backend convenience ----------------------------

def test_register_placeholder_backend_registers_under_canonical_name():
    assert "placeholder" not in ib.list_backends()
    ib.register_placeholder_backend()
    assert "placeholder" in ib.list_backends()
    assert isinstance(ib.BACKENDS["placeholder"], ib.PlaceholderBackend)
