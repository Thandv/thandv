from __future__ import annotations

import pytest

from thandv import image_backend as ib
from thandv.image_cli import main


@pytest.fixture(autouse=True)
def _empty_backend_registry(monkeypatch):
    monkeypatch.setattr(ib, "BACKENDS", {})


def test_no_args_prints_help(thandv_home, capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "backends" in out and "generate" in out


def test_backends_disabled_default(thandv_home, capsys):
    rc = main(["backends"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DISABLED" in out
    assert "(none)" in out
    assert "image_generation_enabled=true" in out


def test_backends_with_placeholder_registered(thandv_home, capsys):
    ib.register_placeholder_backend()
    rc = main(["backends"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "placeholder" in out


def test_register_placeholder_subcommand(thandv_home, capsys):
    rc = main(["register-placeholder"])
    assert rc == 0
    assert "placeholder" in ib.list_backends()
    assert "registered" in capsys.readouterr().out


def test_generate_blocked_without_opt_in(thandv_home, monkeypatch, capsys, tmp_path):
    ib.register_placeholder_backend()
    # Default Config has image_generation_enabled=False
    rc = main(["generate", "a sunset", "-o", str(tmp_path / "out.ppm")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "disabled" in err


def test_generate_blocked_without_backend(thandv_home, monkeypatch, capsys, tmp_path):
    from thandv import config as cfg_mod

    cfg = cfg_mod.Config(image_generation_enabled=True)
    cfg.save()
    rc = main(["generate", "a sunset", "-o", str(tmp_path / "out.ppm")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no image backend" in err


def test_generate_happy_path_with_placeholder(thandv_home, monkeypatch, capsys, tmp_path):
    from thandv import config as cfg_mod

    cfg = cfg_mod.Config(image_generation_enabled=True)
    cfg.save()
    ib.register_placeholder_backend()
    out = tmp_path / "out.ppm"
    rc = main(["generate", "a sunset", "-o", str(out), "--seed", "7"])
    assert rc == 0
    msg = capsys.readouterr().out
    assert "generated:" in msg
    assert str(out) in msg
    assert out.exists()
    data = out.read_bytes()
    assert data.startswith(b"P6\n1 1\n255\n")


def test_generate_picks_named_backend(thandv_home, monkeypatch, capsys, tmp_path):
    from thandv import config as cfg_mod

    cfg = cfg_mod.Config(image_generation_enabled=True)
    cfg.save()
    ib.register_placeholder_backend()

    class _OtherStub:
        name = "other"

        def generate(self, request):
            request.out_path.write_bytes(b"OTHER")
            return ib.GeneratedImage(
                path=request.out_path,
                width=1,
                height=1,
                format="bin",
                backend=self.name,
            )

    ib.register_backend("other", _OtherStub())
    out = tmp_path / "out.bin"
    rc = main(["generate", "x", "-o", str(out), "--backend", "other"])
    assert rc == 0
    assert out.read_bytes() == b"OTHER"


def test_generate_multiple_backends_requires_pick(thandv_home, capsys, tmp_path):
    from thandv import config as cfg_mod

    cfg = cfg_mod.Config(image_generation_enabled=True)
    cfg.save()
    ib.register_placeholder_backend()
    ib.register_backend("other", _StubBackendForTest())
    rc = main(["generate", "x", "-o", str(tmp_path / "x.ppm")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "multiple backends" in err


def test_generate_unknown_named_backend(thandv_home, capsys, tmp_path):
    from thandv import config as cfg_mod

    cfg = cfg_mod.Config(image_generation_enabled=True)
    cfg.save()
    ib.register_placeholder_backend()
    rc = main(
        ["generate", "x", "-o", str(tmp_path / "x.ppm"), "--backend", "nope"]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "not registered" in err


class _StubBackendForTest:
    name = "stub-for-test"

    def generate(self, request):
        request.out_path.write_bytes(b"x")
        return ib.GeneratedImage(
            path=request.out_path,
            width=1,
            height=1,
            format="bin",
            backend=self.name,
        )
