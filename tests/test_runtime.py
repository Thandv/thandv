from thandv.runtime import MODEL_LADDER, HostProfile, describe, pick_model


def _host(**kw) -> HostProfile:
    base = dict(
        os="Darwin",
        arch="arm64",
        apple_silicon=True,
        total_ram_gb=16,
        has_nvidia=False,
        nvidia_vram_gb=0,
    )
    base.update(kw)
    return HostProfile(**base)


def test_pick_model_nvidia_high_end():
    h = _host(os="Linux", arch="x86_64", apple_silicon=False, has_nvidia=True, nvidia_vram_gb=80)
    assert pick_model(h) == "qwen3-coder:30b"


def test_pick_model_apple_silicon_64gb():
    assert pick_model(_host(total_ram_gb=64)) == "qwen3-coder:30b"


def test_pick_model_apple_silicon_16gb():
    # M2 16GB — should land on the 7b tier
    assert pick_model(_host(total_ram_gb=16)) == "qwen2.5-coder:7b"


def test_pick_model_apple_silicon_8gb():
    assert pick_model(_host(total_ram_gb=8)) == "qwen2.5-coder:3b"


def test_pick_model_cpu_fallback():
    h = _host(os="Linux", arch="x86_64", apple_silicon=False, total_ram_gb=4)
    # CPU-only with little RAM should hit the bottom of the ladder
    assert pick_model(h) == MODEL_LADDER[-1][0]


def test_model_ladder_is_descending_by_capability():
    # The ladder is intentionally ordered from largest to smallest. Verify
    # that the RAM/VRAM budgets weakly decrease as we go down.
    rams = [r for _, r, _ in MODEL_LADDER]
    vrams = [v for _, _, v in MODEL_LADDER]
    assert rams == sorted(rams, reverse=True)
    assert vrams == sorted(vrams, reverse=True)


def test_describe_includes_key_fields():
    h = _host()
    out = describe(h, "qwen2.5-coder:7b")
    assert "Darwin" in out
    assert "arm64" in out
    assert "Metal" in out
    assert "qwen2.5-coder:7b" in out
