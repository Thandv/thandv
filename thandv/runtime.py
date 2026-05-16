"""Hardware detection and model selection.

The goal: pick the strongest open-weights model that will plausibly run on the
current machine. Defaults prefer Qwen3-Coder for coding-agent behavior closest
to Claude Code; falls back to smaller variants on tight memory.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass
class HostProfile:
    os: str
    arch: str
    apple_silicon: bool
    total_ram_gb: int
    has_nvidia: bool
    nvidia_vram_gb: int


def _total_ram_gb() -> int:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            return int(int(out) / (1024**3))
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return int(kb / (1024**2))
    except Exception:
        pass
    return 8


def _nvidia_vram_gb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        if out:
            return max(int(x.strip()) for x in out) // 1024
    except Exception:
        pass
    return 0


def detect_host() -> HostProfile:
    sysname = platform.system()
    arch = platform.machine()
    apple_silicon = sysname == "Darwin" and arch in ("arm64", "aarch64")
    vram = _nvidia_vram_gb()
    return HostProfile(
        os=sysname,
        arch=arch,
        apple_silicon=apple_silicon,
        total_ram_gb=_total_ram_gb(),
        has_nvidia=vram > 0,
        nvidia_vram_gb=vram,
    )


# Ordered by capability. The first entry whose budget fits the host wins.
# Names match Ollama tags; users can override via `thandv config set model=...`.
MODEL_LADDER = [
    # (ollama_tag, min_ram_gb_for_unified_memory, min_vram_gb)
    ("qwen3-coder:30b",      48, 24),
    ("qwen2.5-coder:32b",    32, 20),
    ("qwen3-coder:14b",      24, 12),
    ("qwen2.5-coder:14b",    20, 10),
    ("qwen2.5-coder:7b",     12,  6),
    ("qwen2.5-coder:3b",      8,  4),
    ("llama3.2:3b",           6,  3),
]


def pick_model(host: HostProfile) -> str:
    """Pick the best model the host can plausibly serve."""
    for tag, min_ram, min_vram in MODEL_LADDER:
        if host.has_nvidia and host.nvidia_vram_gb >= min_vram:
            return tag
        if host.apple_silicon and host.total_ram_gb >= min_ram:
            return tag
        if not host.has_nvidia and not host.apple_silicon and host.total_ram_gb >= min_ram + 4:
            return tag
    return MODEL_LADDER[-1][0]


def describe(host: HostProfile, model: str) -> str:
    parts = [f"OS={host.os}", f"arch={host.arch}", f"RAM={host.total_ram_gb}GB"]
    if host.has_nvidia:
        parts.append(f"NVIDIA VRAM={host.nvidia_vram_gb}GB")
    if host.apple_silicon:
        parts.append("backend=Metal")
    elif host.has_nvidia:
        parts.append("backend=CUDA")
    else:
        parts.append("backend=CPU")
    parts.append(f"model={model}")
    return " | ".join(parts)
