"""Image-generation backend contract and opt-in gate.

This module ships ZERO heavy ML dependencies. diffusers + SDXL/Flux is
the obvious choice on a GPU box, MLX-Image Gen on Apple Silicon, hosted
APIs (Replicate, Stability, Together) for laptops -- they are all valid
and have very different install footprints and licence implications.
So instead of picking one, we ship the contract and let the user
register a backend in user-land.

Three backend states, mirrored from `paper_trading.py`:

1. **Opt-in not set.** `Config.image_generation_enabled` is False --
   the user hasn't acknowledged the install-size / licence reality.
2. **No backend registered.** Opt-in is set but no `ImageBackend`
   instance has been registered via `register_backend(...)`. We can
   ship a `PlaceholderBackend` (PPM stub, in-tree) so users have
   something to wire end-to-end without 6 GB of model files. Real
   backends are user-land.
3. **Backend registered.** Calls dispatch to the adapter; result is a
   `GeneratedImage` written to a path the caller controls.

The placeholder backend writes a valid 1x1 PPM (portable pixmap) with
the prompt's hash mapped to an RGB colour. That's NOT image generation
-- it's just enough to verify the pipeline end-to-end without GPU
dependence. Honest framing: see USAGE.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from thandv.config import Config


# --- Data types ---------------------------------------------------------

@dataclass(frozen=True)
class GeneratedImage:
    """Result of a successful generation call.

    `path` is where the bytes live on disk; `width`/`height` are the
    pixel dimensions actually produced (may differ from the request
    if the backend snaps to a supported size); `format` is a lowercase
    extension hint (`"png"`, `"ppm"`, `"jpg"`).
    """

    path: Path
    width: int
    height: int
    format: str
    backend: str


@dataclass(frozen=True)
class GenerationRequest:
    """Inputs every backend sees. Backends ignore fields they don't
    support and document so."""

    prompt: str
    out_path: Path
    width: int = 512
    height: int = 512
    seed: int | None = None
    steps: int = 20
    guidance: float = 7.5
    negative_prompt: str = ""


# --- Protocol ----------------------------------------------------------

class ImageBackend(Protocol):
    """User-supplied image-generation backend.

    `name` is a short identifier shown in CLI output; `generate` runs
    the model and returns a GeneratedImage. Backends should write the
    output bytes to `request.out_path`.
    """

    name: str

    def generate(self, request: GenerationRequest) -> GeneratedImage:
        ...


BACKENDS: dict[str, ImageBackend] = {}


def register_backend(name: str, backend: ImageBackend) -> None:
    """User-land hook. Last-one-wins on duplicate names so test
    harnesses can swap backends within a single session."""
    BACKENDS[name] = backend


def unregister_backend(name: str) -> bool:
    return BACKENDS.pop(name, None) is not None


def list_backends() -> list[str]:
    return sorted(BACKENDS)


# --- Opt-in gate -------------------------------------------------------

def is_enabled(cfg: Config | None = None) -> bool:
    """True iff the user has explicitly opted in. Opt-in alone does NOT
    imply a backend is registered -- `status()` reports both."""
    cfg = cfg or Config.load()
    return bool(cfg.image_generation_enabled)


def status(cfg: Config | None = None) -> dict:
    cfg = cfg or Config.load()
    return {
        "opted_in": bool(cfg.image_generation_enabled),
        "backends": list_backends(),
    }


class ImageGenerationDisabled(RuntimeError):
    """Raised when image generation is invoked without opt-in or
    without a registered backend (or with too many)."""


def require_enabled(
    backend_name: str | None = None,
    cfg: Config | None = None,
) -> ImageBackend:
    """Return the registered backend (named or single) or raise with
    a specific reason.

    Three failure modes mirror `paper_trading.require_enabled`:
      - opt-in not set
      - no backends registered
      - multiple backends and no name picked
    """
    cfg = cfg or Config.load()
    if not cfg.image_generation_enabled:
        raise ImageGenerationDisabled(
            "image generation is disabled. Set "
            "`image_generation_enabled=true` in your config "
            "(thandv config --set image_generation_enabled=true) and "
            "register a backend before calling. Install size for a real "
            "backend (diffusers + SDXL/Flux) is ~6 GB."
        )
    if not BACKENDS:
        raise ImageGenerationDisabled(
            "no image backend registered. Thandv ships only the "
            "PlaceholderBackend (a 1x1 PPM stub for pipeline testing); "
            "real backends are user-land. Register one with "
            "image_backend.register_backend(name, backend) -- see the "
            "ImageBackend Protocol for the contract."
        )
    if backend_name is not None:
        if backend_name not in BACKENDS:
            raise ImageGenerationDisabled(
                f"backend {backend_name!r} not registered. "
                f"Known: {list_backends()}"
            )
        return BACKENDS[backend_name]
    if len(BACKENDS) > 1:
        raise ImageGenerationDisabled(
            f"multiple backends registered ({list_backends()}); pass "
            "--backend NAME to pick one. The harness does not guess."
        )
    return next(iter(BACKENDS.values()))


# --- Placeholder backend ------------------------------------------------

class PlaceholderBackend:
    """A 1x1 PPM image where the RGB colour is derived from the prompt
    hash. NOT image generation -- a pipeline-proving stub.

    Why ship it: lets us test the full `thandv-image generate` path
    end-to-end without a GPU or 6 GB of model weights. Users replace
    this with a real backend from user-land. The format choice (PPM)
    is deliberate: stdlib-writable, no Pillow dependency, immediately
    inspectable with `cat` for verification.
    """

    name = "placeholder"

    def generate(self, request: GenerationRequest) -> GeneratedImage:
        # Map prompt → deterministic RGB so identical prompts produce
        # identical bytes. Useful for tests; useless for art.
        seed_input = f"{request.prompt}|{request.seed}".encode()
        digest = hashlib.sha256(seed_input).digest()
        r, g, b = digest[0], digest[1], digest[2]
        ppm = b"P6\n1 1\n255\n" + bytes([r, g, b])
        request.out_path.parent.mkdir(parents=True, exist_ok=True)
        request.out_path.write_bytes(ppm)
        return GeneratedImage(
            path=request.out_path,
            width=1,
            height=1,
            format="ppm",
            backend=self.name,
        )


def register_placeholder_backend() -> None:
    """Convenience: register the placeholder under its canonical name.

    Not called automatically -- the user has to opt in explicitly, even
    for the placeholder, so the CLI failure messages stay clear about
    the gating model.
    """
    register_backend(PlaceholderBackend.name, PlaceholderBackend())
