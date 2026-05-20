"""`thandv-image` CLI entry point.

Separate binary in the v0.9 sense: a distinct script entry registered
in pyproject.toml, sharing the persona + skill + config layer with the
main `thandv` binary but routing all work to an `ImageBackend`.

The CLI does NOT ship a real image-generation backend. The default
`placeholder` backend writes a 1x1 PPM whose colour is the prompt
hash -- enough to verify the pipeline end-to-end, useless for art.
Real backends (diffusers + SDXL/Flux, MLX-Image Gen, hosted APIs)
plug in from user-land via `image_backend.register_backend(...)`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thandv import __version__, image_backend as ib
from thandv.config import Config


def _cmd_backends(_args: argparse.Namespace) -> int:
    st = ib.status()
    opted = "ENABLED" if st["opted_in"] else "DISABLED"
    backends = st["backends"]
    print(f"opt-in:    {opted}")
    print(f"backends:  {backends if backends else '(none)'}")
    if not st["opted_in"]:
        print(
            "\nImage generation is off. Opt in:\n"
            "    thandv config --set image_generation_enabled=true\n"
            "Real backends (diffusers + SDXL/Flux) add ~6 GB to install."
        )
    if not backends:
        print(
            "\nNo backend registered. Thandv ships only PlaceholderBackend\n"
            "(a 1x1 PPM stub). Register a real backend from user code:\n"
            "    from thandv import image_backend\n"
            "    image_backend.register_backend('sdxl', MyDiffusersBackend())\n"
            "Or register the placeholder for pipeline testing:\n"
            "    from thandv import image_backend\n"
            "    image_backend.register_placeholder_backend()"
        )
    return 0


def _cmd_register_placeholder(_args: argparse.Namespace) -> int:
    """Convenience for users who just want to smoke-test the pipeline.

    Writes a tiny shim into the user's config dir under
    `register_on_start.py` so subsequent `thandv-image generate` calls
    in this same process see the backend. Out of scope to persist
    across processes -- a real backend belongs in user code, not in
    Thandv's autoload."""
    ib.register_placeholder_backend()
    print(f"registered: {ib.PlaceholderBackend.name}")
    print("(in-process only; for persistence, register from your own Python startup)")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = Config.load()
    try:
        backend = ib.require_enabled(args.backend, cfg=cfg)
    except ib.ImageGenerationDisabled as e:
        print(str(e), file=sys.stderr)
        return 2
    request = ib.GenerationRequest(
        prompt=args.prompt,
        out_path=Path(args.out).expanduser().resolve(),
        width=args.width,
        height=args.height,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        negative_prompt=args.negative_prompt,
    )
    try:
        result = backend.generate(request)
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(
        f"generated: {result.path}  "
        f"({result.width}x{result.height} {result.format} via {result.backend})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="thandv-image",
        description="Local image-generation coordinator (companion to `thandv`).",
    )
    parser.add_argument("--version", action="version", version=f"thandv {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p_b = sub.add_parser("backends", help="show opt-in status and registered backends")
    p_b.set_defaults(func=_cmd_backends)

    p_rp = sub.add_parser(
        "register-placeholder",
        help="register the in-tree PlaceholderBackend (in-process only)",
    )
    p_rp.set_defaults(func=_cmd_register_placeholder)

    p_g = sub.add_parser("generate", help="generate one image (requires opt-in + a backend)")
    p_g.add_argument("prompt", help="text prompt")
    p_g.add_argument("-o", "--out", required=True, help="output path (file extension is a hint, not enforced)")
    p_g.add_argument("--backend", help="pick one when multiple are registered")
    p_g.add_argument("--width", type=int, default=512)
    p_g.add_argument("--height", type=int, default=512)
    p_g.add_argument("--seed", type=int, default=None)
    p_g.add_argument("--steps", type=int, default=20)
    p_g.add_argument("--guidance", type=float, default=7.5)
    p_g.add_argument("--negative-prompt", default="", dest="negative_prompt")
    p_g.set_defaults(func=_cmd_generate)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
