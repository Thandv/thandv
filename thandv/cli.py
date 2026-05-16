"""Thandv command-line entry point."""

from __future__ import annotations

import argparse
import sys

import requests

from thandv import __version__
from thandv.agent import Agent
from thandv.config import OLLAMA_HOST, Config, ensure_dirs
from thandv.evals import EvalResult, list_suites, run_suite, summarise
from thandv.personas import get_persona, list_personas
from thandv.runtime import describe, detect_host, pick_model


def _check_ollama() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _ensure_model(model: str) -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        tags = [m["name"] for m in r.json().get("models", [])]
        return any(t.startswith(model) for t in tags)
    except requests.RequestException:
        return False


def _resolve_persona(cfg: Config, override: str | None) -> str:
    name = override or cfg.persona
    get_persona(name)  # validates; raises ValueError on unknown
    return name


def cmd_doctor(_: argparse.Namespace) -> int:
    host = detect_host()
    cfg = Config.load()
    model = cfg.model or pick_model(host)
    print(describe(host, model))
    print(f"persona:        {cfg.persona}")
    print(f"ollama running: {_check_ollama()}")
    print(f"model pulled:   {_ensure_model(model)}")
    print(f"thandv version: {__version__}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    ensure_dirs()
    cfg = Config.load()
    if not cfg.model:
        cfg.model = pick_model(detect_host())
        cfg.save()

    try:
        persona_name = _resolve_persona(cfg, getattr(args, "persona", None))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    persona = get_persona(persona_name)

    if not _check_ollama():
        print("ollama is not running. Start it with: `ollama serve`", file=sys.stderr)
        return 2
    if not _ensure_model(cfg.model):
        print(f"model `{cfg.model}` not pulled. Run: `ollama pull {cfg.model}`", file=sys.stderr)
        return 2

    agent = Agent(config=cfg, persona=persona)
    print(
        f"thandv {__version__} | {cfg.model} | persona: {persona.name} | "
        f"session: {agent.session_path.name}"
    )
    print("Type your message. Ctrl-D or /exit to quit.\n")

    if args.prompt:
        for chunk in agent.turn(args.prompt):
            print(chunk, end="", flush=True)
        print()
        return 0

    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return 0
        for chunk in agent.turn(user):
            print(chunk, end="", flush=True)
        print()


def cmd_eval(args: argparse.Namespace) -> int:
    if args.list:
        for s in list_suites():
            print(s)
        print()
        print("personas: " + ", ".join(list_personas()))
        return 0

    ensure_dirs()
    cfg = Config.load()
    if not cfg.model:
        cfg.model = pick_model(detect_host())

    try:
        persona_name = _resolve_persona(cfg, args.persona)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    if not _check_ollama():
        print("ollama is not running. Start it with: `ollama serve`", file=sys.stderr)
        return 2
    if not _ensure_model(cfg.model):
        print(f"model `{cfg.model}` not pulled. Run: `ollama pull {cfg.model}`", file=sys.stderr)
        return 2

    print(f"eval suite={args.suite!r} model={cfg.model} persona={persona_name}")

    def progress(i: int, total: int, r: EvalResult) -> None:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{i}/{total}] {r.task_id:24s} {status} ({r.secs:.1f}s)")

    try:
        results = run_suite(
            args.suite,
            model=cfg.model,
            persona_name=persona_name,
            limit=args.limit,
            on_progress=progress,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    print()
    print(summarise(results))
    return 0 if all(r.passed for r in results) else 1


def cmd_config(args: argparse.Namespace) -> int:
    cfg = Config.load()
    if args.set:
        for kv in args.set:
            if "=" not in kv:
                print(f"bad --set: {kv}", file=sys.stderr)
                return 2
            k, v = kv.split("=", 1)
            if not hasattr(cfg, k):
                print(f"unknown key: {k}", file=sys.stderr)
                return 2
            current = getattr(cfg, k)
            if isinstance(current, bool):
                v = v.lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                v = int(v)
            elif isinstance(current, float):
                v = float(v)
            setattr(cfg, k, v)
        cfg.save()
    for k in cfg.__dataclass_fields__:
        print(f"{k}={getattr(cfg, k)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="thandv", description="Local Claude-style assistant")
    parser.add_argument("--version", action="version", version=f"thandv {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p_chat = sub.add_parser("chat", help="start an interactive chat (default)")
    p_chat.add_argument("prompt", nargs="?", help="one-shot prompt instead of REPL")
    p_chat.add_argument("--persona", help="persona to use (default: from config)")
    p_chat.set_defaults(func=cmd_chat)

    p_doctor = sub.add_parser("doctor", help="show host + model + ollama status")
    p_doctor.set_defaults(func=cmd_doctor)

    p_eval = sub.add_parser("eval", help="run an eval suite")
    p_eval.add_argument("suite", nargs="?", default="smoke", help="suite name (default: smoke)")
    p_eval.add_argument("--persona", help="persona to evaluate (default: suite default)")
    p_eval.add_argument("--limit", type=int, help="max tasks to run")
    p_eval.add_argument("--list", action="store_true", help="list available suites and exit")
    p_eval.set_defaults(func=cmd_eval)

    p_config = sub.add_parser("config", help="show or set config keys")
    p_config.add_argument("--set", action="append", help="key=value", default=[])
    p_config.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        args.cmd = "chat"
        args.prompt = None
        args.persona = None
        return cmd_chat(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
