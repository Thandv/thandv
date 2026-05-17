"""Thandv command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

from thandv import __version__, trainer
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


def cmd_train(args: argparse.Namespace) -> int:
    cfg = Config.load()
    if not cfg.model:
        cfg.model = pick_model(detect_host())

    action = args.train_action or "status"

    if action == "status":
        state = trainer.load_state()
        print(f"running:               {trainer.is_running()}")
        print(f"paused:                {state.paused}")
        print(f"started_at:            {state.started_at or '—'}")
        print(f"last_tick_at:          {state.last_tick_at or '—'}")
        print(f"ticks_completed:       {state.ticks_completed}")
        print(f"baseline_measured:     {state.baseline_measured}")
        print(f"last_eval_pass_rate:   {state.last_eval_pass_rate:.3f}")
        print(f"best_eval_pass_rate:   {state.best_eval_pass_rate:.3f}")
        print(f"active_adapter:        {state.active_adapter or '—'}")
        print(f"queue size:            {trainer.queue_size()}")
        return 0

    if action == "queue":
        for p in sorted(trainer.QUEUE_DIR.glob("*.jsonl")):
            with p.open() as f:
                n = sum(1 for _ in f)
            print(f"{p.name}\t{n} examples")
        return 0

    if action == "enqueue":
        src = Path(args.path).expanduser()
        if not src.exists():
            print(f"not found: {src}", file=sys.stderr)
            return 2
        dst = trainer.enqueue_path(src)
        print(f"queued: {dst.name}")
        return 0

    if action == "tick":
        out = trainer.tick(cfg.model, dry_run=args.dry_run)
        print(json.dumps(out, indent=2))
        return 0

    if action == "pause":
        trainer.pause()
        print("paused")
        return 0

    if action == "resume":
        trainer.resume()
        print("resumed")
        return 0

    if action == "stop":
        if trainer.stop():
            print("sent SIGTERM to running trainer")
            return 0
        print("no running trainer", file=sys.stderr)
        return 1

    if action == "run":
        print(f"trainer starting (model={cfg.model}, interval={args.interval}s). Ctrl-C to stop.")
        trainer.run_forever(cfg.model, interval_s=args.interval)
        print("trainer stopped.")
        return 0

    print(f"unknown train action: {action}", file=sys.stderr)
    return 2


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

    p_train = sub.add_parser("train", help="manage the background training daemon")
    train_sub = p_train.add_subparsers(dest="train_action")
    train_sub.add_parser("status", help="show daemon + state (default action)")
    train_sub.add_parser("queue", help="list queued example files")
    p_enq = train_sub.add_parser("enqueue", help="add a JSONL file to the queue")
    p_enq.add_argument("path", help="path to a .jsonl file of training examples")
    p_tick = train_sub.add_parser("tick", help="run one train→eval→promote iteration now")
    p_tick.add_argument("--dry-run", action="store_true", help="don't actually train")
    train_sub.add_parser("pause", help="pause the daemon (it keeps running but skips ticks)")
    train_sub.add_parser("resume", help="resume a paused daemon")
    train_sub.add_parser("stop", help="send SIGTERM to a running daemon")
    p_run = train_sub.add_parser("run", help="run the daemon in the foreground")
    p_run.add_argument("--interval", type=int, default=300, help="seconds between ticks")
    p_train.set_defaults(func=cmd_train)

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
