"""Thandv command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

from thandv import __version__, rag, trainer
from thandv.agent import Agent
from thandv.config import OLLAMA_HOST, Config, ensure_dirs
from thandv.evals import (
    EvalResult,
    format_regression_line,
    get_suite,
    list_suites,
    load_best_records,
    run_suite,
    summarise,
    update_best_if_improved,
)
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
    print(f"embed model:    {rag.EMBED_MODEL} ({'available' if rag.embed_model_available() else 'NOT pulled — run: ollama pull ' + rag.EMBED_MODEL})")
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

    if args.show_best:
        records = load_best_records()
        if not records:
            print("(no best records yet)")
            return 0
        for key in sorted(records):
            rec = records[key]
            print(
                f"{key:30s} {rec['n_passed']}/{rec['n_total']} "
                f"({rec['pass_rate']:.1%})  model={rec['model']}  at={rec['at']}"
            )
        return 0

    ensure_dirs()
    cfg = Config.load()
    if not cfg.model:
        cfg.model = pick_model(detect_host())

    # For eval, the *suite's* default_persona is authoritative when the
    # user hasn't explicitly passed --persona. Falling back to cfg.persona
    # ("code") would mean running the finance suite under the code
    # persona, which silently invalidates the eval.
    persona_name = args.persona
    if persona_name is not None:
        try:
            get_persona(persona_name)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2

    if not _check_ollama():
        print("ollama is not running. Start it with: `ollama serve`", file=sys.stderr)
        return 2
    if not _ensure_model(cfg.model):
        print(f"model `{cfg.model}` not pulled. Run: `ollama pull {cfg.model}`", file=sys.stderr)
        return 2

    # Compute the effective persona for the status line.
    try:
        effective_persona = persona_name or get_suite(args.suite).default_persona
    except ValueError:
        effective_persona = persona_name or "?"
    print(f"eval suite={args.suite!r} model={cfg.model} persona={effective_persona}")

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

    # Regression tracking — only on full runs, and only if not opted out.
    # A partial (--limit) run can't fairly compare against the full baseline.
    if not args.no_update_best and args.limit is None and results:
        prev_rate, current_rate, improved = update_best_if_improved(
            args.suite, effective_persona, results, cfg.model
        )
        print(format_regression_line(prev_rate, current_rate, improved))

    return 0 if all(r.passed for r in results) else 1


def cmd_ingest(args: argparse.Namespace) -> int:
    ensure_dirs()

    if args.stats:
        persona = args.persona if args.persona and args.persona != "all" else None
        stats = rag.corpus_stats(persona)
        print(json.dumps(stats, indent=2))
        return 0

    if args.clear:
        if not args.persona:
            print("--clear requires --persona NAME", file=sys.stderr)
            return 2
        removed = rag.clear_corpus(args.persona)
        print(f"cleared persona={args.persona} (removed={removed})")
        return 0

    if not args.path:
        print("ingest requires a path, or --stats / --clear", file=sys.stderr)
        return 2

    src = Path(args.path).expanduser()
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        return 2

    persona = args.persona or "all"
    if persona != "all":
        try:
            get_persona(persona)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2

    if not _check_ollama():
        print("ollama is not running. Start it with: `ollama serve`", file=sys.stderr)
        return 2
    if not rag.embed_model_available():
        print(f"embedding model `{rag.EMBED_MODEL}` not pulled. Run: ollama pull {rag.EMBED_MODEL}", file=sys.stderr)
        return 2

    print(f"ingesting {src} into persona={persona} ...")
    files, chunks = rag.ingest_path(src, persona=persona)
    print(f"done. files={files} chunks={chunks}")
    return 0


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

    if action == "enable":
        return _cmd_train_enable(args, cfg)

    if action == "backends":
        return _cmd_train_backends()

    print(f"unknown train action: {action}", file=sys.stderr)
    return 2


def _cmd_train_backends() -> int:
    """List training backends and their availability on this host."""
    from thandv import training_backend as tb
    for b in tb.BACKENDS:
        avail = "available" if b.is_available() else "not available"
        print(f"{b.name:12s} {avail}")
    picked = tb.pick_backend()
    print(f"\nactive: {picked.name if picked else '(none — install: pip install thandv[train-mlx])'}")
    return 0


def _cmd_train_enable(args: argparse.Namespace, cfg: Config) -> int:
    """Set up the LoRA training pipeline end-to-end.

    Steps:
      1. Pick the best available backend (or tell the user to install one).
      2. Resolve a HuggingFace base-model id from `--hf-model` or default.
      3. Download the HF base model (cached under ~/.thandv/training/base/).
      4. Run a tiny verification training step to prove the pipeline works.
    """
    from thandv import training_backend as tb

    backend = tb.pick_backend()
    if backend is None:
        print(
            "no training backend available. On Apple Silicon, install:\n"
            "    pip install thandv[train-mlx]",
            file=sys.stderr,
        )
        return 2

    hf_model = args.hf_model or _default_hf_model_for(cfg.model)
    print(f"backend:    {backend.name}")
    print(f"hf model:   {hf_model}")

    target = tb.base_model_dir(hf_model)
    if target.exists() and any(target.iterdir()):
        print(f"base model: already cached at {target}")
    else:
        print(f"base model: downloading to {target} (this can take a while)...")
        try:
            tb.fetch_hf_base_model(hf_model)
        except Exception as e:
            print(f"download failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        print(f"base model: downloaded to {target}")

    if args.skip_verify:
        print("(skipping verification training run per --skip-verify)")
        return 0

    print("verification: running a 1-iter training step (proves pipeline works)...")
    try:
        info = tb.verify_training_setup(backend, target)
    except Exception as e:
        print(f"verification failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(f"verification: OK ({info})")
    return 0


def _default_hf_model_for(ollama_tag: str) -> str:
    """Map an Ollama tag like 'qwen2.5-coder:7b' to its HF Hub repo id.

    Trainer needs HF format; Ollama only ships GGUF. The mapping is hardcoded
    for the models in our `runtime.MODEL_LADDER` — extend as we adopt more.
    """
    table = {
        "qwen2.5-coder:7b":  "Qwen/Qwen2.5-Coder-7B",
        "qwen2.5-coder:14b": "Qwen/Qwen2.5-Coder-14B",
        "qwen2.5-coder:32b": "Qwen/Qwen2.5-Coder-32B",
        "qwen2.5-coder:3b":  "Qwen/Qwen2.5-Coder-3B",
        "qwen3-coder:30b":   "Qwen/Qwen3-Coder-30B-A3B",
        "qwen3-coder:14b":   "Qwen/Qwen3-Coder-14B",
        "llama3.2:3b":       "meta-llama/Llama-3.2-3B",
    }
    if ollama_tag in table:
        return table[ollama_tag]
    raise ValueError(
        f"no HF mapping for ollama tag '{ollama_tag}'. "
        f"Pass --hf-model <hf-repo-id> explicitly."
    )


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
    p_eval.add_argument(
        "--show-best",
        action="store_true",
        help="show recorded best pass rates for each (suite, persona) and exit",
    )
    p_eval.add_argument(
        "--no-update-best",
        action="store_true",
        help="don't update the best-record file even on a full run",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_ing = sub.add_parser("ingest", help="ingest a file or directory into a persona's corpus")
    p_ing.add_argument("path", nargs="?", help="file or directory to ingest")
    p_ing.add_argument("--persona", help="target persona corpus (default: 'all')")
    p_ing.add_argument("--stats", action="store_true", help="show corpus stats and exit")
    p_ing.add_argument("--clear", action="store_true", help="clear the persona's corpus (requires --persona)")
    p_ing.set_defaults(func=cmd_ingest)

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
    train_sub.add_parser("backends", help="list training backends and which is active")
    p_enable = train_sub.add_parser(
        "enable",
        help="set up the LoRA training pipeline (pulls HF base model + verifies)",
    )
    p_enable.add_argument(
        "--hf-model",
        help="HuggingFace repo id of the base model (default: inferred from config.model)",
    )
    p_enable.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the tiny verification training run",
    )
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
