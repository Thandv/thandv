"""Thandv command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

from dataclasses import asdict

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

    # If the trainer has promoted a persona-specific adapter, use it
    # instead of the raw base. The user can still pin to base via
    # `--model <tag>` (future v0.7.x CLI flag); for now persona-promoted
    # wins automatically.
    state = trainer.load_state()
    persona_model = state.active_models_by_persona.get(persona_name)
    if persona_model:
        active_model = persona_model
        cfg = Config(**{**asdict(cfg), "model": persona_model})
    else:
        active_model = cfg.model

    if not _check_ollama():
        print("ollama is not running. Start it with: `ollama serve`", file=sys.stderr)
        return 2
    if not _ensure_model(active_model):
        print(f"model `{active_model}` not pulled. Run: `ollama pull {active_model}`", file=sys.stderr)
        return 2

    agent = Agent(config=cfg, persona=persona)
    adapter_note = (
        f" [persona adapter: {persona_model}]" if persona_model else ""
    )
    print(
        f"thandv {__version__} | {active_model} | persona: {persona.name}{adapter_note} | "
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
        print(f"last_eval_pass_rate:   {state.last_eval_pass_rate:.3f}")
        print(f"queue size:            {trainer.queue_size()}")
        if state.active_models_by_persona or state.best_by_persona:
            print("per-persona slots:")
            personas = sorted(
                set(state.active_models_by_persona) | set(state.best_by_persona)
            )
            for p in personas:
                model = state.active_models_by_persona.get(p) or "—"
                rate = state.best_by_persona.get(p, 0.0)
                print(f"  {p:8s}  best={rate:.3f}  active={model}")
        else:
            print("per-persona slots:   (none yet — run `thandv train tick` to seed a baseline)")
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
        persona = getattr(args, "persona", None) or cfg.persona
        try:
            get_persona(persona)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        out = trainer.tick(cfg.model, persona, dry_run=args.dry_run)
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
        persona = getattr(args, "persona", None) or cfg.persona
        try:
            get_persona(persona)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        print(
            f"trainer starting (model={cfg.model}, persona={persona}, "
            f"interval={args.interval}s). Ctrl-C to stop."
        )
        trainer.run_forever(cfg.model, persona, interval_s=args.interval)
        print("trainer stopped.")
        return 0

    if action == "enable":
        return _cmd_train_enable(args, cfg)

    if action == "backends":
        return _cmd_train_backends()

    if action == "datasets":
        return _cmd_train_datasets()

    if action == "sample-public":
        return _cmd_train_sample_public(args)

    if action == "sessions":
        return _cmd_train_sessions()

    if action == "promote-session":
        return _cmd_train_promote_session(args)

    if action == "promote-sessions":
        return _cmd_train_promote_sessions(args)

    print(f"unknown train action: {action}", file=sys.stderr)
    return 2


def _cmd_train_sessions() -> int:
    """One-line summary of every session in ~/.thandv/sessions/."""
    from thandv import session_promotion as sp
    sessions = sp.list_sessions()
    if not sessions:
        print("(no sessions on disk)")
        return 0
    print(f"{'session':40s}  pairs  tools  errs  clean")
    for path in sessions:
        s = sp.stats(path)
        marker = "✓" if s.is_clean else "·"
        print(
            f"{path.name:40s}  {s.n_pairs_extracted:5d}  "
            f"{s.n_tool_calls:5d}  {s.n_tool_errors:4d}  {marker}"
        )
    return 0


def _cmd_train_promote_session(args: argparse.Namespace) -> int:
    """Promote one named session to the training queue."""
    from thandv import session_promotion as sp
    src = Path(args.path).expanduser()
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        return 2
    result = sp.promote_session(src)
    if result is None:
        print(f"nothing to promote: {src.name} has zero (user→assistant) pairs", file=sys.stderr)
        return 1
    queue_path, provenance = result
    print(f"queued:     {queue_path.name}  ({provenance['n_pairs']} pairs)")
    print(f"provenance: {queue_path.name}.provenance.json")
    return 0


def _cmd_train_promote_sessions(args: argparse.Namespace) -> int:
    """Bulk-promote all clean, unpromoted sessions."""
    from thandv import session_promotion as sp
    promoted = sp.promote_clean_sessions(max_sessions=args.max_sessions)
    if not promoted:
        print("(no clean unpromoted sessions found)")
        return 0
    total_pairs = sum(p["n_pairs"] for _, p in promoted)
    print(f"promoted {len(promoted)} sessions, {total_pairs} pairs total:")
    for queue_path, provenance in promoted:
        print(f"  {queue_path.name}  ({provenance['n_pairs']} pairs)")
    return 0


def _cmd_train_datasets() -> int:
    from thandv import training_data as td
    for d in td.list_public_datasets():
        print(f"{d.name:14s} [{d.license:12s}] persona={d.persona_hint:6s} {d.description}")
    return 0


def _cmd_train_sample_public(args: argparse.Namespace) -> int:
    from thandv import training_data as td
    try:
        path = td.queue_public_dataset(args.dataset, args.n, seed=args.seed)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"queued: {path.name}  ({args.n} examples from {args.dataset!r})")
    return 0


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

    hf_model = args.hf_model or tb.ollama_to_hf(cfg.model)
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

    # llama.cpp's converter is what bridges MLX-fused HF safetensors to GGUF
    # for Ollama serving. Clone once on first enable.
    if not (tb.LLAMA_CPP_DIR / "convert_hf_to_gguf.py").exists():
        print(f"llama.cpp: cloning to {tb.LLAMA_CPP_DIR} (for HF→GGUF conversion)...")
        try:
            tb.ensure_llama_cpp()
        except Exception as e:
            print(f"llama.cpp clone failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        print(f"llama.cpp: ready at {tb.LLAMA_CPP_DIR}")
    else:
        print(f"llama.cpp: already cloned at {tb.LLAMA_CPP_DIR}")

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




def cmd_distill(args: argparse.Namespace) -> int:
    """Generate (prompt, completion) pairs from a free-tier teacher LLM."""
    from thandv import teachers, training_data

    if args.list:
        for t in teachers.list_teachers():
            key_set = "✓" if os.environ.get(t.api_key_env) else " "
            print(
                f"[{key_set}] {t.name:30s} persona={t.persona_hint:6s} "
                f"key=${t.api_key_env}"
            )
            print(f"     {t.description}")
            print(f"     {t.free_tier_note}")
            print(f"     ToS: {t.tos_url}")
        return 0

    # Validate the teacher first — refused providers must surface ToS
    # context regardless of whether the prompts source is valid.
    if not args.teacher:
        print("--teacher is required (or pass --list)", file=sys.stderr)
        return 2
    try:
        teachers.get_teacher(args.teacher)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    # Resolve prompts.
    prompts: list[str] = []
    if args.prompts_file:
        src = Path(args.prompts_file).expanduser()
        if not src.exists():
            print(f"not found: {src}", file=sys.stderr)
            return 2
        for line in src.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"skipping non-JSON line in {src.name}", file=sys.stderr)
                continue
            p = obj.get("prompt") if isinstance(obj, dict) else None
            if isinstance(p, str) and p:
                prompts.append(p)
    elif args.prompts_from:
        try:
            examples = training_data.sample_public_dataset(
                args.prompts_from, args.n, seed=args.seed
            )
        except (ValueError, RuntimeError) as e:
            print(str(e), file=sys.stderr)
            return 2
        prompts = [ex["prompt"] for ex in examples]
    else:
        print(
            "distill needs prompts: --prompts-file <jsonl> OR --prompts-from <dataset>",
            file=sys.stderr,
        )
        return 2

    if args.n and len(prompts) > args.n:
        prompts = prompts[: args.n]
    if not prompts:
        print("no prompts to distill", file=sys.stderr)
        return 2

    def progress(i: int, total: int, status: str) -> None:
        marker = "." if status == "ok" else "x"
        print(f"[{i:4d}/{total}] {marker} {status}", flush=True)

    try:
        queue_path, prov_path = teachers.distill(
            args.teacher, prompts, on_progress=progress
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2
    print()
    print(f"queued:     {queue_path.name}  ({len(prompts)} prompts attempted)")
    print(f"provenance: {prov_path.name}")
    return 0


def cmd_distill_filtered(args: argparse.Namespace) -> int:
    """Distill N completions per eval task; keep only those that pass."""
    from thandv import teachers, verifier_filtered

    try:
        teachers.get_teacher(args.teacher)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    try:
        get_suite(args.suite)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    def progress(i: int, total: int, status: str) -> None:
        marker = "+" if status.startswith("pass") else (
            "-" if status.startswith("reject") else "x"
        )
        print(f"[{i:4d}/{total}] {marker} {status}", flush=True)

    try:
        queue_path, stats = verifier_filtered.distill_and_filter(
            args.teacher,
            args.suite,
            n_per_task=args.n,
            limit=args.limit,
            persona=args.persona,
            on_progress=progress,
            temperature=args.temperature,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    print()
    print(f"queued:     {queue_path.name}")
    print(
        f"yield:      {stats['n_passed']}/{stats['n_attempted']}  "
        f"({stats['yield']:.1%})"
    )
    print(f"provenance: {queue_path.name}.provenance.json")
    return 0


def cmd_writer(args: argparse.Namespace) -> int:
    """`thandv writer ...` subcommands for the writer persona."""
    if args.writer_action == "bundle-style-corpus":
        from thandv import style_corpus

        try:
            chunks = style_corpus.bundle_writer_corpus(clear=args.clear)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 2
        action = "re-ingested" if args.clear else "ingested"
        print(
            f"{action} {len(style_corpus.STYLE_CORPUS)} snippets into "
            f"writer corpus ({chunks} chunks)."
        )
        return 0
    print(
        "writer needs an action: bundle-style-corpus [--clear]",
        file=sys.stderr,
    )
    return 2


def cmd_finance(args: argparse.Namespace) -> int:
    """`thandv finance ...` subcommands for the finance persona."""
    action = getattr(args, "finance_action", None)

    if action == "metrics":
        from thandv import finance_tools as ft

        try:
            prices = ft.read_prices_csv(args.prices)
            returns = ft.returns_from_prices(prices)
            metrics = ft.portfolio_metrics(returns)
        except (FileNotFoundError, ValueError) as e:
            print(str(e), file=sys.stderr)
            return 2
        for k, v in metrics.items():
            print(f"{k:14s} {v}")
        print()
        print("Educational only — not investment advice. Local model, no market edge.")
        return 0

    if action == "exposure":
        from thandv import finance_tools as ft

        try:
            positions = ft.parse_positions_csv(args.positions)
        except (FileNotFoundError, ValueError) as e:
            print(str(e), file=sys.stderr)
            return 2
        summary = ft.exposure_summary(positions)
        print(f"gross:  {summary['gross']:.2f}")
        print(f"net:    {summary['net']:.2f}")
        print("by_asset_class:")
        for cls, val in sorted(summary["by_asset_class"].items()):
            print(f"  {cls:12s} {val:.2f}")
        print("by_symbol:")
        for sym, val in sorted(summary["by_symbol"].items()):
            print(f"  {sym:12s} {val:.2f}")
        return 0

    if action == "backtest":
        from thandv import finance_tools as ft

        try:
            prices = ft.read_prices_csv(args.prices)
            signals = ft.read_signals_csv(args.signals)
            result = ft.backtest(prices, signals, commission_bps=args.commission_bps)
        except (FileNotFoundError, ValueError) as e:
            print(str(e), file=sys.stderr)
            return 2
        print(f"total_return  {result['total_return']:.4f}")
        print(f"sharpe        {result['sharpe']:.4f}")
        print(f"max_drawdown  {result['max_drawdown']:.4f}")
        print(f"n_trades      {result['n_trades']}")
        print(f"turnover      {result['turnover']:.4f}")
        print()
        print("Run the strategy-critique checklist before quoting this number.")
        print("Educational only — not investment advice. Local model, no market edge.")
        return 0

    if action == "ingest-filing":
        from thandv import filings

        try:
            chunks = filings.ingest_filing(
                args.path,
                filing_type=args.type,
                ticker=args.ticker,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(str(e), file=sys.stderr)
            return 2
        print(
            f"ingested {args.path} as {args.type} ({args.ticker or 'no-ticker'}): "
            f"{chunks} chunks into the finance corpus."
        )
        return 0

    if action == "filings":
        from thandv import filings

        rows = filings.list_filings()
        if not rows:
            print("(no filings ingested)")
            return 0
        for row in rows:
            print(
                f"{row['filing_type']:14s} {row['ticker']:10s} "
                f"{row['chunks']:4d} chunks  {row['filename']}"
            )
        return 0

    if action == "paper-trade":
        from thandv import paper_trading as pt

        st = pt.status()
        opted = "ENABLED" if st["opted_in"] else "DISABLED"
        adapters = st["adapters"]
        print(f"opt-in:    {opted}")
        print(f"adapters:  {adapters if adapters else '(none)'}")
        if not st["opted_in"]:
            print(
                "\nPaper trading is off. Read NOT_FINANCIAL_ADVICE.md, then opt in:\n"
                "    thandv config --set finance_paper_trading_enabled=true"
            )
        if not adapters:
            print(
                "\nNo broker adapter is bundled. Register your own from user code:\n"
                "    from thandv import paper_trading\n"
                "    paper_trading.register_adapter('alpaca', MyAlpacaAdapter())"
            )
        return 0

    print(
        "finance needs an action: metrics, exposure, backtest, "
        "ingest-filing, filings, paper-trade",
        file=sys.stderr,
    )
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
    p_tick.add_argument("--persona", help="which persona slot to train (default: from config)")
    train_sub.add_parser("pause", help="pause the daemon (it keeps running but skips ticks)")
    train_sub.add_parser("resume", help="resume a paused daemon")
    train_sub.add_parser("stop", help="send SIGTERM to a running daemon")
    p_run = train_sub.add_parser("run", help="run the daemon in the foreground")
    p_run.add_argument("--interval", type=int, default=300, help="seconds between ticks")
    p_run.add_argument("--persona", help="which persona slot to train (default: from config)")
    train_sub.add_parser("backends", help="list training backends and which is active")
    train_sub.add_parser("datasets", help="list registered public datasets")
    p_sample = train_sub.add_parser(
        "sample-public",
        help="sample N rows from a public dataset and queue them for training",
    )
    p_sample.add_argument("dataset", help="dataset name (see `train datasets`)")
    p_sample.add_argument("--n", type=int, default=100, help="how many rows (default 100)")
    p_sample.add_argument("--seed", type=int, default=0, help="random seed (default 0)")

    train_sub.add_parser(
        "sessions",
        help="list recorded chat sessions and their pair / tool-error counts",
    )
    p_promote_one = train_sub.add_parser(
        "promote-session",
        help="extract (user→assistant) pairs from one session and queue them",
    )
    p_promote_one.add_argument("path", help="path to a session .jsonl under ~/.thandv/sessions/")
    p_promote_many = train_sub.add_parser(
        "promote-sessions",
        help="bulk-promote every clean (no tool errors) session not yet promoted",
    )
    p_promote_many.add_argument(
        "--max-sessions", type=int,
        help="cap on how many sessions to promote in one run",
    )
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

    p_distill = sub.add_parser(
        "distill",
        help="generate (prompt, completion) pairs from a free-tier teacher LLM",
    )
    p_distill.add_argument(
        "--teacher",
        help="teacher name (see `thandv distill --list`)",
    )
    p_distill.add_argument(
        "--list", action="store_true",
        help="list registered teachers and which have API keys set, then exit",
    )
    p_distill.add_argument(
        "--prompts-from",
        help="pull prompts from a registered public dataset (see `thandv train datasets`)",
    )
    p_distill.add_argument(
        "--prompts-file",
        help="path to a JSONL file; each line has a 'prompt' field",
    )
    p_distill.add_argument(
        "--n", type=int, default=100,
        help="max number of prompts to distill (default 100)",
    )
    p_distill.add_argument(
        "--seed", type=int, default=0,
        help="random seed when pulling prompts from a public dataset",
    )
    p_distill.set_defaults(func=cmd_distill)

    p_dfilt = sub.add_parser(
        "distill-filtered",
        help="generate completions from a teacher; keep only those that pass the suite's verifier",
    )
    p_dfilt.add_argument("--teacher", required=True, help="teacher name (see `thandv distill --list`)")
    p_dfilt.add_argument(
        "--suite", required=True,
        help="eval suite to use as the source of (prompt, verifier) pairs",
    )
    p_dfilt.add_argument(
        "--n", type=int, default=3,
        help="completions to request per task (default 3 — higher means more chances to pass)",
    )
    p_dfilt.add_argument(
        "--limit", type=int,
        help="cap the number of tasks (default: all)",
    )
    p_dfilt.add_argument(
        "--persona",
        help="record persona in provenance (informational)",
    )
    p_dfilt.add_argument(
        "--temperature", type=float, default=0.7,
        help="teacher sampling temperature (default 0.7 — higher = more diverse candidates)",
    )
    p_dfilt.set_defaults(func=cmd_distill_filtered)

    p_fin = sub.add_parser("finance", help="finance-persona helpers")
    fin_sub = p_fin.add_subparsers(dest="finance_action")
    p_fm = fin_sub.add_parser(
        "metrics",
        help="Sharpe / Sortino / max drawdown / volatility from a prices CSV",
    )
    p_fm.add_argument("prices", help="CSV with a `price` column")
    p_fe = fin_sub.add_parser(
        "exposure",
        help="gross / net / by-asset-class exposure from a positions CSV",
    )
    p_fe.add_argument("positions", help="CSV with symbol,qty,avg_price[,asset_class]")
    p_fb = fin_sub.add_parser(
        "backtest",
        help="pure-Python next-day-execution backtest (sanity check, not production)",
    )
    p_fb.add_argument("--prices", required=True, help="CSV with a `price` column")
    p_fb.add_argument("--signals", required=True, help="CSV with a `signal` column (-1/0/1)")
    p_fb.add_argument(
        "--commission-bps", type=float, default=0.0,
        help="round-trip commission per unit of turnover, in basis points",
    )
    p_fi = fin_sub.add_parser(
        "ingest-filing",
        help="add a plain-text 10-K / earnings transcript to the finance RAG corpus",
    )
    p_fi.add_argument("path", help="path to a plain-text filing")
    p_fi.add_argument(
        "--type", default="10-K",
        help="filing type (10-K, 10-Q, 8-K, earnings-call, annual-report, other)",
    )
    p_fi.add_argument("--ticker", help="ticker symbol, e.g. AAPL")
    fin_sub.add_parser(
        "filings",
        help="list filings already ingested under the finance persona",
    )
    fin_sub.add_parser(
        "paper-trade",
        help="show paper-trading opt-in status and registered adapters",
    )
    p_fin.set_defaults(func=cmd_finance)

    p_writer = sub.add_parser("writer", help="writer-persona helpers")
    writer_sub = p_writer.add_subparsers(dest="writer_action")
    p_bundle = writer_sub.add_parser(
        "bundle-style-corpus",
        help="ingest the bundled public-domain style snippets into the writer corpus",
    )
    p_bundle.add_argument(
        "--clear",
        action="store_true",
        help="clear the writer corpus before ingesting (keeps the bundle idempotent)",
    )
    p_writer.set_defaults(func=cmd_writer)

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
