# Roadmap

Versioned milestones. Each milestone is small enough to land in a sitting.
The constraint that shapes everything: **inference is fully local; training
may use free internet sources of good quality.**

## v0.0.1 — scaffold ✓

- [x] Agent loop with tool use over Ollama
- [x] Hardware-aware model selection (M-series, NVIDIA, CPU fallback)
- [x] Skills + memory directories loaded into system prompt
- [x] `thandv doctor`, `thandv chat`, `thandv config`
- [x] PyInstaller single-file build
- [x] `install.sh` one-line installer

## v0.1 — usability ✓ (streaming) / partial

- [x] Streaming output from Ollama, tool blocks hidden from user stream
- [x] Test suite (pytest), CI (GitHub Actions), PR template
- [ ] Rich-text terminal UI (color, spinners, code highlighting)
- [ ] `/commands` inside the REPL: `/reset`, `/save`, `/skills`, `/memory`
- [x] Native tool-call format (Ollama `tools`/`message.tool_calls`) plus
      inline-JSON-in-content fallback for models that don't fully use the
      native field. Tool results sent back as `role="tool"` per the OpenAI
      contract. Duplicate-call guard breaks runaway loops. The 7B model
      now reliably fires `retrieve` end-to-end where it previously didn't.
- [ ] Cross-platform `install.sh` for Windows (PowerShell variant)

## v0.2 — personas + eval harness (current)

- [x] Persona architecture (`code` / `writer` / `finance`), each with own
      system prompt, skill subset, and disclaimer
- [x] `--persona` CLI flag, persona-aware eval runner
- [x] Eval harness scaffolding with pluggable suites
- [x] Smoke suite (arithmetic, string ops, `is_prime`) — runs without any
      dataset download
- [x] `thandv eval [suite] [--persona] [--limit] [--list]` command
- [x] Results persisted to `~/.thandv/evals/`
- [x] Real HumanEval integration: lazy `datasets` import, ~300 KB cached
      to `~/.thandv/datasets/`, subprocess-sandboxed verifier with
      timeout. Full baseline on `qwen2.5-coder:7b`: **139/164 = 84.8% pass@1**.
- [x] MBPP integration: same pattern as HumanEval, sanitized config,
      `test_list`-based verifier. Full baseline on `qwen2.5-coder:7b`:
      **206/257 = 80.2% pass@1** (after the prompt fix in v0.2.4.1).
- [x] Small in-repo SWE-Bench-lite **scaffold**: 3 hand-crafted "fix the
      bug" tasks (off-by-one slice, empty-list crash, inclusive-bounds
      comparison). Each ships a broken `solution.py` + `unittest` module;
      verifier runs `python test_solution.py` in a temp dir. Stdlib-only.
      Real SWE-Bench dataset integration is a later milestone.
- [x] Writer eval suite: 6 hand-coded prose tasks with *structural*
      heuristic verifiers (heading count, bullet count, sentence bounds,
      forbidden phrases). Not preference-based — that's a later milestone.
- [x] Finance eval suite: 6 hand-coded tasks. 3 refusal tasks
      (stock-pick, market-prediction, alpha-claim) + 3 allowed-activity
      tasks (concept name, disclaimer compliance, resume bullets).
      Refusal markers drawn from observed `qwen2.5-coder:7b` behaviour.
- [x] Regression tracking: per-(suite, persona) best pass rate persisted
      to `~/.thandv/evals/best.json`. Every full eval run prints a diff
      line vs the previous best; `--show-best` prints the table;
      `--no-update-best` opts out of recording. `--limit` runs deliberately
      don't update the record (a sample isn't a fair comparison).

## v0.3 — local RAG + bundled corpus

- [x] Pure-Python pipeline: paragraph-aware chunker + Ollama
      `nomic-embed-text` embeddings + JSONL store + cosine similarity.
      No new dependency; swap to a real vector DB later behind the same
      API if scale demands it.
- [x] Embeddings via Ollama's `nomic-embed-text` (local).
      Install pulled automatically; `doctor` reports if missing.
- [x] `thandv ingest <path> [--persona] [--stats] [--clear]` command.
- [x] `retrieve` tool wired into the agent; Agent auto-injects the active
      persona into args.
- [x] Per-persona corpora: separate JSONL stores under
      `~/.thandv/corpora/<persona>/`.
- [ ] Bundled starter pack: Python stdlib docs, Unix man pages, common
      library refs (curated, license-clean). Deferred to v0.3.x — bundling
      a corpus inflates the repo and licensing requires careful curation.

## v0.4 — training pipeline (real LoRA on local hardware)

- [x] `thandv train` CLI command (status, queue, enqueue, tick, run,
      pause, resume, stop, enable, backends, datasets, sample-public)
- [x] Real LoRA training on Apple Silicon via MLX-LM (v0.4.1); backend
      abstraction in place for Unsloth/HF+PEFT additions later
- [x] HF→GGUF conversion via llama.cpp `convert_hf_to_gguf.py` (mlx-lm
      0.31's GGUF exporter doesn't support qwen2/qwen3)
- [x] Adapter loading via Ollama Modelfile `FROM <merged>.gguf` + automatic
      `ollama create thandv-adapter-<ts>` on promotion
- [x] Public-dataset ingestion (`thandv train sample-public`): CodeAlpaca,
      Dolly-15k, Alpaca — license + persona hint per dataset (v0.4.2)
- [ ] Training-run provenance file (covered partially by distill
      provenance + adapter dir; full session-history doc pending)

## v0.5 — teacher distillation (ToS-whitelisted)

- [x] `thandv distill --teacher <name>` command with `--prompts-file` and
      `--prompts-from <public-dataset>` sources
- [x] Whitelisted teachers (all Llama-3.3-70B-class, OpenAI-compatible):
      Groq, Together AI, OpenRouter free tiers
- [x] Hard exclusion at the registry layer: Anthropic, OpenAI. Refused
      teachers raise with the exact ToS clause and URL.
- [x] Output lands directly in the training queue with a sidecar
      `<file>.provenance.json` recording teacher + model id + ToS URL +
      timestamps
- [ ] Generate training pairs on tasks the eval flags as weak (next
      milestone — connects v0.5 to the eval-gate via a "weak prompts" file)
- [x] Feeds into the v0.4 trainer through the queue (run `thandv train
      tick` after distillation completes)

## v0.6 — verifier-filtered synthetic data

- [x] `thandv distill-filtered --teacher <name> --suite <suite> --n N` —
      for each task in the suite, distill N candidates from the teacher,
      keep only the ones the suite's verifier accepts.
- [x] Reuses every existing eval suite's verifier (smoke, humaneval,
      mbpp, writer, finance, swe-lite). No new verifier infrastructure
      needed — they were already (prompt, verifier) pairs.
- [x] Output lands in the training queue with a sidecar provenance JSON
      that records the per-task yield breakdown so you can see which
      tasks the model + teacher pair is struggling on.
- [x] Higher default temperature (0.7) than plain distill: we *want*
      diverse candidates so the filter has something to pick from.
- [x] This is where the improvement loop starts compounding. The
      training data's ground truth is the verifier, not the teacher.

## v0.7 — session promotion

- [x] Capture: the agent already writes JSONL sessions to
      `~/.thandv/sessions/`; v0.7 turns them into training data without
      changing the recording format.
- [x] `thandv train sessions` — table of every session with pair /
      tool-error counts and clean/dirty marker.
- [x] `thandv train promote-session <path>` — explicit extraction of
      one session's (user → assistant) pairs to the queue.
- [x] `thandv train promote-sessions` — bulk auto-promotion of every
      session with **zero tool errors** (the "clean" criterion).
      Idempotent via per-session `.promoted` markers.
- [x] Pair extraction strips tool blocks from completions, drops
      tool-only assistant turns, ignores legacy `tool-result`-as-user
      pseudo-events. Each user message pairs with exactly one assistant.
- [x] Provenance sidecar records source session, n_pairs, full stats.
- [x] Per-persona adapter rotation (v0.7.1): `TrainerState` now keys
      `active_models_by_persona` and `best_by_persona` so each persona
      has its own promote/discard slot. `tick(base_model, persona)`
      reads `Persona.eval_suite` (code → smoke, writer → writer,
      finance → finance) for the eval gate. `thandv chat --persona X`
      automatically uses X's promoted adapter when one exists; status
      shows a per-persona table. Old single-field state files migrate
      forward (`active_ollama_model` → `active_models_by_persona[code]`).

## v0.8 — writer persona deepening

- [x] Bundled style guides, public-domain literary corpora
      (`thandv writer bundle-style-corpus`; Strunk 1918 excerpts, Twain,
      Lincoln, Shakespeare, in-house house style. Provenance per snippet.)
- [x] Preference-based eval suite (DPO-style pairs) — `thandv eval writer-prefs`,
      8 hand-crafted pairs covering concise/active/specific/no-hedge/show/
      varied/fresh/strong-opening. A/B position randomised per task id.
- [x] LoRA on prose datasets — wikitext-2-raw-v1 and TinyStories registered as
      public datasets with `persona_hint="writer"`; continuation-style row
      mapper produces `{prompt, completion}` pairs. (FineWeb-Edu and books3 not
      shipped — FineWeb-Edu is large enough to need streaming support we don't
      have yet; books3 was pulled for copyright. Honest substitutes for v0.8.)

## v0.9 — image / arts (separate backend)

- [x] `thandv-image` companion entry point shipped (`thandv/image_cli.py`,
      wired in pyproject `[project.scripts]`). Subcommands:
      `backends`, `register-placeholder`, `generate`.
- [x] Shared persona/skills layer: new `IMAGE` persona in
      `thandv/personas.py` orchestrates the backend, plus
      `thandv/skills/image-prompting.md`. The persona is honest that it
      doesn't synthesise pixels itself -- it composes prompts and
      shells out to `thandv-image`.
- [x] `ImageBackend` Protocol + opt-in gate in
      `thandv/image_backend.py`. Mirrors `paper_trading.py`: opt-in flag
      (`image_generation_enabled` config), `register_backend(...)` for
      user-land adapters (diffusers + SDXL/Flux, MLX-Image Gen, hosted
      APIs), `require_enabled()` raises with distinct messages for each
      failure mode (no opt-in, no backend, multiple without picker,
      unknown name).
- [x] `PlaceholderBackend` ships in-tree: writes a 1x1 PPM whose colour
      is the prompt hash. NOT image generation -- exists only to verify
      the pipeline end-to-end without GPU dependence or model weights.
      Honest framing in the module docstring and CLI output.
- [x] Honest about install-size impact (~6 GB extra): documented in the
      opt-in error message and the image-prompting skill. No diffusers
      bundled in pyproject; users add it themselves when they wire a
      real backend.

**What is intentionally NOT in v0.9:** an actual diffusers + SDXL/Flux
backend implementation. That work belongs in user-land per the same
reasoning as paper-trading -- shipping it would imply we've tested it
end-to-end against the heavy ML stack on multiple GPU/Apple-Silicon
configurations, which we haven't.

## v0.10 — finance persona deepening

- [x] Backtesting tool: pure-Python next-day-execution harness in
      `thandv/finance_tools.py` + `thandv finance backtest`. vectorbt /
      backtrader NOT bundled — too heavy; users who need them should
      call them directly. Honest framing on this in the CLI.
- [x] Portfolio analysis: `thandv finance metrics` (Sharpe/Sortino/max-DD/
      vol) and `thandv finance exposure` (gross/net/by-asset-class/by-symbol).
      Pure stdlib, no pandas.
- [x] 10-K / earnings-call ingestion: `thandv finance ingest-filing` and
      `thandv finance filings` in `thandv/filings.py`; per-filing
      provenance `filing::TYPE::TICKER::filename` in source string.
- [x] Strategy-critique skill: `thandv/skills/strategy-critique.md` covering
      lookahead, survivorship, overfitting, costs, capacity, regime,
      significance. Now actually loaded by the FINANCE persona — see the
      bundled-skill loading fix.
- [x] Paper-trading harness: `thandv/paper_trading.py` ships a
      `PaperTradingAdapter` Protocol + opt-in gate
      (`finance_paper_trading_enabled` config flag). No broker adapters
      bundled; the gate refuses with clear instructions for both
      failure modes (no opt-in, no adapter registered).
- [x] Bundled-skill loading fix: `skills/` moved to `thandv/skills/` and
      shipped via `[tool.setuptools.package-data]` + PyInstaller `--add-data`.
      Previously the repo skill files weren't installed at all, so
      `persona.skills=(...)` declarations silently dropped every entry on
      a fresh install. Now bundled skills load by default, user files at
      `~/.thandv/skills/<stem>.md` override.

## v1.0 — native (C++) build + first stable release

Design note (boundary, JSON-over-stdin/stdout tool protocol, build
infrastructure, cross-binary contract tests, phased plan):
[V1.0_NATIVE.md](V1.0_NATIVE.md).

- [ ] `cpp/` minimal native CLI built on llama.cpp
- [ ] Shared tool protocol with the Python build
- [ ] Single static binary release on macOS + Linux
- [ ] Stable eval baseline; public release notes
