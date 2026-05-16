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
- [ ] Native tool-call format (Qwen / Llama function-calling) instead of
      fenced JSON blocks — fewer parse failures
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
- [ ] Real HumanEval and MBPP integrations (download via `datasets`)
- [ ] Small in-repo SWE-Bench-lite
- [ ] Writer + finance eval suites (preference-based and finance-discipline)
- [ ] Regression tracking across model + prompt versions

## v0.3 — local RAG + bundled corpus

- [ ] Embedded vector DB (LanceDB — pure Python, fits PyInstaller cleanly)
- [ ] Embeddings via Ollama's `nomic-embed-text` (local)
- [ ] `thandv ingest <path>` command for user corpora
- [ ] `retrieve` tool wired into the agent
- [ ] Bundled starter pack: Python stdlib docs, Unix man pages, common
      library refs (curated, license-clean)
- [ ] Per-persona corpora: `code/` (language refs), `writer/` (style
      guides, public-domain literature), `finance/` (educational only)

## v0.4 — training pipeline scaffolding

- [ ] `thandv train` CLI command
- [ ] Unsloth-based LoRA training on local hardware
- [ ] Initial training data: license-filtered slices of The Stack v2 +
      CodeAlpaca (HuggingFace `datasets`)
- [ ] Adapter loading via Ollama Modelfiles
- [ ] Training-run provenance (`training/sources.yml`)

## v0.5 — teacher distillation (ToS-whitelisted)

- [ ] `thandv distill --teacher <name>` command
- [ ] Whitelisted teachers: Gemini free tier, Groq free hosting,
      Together AI free, OpenRouter free models
- [ ] Hard exclusion: Anthropic, OpenAI (ToS forbids competing-model
      training)
- [ ] Generate training pairs on tasks the eval flags as weak
- [ ] Feed into the v0.4 trainer

## v0.6 — verifier-filtered synthetic data

- [ ] Generate candidates at scale via free teachers
- [ ] Filter locally: unit tests pass, type-check, lint, smoke runs
- [ ] Train on survivors only
- [ ] This is where the improvement loop starts compounding

## v0.7 — session promotion

- [ ] Capture verifier-passed user sessions
- [ ] Auto-promote to the training set
- [ ] `thandv train --from-sessions` flag
- [ ] Per-persona adapter rotation

## v0.8 — writer persona deepening

- [ ] Bundled style guides, public-domain literary corpora
- [ ] Preference-based eval suite (DPO-style pairs)
- [ ] LoRA on prose datasets (selected slices of FineWeb-Edu, books3-clean)

## v0.9 — image / arts (separate backend)

- [ ] `thandv-image` companion binary using `diffusers` + SDXL / Flux
- [ ] Shared persona/skills layer, separate model backend
- [ ] Honest about install-size impact (~6GB extra)

## v0.10 — finance persona deepening

- [ ] Backtesting tools: vectorbt / backtrader wrappers
- [ ] Portfolio analysis: CSV parsing → Sharpe, Sortino, drawdown, exposure
- [ ] 10-K / earnings-call ingestion pipeline (RAG)
- [ ] Strategy-critique skill: lookahead, survivorship, overfitting checks
- [ ] Paper-trading harness (user-configured broker keys; not bundled)
- [ ] Refer to [NOT_FINANCIAL_ADVICE.md](../NOT_FINANCIAL_ADVICE.md)

## v1.0 — native (C++) build + first stable release

- [ ] `cpp/` minimal native CLI built on llama.cpp
- [ ] Shared tool protocol with the Python build
- [ ] Single static binary release on macOS + Linux
- [ ] Stable eval baseline; public release notes
