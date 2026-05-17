# Thandv

A local, open-weights, Claude-style coding assistant. Research scaffolding for
exploring how far a small, self-hostable agent can be pushed toward
frontier-model quality through skills, memory, distillation, and adapter
fine-tuning — without ever sending a token to a paid API.

> **Status:** v0.2 — usable scaffold with personas, eval harness, and a
> background training daemon. Not Claude. See
> [`research/SELF_IMPROVEMENT.md`](research/SELF_IMPROVEMENT.md) for the
> honest gap analysis and the roadmap to close (some of) it.
>
> **Docs:** [`docs/README.md`](docs/README.md) is the index. Start with
> [`docs/USAGE.md`](docs/USAGE.md).

## What you get

- A `thandv` CLI with an agent loop, tool use (`read_file`, `write_file`,
  `edit_file`, `list_dir`, `run_bash`), persistent sessions, a skills
  directory, and a memory directory.
- **Personas**: `code` (default), `writer`, `finance`. Same base model,
  different system prompt + skill subset. Switch with `--persona`.
- **Eval harness**: `thandv eval` runs pluggable suites against the active
  model and persona, with results persisted under `~/.thandv/evals/`.
- **Local RAG**: `thandv ingest <path>` chunks your files, embeds them via
  Ollama's `nomic-embed-text`, and stores per-persona corpora under
  `~/.thandv/corpora/`. The agent can call a `retrieve` tool that's
  auto-scoped to the active persona. Pure-Python, no new deps.
- **Background training daemon**: `thandv train run` reads from a local
  queue, runs an eval-gated training cycle, and only promotes a new
  adapter if it beats the current best on the chosen suite. Pause,
  resume, and stop are first-class commands. macOS launchd and Linux
  systemd templates ship in `scripts/`. The inner LoRA step is stubbed
  in v0.2.x; the orchestration is real. See
  [`docs/TRAINING.md`](docs/TRAINING.md).
- Hardware-aware model selection: picks the strongest Qwen3-Coder /
  Qwen2.5-Coder variant that fits the host (Apple Silicon, NVIDIA, or CPU).
- Streaming output (tool-block JSON hidden from the user stream).
- One-line install that pulls Ollama and the right model.
- A PyInstaller build script that produces a single-file `thandv` binary.
- A parallel C++ track ([`cpp/`](cpp/)) for an eventual native build on top
  of `llama.cpp`.

> The `finance` persona is educational and analytical only — see
> [`NOT_FINANCIAL_ADVICE.md`](NOT_FINANCIAL_ADVICE.md).

## Quick start (Apple Silicon, Linux, Windows-WSL)

```bash
git clone git@github.com:Thandv/thandv.git
cd thandv
./install.sh           # installs ollama if missing, pulls the right model, installs `thandv`
thandv doctor          # confirms host, model, ollama status, active persona
thandv chat            # REPL (uses code persona by default)
thandv chat --persona writer "draft an outline for a short essay on focus"
thandv chat --persona finance "summarise the key risks in this 10-K"
thandv eval            # run the smoke suite against the current model + persona
thandv eval --list     # show available suites and personas
thandv train status    # show training daemon + queue + eval-gate state
```

## Documentation

| Document | Purpose |
|---|---|
| [docs/README.md](docs/README.md) | Index. Start here. |
| [docs/USAGE.md](docs/USAGE.md) | End-user guide: every CLI command + config. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module map, data flow, key decisions. |
| [docs/TRAINING.md](docs/TRAINING.md) | Training pipeline + background daemon setup. |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Dev environment, tests, CI, adding personas/tools/suites. |
| [research/ROADMAP.md](research/ROADMAP.md) | Versioned milestones through v1.0. |
| [research/SELF_IMPROVEMENT.md](research/SELF_IMPROVEMENT.md) | Honest take on local self-improvement. |
| [NOT_FINANCIAL_ADVICE.md](NOT_FINANCIAL_ADVICE.md) | Finance persona scope + legal frame. |

## Layout

```
thandv/                 core Python package
  cli.py                CLI entry: chat, doctor, eval, train, config
  agent.py              agent loop + streaming + tool dispatch
  personas.py           code / writer / finance persona definitions
  evals.py              eval harness (smoke suite included)
  trainer.py            background training daemon (queue, eval-gate, state)
  rag.py                chunking, embeddings, per-persona corpora, retrieve
  runtime.py            host detection + model ladder
  tools.py              read_file, write_file, edit_file, list_dir, run_bash
  memory.py             skills + memory + session log
  config.py             on-disk config
skills/                 seed skills loaded by personas (markdown)
tests/                  pytest suite (no Ollama needed; all I/O mocked)
docs/                   user + developer + training docs
research/               roadmap + self-improvement notes
cpp/                    native build track (llama.cpp-based, future)
scripts/                launchd plist, systemd unit, distill/eval stubs
NOT_FINANCIAL_ADVICE.md plain-language statement of the finance persona's scope
install.sh              one-line installer
build.sh                PyInstaller single-file binary build
```

## Why "as close to Claude as possible"?

Frontier models like Claude Opus 4.7 are closed-weights, hundreds of
billions of parameters, trained on compute budgets in the nine-figure range.
Nothing you install locally will be Claude. What you *can* do:

1. **Pick the right open base.** Qwen3-Coder is currently the strongest open
   coding model; we default to it.
2. **Lean on agent scaffolding.** Most of Claude Code's practical usefulness
   isn't raw model IQ — it's the loop, the tools, and the system prompt.
   That part we can copy.
3. **Specialise via skills, memory, and LoRA.** A 14B model with strong
   skills for *your* codebase often beats a generic 70B on *your* tasks.
4. **Distill from a frontier teacher** (carefully, and only where licensing
   permits) to push the small model further on tasks you care about.

See [`research/SELF_IMPROVEMENT.md`](research/SELF_IMPROVEMENT.md) for the
candid version of what's realistic and what isn't.

## License

MIT.
