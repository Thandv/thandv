# Roadmap

Versioned milestones. Each milestone is small enough to land in a sitting.

## v0.0.1 — scaffold (current)

- [x] Agent loop with tool use over Ollama
- [x] Hardware-aware model selection (M-series, NVIDIA, CPU fallback)
- [x] Skills + memory directories loaded into system prompt
- [x] `thandv doctor`, `thandv chat`, `thandv config`
- [x] PyInstaller single-file build
- [x] `install.sh` one-line installer

## v0.1 — usability

- [x] Streaming output from Ollama (`stream=True`), tool blocks still hidden
- [ ] Rich-text terminal UI (color, spinners, code highlighting)
- [ ] `/commands` inside the REPL: `/reset`, `/save`, `/skills`, `/memory`
- [ ] Native tool-call format (Qwen / Llama function-calling) instead of
      fenced JSON blocks — fewer parse failures
- [ ] Cross-platform `install.sh` for Windows (PowerShell variant)

## v0.2 — evals

- [ ] Eval harness: HumanEval, MBPP, and a small in-repo SWE-Bench-lite
- [ ] `thandv eval <suite>` command, results to `~/.thandv/evals/`
- [ ] Regression tracking across model + prompt versions

## v0.3 — adapters

- [ ] `scripts/distill.py`: capture Claude/GPT teacher traces (when the user
      provides their own API key) into a JSONL training set
- [ ] `scripts/train_lora.py`: train a LoRA on the captured traces (Qwen 7B
      first, runs on a single 24GB GPU or M-series with 32GB)
- [ ] Adapter loading via Ollama Modelfiles or a llama.cpp side-path

## v0.4 — native (C++) build

- [ ] `cpp/` minimal native CLI built on llama.cpp
- [ ] Shared tool protocol with the Python build (so skills/memory transfer)
- [ ] Single static binary release on macOS + Linux

## v0.5 — self-improvement loop

- [ ] Per-session success/failure labelling (explicit user feedback +
      heuristic from tool exit codes)
- [ ] Auto-promote successful traces into the distillation set
- [ ] Auto-prune skills/memory that correlate with failures
