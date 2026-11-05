# Status

v0.10.0, updated 2026-11-05 (epic THV-E2). Keep this file current with each landing.

## Works (tested)

- `thandv chat` REPL and one-shot. Tool calls are parsed from model output only.
  `run_bash`, `write_file` and `edit_file` need a human `y`. Writes are confined
  to the start directory, `run_bash` has a denylist and a process-group timeout,
  and Ctrl-C never approves a tool. These were reviewed over three rounds plus a
  Ctrl-C injection harness.
- Exit codes of a one-shot `thandv chat "..."`: 0 ok, 1 model error, 2 setup error
  (Ollama down, model missing, bad persona), 130 Ctrl-C. These hold for the `thandv` script, `python -m thandv`
  and the PyInstaller binary (`thandv/__main__.py` exits with `main()`'s status).
- A session log that cannot be written (disk full, permissions) prints one
  warning on stderr, and the chat continues without logging.
- `thandv doctor`, `eval` (smoke suite), `ingest`/`retrieve`, `train status|queue|enqueue|pause|resume|stop`,
  `config`, personas, and opt-in `THANDV_DEBUG=1` logging.
- Test suite: `pytest -q` runs 698 tests, hermetic (no network, no real `~/.thandv`),
  in about 20–30 s on the owner's Mac. CI (`.github/workflows/ci.yml`) runs ruff and
  pytest on ubuntu-22.04 and macos-15 with Python 3.10–3.12, plus a PyInstaller build smoke.

## Untested or not run

- No real chat turn against a model has been recorded since 2026-05-18 (the last
  session and eval in `~/.thandv`). The v0.10 approval gate and Ctrl-C handling
  have only been run against stubbed models.
- The training path (MLX-LM LoRA, then merged GGUF, then `ollama create`, then the
  eval gate) is implemented and covered by mocked tests but has never been run end
  to end. The HF+PEFT fallback is not implemented.
- humaneval and mbpp evals (they need `datasets` and a model). Linux/WSL `install.sh`.
  The PyInstaller binary is built only in CI.
- Wrong config field types (for example `"model": 5`) are accepted without a
  warning. This is not fixed.

## Held by the owner

- **Real use:** use the v0.10 REPL for real coding sessions on this Mac and keep
  the session logs and one smoke/humaneval eval as the run record.
- **MLX training path:** decide whether to spend the download and compute on one
  real end-to-end training run, or re-scope the goal to exclude it.

## `thandv doctor` (2026-11-05)

This was run with the venv's `thandv doctor`, from outside the repo:

```
OS=Darwin | arch=arm64 | RAM=16GB | backend=Metal | model=qwen2.5-coder:7b
persona:        code
ollama running: True
model pulled:   True
embed model:    nomic-embed-text (available)
thandv version: 0.10.0
```

## Repairing the launcher after a move

A venv records absolute paths. After the checkout moved, the venv's `thandv`
failed with `bad interpreter`, and `thandv` was not on PATH. Repair (dev install,
keeps pytest and ruff for the pre-commit hook):

```bash
cd path/to/thandv
python3.14 -m venv --clear .venv
.venv/bin/python -m pip install -e '.[dev]'
mkdir -p ~/.local/bin
ln -sfn "$PWD/.venv/bin/thandv" ~/.local/bin/thandv
thandv doctor
```

For a user install without dev tools, re-run `./install.sh` instead.
