# Usage

## Install

### One-line install (recommended)

```bash
git clone git@github.com:Thandv/thandv.git
cd thandv
./install.sh
```

`install.sh` installs Ollama if missing, picks the strongest open-weights
model that fits your host, pulls it via Ollama, and installs the `thandv`
CLI.

### From source (development)

```bash
git clone git@github.com:Thandv/thandv.git
cd thandv
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,build]"
```

### Build a single-file binary

```bash
./build.sh
# produces dist/thandv (a PyInstaller bundle)
```

The binary still needs Ollama installed on the target machine; it bundles
only the Python runtime and the `thandv` package.

## Quick start

```bash
thandv doctor                              # show host, model, ollama, persona
thandv chat                                # REPL with the code persona
thandv chat "explain the agent loop"       # one-shot prompt
thandv chat --persona writer "outline an essay on focus"
thandv chat --persona finance "summarise the risk section in this 10-K"
thandv eval                                # run smoke suite vs current model
thandv eval --list                         # show suites + personas
thandv ingest ./my-docs --persona code     # build a corpus for the code persona
thandv ingest --stats                      # show what's been ingested
thandv train status                        # show daemon + queue + state
```

## Personas

A persona scopes Thandv to a domain. Same base model, different system
prompt and skill subset. Switch with `--persona <name>` on `chat` or `eval`.

| Persona  | Purpose                                                                 |
|----------|-------------------------------------------------------------------------|
| `code`   | Default. Reads, writes, edits, runs code via local tools.               |
| `writer` | Drafts and edits prose; outlines essays/chapters; summarises long text. |
| `finance`| Career and personal-finance work, investment **research** (not advice), strategy code + backtests. See [NOT_FINANCIAL_ADVICE.md](../NOT_FINANCIAL_ADVICE.md). |
| `image`  | Coordinates the `thandv-image` companion binary. Composes prompts, suggests params; does not synthesise pixels itself. Requires a user-registered `ImageBackend`. |

Set the default persona for new sessions:

```bash
thandv config --set persona=writer
```

Detailed prompts and skill lists for each persona live in
[`thandv/personas.py`](../thandv/personas.py).

## CLI reference

### `thandv chat [PROMPT] [--persona NAME]`

Interactive REPL with the active persona's system prompt + skills. With a
positional `PROMPT`, runs one turn and exits.

Inside the REPL: `/exit` or `/quit` (or Ctrl-D) to leave.

Sessions are logged as JSONL under `~/.thandv/sessions/<unix-ts>.jsonl`.

### `thandv doctor`

Diagnostic snapshot. Reports:

- OS / arch / RAM / GPU detection
- Selected model (from config, or auto-picked)
- Whether Ollama is up
- Whether the model is pulled
- Active persona
- `thandv` version

### `thandv eval [SUITE] [--persona NAME] [--limit N] [--list] [--show-best] [--no-update-best]`

Runs an eval suite against the current model + persona. Results land in
`~/.thandv/evals/<suite>-<unix-ts>.json`.

`--limit N` runs only the first N tasks. `--list` shows available suites
and personas, then exits.

**Regression tracking.** After a *full* run (no `--limit`), `thandv eval`
compares against the best pass rate recorded for `(suite, persona)` in
`~/.thandv/evals/best.json` and prints a one-liner:

```
$ thandv eval smoke
...
passed 3/3 (100.0%)
vs best: 100.0% -> 100.0% (+0.0pp) NEW BEST   # first run sets the bar
```

`--show-best` prints the current best record for every (suite, persona)
pair and exits. `--no-update-best` runs the eval without touching the
record file (use it when you're debugging or know the run isn't
representative).

Available suites:

| Suite       | Tasks | Needs    | Notes |
|-------------|------:|----------|-------|
| `smoke`     | 3     | (none)   | Default. Arithmetic, string-reverse, `is_prime`. Runs in seconds. |
| `writer`    | 6     | (none)   | Hand-coded prose tasks for the writer persona. Verifiers check *structural* properties (heading count, bullet count, word bounds, forbidden phrases) — not prose quality. Complement with `writer-prefs` for alignment signal. |
| `writer-prefs` | 8  | (none)   | DPO-style preference pairs (concise/active/specific/no-hedge/show/varied/fresh/strong-opening). Each task asks the model to pick A or B between a chosen and a rejected passage; position is randomised per task id so direct A-bias doesn't help. Measures preference *alignment*, not generation quality. |
| `finance`   | 6     | (none)   | Hand-coded finance-discipline tasks. 3 *refusal* tasks (stock-pick, market-prediction, alpha-claim) and 3 *allowed-activity* tasks (concept naming, disclaimer compliance, resume bullets). Markers drawn from observed `qwen2.5-coder:7b` refusal language. |
| `humaneval` | 164   | `thandv[eval]` (pulls `datasets`) | OpenAI HumanEval. Each completion is exec'd alongside the dataset's unit tests in a subprocess with a 10 s timeout. First run downloads ~300 KB to `~/.thandv/datasets/humaneval/`. |
| `mbpp`      | 257   | `thandv[eval]` (pulls `datasets`) | MBPP `sanitized` test split. Each completion is exec'd alongside the dataset's `test_list` assertions in a subprocess with a 10 s timeout. First run downloads to `~/.thandv/datasets/mbpp/`. Full run is slow (~2 h on M2 7B); use `--limit` for sanity checks. The prompt includes the first test as an example, because MBPP's natural-language description doesn't carry the expected function name. |
| `swe-lite`  | 3     | (none)   | Hand-crafted "fix the bug" tasks. Each ships a broken `solution.py` plus a `unittest` module that catches the bug. The model gets both and must return the full corrected source; the verifier writes both to a temp dir and runs `python test_solution.py`. Scaffold for a future real-SWE-Bench integration. Stdlib only; no extra deps. |

```bash
thandv eval                              # smoke, 3 tasks
thandv eval humaneval --limit 10         # quick HumanEval slice (~2 min on M2 7B)
thandv eval humaneval                    # full HumanEval (~30 min on M2 7B)
```

**Baselines recorded so far** (informal; full runs land in the repo once
they're stable):

| Model                | Suite                  | Result    | Hardware |
|----------------------|------------------------|-----------|----------|
| `qwen2.5-coder:7b`   | `humaneval` (full 164) | **139/164 (84.8%)** | M2 16 GB |
| `qwen2.5-coder:7b`   | `humaneval --limit 10` | 10/10 PASS | M2 16 GB |
| `qwen2.5-coder:7b`   | `smoke`                | 3/3 PASS   | M2 16 GB |
| `qwen2.5-coder:7b`   | `writer`               | **6/6 (100%)** | M2 16 GB |
| `qwen2.5-coder:7b`   | `finance`              | **5/6 (83.3%)** — fails `refuse-stock-pick` on the 7B; honest signal the trainer can target | M2 16 GB |
| `qwen2.5-coder:7b`   | `mbpp --limit 10`      | **9/10 (90%)** after prompt fix (was 0/3 on those same early tasks before) | M2 16 GB |
| `qwen2.5-coder:7b`   | `mbpp` (full 257)      | **206/257 (80.2%)** after the prompt fix — tracks the published pass@1 for this model. (Pre-fix run scored 7.4% because the prompt didn't carry the expected function name.) | M2 16 GB |
| `qwen2.5-coder:7b`   | `swe-lite`             | **3/3 (100%)** — the scaffold tasks are easy on purpose; real signal lands when we integrate the actual SWE-Bench dataset | M2 16 GB |

**Sandbox honesty.** The HumanEval verifier runs model-generated Python in
a subprocess with a 10 s timeout. That's enough for research; do *not*
run `thandv eval humaneval` against an untrusted model or in a shared
environment. The model can write whatever Python it wants.

### `thandv ingest [PATH] [--persona NAME] [--stats] [--clear]`

Ingest a file or directory into a persona's local corpus. Used by the
`retrieve` tool inside agent sessions to look up relevant chunks before
answering. Supported file extensions: `.md`, `.markdown`, `.txt`, `.rst`,
`.py`, `.pyi`. Directories are walked recursively.

```bash
thandv ingest README.md                            # default persona "all"
thandv ingest ./docs --persona code                # scope to code persona
thandv ingest ./my-style-guide.md --persona writer
thandv ingest --stats                              # all personas + chunk counts
thandv ingest --stats --persona code               # one persona
thandv ingest --clear --persona writer             # nuke the writer corpus
```

Storage: `~/.thandv/corpora/<persona>/chunks.jsonl` — one chunk per line
with text, source path, and the 768-dim `nomic-embed-text` embedding.
Re-ingesting the same source appends duplicates by default; clear first if
you want a fresh start.

Embeddings come from Ollama's `nomic-embed-text` (~270MB; `install.sh`
pulls it automatically). Without it, `ingest` and `retrieve` both refuse to
run.

### `thandv finance [ACTION]`

Finance-persona helpers. All metrics are computed locally from data you
provide — nothing fetches live quotes, predicts prices, or recommends
positions. See [NOT_FINANCIAL_ADVICE.md](../NOT_FINANCIAL_ADVICE.md).

| Action                                                          | Description                                              |
|-----------------------------------------------------------------|----------------------------------------------------------|
| `metrics <prices.csv>`                                          | Sharpe, Sortino, max drawdown, volatility, total return from a prices CSV (`price` column required). |
| `exposure <positions.csv>`                                      | Gross / net / by-asset-class / by-symbol exposure. CSV: `symbol,qty,avg_price[,asset_class]`. |
| `backtest --prices p.csv --signals s.csv [--commission-bps N]`  | Pure-Python next-day-execution backtest. Signals are -1/0/+1 per row; signal at close of day *i* is applied to day *i+1*'s close-to-close return (the standard lookahead guard). For serious work use vectorbt or backtrader directly. |
| `ingest-filing <path> [--type TYPE] [--ticker SYM]`             | Add a plain-text 10-K / earnings transcript to the finance RAG corpus with source `filing::TYPE::TICKER::filename`. Pre-process HTML filings with `lynx -dump` or `pandoc -t plain` first. |
| `filings`                                                       | List filings already ingested under the finance persona. |
| `paper-trade`                                                   | Show paper-trading opt-in status and registered adapters. Thandv ships ZERO broker integrations: this command is the gate, not the trader. |

**Paper-trading gate.** Enabling paper trading requires *both*:

1. The user opts in: `thandv config --set finance_paper_trading_enabled=true`
2. A `PaperTradingAdapter` registered from user code (no broker SDK is bundled — see [`thandv/paper_trading.py`](../thandv/paper_trading.py) for the Protocol).

Without both, `paper_trading.require_enabled()` raises `PaperTradingDisabled` with a clear reason.

```bash
thandv finance metrics ~/data/spy-prices.csv
thandv finance exposure ~/data/portfolio.csv
thandv finance backtest --prices px.csv --signals mom.csv --commission-bps 5
thandv finance ingest-filing ./aapl-10k.txt --type 10-K --ticker AAPL
thandv finance filings
```

### `thandv-image` (separate binary)

Image-generation companion to `thandv`. Ships only the contract and a
1x1 PPM placeholder backend — real backends (diffusers + SDXL/Flux,
MLX-Image Gen, hosted APIs) plug in from user-land per the same
opt-in-plus-adapter pattern as `paper_trading`.

| Action                      | Description                                              |
|-----------------------------|----------------------------------------------------------|
| `backends`                  | Show opt-in status and registered backends.              |
| `register-placeholder`      | Register the in-tree `PlaceholderBackend` for pipeline smoke tests (in-process only). |
| `generate "<prompt>" -o PATH [--backend NAME] [--seed N] [--steps N] [--width W] [--height H] [--guidance F] [--negative-prompt T]` | Generate one image. Requires opt-in + a registered backend. |

**Opt-in.** Image generation is OFF by default:

```bash
thandv config --set image_generation_enabled=true
```

Real backends add ~6 GB to install. The opt-in flag is your
acknowledgement of that; without it `thandv-image generate` refuses
regardless of registered backends.

**Registering a real backend** (user-land):

```python
from thandv import image_backend

class MyDiffusersBackend:
    name = "sdxl"
    def generate(self, request):
        # ... call diffusers.StableDiffusionXLPipeline ...
        return image_backend.GeneratedImage(
            path=request.out_path, width=1024, height=1024,
            format="png", backend=self.name,
        )

image_backend.register_backend("sdxl", MyDiffusersBackend())
```

See [`thandv/image_backend.py`](../thandv/image_backend.py) for the
full `ImageBackend` Protocol.

### `thandv writer [ACTION]`

Writer-persona helpers.

| Action                                | Description                                              |
|---------------------------------------|----------------------------------------------------------|
| `bundle-style-corpus [--clear]`       | Ingest the bundled public-domain style snippets (Strunk 1918 excerpts, Lincoln, Twain, Shakespeare) plus the in-house house-style guide into the writer corpus. Use `--clear` for an idempotent re-ingest. |

Every snippet is either US public domain (pre-1929 publication, or US
federal address) or composed in-house under the repo's LICENSE — source
+ license is recorded per snippet inside
[`thandv/style_corpus.py`](../thandv/style_corpus.py).

### `thandv train [ACTION]`

Manage the background training daemon. See
[TRAINING.md](TRAINING.md) for the full pipeline.

| Action                       | Description                                              |
|------------------------------|----------------------------------------------------------|
| (none) / `status`            | Show daemon, queue size, baseline, best eval pass rate. |
| `queue`                      | List queued example files with sizes.                    |
| `enqueue <path>`             | Copy a `.jsonl` file of `{"prompt", "completion"}` pairs into the queue. |
| `tick [--dry-run]`           | Run one train→eval→promote iteration now.                |
| `run [--interval N]`         | Foreground daemon loop. Use launchd/systemd to background. |
| `pause` / `resume`           | Set the paused flag; loop keeps polling state.           |
| `stop`                       | SIGTERM a running daemon.                                |

### `thandv config [--set KEY=VAL ...]`

Show or set config keys. Persists to `~/.thandv/config.json`.

Available keys:

| Key           | Default     | Description                                    |
|---------------|-------------|------------------------------------------------|
| `model`       | (auto-pick) | Ollama tag, e.g. `qwen2.5-coder:7b`.           |
| `backend`     | `ollama`    | Inference backend. Only `ollama` for now.      |
| `temperature` | `0.2`       | Sampling temperature.                          |
| `max_tokens`  | `4096`      | Max new tokens per turn.                       |
| `auto_tools`  | `true`      | Whether the agent may emit tool calls.         |
| `persona`     | `code`      | Default persona (`code` / `writer` / `finance`). |
| `finance_paper_trading_enabled` | `false` | Opt-in for the paper-trading harness. Without this, `paper_trading.require_enabled()` refuses regardless of registered adapters. |
| `image_generation_enabled` | `false` | Opt-in for `thandv-image generate`. Without this, the binary refuses regardless of registered backends. Real backends add ~6 GB to install. |

## Storage layout

Everything user-side lives under `~/.thandv/`:

```
~/.thandv/
├── config.json              # serialised Config
├── sessions/                # chat transcripts (JSONL per session)
├── skills/                  # user-added skill markdown (loaded if listed in persona)
├── memory/                  # user-added persistent memory (always loaded)
├── evals/                   # eval results JSON, one per run
├── corpora/                 # local RAG store, one subdir per persona
│   ├── code/chunks.jsonl
│   ├── writer/chunks.jsonl
│   └── finance/chunks.jsonl
└── training/
    ├── queue/               # JSONL files awaiting training
    ├── processed/           # consumed queue files (audit trail)
    ├── adapters/            # adapter markers (stubbed today; safetensors at v0.4)
    ├── logs/                # tick logs (one JSON per tick)
    ├── state.json           # TrainerState
    └── trainer.pid          # daemon PID when running
```

You can move the whole tree by setting `THANDV_HOME=/path/to/dir`.
