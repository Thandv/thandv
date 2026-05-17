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

### `thandv eval [SUITE] [--persona NAME] [--limit N] [--list]`

Runs an eval suite against the current model + persona. Defaults to the
`smoke` suite (3 tasks, no dataset download needed). Results land in
`~/.thandv/evals/<suite>-<unix-ts>.json`.

`--limit N` runs only the first N tasks. `--list` shows available suites
and personas, then exits.

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
