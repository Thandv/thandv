# Architecture

## High-level data flow

```
            ┌─────────────────────┐
            │       user          │
            └────────┬────────────┘
                     │ stdin
                     ▼
            ┌─────────────────────┐
   chat ──▶ │  cli.py             │ ◀── eval, doctor, train, config
            └────────┬────────────┘
                     │
                     ▼
            ┌─────────────────────┐
            │  agent.Agent        │── load persona + skills + memory
            └────────┬────────────┘
                     │ stream
                     ▼
            ┌─────────────────────┐
            │  ollama /api/chat   │── streams NDJSON, content chunks
            └────────┬────────────┘
                     │ chunks
                     ▼
            ┌─────────────────────┐
            │  tool-block parser  │── hides ```tool``` from user stream
            └────────┬────────────┘
                     │ if tool block:
                     ▼
            ┌─────────────────────┐
            │  approval gate      │── run_bash/write_file/edit_file:
            │                     │   ask the person y/N first
            └────────┬────────────┘
                     │ approved (or read-only tool)
                     ▼
            ┌─────────────────────┐
            │  tools.dispatch     │── read/write/edit/list/bash
            └────────┬────────────┘
                     │ result JSON
                     └──▶ fed back as next user message; loop
```

The CLI is a thin shell around the agent loop, plus three side commands
(`doctor`, `config`, `eval`) and the training daemon control surface
(`train`).

## Module map

```
thandv/
├── __init__.py     version
├── __main__.py     `python -m thandv` entry
├── cli.py          argparse + subcommand handlers
├── runtime.py      detect_host(), pick_model() — hardware → model ladder
├── config.py       Config dataclass + on-disk persistence + paths
├── personas.py     CODE / WRITER / FINANCE definitions + registry
├── prompts.py      (deleted in v0.2; personas replaced it)
├── agent.py        Agent — streaming chat, tool-block parsing, tool-call hop budget
├── tools.py        local tool implementations + dispatch()
├── memory.py       skills loader (persona-filtered), memory loader, session logger
├── evals.py        EvalTask/EvalSuite/EvalResult, smoke suite, run_suite()
├── trainer.py      daemon scaffold: queue, tick, eval-gate, state, pause/resume
└── rag.py          chunking, embedding (Ollama nomic-embed-text), per-persona JSONL store, retrieve
```

## Key design decisions

### 1. Inference local, training open

Inference must be reachable with no network. Training may use free
high-quality internet sources (HuggingFace datasets, ToS-clean teacher
APIs). The binary itself never phones home at runtime. This shapes which
features go where.

### 2. Persona as a layer, not a model

The base model is shared. A persona only changes the system prompt and
the skill subset loaded into context. Switching personas is free.
Adding a persona is ~30 lines.

### 3. Streaming with three tool-call protocols

Ollama's `/api/chat?stream=true` returns NDJSON, one content chunk per
line. The agent recognises three ways a model can signal a tool call,
tried in priority order:

1. **Native `message.tool_calls`** — the OpenAI / Ollama function-calling
   field. Activated by sending `tools=TOOL_SCHEMAS` in the chat request.
   This is the cleanest signal: structured, schema-validated, no parsing.
2. **Text-based `\`\`\`tool\n{...}\n\`\`\`` block** — our explicit fallback
   protocol documented in `thandv/skills/tool-use.md`. Caught by a regex over
   the accumulated content stream. An 8-char tail buffer over chunks
   ensures the marker is detected even when it straddles a boundary.
3. **Inline JSON in `message.content`** — some smaller models (e.g.
   certain `qwen2.5-coder` builds) emit the call as a raw JSON object
   in content instead of using the native field. We detect this by:
   - watching whether the first non-whitespace char of the streamed
     content is `{`, and if so, buffering the whole reply rather than
     yielding incrementally;
   - at end-of-stream, parsing the buffered content and matching against
     `TOOLS`. If it's a recognised tool, suppress the raw JSON entirely
     (only the `[tool]` line shows); if not, deliver it as legitimate
     output.

Tool-block JSON never reaches the user's stream in cases (2) and (3); a
single-line `[tool] name(args)` summary appears instead.

After a tool dispatches, its result is appended to history with
`role="tool"` (not `role="user"` as in earlier versions) and a `name`
field. This is the OpenAI / Ollama contract and keeps the model from
treating the result as a fresh user turn — which used to cause smaller
models to loop on the same tool call without synthesising an answer.

A duplicate-call guard short-circuits the agent if the model emits the
*same* tool call twice in a row, instead of silently chewing through the
hop budget.

### 4. Eval-gated training, always

The trainer never promotes a new adapter blindly. Every adapter has to
beat the *current best* pass rate on the configured eval suite. If the
eval can't run (e.g. Ollama down), the tick is skipped, not penalised.
This is the only honest version of "self-improvement" — see
[`../research/SELF_IMPROVEMENT.md`](../research/SELF_IMPROVEMENT.md).

### 5. State on disk, not in memory

Everything important the daemon does is reflected in
`~/.thandv/training/state.json` and one JSON log per tick under
`~/.thandv/training/logs/`. Crash recovery is "read the JSON, resume".
There is no in-memory state of interest.

### 6. RAG without a vector DB

`thandv/rag.py` does retrieval with a pure-Python pipeline: paragraph-aware
chunker, Ollama `nomic-embed-text` for embeddings, JSONL store under
`~/.thandv/corpora/<persona>/chunks.jsonl`, cosine similarity computed in
pure Python. No `numpy`, no `lancedb`, no `chromadb`. This stays fast
enough up to ~50k chunks per persona (~100 ms / retrieve). When we outgrow
it, the same `ingest_text` / `retrieve` API hides the storage swap.

The agent auto-injects the active persona into the `retrieve` tool's args
so the model doesn't need to know its own name; the model can override
with `persona="all"` to search across every corpus.

### 7. Human gate for side-effecting tools

Anything the model reads (a retrieved chunk, a file, a tool result) can
carry prompt-injection text, so the model is never the one who decides
whether a side effect happens.

- **Gated tools.** `run_bash`, `write_file` and `edit_file`
  (`tools.GATED_TOOLS`) go through `Agent._gate` before dispatch. The
  person sees the full command (with control characters such as ANSI
  escapes, `\r` and bidi overrides escaped) or the resolved path plus the
  full diff, and must type `y`/`yes`. Commands over 4000 characters or 40
  lines and changes over 2000 diff lines are refused rather than shown in
  part; the REPL prints as much of a longer summary as fits in 60 screen
  rows (long lines count by how far they wrap) and an `f` answer prints
  all of it. Files that make code run later (`.git/` hooks, `conftest.py`,
  `Makefile`, shell start-up files, `.claude/` and `.cursor/` agent
  settings, executables, ...; matched case-insensitively) get a WARNING
  line. Everything shown
  is escaped unless plainly printable (controls, bidi overrides, lone
  surrogates, non-ASCII spaces), and commands are shown as pure ASCII so
  a look-alike such as a no-break space cannot disguise what runs. Model
  output itself is printed with the same escaping, and the prompt resets terminal attributes, so a
  reply cannot conceal or forge the real prompt. Anything else — Enter, `n`,
  EOF, Ctrl-C, an approver that raises — returns
  `{"error": "denied by user"}` to the model. `read_file`, `list_dir` and
  `retrieve` are ungated (`tools.UNGATED_TOOLS`); an import-time tripwire
  requires every tool to be in exactly one of the two sets.
- **Who approves.** The approver is a callback on `Agent` set by the
  CLI, never by the model. The REPL asks at the terminal, flushing any
  type-ahead first so a pasted line can't answer; if stdin is not a
  terminal it denies without reading. A one-shot `thandv chat PROMPT`, a
  `thandv eval` run and the training daemon have no approver, so gated
  tools are denied (`"denied: side-effecting tools need the user's
  approval..."`). The person can opt in with `--allow-tools` on the
  command line (`chat PROMPT` and `eval` only).
- **No model-settable bypass.** `dispatch` rejects any argument that is
  not in the tool's schema (`confirm`, `force`, `root`, ...) and any
  non-string value for a string argument. Tool names are matched exactly;
  `Run_Bash` or `bash` is an unknown tool.
- **Write confinement.** `write_file`/`edit_file` resolve the path
  (relative paths from the session root; `~`, `..` and symlinks —
  including a symlinked parent directory — resolved first) and refuse it
  unless it is inside the session's working-directory root (the cwd when
  the session started; `run_bash` also runs there). Paths ending in `/`
  are refused. The file is opened without following symlinks
  (`O_NOFOLLOW_ANY` on macOS, so no component may be a symlink;
  `O_NOFOLLOW` plus a re-check of the resolved path elsewhere) and must
  be a regular file with one hard link; it is only truncated after those
  checks. Missing parent directories are created before the open, so a
  refused write can leave new empty directories inside the root, and a
  concurrent local process that swaps a directory for a symlink at that
  moment can still make empty directories outside it.
- **Second layer only.** `run_bash` still refuses commands containing
  `rm -rf`, `mkfs`, `dd if=`, `:(){:|`, `shutdown`, `reboot`, with no
  override. That substring list is trivially bypassed and is not a
  security boundary; the gate is. Commands get `/dev/null` as stdin and
  run in their own process group, which is killed as a whole on the 30 s
  timeout.

`read_file` can read any regular file the user can (FIFOs and devices are refused), and an approved `run_bash`
runs with the user's full privileges: this is not a sandbox. The eval
verifiers are a separate path: they run model-written Python to grade it
(including from the training daemon) without asking, contained only by a
temporary directory — see "Sandbox honesty" in USAGE.md. Don't run
Thandv as root, don't expose its HTTP surface, don't trust untrusted
skill markdown.

## Tools

| Tool         | Args                                       | Use                                  |
|--------------|--------------------------------------------|--------------------------------------|
| `read_file`  | `path`                                     | Read up to 200 KB; UTF-8 decoded.    |
| `write_file` | `path`, `content`                          | **Gated.** Under the session root only; create parents; full overwrite. |
| `edit_file`  | `path`, `old`, `new`                       | **Gated.** Under the session root only; single-occurrence replace; errors if `old` is non-unique. |
| `list_dir`   | `path`                                     | Lists entries with `type ∈ {file, dir}`. |
| `run_bash`   | `command`                                  | **Gated.** Runs in the session root; 30-second timeout, captures stdout/stderr (tail-truncated). |
| `retrieve`   | `query`, `persona` (auto), `k` (1–20)      | Top-k chunks from the active persona's corpus by cosine similarity. Agent auto-fills `persona`. |

Tools live in [`thandv/tools.py`](../thandv/tools.py); dispatcher is
`dispatch(name, args, *, root)`. `dispatch` does not ask the person — the
agent loop gates first — and refuses gated tools when no `root` is given.

## Storage layout

End-user data: `~/.thandv/` (overridable with `$THANDV_HOME`).

Repo-side data (loaded at install time):

```
thandv/skills/            default skill markdown bundled with the wheel
                          (shipped via [tool.setuptools.package-data])
personas defined in       thandv/personas.py
eval suites defined in    thandv/evals.py
```

Skill loading: `memory.load_skills(only=...)` reads the bundled
`thandv/skills/` directory first, then overlays user files from
`~/.thandv/skills/` (same stem in user-land overrides bundled). Personas
reference skill *stems* (file names without `.md`); a skill not present
in either location is silently skipped, so user-added skills can extend
a persona without modifying the repo.
