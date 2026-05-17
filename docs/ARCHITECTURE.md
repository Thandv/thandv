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
   protocol documented in `skills/tool-use.md`. Caught by a regex over
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

### 7. Tools are sandboxed-by-convention, not by mechanism

`run_bash` refuses commands containing `rm -rf`, `mkfs`, `:(){:|`,
`shutdown`, `reboot` unless the caller passes `confirm=True`. That's a
soft seatbelt, not a sandbox. Don't run Thandv as root, don't expose its
HTTP surface, don't trust untrusted skill markdown.

## Tools

| Tool         | Args                                       | Use                                  |
|--------------|--------------------------------------------|--------------------------------------|
| `read_file`  | `path`                                     | Read up to 200 KB; UTF-8 decoded.    |
| `write_file` | `path`, `content`                          | Create parents; full overwrite.      |
| `edit_file`  | `path`, `old`, `new`                       | Single-occurrence replace; errors if `old` is non-unique. |
| `list_dir`   | `path`                                     | Lists entries with `type ∈ {file, dir}`. |
| `run_bash`   | `command`, `confirm` (optional)            | 30-second timeout, captures stdout/stderr (tail-truncated). |
| `retrieve`   | `query`, `persona` (auto), `k` (1–20)      | Top-k chunks from the active persona's corpus by cosine similarity. Agent auto-fills `persona`. |

Tools live in [`thandv/tools.py`](../thandv/tools.py); dispatcher is
`dispatch(name, args)`.

## Storage layout

End-user data: `~/.thandv/` (overridable with `$THANDV_HOME`).

Repo-side data (loaded at install time):

```
skills/                   default skill markdown shipped with the binary
personas defined in       thandv/personas.py
eval suites defined in    thandv/evals.py
```

Personas reference skill *stems* (file names without `.md`). A skill not
present is silently skipped; this is intentional so user-added skills in
`~/.thandv/skills/` can extend a persona without modifying the repo.
