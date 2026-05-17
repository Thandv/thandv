Tool-use protocol reminder.

Emit exactly one fenced block per turn:

```tool
{"name": "<tool>", "args": { ... }}
```

Available tools:
- `read_file`  args: {"path": str}
- `write_file` args: {"path": str, "content": str}
- `edit_file`  args: {"path": str, "old": str, "new": str}
- `list_dir`   args: {"path": str}
- `run_bash`   args: {"command": str, "confirm": bool (optional)}
- `retrieve`   args: {"query": str, "k": int (1-20, default 5)}
    Looks up the top-k most relevant chunks from your active persona's
    corpus (a local vector store of documents the user has ingested).
    The persona is filled in automatically; you can override it with
    "persona": "all" to search across every persona's corpus, or
    "persona": "<other-persona>" for a specific one. Use this *before*
    answering a question whose answer might live in the corpus, e.g.
    documentation, the user's own files, reference material.

After the tool runs, you'll see the JSON result in the next user message
wrapped in a ```tool-result``` block. Read it and proceed.

If the task is purely conversational, do not emit a tool block at all.
