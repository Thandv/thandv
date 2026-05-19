Tool-use protocol reminder.

You may emit tool calls in any of these forms — the runtime accepts all
three. Pick whichever your training prefers:

1. Native function-calling (preferred): use the model API's structured
   `tool_calls` field. The runtime sends tool schemas with each request.
2. Fenced JSON block:

   ```tool
   {"name": "<tool>", "args": { ... }}
   ```

3. Inline JSON in your reply (last-resort): a single `{"name": ..., "arguments": ...}`
   object as your entire content.

After a tool returns, you'll see its result as a `role="tool"` message.
Read it and USE it to answer the user. Do NOT call the same tool again
with the same args — the runtime detects loops and stops.

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
