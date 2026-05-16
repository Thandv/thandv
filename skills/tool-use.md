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

After the tool runs, you'll see the JSON result in the next user message
wrapped in a ```tool-result``` block. Read it and proceed.

If the task is purely conversational, do not emit a tool block at all.
