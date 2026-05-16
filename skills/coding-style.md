When writing code, prefer the existing style of the surrounding file.

- No comments unless the *why* is non-obvious.
- No trailing summaries of what you just did — the diff speaks for itself.
- Single responsibility per function. Pull helpers out when a function grows
  past ~40 lines.
- Trust internal callers; validate only at boundaries (user input, network).
- Don't introduce a dependency for a one-liner.
