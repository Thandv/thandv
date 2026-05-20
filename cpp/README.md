# cpp/ — native build track

This directory is a placeholder for the native (C++) Thandv build.
Design note (boundary, protocol, build, phased plan): see
[`../research/V1.0_NATIVE.md`](../research/V1.0_NATIVE.md).

## Why a C++ track at all?

The Python CLI is the right place to iterate on the agent loop, tools, and
skills system — fast to change, easy to debug. But for distribution and for
on-device performance there are real wins in going native:

- **One static binary, no Python runtime.** `thandv` becomes a single file
  you `chmod +x` and run.
- **Direct llama.cpp integration.** Cut the Ollama HTTP hop entirely and
  call `llama.cpp` in-process. Lower latency, smaller install footprint.
- **Tighter memory control.** Quantisation, mmap'd weights, prompt caching
  are all first-class in llama.cpp.

## Plan

1. Vendor llama.cpp as a submodule (`cpp/external/llama.cpp`).
2. Build a tiny C++ CLI (`cpp/src/main.cpp`) that mirrors the Python tool
   protocol: read prompt from stdin / argv, emit fenced `tool` blocks, read
   tool results from stdin, repeat.
3. Reuse the Python tools layer initially by shelling out — that keeps the
   tool implementations in one place. Later, port the hot ones to C++.
4. Share the skills/ and memory/ directories with the Python build so both
   front-ends see the same brain.

## Non-goals (for now)

- Reimplementing the whole agent in C++. The agent loop is cheap; the model
  is what costs. Native build is about removing the Python and HTTP cost,
  not about rewriting the assistant.
- Windows native support. Start with macOS + Linux; Windows comes later.
