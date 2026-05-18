# Thandv documentation

Thandv is a local, open-weights, Claude-style assistant. Inference runs
entirely on your machine. Training may reach for free, high-quality
internet sources — the *binary* stays self-sufficient at runtime.

## Reading order

If you're new, read them in this order:

1. **[USAGE.md](USAGE.md)** — install, run, and use Thandv. Every CLI
   command with examples. Persona reference. Configuration keys.
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** — what the modules are and how
   they fit together. Storage layout. Key design decisions.
3. **[TRAINING.md](TRAINING.md)** — the training pipeline, the background
   daemon (`thandv train`), launchd / systemd setup, what's stubbed vs
   real today.
4. **[DEVELOPMENT.md](DEVELOPMENT.md)** — dev environment, tests, CI, PR
   workflow, and how to add a new persona / eval suite / tool.

## Related documents outside `docs/`

- **[`../research/ROADMAP.md`](../research/ROADMAP.md)** — versioned
  milestones from v0.0.1 through v1.0. Read this to know what's coming.
- **[`../research/SELF_IMPROVEMENT.md`](../research/SELF_IMPROVEMENT.md)** —
  honest take on what self-improvement does and doesn't mean for a local
  open-weights model. The philosophy that constrains the roadmap.
- **[`../research/V0.4_DESIGN.md`](../research/V0.4_DESIGN.md)** —
  open design memo for the real LoRA training pipeline (trainer backend,
  HF↔GGUF round-trip, adapter serving, hardware budget on M2 16 GB).
- **[`../NOT_FINANCIAL_ADVICE.md`](../NOT_FINANCIAL_ADVICE.md)** — the
  canonical statement of the `finance` persona's scope, capability
  honesty, and legal frame. Read before using the finance persona.

## Honest framing in one paragraph

Thandv will not be Claude. The closed weights and ~$100M compute budget
behind a frontier model can't be reproduced by scaffolding. What Thandv
*can* be: a strong **specialised** local assistant on your code, your
prose, and your finance research — sharpened over time by a verifier-
gated training loop that runs locally on whatever data you supply,
including free internet sources. The roadmap is built around that
distinction.
