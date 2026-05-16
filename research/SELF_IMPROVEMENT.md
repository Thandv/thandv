# Self-improvement: what's real and what isn't

Honest framing first, because the "AI improves itself into superintelligence"
trope is doing a lot of damage to clear thinking in this space.

## What does *not* work

1. **Pure self-distillation.** Fine-tuning a small model on its own outputs
   degrades quality ("model collapse"). The model's output distribution is a
   lossy projection of its weights; training on it shrinks the support.
2. **Bootstrapping past your teacher.** No supervised method moves a student
   meaningfully past a frozen teacher's quality. The student asymptotes.
3. **"Self-aware" prompt editing.** A model rewriting its own system prompt
   can find local wins but doesn't add capability; it just reshuffles bias.
4. **Recursive self-improvement, in the Yudkowsky/Schmidhuber sense.** Active
   research topic for 20+ years; no working production system exists.

## What *does* work, in order of bang-for-buck

### 1. Skills and memory (zero training)

Concrete, durable facts in the system prompt outperform clever prompting.
A 14B base + a well-curated `skills/` library can beat a 70B base on the
specific domain those skills cover. Cheap, immediate, reversible.

**Mechanism:** the model's context window *is* a kind of weights at
inference time. Putting the right facts there is free fine-tuning.

### 2. Verifiable-reward fine-tuning

Where there's an automatic scorer — unit tests pass / fail, math answer
right / wrong, code compiles / doesn't — RL or rejection sampling on the
verified signal genuinely improves the model. This is how DeepSeek-R1 and
similar reasoning models were trained. It works because the signal is
*external*, not self-generated.

**For Thandv:** the unit-test signal is the cheapest verifier we can wire
up. A "solve the bug, run the tests, keep traces that pass" loop is a
realistic self-improvement loop bounded by the test suite's coverage.

### 3. Teacher distillation (with care)

If the user supplies their own frontier-model API key, capture
input → output traces on tasks the small model fails. Fine-tune on the
deltas. The student approaches but never matches the teacher; on narrow
domains it can get close.

**Caveats:**
- Anthropic and OpenAI ToS both restrict using their outputs to train
  competing models. For personal / research / non-competitive use the line
  is fuzzier — read the current ToS before doing this commercially.
- The student inherits the teacher's mistakes and biases.

### 4. LoRA adapters per domain

Instead of one fine-tune, train many small LoRA adapters (one per repo,
language, or task family) and route between them. Adapter swap is cheap.
Storage is cheap. Mistakes in one adapter don't poison the others.

### 5. Continual evaluation, not continual training

The trap is treating "training" as the goal. The actual goal is **a
measurably better assistant**. So: invest disproportionately in evals
(`thandv eval`), and only ship a change — a new prompt, a new skill, a new
adapter — when the eval moves. Most "improvements" don't survive a real
eval.

## A realistic improvement loop

```
                +------------------------+
                |  user runs thandv chat |
                +-----------+------------+
                            |
                            v
            +---------------+---------------+
            |  agent solves task with tools |
            |  unit tests / lints / user OK |
            +---------------+---------------+
                            |
              success?      |     failure?
                  v         v          v
        +---------+----+  +-+----------+-----+
        | trace -> SFT |  | trace -> diff vs |
        |    pool      |  | teacher (opt.)   |
        +---------+----+  +--+---------------+
                  |          |
                  v          v
            +-----+----------+----+
            | weekly LoRA refresh |
            | + skills auto-edit  |
            +-----+---------------+
                  |
                  v
            +-----+--------+
            |  thandv eval |
            |  ship if up  |
            +--------------+
```

Each box is a script under `scripts/` we can build incrementally. Nothing
in the loop requires a breakthrough.

## What "Claude quality" means here

We will not match Opus 4.7 on open-ended reasoning. We can plausibly match
it on **specific, scoped tasks** where:
- the user's codebase is in the skills/memory directory,
- the relevant facts fit in 32k tokens of context,
- success has a cheap automated check,
- and we've trained a small LoRA on the user's prior accepted edits.

That's a real, defensible goal. Anything broader is marketing.
