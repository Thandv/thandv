# Training

## Philosophy in one paragraph

Inference is local; training may use the internet. The training pipeline
exists to make the local model **measurably better** on tasks the user
cares about, without losing the "self-sufficient binary at runtime"
property. Every adapter has to beat the current best score on an eval
suite to be promoted. Adapters that don't are deleted. This is the
verifier-driven loop from
[`../research/SELF_IMPROVEMENT.md`](../research/SELF_IMPROVEMENT.md) —
the only kind of self-improvement that actually works.

## State of the trainer

| Component                   | Today (v0.4.1)                       |
|-----------------------------|--------------------------------------|
| Queue management            | ✅ JSONL files in `queue/`            |
| Eval gate                   | ✅ smoke suite (configurable later)  |
| Adapter promote / discard   | ✅ real GGUF + Ollama registration   |
| **Actual LoRA training**    | ✅ real (MLX-LM on M-series; HF+PEFT fallback stubbed for now) |
| Daemon control              | ✅ run / stop / pause / resume / status |
| launchd / systemd templates | ✅ provided                           |

**What's real.** The orchestration *and* the inner training step both
work. A tick now downloads (on first run) the HF base model, trains a
LoRA adapter via MLX-LM, merges + exports a GGUF via `mlx_lm fuse
--export-gguf`, registers the new model in Ollama as
`thandv-adapter-<ts>`, runs the eval gate against the new model, and
either promotes (recording it as `active_ollama_model`) or discards it
(`ollama rm`).

**Honest expectations.** A single LoRA on a 7B base via MLX-LM on M2
16 GB is tight — small batch sizes, short context windows, ~30-60 min
per small adapter. If you hit OOM, drop to a smaller training base via
`thandv train enable --hf-model <smaller-repo>`.

## Pipeline

```
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│  data source   │───▶│  queue/*.jsonl │───▶│  trainer.tick  │
│  (HF datasets, │    │ (FIFO by name) │    │   per cycle    │
│  free teachers,│    └────────────────┘    └───────┬────────┘
│  user sessions)│                                  │
└────────────────┘                                  ▼
                                          ┌─────────────────┐
                                          │   stub_train    │── adapter id
                                          └────────┬────────┘
                                                   ▼
                                          ┌─────────────────┐
                                          │ run_suite(smoke)│── pass rate
                                          └────────┬────────┘
                                                   ▼
                                       ┌─────────────────────┐
                                       │ rate > best?         │
                                       │  yes → promote       │
                                       │  no  → discard       │
                                       │  err → skip          │
                                       └─────────────────────┘
```

## Background mode (no active input required)

The whole point of "improving while you sleep" is the daemon. Two
supported routes, pick whichever fits your OS.

### macOS — launchd

```bash
# Copy the template into the LaunchAgents dir.
cp scripts/com.thandv.trainer.plist ~/Library/LaunchAgents/

# Edit the two __HOME__ placeholders to your real $HOME, and __THANDV__
# to your `thandv` binary path (`which thandv`).
$EDITOR ~/Library/LaunchAgents/com.thandv.trainer.plist

# Load. RunAtLoad=true means it starts immediately.
launchctl load ~/Library/LaunchAgents/com.thandv.trainer.plist

# Check it's running:
thandv train status
```

To unload:

```bash
launchctl unload ~/Library/LaunchAgents/com.thandv.trainer.plist
```

### Linux — systemd user unit

```bash
mkdir -p ~/.config/systemd/user
cp scripts/thandv-trainer.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now thandv-trainer.service

systemctl --user status thandv-trainer.service
journalctl --user -u thandv-trainer.service -f
```

### Anywhere — cron + `thandv train tick`

If you'd rather not run a long-lived process, schedule single ticks:

```cron
# crontab -e — runs one training tick every 30 minutes
*/30 * * * * /usr/local/bin/thandv train tick >> ~/.thandv/training/logs/cron.log 2>&1
```

### Foreground (good for debugging)

```bash
thandv train run --interval 300
```

## Feeding the queue

You give the trainer something to train on by putting JSONL files in
`~/.thandv/training/queue/`. Each line is a JSON object with at minimum
`prompt` and `completion`:

```json
{"prompt": "Reverse this string: 'foo'", "completion": "oof"}
{"prompt": "What is 7 * 8?", "completion": "56"}
```

Add a file:

```bash
thandv train enqueue path/to/examples.jsonl
```

Once v0.4 lands, additional ingestion paths are planned:

- **Public datasets** (license-clean slices of The Stack v2, CodeAlpaca,
  FineWeb-Edu).
- **Teacher distillation** from ToS-whitelisted free-tier APIs
  (Gemini free, Groq free, Together free, OpenRouter free models).
  Excluded: Anthropic, OpenAI — their ToS forbids competing-model
  training.
- **Session promotion** — verifier-passed user sessions auto-converted
  to training pairs.

## Observability

```bash
thandv train status        # one-screen snapshot
thandv train queue         # what's in line
ls ~/.thandv/training/logs # one JSON file per tick
cat ~/.thandv/training/state.json
```

Every tick writes a log entry with `action` ∈ `{baseline, skip, promote,
discard}` plus context. Greppable. Diff-able.

## Pause / resume

```bash
thandv train pause   # daemon keeps running but skips ticks
thandv train resume  # un-pauses
thandv train stop    # SIGTERM the daemon entirely
```

Pausing is useful when you're about to do something GPU-heavy in the
foreground.

## What to read next

- [`../research/SELF_IMPROVEMENT.md`](../research/SELF_IMPROVEMENT.md)
  for *why* this pipeline is shaped the way it is and what its limits
  are.
- [`../research/ROADMAP.md`](../research/ROADMAP.md) for the path from
  the v0.2 stub to a real v0.4 trainer and beyond.
