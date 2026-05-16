## Summary

<!-- One or two sentences: what does this PR change and why. -->

## Motivation

<!-- Link to the roadmap item, issue, or research note this advances. -->
<!-- e.g. research/ROADMAP.md v0.1 "Streaming output" -->

## Changes

<!-- Bulleted list of the substantive changes. Skip noise. -->

-
-

## How I tested this

<!-- Be specific. "Ran pytest" is not enough; what did you actually exercise? -->

- [ ] `pytest -q` passes locally
- [ ] `ruff check thandv tests` passes
- [ ] Ran `thandv doctor` and `thandv chat "..."` against a local Ollama
- [ ] Added/updated tests for the behaviour this PR changes
- [ ] (if relevant) Ran `bash build.sh` and the produced binary still works

## Eval impact

<!-- Once `thandv eval` lands (v0.2), include before/after numbers here. -->
<!-- Until then, describe the qualitative effect on a real session. -->

## Breaking changes

<!-- Config keys, CLI flags, tool protocol, or skill format. -->
<!-- "None" is a fine answer. -->

## Honesty check

<!-- Cribbed from research/SELF_IMPROVEMENT.md. Tick whichever apply. -->

- [ ] This is a real capability improvement (a measurable win on a task we
      couldn't do before, or did worse).
- [ ] This is a UX / scaffolding change (no capability gain claimed).
- [ ] This is a research probe (we expect to learn whether it helps; the
      eval will decide).
