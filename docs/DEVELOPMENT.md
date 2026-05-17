# Development

## Setup

```bash
git clone git@github.com:Thandv/thandv.git
cd thandv
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,build]"
```

`requirements:` Python 3.10+, `ollama` installed locally for live runs
(tests don't need it — HTTP and eval are mocked at the seams).

## Running tests

```bash
pytest -q                      # full suite (~1.2s, no network)
pytest tests/test_trainer.py   # one file
pytest -k tool                 # match by name
pytest -x --pdb                # stop on first failure, drop into pdb
```

Test conventions:

- `tests/conftest.py::thandv_home` is the canonical fixture. It creates
  a temp `~/.thandv`-equivalent and monkeypatches paths in `config`,
  `memory`, and `trainer`. Anything that touches the home dir must use
  this fixture.
- HTTP is mocked at `Agent._raw_stream`. Don't mock `requests` globally.
- Eval is mocked at `thandv.evals.Agent` (the module-level name the
  runner imports). Test reads of `eval_mod.Agent = FakeAgent` are
  scoped per test.
- The trainer mocks `trainer._run_eval` rather than going through the
  real eval runner — keeps trainer tests independent of eval contents.

## Linting

```bash
ruff check thandv tests
ruff check --fix thandv tests   # autofix unused imports etc.
```

CI runs the same `ruff check`. Don't disable rules silently; if a check
is wrong for the codebase, edit `pyproject.toml`.

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on push to
`main` and on every PR. Matrix:

- `ubuntu-22.04 × py3.10/3.11/3.12` — ruff + pytest
- `macos-14 × py3.10/3.11/3.12` — ruff + pytest
- Build smoke: `bash build.sh && ./dist/thandv --version` on both OSes

Failures block merge. Tests must pass before review.

## PR workflow

1. Branch off `main`.
2. Write the change + the tests in the same commit.
3. Push and open a PR — `.github/pull_request_template.md` is loaded
   automatically.
4. Fill in the template, including the **honesty check**: capability gain
   / UX / research probe.
5. Wait for CI green.
6. Squash-merge when reviewed.

## Adding a new persona

1. Define a `Persona(...)` in [`thandv/personas.py`](../thandv/personas.py)
   and add it to the `PERSONAS` registry tuple.
2. Add any skill files it references to [`skills/`](../skills/) (the file
   stem must match the entry in `persona.skills`).
3. Add tests in [`tests/test_personas.py`](../tests/test_personas.py) —
   at minimum: registration, system-prompt invariants, skill list.

## Adding a new eval suite

1. Build an `EvalSuite(name=..., tasks=[...])` in
   [`thandv/evals.py`](../thandv/evals.py). Verifiers must be pure
   functions `reply: str → bool`.
2. Register the suite in the `SUITES` dict.
3. Add tests:
   - the suite is discoverable (`get_suite("yours")`),
   - each verifier accepts the canonical correct reply and rejects an
     obviously wrong one,
   - the runner produces `EvalResult` rows with the expected `task_id`s.

Real benchmarks (HumanEval, MBPP) will come behind a lazy import of
`datasets` so the smoke suite stays no-dep.

## Adding a new tool

1. Add the function to [`thandv/tools.py`](../thandv/tools.py). Signature:
   takes JSON-able kwargs, returns a JSON-able dict. Errors live in an
   `"error"` key.
2. Register it in the `TOOLS` dict.
3. Add a `# Skill: tool-use` entry to
   [`skills/tool-use.md`](../skills/tool-use.md) so the model learns
   about it.
4. Tests: success path, missing-arg path, error path. Reuse `tmp_path`.

## Releases

Release process today is light: bump `__version__` and `pyproject.toml`,
tag `vX.Y.Z`, push the tag. The `release.yml` workflow builds the
PyInstaller binary for macOS and Linux and attaches them to the GitHub
release.

## Folder layout reminder

```
thandv/        the package
skills/        shipped skill markdown (loaded by personas that list them)
tests/         pytest suite — fast, hermetic, no network
research/      ROADMAP.md, SELF_IMPROVEMENT.md
docs/          you are here
cpp/           placeholder for the eventual native build
scripts/       launchd plist, systemd unit, distill/eval stubs
```
