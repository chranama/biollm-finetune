# Testing

The test suite focuses on package behavior, CLI failure modes, data conversion,
scoring, and a small generation smoke path.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Test Command

```bash
pytest -q
```

CI runs tests with `PYTHONPATH=src`. The equivalent local command is:

```bash
PYTHONPATH=src python -m pytest -q
```

## Test Coverage Areas

Current tests cover:

- YAML config loading
- BioASQ-style data loading
- preprocessing to JSONL
- prompt construction
- metrics and postprocessing
- manifest writing
- missing-input and missing-adapter failure behavior
- generation CLI smoke behavior

## Model-Loading Smoke Test

`tests/test_generate_cli.py` runs the generation entry point with the tiny prompt
fixture. Depending on local cache state, this can require loading or downloading
the configured Hugging Face model.

Use the other tests and the evidence manifest validation for faster checks when
model execution is not needed.

## Validation Command

```bash
python proof/validate_evidence_manifest.py
```

This command checks that the latest saved evidence manifest is internally
consistent and that referenced artifact paths exist.

## CI Checks

GitHub Actions runs:

- Ruff linting
- Black format check
- pytest
- evidence manifest validation

The CI workflow is defined in `.github/workflows/ci.yaml`.
