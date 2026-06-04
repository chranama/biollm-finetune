# Testing

The test suite focuses on package behavior, CLI failure modes, data conversion,
scoring, and a small generation smoke path.

## Setup

Use the runbook for local environment setup.

## Local Tests

Use the runbook for local test commands. CI runs tests with `PYTHONPATH=src`.

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

## Manifest Validation

Evidence manifest validation checks that the latest saved evidence manifest is
internally consistent and that referenced artifact paths exist. Use the runbook
for the local validation command.

## CI Checks

GitHub Actions runs:

- Ruff linting
- Black format check
- pytest
- evidence manifest validation

The CI workflow is defined in `.github/workflows/ci.yaml`.
