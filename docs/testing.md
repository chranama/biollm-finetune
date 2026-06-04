# Testing

The test suite focuses on package behavior, config validation, CLI failure
modes, data conversion, scoring, adapter wiring, and a small generation smoke
path.

## Setup

Use the runbook for local environment setup.

## Local Tests

Use the runbook for local test commands. CI runs tests with `PYTHONPATH=src`.

## Test Coverage Areas

Current tests cover:

- YAML config loading
- training-only config validation
- BioASQ-style data loading
- preprocessing to JSONL
- prompt construction
- metrics and postprocessing
- manifest writing
- missing-input and missing-adapter failure behavior
- adapter path resolution from experiment configs
- quantized model loading guardrails
- generation CLI smoke behavior

## Fine-Tuning Tests

Fast tests validate that `configs/finetune_tiny.yaml` loads through the
training-only schema and that the adapter-aware experiment config points to the
expected generated adapter directory.

The suite does not run full model fine-tuning by default because that path can
download model weights and write checkpoint artifacts. Use the runbook's tiny
fine-tuning command when model execution should be validated locally.

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
