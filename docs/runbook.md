# Runbook

This runbook covers local setup, validation, a small experiment run, aggregation,
and cleanup.

For config, input, and output contracts, see [Workflow Interface](interface.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If dependencies are already installed and the package is not installed in
editable mode, prefix package commands with `PYTHONPATH=src`.

## Validate The Current Repository State

```bash
PYTHONPATH=src python -m pytest -q
python proof/validate_evidence_manifest.py
```

## Run A Tiny Fine-Tuning Job

This example trains a small LoRA adapter on the committed smoke training sample.
It may download model weights from Hugging Face if they are not cached.

```bash
PYTHONPATH=src python -m biollm_finetune.training.finetune \
  --config configs/finetune_tiny.yaml
```

Expected training outputs:

- `results/ckpts/tiny_run/run.json`
- `results/ckpts/tiny_adapter/`
- `results/ckpts/tiny_adapter/run.json`

Checkpoint and adapter outputs under `results/ckpts/` are generated artifacts
and are not tracked by git.

For CUDA QLoRA runs, install the optional GPU dependencies in a CUDA
environment:

```bash
pip install -e ".[dev,gpu]"
```

Then use `configs/finetune.yaml` after providing the private BioASQ-style JSONL
training file referenced by that config.

## Evaluate A Trained Adapter

After `results/ckpts/tiny_adapter/` exists, run the adapter-aware experiment
config:

```bash
PYTHONPATH=src python scripts/run_experiment.py \
  --config configs/experiments/bioasq_TINY_mps_fp32_lora_clean_seed42.yaml
```

The runner passes the adapter path into generation, writes the exact inference
inputs, scores the predictions, and saves the same per-run artifacts as the
inference-only experiment path.

## Run A Single Experiment

This example runs one clean BioASQ sample experiment. It may download model
weights from Hugging Face if they are not cached.

```bash
PYTHONPATH=src python scripts/run_experiment.py \
  --config configs/experiments/bioasq_TINY_mps_fp32_clean_seed42.yaml
```

Inspect the generated run directory under:

```text
results/experiments/
```

Expected run files include:

- `inputs.jsonl`
- `predictions.jsonl`
- `metrics.json`
- `phenotypes.json`
- `run_metadata.json`
- `manifest.json`

## Run A Selected Experiment Set

`scripts/run_experiments.py` can select configs by YAML fields.

Example:

```bash
PYTHONPATH=src python scripts/run_experiments.py \
  --configs configs/experiments \
  --select perturbation=clean
```

Use selectors carefully because model execution can be slow and can download
weights if the model cache is empty.

## Rebuild Phase-Level Outputs

After experiment directories exist under `results/experiments/`, rebuild the main
aggregate outputs:

```bash
PYTHONPATH=src python scripts/aggregate_experiments.py
PYTHONPATH=src python scripts/compute_deltas.py \
  --experiments-csv results/phase4/experiments.csv \
  --out-dir results/phase4/deltas
PYTHONPATH=src python scripts/analyze_phase4_results.py
PYTHONPATH=src python scripts/generate_phase4_tables_and_figures.py
python proof/generate_canonical_manifest.py
python proof/validate_evidence_manifest.py
```

## Inspect Outputs

Start with:

- `results/phase4/summary.json`
- `results/phase4/analysis/phase4_findings.md`
- `results/phase4/analysis/phenotype_findings.md`
- `results/phase4/report_artifacts/tables/perturbation_ranking_macro_avg.md`
- `proof/evidence_manifest.latest.json`

## Shutdown

There is no long-running service to stop. Local runs end when the Python command
exits.

If a run is interrupted, inspect the partially written directory under
`results/experiments/` before deleting or rerunning it.
