# Runbook

This runbook covers local setup, validation, fine-tuning, adapter-aware
evaluation, aggregation, proof artifact generation, and cleanup.

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
- `results/ckpts/tiny_adapter/adapter_manifest.json`

Checkpoint and adapter outputs under `results/ckpts/` are generated artifacts
and are not tracked by git.

For CUDA QLoRA runs, install the optional GPU dependencies in a CUDA
environment:

```bash
pip install -e ".[dev,gpu]"
```

Then use `configs/finetune.yaml` after providing the private BioASQ-style JSONL
training file referenced by that config.

## Run The Local LoRA Proof Workflow

This sequence trains the tiny local LoRA adapter, runs the matching adapter and
base-model perturbation sets, refreshes aggregate outputs, publishes a
lightweight PEFT manifest, and validates the current proof manifest.

Use offline Hugging Face mode when the TinyLlama model is already cached:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m biollm_finetune.training.finetune \
  --config configs/finetune_tiny.yaml
```

Run the adapter-backed clean and perturbation set:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python scripts/run_experiments.py \
  --configs configs/experiments \
  --select model.name=tinyllama-1.1b-chat-lora
```

Run the matching seed-42 base-model controls:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python scripts/run_experiments.py \
  --configs configs/experiments \
  --select model.name=tinyllama-1.1b-chat \
  --select seed=42 \
  --select perturbation=clean,shuffle_snippets,lexical_noise,irrelevant_noise,contradiction
```

Refresh the current evidence outputs:

```bash
PYTHONPATH=src python scripts/aggregate_experiments.py
PYTHONPATH=src python scripts/compute_deltas.py \
  --experiments-csv results/phase4/experiments.csv \
  --out-dir results/phase4/deltas
PYTHONPATH=src python scripts/analyze_phase4_results.py
PYTHONPATH=src python scripts/generate_phase4_tables_and_figures.py
PYTHONPATH=src python scripts/summarize_peft_adapter.py \
  --adapter-dir results/ckpts/tiny_adapter \
  --out results/phase4/peft/tiny_adapter_manifest.json
PYTHONPATH=src python scripts/summarize_runtime_manifests.py \
  --experiments-csv results/phase4/experiments.csv \
  --out results/phase4/runtime/runtime_summary.json
python scripts/validate_experiment_integrity.py \
  --only-configured \
  --out results/analysis/integrity_report.json
python proof/generate_canonical_manifest.py
python proof/validate_evidence_manifest.py
```

The experiment configs request MPS for local Apple hardware, but the actual
runtime depends on the installed PyTorch build. Inspect each run's
`inference_manifest.json` to confirm the resolved device and dtype.

## Evaluate A Trained Adapter

After `results/ckpts/tiny_adapter/` exists, run the adapter-aware experiment
config:

```bash
PYTHONPATH=src python scripts/run_experiment.py \
  --config configs/experiments/bioasq_TINY_mps_fp32_lora_clean_seed42.yaml
```

The runner passes the adapter path into generation, writes the exact inference
inputs, scores the predictions, and saves the same per-run artifacts as the
inference-only experiment path. It also writes `inference_manifest.json` into the
experiment directory so the resolved device, dtype, model id, adapter path, seed,
and git commit are inspectable.

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
PYTHONPATH=src python scripts/summarize_peft_adapter.py \
  --adapter-dir results/ckpts/tiny_adapter \
  --out results/phase4/peft/tiny_adapter_manifest.json
PYTHONPATH=src python scripts/summarize_runtime_manifests.py \
  --experiments-csv results/phase4/experiments.csv \
  --out results/phase4/runtime/runtime_summary.json
python scripts/validate_experiment_integrity.py \
  --only-configured \
  --out results/analysis/integrity_report.json
python proof/generate_canonical_manifest.py
python proof/validate_evidence_manifest.py
```

## Inspect Outputs

Start with:

- `results/phase4/summary.json`
- `results/phase4/analysis/phase4_findings.md`
- `results/phase4/report_artifacts/tables/perturbation_ranking_macro_avg.md`
- `results/phase4/peft/tiny_adapter_manifest.json`
- `results/phase4/runtime/runtime_summary.json`
- `proof/evidence_manifest.latest.json`

## Shutdown

There is no long-running service to stop. Local runs end when the Python command
exits.

If a run is interrupted, inspect the partially written directory under
`results/experiments/` before deleting or rerunning it.
