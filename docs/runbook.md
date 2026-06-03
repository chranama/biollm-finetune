# Runbook

This runbook covers local setup, validation, a small experiment run, aggregation,
and cleanup.

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
- `phenotypes.jsonl`
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
