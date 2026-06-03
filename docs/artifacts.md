# Artifacts

BioLLM-Finetune writes experiment state to disk so results can be inspected
without rerunning every step.

## Run Artifacts

Each run under `results/experiments/` contains files such as:

- `inputs.jsonl`: exact inputs passed to inference
- `predictions.jsonl`: model outputs
- `metrics.json`: BioASQ-style scores
- `phenotypes.jsonl`: phenotype tags for the run inputs
- `manifest.json`: run metadata, config path, seed, model, perturbation, and
  timing information

These files are the lowest-level artifacts for checking a single run.

## Phase 4 Artifacts

Phase-level aggregation writes to `results/phase4/`.

Useful files:

- `summary.json`: grouped run summary
- `experiments.csv`: flat table of experiment metrics and metadata
- `deltas/deltas_long.csv`: long-form clean-vs-perturbed deltas
- `deltas/deltas_wide.csv`: wide-form clean-vs-perturbed deltas
- `deltas/deltas_summary.json`: aggregate delta summary
- `analysis/phase4_findings.md`: human-readable robustness findings
- `analysis/phenotype_findings.md`: human-readable phenotype findings
- `report_artifacts/tables/`: Markdown and CSV tables
- `report_artifacts/figures/`: generated figures

## Evidence Manifest

The `proof/` directory contains validation scripts and metadata for the latest
saved artifact set. The directory name is retained for compatibility with the
existing scripts.

Primary files:

- `proof/evidence_contract.schema.json`
- `proof/evidence_manifest.latest.json`
- `proof/proof_points.latest.md`
- `proof/validate_evidence_manifest.py`

Validate the manifest:

```bash
python proof/validate_evidence_manifest.py
```

The manifest validation checks that referenced artifacts exist and that expected
metadata, such as seed configuration, is present.

## What Artifacts Do Not Show

The saved artifacts do not establish clinical correctness, model safety, or
production reliability. They show that the repository can run a bounded
evaluation workflow, persist its intermediate state, and validate references to
the current saved outputs.
