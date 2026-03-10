# Canonical Proof Points (Latest)

## Proof 1: Phase 4 Summary
- Claim: deterministic phase-level robustness summary is available.
- Command: `python proof/generate_canonical_manifest.py`
- Artifacts:
  - `results/phase4/summary.json`
  - `results/phase4/experiments.csv`
- Validation signal: both artifacts exist and can be inspected directly.

## Proof 2: Findings + Ranking
- Claim: reproducible robustness findings and perturbation ranking are persisted.
- Command: `python proof/generate_canonical_manifest.py`
- Artifacts:
  - `results/phase4/analysis/phase4_findings.md`
  - `results/phase4/report_artifacts/tables/perturbation_ranking_macro_avg.md`
- Validation signal: findings and ranking table exist and are non-empty.

## Proof 3: Integrity Metadata
- Claim: canonical reproducibility checks are represented in persisted reports.
- Command: `python proof/generate_canonical_manifest.py`
- Artifacts:
  - `results/analysis/integrity_report.json`
  - `results/phase4/deltas/deltas_summary.json`
- Validation signal: integrity and delta summary reports both exist.
