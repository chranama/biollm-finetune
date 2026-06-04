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
- Claim: canonical reproducibility checks are represented in persisted reports for the configured current run set.
- Command: `python scripts/validate_experiment_integrity.py --only-configured --out results/analysis/integrity_report.json`
- Artifacts:
  - `results/analysis/integrity_report.json`
  - `results/phase4/deltas/deltas_summary.json`
- Validation signal: integrity and delta summary reports both exist.

## Proof 4: PEFT Adapter Evidence
- Claim: local LoRA adapter training is represented without committing generated adapter weights.
- Command: `PYTHONPATH=src python scripts/summarize_peft_adapter.py --adapter-dir results/ckpts/tiny_adapter --out results/phase4/peft/tiny_adapter_manifest.json`
- Artifacts:
  - `results/phase4/peft/tiny_adapter_manifest.json`
- Validation signal: adapter manifest records base model, LoRA settings, quantization flags, training data count, runtime, and checksum metadata.

## Proof 5: Resolved Runtime Evidence
- Claim: adapter-aware and base inference runs expose actual runtime state.
- Command: `PYTHONPATH=src python scripts/summarize_runtime_manifests.py --experiments-csv results/phase4/experiments.csv --out results/phase4/runtime/runtime_summary.json`
- Artifacts:
  - `results/phase4/runtime/runtime_summary.json`
- Validation signal: runtime summary shows requested device/dtype, resolved device/dtype, model id, adapter path, and per-run manifest locations.
