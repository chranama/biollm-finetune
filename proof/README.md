# Evidence Manifest

This directory stores validation metadata for the latest saved artifact set. The
directory name remains `proof/` for compatibility with the existing scripts.

## Files
- `evidence_contract.schema.json`
- `evidence_manifest.latest.json`
- `proof_points.latest.md`
- `generate_canonical_manifest.py`
- `validate_evidence_manifest.py`

## Deterministic Validation Run

Run from repository root.

```bash
python scripts/run_experiment.py --config configs/experiments/bioasq_TINY_mps_fp32_clean_seed42.yaml
python scripts/run_experiment.py --config configs/experiments/bioasq_TINY_mps_fp32_lexical_noise_seed42.yaml
python scripts/aggregate_experiments.py
python scripts/analyze_phase4_results.py
python scripts/validate_experiment_integrity.py --strict --only-configured
python proof/generate_canonical_manifest.py
```

## Validate The Saved Manifest

```bash
python proof/validate_evidence_manifest.py
```
