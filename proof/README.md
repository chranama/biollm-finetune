# Proof System

This directory stores the canonical latest-only evidence bundle.

## Files
- `evidence_contract.schema.json`
- `evidence_manifest.latest.json`
- `proof_points.latest.md`
- `generate_canonical_manifest.py`
- `validate_evidence_manifest.py`

## Canonical Proof Run (deterministic subset)

```bash
python scripts/run_experiments.py
python scripts/aggregate_experiments.py
python scripts/analyze_phase4_results.py
python scripts/validate_experiment_integrity.py
python proof/generate_canonical_manifest.py
```

## Validate

```bash
python proof/validate_evidence_manifest.py
```
