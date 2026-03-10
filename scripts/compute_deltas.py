#!/usr/bin/env python3
"""
Compute deltas vs clean baselines from an experiments.csv produced by aggregate_experiments.

This script expects (at minimum) these columns:
- experiment, dataset, model, seed, perturbation
- macro_avg and/or per-type metrics (yesno_acc, factoid_f1, list_f1, summary_rougeL)

If 'runtime' is missing, it will be DERIVED from the experiment naming convention:
  <dataset>_<runtime>_<perturbation>_seed<seed>

Example:
  uv run scripts/compute_deltas.py \
    --experiments-csv results/phase4/experiments.csv \
    --out-dir results/phase4/deltas
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----------------------------
# I/O
# ----------------------------


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    return header, rows


def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ----------------------------
# Helpers
# ----------------------------


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _to_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        s = str(x).strip()
        if s == "":
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def derive_runtime_from_experiment(
    experiment: str,
    dataset: str,
    perturbation: str,
    seed: int,
) -> str:
    """
    Robustly derive runtime from:
      experiment = f"{dataset}_{runtime}_{perturbation}_seed{seed}"

    We use dataset/perturbation/seed columns to carve out the runtime substring.
    This handles perturbations with underscores (e.g., irrelevant_noise_heavy).
    """
    exp = (experiment or "").strip()
    ds = (dataset or "").strip()
    pert = (perturbation or "").strip()
    seed_sfx = f"_seed{seed}"

    # Expected suffix includes perturbation and seed
    suffix = f"_{pert}{seed_sfx}"

    # Expected prefix includes dataset
    prefix = f"{ds}_"

    if exp.startswith(prefix) and exp.endswith(suffix):
        mid = exp[len(prefix) : -len(suffix)]
        mid = mid.strip("_")
        if mid:
            return mid

    # Fallback: parse from right assuming runtime is two tokens (e.g., mps_fp32)
    parts = exp.split("_")
    if len(parts) >= 4 and parts[-1].startswith("seed"):
        # runtime commonly "mps_fp32" or "cuda_bf16"
        return "_".join(parts[-4:-2])

    return ""


# ----------------------------
# Core
# ----------------------------

REQUIRED_KEYS = ["experiment", "dataset", "model", "seed", "perturbation"]
METRIC_KEYS = ["macro_avg", "yesno_acc", "factoid_f1", "list_f1", "summary_rougeL"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-csv", required=True, help="Path to experiments.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    experiments_csv = Path(args.experiments_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header, rows = read_csv_rows(experiments_csv)

    # Basic header validation
    for k in REQUIRED_KEYS:
        if k not in header:
            raise SystemExit(f"[error] key col '{k}' not in CSV header. Available: {header}")

    # If runtime is missing, derive it
    if "runtime" not in header:
        for r in rows:
            seed = _to_int(r.get("seed", 0))
            runtime = derive_runtime_from_experiment(
                experiment=r.get("experiment", ""),
                dataset=r.get("dataset", ""),
                perturbation=r.get("perturbation", ""),
                seed=seed,
            )
            r["runtime"] = runtime
        header = header + ["runtime"]

    # Index clean baselines by (dataset, runtime, seed, model)
    clean_index: Dict[Tuple[str, str, int, str], Dict[str, str]] = {}
    for r in rows:
        if (r.get("perturbation") or "").strip().lower() != "clean":
            continue
        key = (
            (r.get("dataset") or "").strip(),
            (r.get("runtime") or "").strip(),
            _to_int(r.get("seed", 0)),
            (r.get("model") or "").strip(),
        )
        clean_index[key] = r

    missing_baselines = 0
    out_long: List[Dict[str, Any]] = []
    out_wide: List[Dict[str, Any]] = []

    for r in rows:
        ds = (r.get("dataset") or "").strip()
        rt = (r.get("runtime") or "").strip()
        seed = _to_int(r.get("seed", 0))
        model = (r.get("model") or "").strip()
        pert = (r.get("perturbation") or "").strip()

        key = (ds, rt, seed, model)
        base = clean_index.get(key)

        if base is None:
            # For clean rows themselves, we skip baseline requirement
            if pert.lower() != "clean":
                missing_baselines += 1
            base = None

        wide_row: Dict[str, Any] = dict(r)
        wide_row["baseline_experiment"] = base.get("experiment") if base else ""
        wide_row["baseline_found"] = bool(base)

        for mk in METRIC_KEYS:
            if mk not in r:
                continue
            val = _to_float(r.get(mk))
            base_val = _to_float(base.get(mk)) if base else float("nan")
            delta = val - base_val if (base is not None) else float("nan")
            wide_row[f"{mk}_delta"] = delta

            out_long.append(
                {
                    "experiment": r.get("experiment"),
                    "dataset": ds,
                    "runtime": rt,
                    "model": model,
                    "seed": seed,
                    "perturbation": pert,
                    "metric": mk,
                    "value": val,
                    "baseline_value": base_val if base else None,
                    "delta": delta,
                    "baseline_experiment": base.get("experiment") if base else None,
                    "baseline_found": bool(base),
                }
            )

        out_wide.append(wide_row)

    # Write outputs
    wide_header = list(dict.fromkeys(list(out_wide[0].keys()) if out_wide else header))
    write_csv(out_dir / "deltas_wide.csv", out_wide, header=wide_header)

    long_header = [
        "experiment",
        "dataset",
        "runtime",
        "model",
        "seed",
        "perturbation",
        "metric",
        "value",
        "baseline_value",
        "delta",
        "baseline_experiment",
        "baseline_found",
    ]
    write_csv(out_dir / "deltas_long.csv", out_long, header=long_header)

    summary = {
        "experiments_csv": str(experiments_csv),
        "n_rows": len(rows),
        "n_long_rows": len(out_long),
        "missing_baselines": missing_baselines,
        "metrics": [k for k in METRIC_KEYS if any(k in rr for rr in rows)],
    }
    write_json(out_dir / "deltas_summary.json", summary)

    print(
        f"[deltas] rows={len(rows)} long={len(out_long)} "
        f"missing_baselines={missing_baselines} → {out_dir}"
    )


if __name__ == "__main__":
    main()
