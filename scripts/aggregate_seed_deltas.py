#!/usr/bin/env python3
"""
Aggregate deltas across seeds for phenotype robustness runs.

Input:
  phenotype_deltas_vs_clean.csv  (from reaggregate_by_phenotype.py)

Output:
  - seed_agg_deltas.csv   (grouped mean/std/min/max/count)
  - seed_agg_deltas.json  (same content, records format)

Example:
  uv run scripts/aggregate_seed_deltas.py \
    --deltas-csv results/aggregates_re/phenotype_deltas_vs_clean.csv \
    --out-dir results/phase4/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DELTA_COLS = [
    "delta__macro_avg",
    "delta__yesno_acc",
    "delta__factoid_em",
    "delta__factoid_f1",
    "delta__list_f1",
    "delta__list_precision",
    "delta__list_recall",
    "delta__summary_rougeL",
]


GROUP_COLS = ["dataset", "runtime", "model", "phenotype", "perturbation"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas-csv", required=True, help="phenotype_deltas_vs_clean.csv")
    ap.add_argument("--out-dir", default="results/phase4/analysis", help="output directory")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.deltas_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    missing = [c for c in GROUP_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] missing required columns: {missing}")

    present_delta_cols = [c for c in DELTA_COLS if c in df.columns]
    if not present_delta_cols:
        raise SystemExit(f"[error] no delta__* metric columns found. available={list(df.columns)}")

    # ensure numeric
    for c in present_delta_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby(GROUP_COLS, dropna=False)

    agg_parts = []
    for metric in present_delta_cols:
        a = g[metric].agg(["count", "mean", "std", "min", "max"]).reset_index()
        a.insert(len(GROUP_COLS), "metric", metric.replace("delta__", ""))
        a = a.rename(
            columns={
                "count": "n",
                "mean": "delta_mean",
                "std": "delta_std",
                "min": "delta_min",
                "max": "delta_max",
            }
        )
        agg_parts.append(a)

    out = pd.concat(agg_parts, ignore_index=True)

    # Helpful ordering
    out = out.sort_values(["phenotype", "metric", "perturbation"], kind="stable").reset_index(
        drop=True
    )

    out_csv = out_dir / "seed_agg_deltas.csv"
    out_json = out_dir / "seed_agg_deltas.json"

    out.to_csv(out_csv, index=False)
    out.to_json(out_json, orient="records", indent=2)

    print(f"[seed-agg] rows={len(df)} groups={g.ngroups} metrics={len(present_delta_cols)}")
    print(f"[done] wrote:")
    print(f"  - {out_csv}")
    print(f"  - {out_json}")


if __name__ == "__main__":
    main()
