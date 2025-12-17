#!/usr/bin/env python3
"""
Rank perturbations by severity (worst -> best) per phenotype + metric,
using seed-aggregated deltas.

Input:
  seed_agg_deltas.csv (from aggregate_seed_deltas.py)

Output:
  - perturbation_ranking.csv
  - perturbation_ranking.md   (compact tables)

Example:
  uv run scripts/rank_perturbations.py \
    --seed-agg-csv results/phase4/analysis/seed_agg_deltas.csv \
    --out-dir results/phase4/analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-agg-csv", required=True, help="seed_agg_deltas.csv")
    ap.add_argument("--out-dir", default="results/phase4/analysis", help="output directory")
    ap.add_argument("--top-k", type=int, default=10, help="top-k per (phenotype, metric)")
    return ap.parse_args()


def _to_md_table(df: pd.DataFrame, cols: list[str]) -> str:
    # very small, dependency-free markdown table
    head = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows = []
    for _, r in df[cols].iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return head + sep + "\n".join(rows) + "\n"


def main() -> None:
    args = parse_args()
    in_path = Path(args.seed_agg_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    required = ["dataset", "runtime", "model", "phenotype", "metric", "perturbation", "delta_mean", "n"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] missing required columns: {missing}")

    # Rank: more negative delta_mean = worse
    df["rank_worst"] = df.groupby(["dataset", "runtime", "model", "phenotype", "metric"])["delta_mean"] \
        .rank(method="dense", ascending=True)

    df_rank = df.sort_values(
        ["phenotype", "metric", "rank_worst", "perturbation"],
        kind="stable"
    ).reset_index(drop=True)

    out_csv = out_dir / "perturbation_ranking.csv"
    df_rank.to_csv(out_csv, index=False)

    # Markdown: top-k per phenotype+metric (keeps it readable)
    top_k = max(1, int(args.top_k))
    md_lines = ["# Perturbation Severity Ranking", ""]
    for (phenotype, metric), sub in df_rank.groupby(["phenotype", "metric"], dropna=False):
        sub_top = sub.nsmallest(top_k, "rank_worst").copy()
        sub_top["delta_mean"] = sub_top["delta_mean"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        sub_top["delta_std"] = sub_top.get("delta_std", pd.Series([None]*len(sub_top))).map(
            lambda x: f"{x:.4f}" if pd.notna(x) else ""
        )

        md_lines.append(f"## phenotype={phenotype} | metric={metric}")
        md_lines.append("")
        md_lines.append(_to_md_table(
            sub_top,
            cols=["rank_worst", "perturbation", "delta_mean", "delta_std", "delta_min", "delta_max", "n"]
        ))
        md_lines.append("")

    out_md = out_dir / "perturbation_ranking.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[rank] rows={len(df_rank)}")
    print(f"[done] wrote:")
    print(f"  - {out_csv}")
    print(f"  - {out_md}")


if __name__ == "__main__":
    main()