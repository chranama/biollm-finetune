#!/usr/bin/env python3
"""
Generate Phase 4 tables + figures (robustness deltas + phenotype deltas).

Inputs (defaults):
  - results/phase4/deltas/deltas_wide.csv
  - results/aggregates_re/phenotype_deltas_vs_clean.csv

Outputs (default outdir: results/phase4/report_artifacts):
  tables/
    - perturbation_ranking_macro_avg.csv
    - perturbation_ranking_macro_avg.md
    - perturbation_ranking_by_metric.csv
    - perturbation_ranking_by_metric.md
    - phenotype_delta_macro_avg.csv
    - phenotype_delta_macro_avg.md
  figures/
    - delta_macro_avg_by_perturbation.png
    - delta_by_metric_top_perturbations.png
    - phenotype_heatmap_delta_macro_avg.png

Usage:
  uv run scripts/generate_phase4_tables_and_figures.py

  # If your paths differ:
  uv run scripts/generate_phase4_tables_and_figures.py \
    --deltas-wide results/phase4/deltas/deltas_wide.csv \
    --phenotype-deltas results/aggregates_re/phenotype_deltas_vs_clean.csv \
    --outdir results/phase4/report_artifacts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------

METRIC_ORDER = [
    "macro_avg",
    "yesno_acc",
    "factoid_em",
    "factoid_f1",
    "list_f1",
    "list_precision",
    "list_recall",
    "summary_rougeL",
]

METRIC_LABEL = {
    "macro_avg": "Macro Avg",
    "yesno_acc": "Yes/No Acc",
    "factoid_em": "Factoid EM",
    "factoid_f1": "Factoid F1",
    "list_f1": "List F1",
    "list_precision": "List Precision",
    "list_recall": "List Recall",
    "summary_rougeL": "Summary ROUGE-L",
}

DELTA_PREFIX = "delta__"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_md_table(path: Path, df: pd.DataFrame, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    md = df.to_markdown(index=index)
    path.write_text(md + "\n", encoding="utf-8")


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_col(df: pd.DataFrame, col: str, default=None):
    return df[col] if col in df.columns else default


# ----------------------------
# Core aggregations
# ----------------------------

def perturbation_ranking(deltas_wide: pd.DataFrame) -> pd.DataFrame:
    # Detect delta naming convention
    if any(c.startswith("delta__") for c in deltas_wide.columns):
        delta_cols = [c for c in deltas_wide.columns if c.startswith("delta__")]
        metrics = [c.replace("delta__", "") for c in delta_cols]
        col_for = lambda m: f"delta__{m}"
    elif any(c.endswith("_delta") for c in deltas_wide.columns):
        delta_cols = [c for c in deltas_wide.columns if c.endswith("_delta")]
        metrics = [c[: -len("_delta")] for c in delta_cols]
        col_for = lambda m: f"{m}_delta"
    else:
        raise ValueError("No delta columns found. Expected either delta__<metric> or <metric>_delta columns.")

    if "perturbation" not in deltas_wide.columns:
        raise ValueError("deltas_wide.csv must contain a 'perturbation' column.")

    out_rows = []
    for pert, g in deltas_wide.groupby("perturbation", dropna=False):
        row = {"perturbation": str(pert), "n_runs": int(len(g))}
        for m in metrics:
            col = col_for(m)
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"mean__{m}"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"std__{m}"] = float(vals.std(ddof=1)) if vals.notna().sum() >= 2 else np.nan
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    if "mean__macro_avg" in out.columns:
        out = out.sort_values(by="mean__macro_avg", ascending=True)
    return out


def phenotype_macro_table(phenotype_deltas: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize macro_avg delta by (phenotype, perturbation) averaged over seeds.
    """
    need = {"phenotype", "perturbation", "delta__macro_avg"}
    missing = [c for c in need if c not in phenotype_deltas.columns]
    if missing:
        raise ValueError(f"phenotype_deltas_vs_clean.csv missing columns: {missing}")

    df = phenotype_deltas.copy()
    df["delta__macro_avg"] = pd.to_numeric(df["delta__macro_avg"], errors="coerce")

    agg = (
        df.groupby(["phenotype", "perturbation"], dropna=False)["delta__macro_avg"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_delta_macro_avg", "std": "std_delta_macro_avg", "count": "n_runs"})
    )

    # Sort: largest degradation first within phenotype
    agg = agg.sort_values(["phenotype", "mean_delta_macro_avg"], ascending=[True, True])
    return agg


# ----------------------------
# Plotting
# ----------------------------

def plot_macro_by_perturbation(ranking: pd.DataFrame, outpath: Path, topk: Optional[int] = None) -> None:
    df = ranking.copy()
    if "mean__macro_avg" not in df.columns:
        raise ValueError("ranking table missing mean__macro_avg")

    if topk is not None:
        df = df.head(int(topk))

    x = df["perturbation"].tolist()
    y = df["mean__macro_avg"].to_numpy()

    plt.figure(figsize=(10, max(4, 0.35 * len(df))))
    plt.barh(x, y)
    plt.axvline(0.0)
    plt.title("Mean Δ vs Clean (Macro Avg) by Perturbation")
    plt.xlabel("Δ Macro Avg (perturbed - clean)")
    plt.ylabel("Perturbation")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_metric_grid_topperts(ranking: pd.DataFrame, outpath: Path, topk: int = 6) -> None:
    """
    Compact figure: for top-k most harmful perturbations (by macro), show deltas across metrics.
    (Single figure; avoids making 8 separate charts.)
    """
    df = ranking.copy()
    # pick topk by macro degradation
    if "mean__macro_avg" in df.columns:
        df = df.sort_values("mean__macro_avg", ascending=True).head(int(topk))
    else:
        df = df.head(int(topk))

    perts = df["perturbation"].tolist()

    metrics = [m for m in METRIC_ORDER if f"mean__{m}" in df.columns]
    if not metrics:
        raise ValueError("No mean__<metric> columns found in ranking table.")

    # Build matrix (metrics x perts)
    mat = np.vstack([df[f"mean__{m}"].to_numpy() for m in metrics])

    plt.figure(figsize=(10, 0.6 * len(metrics) + 2))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(range(len(metrics)), [METRIC_LABEL.get(m, m) for m in metrics])
    plt.xticks(range(len(perts)), perts, rotation=30, ha="right")
    plt.title(f"Mean Δ vs Clean across Metrics (Top {topk} most harmful by Macro Avg)")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_phenotype_heatmap(phenotype_deltas: pd.DataFrame, outpath: Path) -> None:
    """
    Heatmap: phenotype x perturbation mean delta macro_avg.
    """
    df = phenotype_deltas.copy()
    df["delta__macro_avg"] = pd.to_numeric(df["delta__macro_avg"], errors="coerce")

    pivot = (
        df.groupby(["phenotype", "perturbation"], dropna=False)["delta__macro_avg"]
        .mean()
        .reset_index()
        .pivot(index="phenotype", columns="perturbation", values="delta__macro_avg")
    )

    # Nice ordering: put ALL first if present
    idx = list(pivot.index)
    if "ALL" in idx:
        idx = ["ALL"] + [x for x in idx if x != "ALL"]
        pivot = pivot.loc[idx]

    plt.figure(figsize=(max(8, 1.2 * len(pivot.columns)), max(3, 0.6 * len(pivot.index))))
    im = plt.imshow(pivot.to_numpy(), aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(range(len(pivot.index)), pivot.index.tolist())
    plt.xticks(range(len(pivot.columns)), pivot.columns.tolist(), rotation=30, ha="right")
    plt.title("Phenotype × Perturbation: Mean Δ Macro Avg vs Clean")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas-wide", default="results/phase4/deltas/deltas_wide.csv")
    ap.add_argument("--phenotype-deltas", default="results/aggregates_re/phenotype_deltas_vs_clean.csv")
    ap.add_argument("--outdir", default="results/phase4/report_artifacts")
    ap.add_argument("--topk", type=int, default=10, help="Top-k perturbations to display in some outputs.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    tables_dir = outdir / "tables"
    figs_dir = outdir / "figures"
    ensure_dir(tables_dir)
    ensure_dir(figs_dir)

    deltas_path = Path(args.deltas_wide)
    phen_path = Path(args.phenotype_deltas)

    if not deltas_path.exists():
        raise SystemExit(f"[error] deltas-wide not found: {deltas_path}")
    if not phen_path.exists():
        raise SystemExit(f"[error] phenotype-deltas not found: {phen_path}")

    deltas_wide = pd.read_csv(deltas_path)
    phenotype_deltas = pd.read_csv(phen_path)

    # Coerce delta columns
    delta_cols = [c for c in deltas_wide.columns if c.startswith(DELTA_PREFIX)]
    deltas_wide = coerce_numeric(deltas_wide, delta_cols)
    phenotype_deltas = coerce_numeric(phenotype_deltas, [c for c in phenotype_deltas.columns if c.startswith(DELTA_PREFIX)])

    # ----------------------------
    # Tables: perturbation ranking
    # ----------------------------
    ranking = perturbation_ranking(deltas_wide)

    # Simple macro-only ranking table
    macro_cols = ["perturbation", "n_runs"]
    if "mean__macro_avg" in ranking.columns:
        macro_cols += ["mean__macro_avg"]
    if "std__macro_avg" in ranking.columns:
        macro_cols += ["std__macro_avg"]
    rank_macro = ranking[macro_cols].copy()

    rank_macro.to_csv(tables_dir / "perturbation_ranking_macro_avg.csv", index=False)
    write_md_table(tables_dir / "perturbation_ranking_macro_avg.md", rank_macro, index=False)

    # Full ranking by metric (wide)
    ranking.to_csv(tables_dir / "perturbation_ranking_by_metric.csv", index=False)
    write_md_table(tables_dir / "perturbation_ranking_by_metric.md", ranking, index=False)

    # ----------------------------
    # Tables: phenotype macro deltas
    # ----------------------------
    ph_macro = phenotype_macro_table(phenotype_deltas)
    ph_macro.to_csv(tables_dir / "phenotype_delta_macro_avg.csv", index=False)
    write_md_table(tables_dir / "phenotype_delta_macro_avg.md", ph_macro, index=False)

    # ----------------------------
    # Figures
    # ----------------------------
    plot_macro_by_perturbation(
        ranking=ranking,
        outpath=figs_dir / "delta_macro_avg_by_perturbation.png",
        topk=args.topk,
    )

    plot_metric_grid_topperts(
        ranking=ranking,
        outpath=figs_dir / "delta_by_metric_top_perturbations.png",
        topk=min(int(args.topk), 10),
    )

    plot_phenotype_heatmap(
        phenotype_deltas=phenotype_deltas,
        outpath=figs_dir / "phenotype_heatmap_delta_macro_avg.png",
    )

    print("[done] wrote artifacts to:", outdir)
    print("  tables:", tables_dir)
    print("  figures:", figs_dir)


if __name__ == "__main__":
    main()