#!/usr/bin/env python3
"""
Re-aggregate experiment results into analysis-friendly tables.

What this script produces (in --outdir):
  1) runs_reaggregated.csv
     - one row per run, extracted from manifest.json + metrics.json (+ phenotype counts)

  2) deltas_vs_clean.csv
     - for every non-clean run, compute deltas vs the matched clean baseline run
       (same dataset/runtime/model/seed)

  3) summary_by_perturbation.csv
     - mean/std across seeds for (raw metrics and deltas) grouped by dataset/runtime/model/perturbation

Notes:
- This does NOT recompute metrics from predictions; it trusts metrics.json already produced.
- It DOES sanity-check that each run has the required files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# -------------------------
# IO helpers
# -------------------------


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    safe_get(d, "factoid.f1") => d["factoid"]["f1"] if present else default
    """
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def count_phenotype_tags(phenotypes_json: Dict[str, Any]) -> Dict[str, int]:
    """
    phenotypes.json format:
      {
        "schema": {...},
        "tags": { "<qid>": ["tag1", "tag2", ...], ... }
      }
    """
    tags = phenotypes_json.get("tags") or {}
    counts: Dict[str, int] = {}
    if not isinstance(tags, dict):
        return counts

    for _, tag_list in tags.items():
        if not isinstance(tag_list, list):
            continue
        for t in tag_list:
            if not isinstance(t, str):
                continue
            counts[t] = counts.get(t, 0) + 1
    return counts


# -------------------------
# Run discovery
# -------------------------


@dataclass
class RunPaths:
    run_dir: Path
    manifest: Path
    metrics: Path
    phenotypes: Optional[Path]


def iter_run_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            yield p


def find_run_files(run_dir: Path) -> Optional[RunPaths]:
    manifest = run_dir / "manifest.json"
    metrics = run_dir / "metrics.json"
    phenotypes = run_dir / "phenotypes.json"

    if not manifest.exists() or not metrics.exists():
        return None

    return RunPaths(
        run_dir=run_dir,
        manifest=manifest,
        metrics=metrics,
        phenotypes=phenotypes if phenotypes.exists() else None,
    )


# -------------------------
# Canonical keys / matching
# -------------------------


def baseline_key(row: Dict[str, Any]) -> Tuple[str, str, str, int]:
    """
    Key used to match non-clean runs to the correct clean baseline.
    Must be identical for clean and perturbed variants you want to compare.
    """
    return (
        str(row.get("dataset", "")),
        str(row.get("runtime", "")),
        str(row.get("model", "")),
        int(row.get("seed", -1)),
    )


# -------------------------
# Extraction
# -------------------------


def extract_row(paths: RunPaths) -> Dict[str, Any]:
    man = read_json(paths.manifest)
    met = read_json(paths.metrics)

    # core identifiers
    row: Dict[str, Any] = {
        "run_id": man.get("experiment") or paths.run_dir.name,
        "run_dir": str(paths.run_dir),
        "config": man.get("config"),
        "dataset": man.get("dataset"),
        "model": man.get("model"),
        "seed": man.get("seed"),
        "perturbation": man.get("perturbation"),
        "task": man.get("task"),
        "runtime": None,  # try to infer from run_id if not stored
        "start_time_utc": man.get("start_time_utc"),
        "end_time_utc": man.get("end_time_utc"),
        "n_examples": man.get("n_examples"),
        "n_changed_vs_clean": man.get("n_changed_vs_clean"),
    }

    # infer runtime tag from canonical naming if possible: dataset_runtime_pert_seedX
    rid = str(row["run_id"])
    parts = rid.split("_")
    if len(parts) >= 4:
        # Example: bioasq_TINY_mps_fp32_clean_seed42
        row["runtime"] = "_".join(parts[2:4])
    else:
        row["runtime"] = None

    # metrics (common fields from your metrics.py)
    row.update(
        {
            "macro_avg": safe_get(met, "macro_avg", None),
            "yesno_acc": safe_get(met, "yesno.accuracy", None),
            "factoid_em": safe_get(met, "factoid.em", None),
            "factoid_f1": safe_get(met, "factoid.f1", None),
            "list_prec": safe_get(met, "list.precision", None),
            "list_rec": safe_get(met, "list.recall", None),
            "list_f1": safe_get(met, "list.f1", None),
            "summary_rougeL": safe_get(met, "summary.rougeL", None),
            "count_yesno": safe_get(met, "counts.yesno", None),
            "count_factoid": safe_get(met, "counts.factoid", None),
            "count_list": safe_get(met, "counts.list", None),
            "count_summary": safe_get(met, "counts.summary", None),
            "missing_pred_without_gold": safe_get(met, "missing.pred_without_gold", None),
        }
    )

    # phenotype counts (optional)
    if paths.phenotypes is not None:
        ph = read_json(paths.phenotypes)
        ph_counts = count_phenotype_tags(ph)
        # store as columns phenotype__<tag>
        for k, v in ph_counts.items():
            row[f"phenotype__{k}"] = int(v)

    return row


# -------------------------
# Aggregations
# -------------------------

METRIC_COLS = [
    "macro_avg",
    "yesno_acc",
    "factoid_em",
    "factoid_f1",
    "list_f1",
    "summary_rougeL",
]

DELTA_COLS = [f"delta__{c}" for c in METRIC_COLS]


def compute_deltas_vs_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each non-clean run, find clean baseline with same (dataset, runtime, model, seed).
    """
    # split
    clean = df[df["perturbation"] == "clean"].copy()
    pert = df[df["perturbation"] != "clean"].copy()

    # index clean by key
    clean_index: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for _, r in clean.iterrows():
        key = (str(r["dataset"]), str(r["runtime"]), str(r["model"]), int(r["seed"]))
        clean_index[key] = r.to_dict()

    rows: List[Dict[str, Any]] = []
    for _, r in pert.iterrows():
        key = (str(r["dataset"]), str(r["runtime"]), str(r["model"]), int(r["seed"]))
        base = clean_index.get(key)
        out = r.to_dict()
        out["matched_clean_run_id"] = base.get("run_id") if base else None
        out["matched_clean_found"] = bool(base)

        for c in METRIC_COLS:
            rv = out.get(c)
            bv = base.get(c) if base else None
            if isinstance(rv, (int, float)) and isinstance(bv, (int, float)):
                out[f"delta__{c}"] = float(rv) - float(bv)
            else:
                out[f"delta__{c}"] = None

        rows.append(out)

    return pd.DataFrame(rows)


def summarize_by_perturbation(df_all: pd.DataFrame, df_deltas: pd.DataFrame) -> pd.DataFrame:
    """
    Group summary stats by (dataset, runtime, model, perturbation).
    """
    group_keys = ["dataset", "runtime", "model", "perturbation"]

    # raw metrics (includes clean + perturbed)
    raw = df_all.groupby(group_keys)[METRIC_COLS].agg(["mean", "std", "count"]).reset_index()
    raw.columns = ["__".join([c for c in col if c]) for col in raw.columns.to_flat_index()]

    # deltas (perturbations only)
    if len(df_deltas) > 0:
        deltas = (
            df_deltas.groupby(group_keys)[DELTA_COLS].agg(["mean", "std", "count"]).reset_index()
        )
        deltas.columns = [
            "__".join([c for c in col if c]) for col in deltas.columns.to_flat_index()
        ]
        # outer merge so clean rows exist even if no deltas
        out = pd.merge(raw, deltas, how="left", on=group_keys)
    else:
        out = raw

    return out


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiments-root",
        default="results/experiments",
        help="Root directory of experiment runs",
    )
    ap.add_argument("--outdir", default="results/aggregates_re", help="Output directory")
    ap.add_argument(
        "--strict", action="store_true", help="Fail if any run is missing required files"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.experiments_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect rows
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for run_dir in iter_run_dirs(root):
        files = find_run_files(run_dir)
        if files is None:
            missing.append(str(run_dir))
            continue
        rows.append(extract_row(files))

    if args.strict and missing:
        raise SystemExit(f"[error] missing manifest/metrics in {len(missing)} run dirs")

    if not rows:
        raise SystemExit(f"[error] no runs found under {root}")

    df = pd.DataFrame(rows)

    # Write run-level table
    runs_path = outdir / "runs_reaggregated.csv"
    df.sort_values(
        ["dataset", "runtime", "model", "seed", "perturbation", "run_id"],
        inplace=True,
        na_position="last",
    )
    df.to_csv(runs_path, index=False)

    # Deltas vs clean
    df_deltas = compute_deltas_vs_clean(df)
    deltas_path = outdir / "deltas_vs_clean.csv"
    df_deltas.sort_values(
        ["dataset", "runtime", "model", "seed", "perturbation", "run_id"],
        inplace=True,
        na_position="last",
    )
    df_deltas.to_csv(deltas_path, index=False)

    # Summary by perturbation
    df_summary = summarize_by_perturbation(df, df_deltas)
    summary_path = outdir / "summary_by_perturbation.csv"
    df_summary.sort_values(
        ["dataset", "runtime", "model", "perturbation"], inplace=True, na_position="last"
    )
    df_summary.to_csv(summary_path, index=False)

    # Tiny json “index” for convenience
    index = {
        "experiments_root": str(root),
        "n_runs_total": int(len(df)),
        "n_runs_clean": int((df["perturbation"] == "clean").sum()),
        "n_runs_perturbed": int((df["perturbation"] != "clean").sum()),
        "outputs": {
            "runs_reaggregated": str(runs_path),
            "deltas_vs_clean": str(deltas_path),
            "summary_by_perturbation": str(summary_path),
        },
    }
    with (outdir / "reaggregate_index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(
        f"[done] wrote:\n  - {runs_path}\n  - {deltas_path}\n  - {summary_path}\n  - {outdir / 'reaggregate_index.json'}"
    )


if __name__ == "__main__":
    main()
