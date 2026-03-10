#!/usr/bin/env python3
"""
Analyze phenotype effects from phenotype_deltas_vs_clean.csv.

Inputs
------
Produced by scripts/reaggregate_by_phenotype.py:
  - phenotype_deltas_vs_clean.csv

This file contains rows for non-clean runs only, with delta__<metric> columns
computed vs the matching clean baseline for (dataset, runtime, model, seed, phenotype).

Outputs
-------
Writes to --outdir:
  - phenotype_effects.json
  - phenotype_effects.md

Usage
-----
uv run scripts/analyze_phenotype_effects.py \
  --phenotype-deltas results/aggregates_re/phenotype_deltas_vs_clean.csv \
  --outdir results/aggregates_re/findings

Optional:
  --include-all
  --min-n 1
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Small helpers
# ----------------------------


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        header = list(r.fieldnames or [])
        rows = [dict(row) for row in r]
    return header, rows


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _std(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return 0.0 if xs else None
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return var**0.5


# ----------------------------
# Core analysis
# ----------------------------

DEFAULT_METRICS = [
    "delta__macro_avg",
    "delta__yesno_acc",
    "delta__factoid_f1",
    "delta__factoid_em",
    "delta__list_f1",
    "delta__summary_rougeL",
]


@dataclass
class SliceKey:
    dataset: str
    runtime: str
    model: str
    phenotype: str
    perturbation: str


def analyze(
    rows: List[Dict[str, Any]], metrics: List[str], include_all: bool, min_n: int
) -> Dict[str, Any]:
    # Normalize + filter
    normed: List[Dict[str, Any]] = []
    for r in rows:
        phenotype = _safe_str(r.get("phenotype")).strip()
        if (not include_all) and phenotype.upper() == "ALL":
            continue

        n_gold = _to_int(r.get("n_gold"))
        if n_gold is not None and n_gold < min_n:
            continue

        rr = dict(r)
        rr["dataset"] = _safe_str(r.get("dataset")).strip()
        rr["runtime"] = _safe_str(r.get("runtime")).strip()
        rr["model"] = _safe_str(r.get("model")).strip()
        rr["phenotype"] = phenotype
        rr["perturbation"] = _safe_str(r.get("perturbation")).strip()

        # numericize metrics
        for m in metrics:
            rr[m] = _to_float(r.get(m))

        # also numericize seed for possible grouping/diagnostics
        rr["seed"] = _to_int(r.get("seed"))

        normed.append(rr)

    # Group: (dataset, runtime, model, phenotype, perturbation)
    buckets: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for r in normed:
        key = (r["dataset"], r["runtime"], r["model"], r["phenotype"], r["perturbation"])
        buckets.setdefault(key, []).append(r)

    # Summaries per bucket
    per_bucket: List[Dict[str, Any]] = []
    for (dataset, runtime, model, phenotype, perturbation), rs in sorted(buckets.items()):
        out: Dict[str, Any] = {
            "dataset": dataset,
            "runtime": runtime,
            "model": model,
            "phenotype": phenotype,
            "perturbation": perturbation,
            "n_rows": len(rs),
            "seeds": sorted({r["seed"] for r in rs if r.get("seed") is not None}),
        }

        for m in metrics:
            vals = [r[m] for r in rs if isinstance(r.get(m), float)]
            vals = [v for v in vals if v is not None]
            out[m] = _mean(vals)
            out[m.replace("delta__", "std__")] = _std(vals)
            out[m.replace("delta__", "n__")] = len(vals)

        per_bucket.append(out)

    # For each phenotype, rank perturbations by worst mean delta (more negative = worse)
    by_pheno: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for b in per_bucket:
        k = (b["dataset"], b["runtime"], b["model"], b["phenotype"])
        by_pheno.setdefault(k, []).append(b)

    phenotype_rankings: List[Dict[str, Any]] = []
    for (dataset, runtime, model, phenotype), bs in sorted(by_pheno.items()):
        entry: Dict[str, Any] = {
            "dataset": dataset,
            "runtime": runtime,
            "model": model,
            "phenotype": phenotype,
            "worst_by_metric": {},
        }
        for m in metrics:
            # sort by mean delta ascending (most negative first); None goes last
            def keyfn(x: Dict[str, Any]) -> float:
                v = x.get(m)
                return v if isinstance(v, float) else 1e9

            ranked = sorted(bs, key=keyfn)
            top = ranked[:5]
            entry["worst_by_metric"][m] = [
                {
                    "perturbation": t["perturbation"],
                    "mean": t.get(m),
                    "std": t.get(m.replace("delta__", "std__")),
                    "n": t.get(m.replace("delta__", "n__")),
                    "seeds": t.get("seeds", []),
                }
                for t in top
                if t.get(m) is not None
            ]
        phenotype_rankings.append(entry)

    return {
        "n_input_rows": len(rows),
        "n_used_rows": len(normed),
        "n_buckets": len(per_bucket),
        "metrics": metrics,
        "per_bucket": per_bucket,
        "phenotype_rankings": phenotype_rankings,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Phenotype Effects Summary")
    lines.append("")
    lines.append(f"- Rows (input): {report.get('n_input_rows')}")
    lines.append(f"- Rows (used): {report.get('n_used_rows')}")
    lines.append(f"- Buckets: {report.get('n_buckets')}")
    lines.append("")
    lines.append("## Worst perturbations by phenotype")
    lines.append("")
    metrics = report.get("metrics", [])
    for ph in report.get("phenotype_rankings", []):
        lines.append(f"### {ph['phenotype']}")
        lines.append(
            f"- dataset: `{ph['dataset']}` | runtime: `{ph['runtime']}` | model: `{ph['model']}`"
        )
        lines.append("")
        for m in metrics:
            worst = ph["worst_by_metric"].get(m, [])
            if not worst:
                continue
            lines.append(f"**{m}**")
            for w in worst:
                lines.append(
                    f"- {w['perturbation']}: mean={w['mean']:.4f} std={w['std']:.4f} n={w['n']} seeds={w['seeds']}"
                    if (isinstance(w.get("mean"), float) and isinstance(w.get("std"), float))
                    else f"- {w['perturbation']}: mean={w.get('mean')} std={w.get('std')} n={w.get('n')} seeds={w.get('seeds')}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ----------------------------
# CLI
# ----------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenotype-deltas", required=True, help="phenotype_deltas_vs_clean.csv")
    ap.add_argument("--outdir", default="results/aggregates_re/findings")
    ap.add_argument(
        "--include-all", action="store_true", help="Include phenotype='ALL' in rankings"
    )
    ap.add_argument("--min-n", type=int, default=1, help="Minimum n_gold slice size to include")
    ap.add_argument(
        "--metrics", nargs="*", default=None, help="Override metric columns (delta__...)"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.phenotype_deltas)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise SystemExit(f"[error] file not found: {in_path}")

    header, rows = _read_csv(in_path)

    metrics = args.metrics if args.metrics else DEFAULT_METRICS
    # keep only metrics that actually exist in the file
    metrics = [m for m in metrics if m in header]
    if not metrics:
        raise SystemExit(f"[error] no requested delta metrics exist in CSV header. header={header}")

    report = analyze(
        rows=rows,
        metrics=metrics,
        include_all=bool(args.include_all),
        min_n=int(args.min_n),
    )

    out_json = outdir / "phenotype_effects.json"
    out_md = outdir / "phenotype_effects.md"

    _write_json(out_json, report)
    _write_text(out_md, render_markdown(report))

    print(
        f"[phenotype-effects] rows_in={report['n_input_rows']} rows_used={report['n_used_rows']} buckets={report['n_buckets']}"
    )
    print(f"  - {out_json}")
    print(f"  - {out_md}")


if __name__ == "__main__":
    main()
