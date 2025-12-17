#!/usr/bin/env python3
"""
Analyze Phase 4 robustness results (no new experiments).

Inputs (typical):
  - results/phase4/experiments.csv
  - results/phase4/deltas/deltas_long.csv
  - results/phase4/deltas/deltas_summary.json

Outputs:
  - results/phase4/analysis/phase4_findings.json
  - results/phase4/analysis/phase4_findings.md

This script is intentionally schema-tolerant:
- does NOT require a 'runtime' column
- identifies clean baselines by perturbation=='clean'
- summarizes deltas vs clean by metric, perturbation, and seed
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


METRICS = ["macro_avg", "yesno_acc", "factoid_f1", "list_f1", "summary_rougeL"]


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    # group by dataset+model+seed (no runtime in your CSV)
    return (
        str(row.get("dataset", "")).strip(),
        str(row.get("model", "")).strip(),
        str(row.get("seed", "")).strip(),
    )


def summarize_experiments(experiments_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Build clean baseline per (dataset, model, seed)
    clean_by = {}
    runs_by_pert = defaultdict(list)

    for r in experiments_rows:
        pert = str(r.get("perturbation", "")).strip()
        runs_by_pert[pert].append(r)
        if pert == "clean":
            clean_by[_key(r)] = r

    # Compute deltas vs clean directly from experiments.csv (fallback if deltas are missing)
    deltas: List[Dict[str, Any]] = []
    missing_baseline = 0

    for r in experiments_rows:
        pert = str(r.get("perturbation", "")).strip()
        if pert == "clean":
            continue
        base = clean_by.get(_key(r))
        if not base:
            missing_baseline += 1
            continue

        entry = {
            "experiment": r.get("experiment"),
            "dataset": r.get("dataset"),
            "model": r.get("model"),
            "seed": int(r.get("seed")) if str(r.get("seed", "")).isdigit() else r.get("seed"),
            "perturbation": pert,
            "deltas": {},
        }

        for m in METRICS:
            pv = _to_float(r.get(m))
            bv = _to_float(base.get(m))
            if pv is None or bv is None:
                continue
            entry["deltas"][m] = pv - bv

        deltas.append(entry)

    # Aggregate by perturbation (mean across seeds)
    by_pert: Dict[str, Dict[str, Any]] = {}
    for d in deltas:
        p = d["perturbation"]
        by_pert.setdefault(p, {"n": 0, "by_metric": defaultdict(list)})
        by_pert[p]["n"] += 1
        for m, dv in d["deltas"].items():
            by_pert[p]["by_metric"][m].append(dv)

    for p, agg in by_pert.items():
        out = {"n": agg["n"], "mean_delta": {}, "min_delta": {}, "max_delta": {}}
        for m, xs in agg["by_metric"].items():
            xs2 = [x for x in xs if isinstance(x, (int, float))]
            if not xs2:
                continue
            out["mean_delta"][m] = sum(xs2) / len(xs2)
            out["min_delta"][m] = min(xs2)
            out["max_delta"][m] = max(xs2)
        by_pert[p] = out

    # Rank perturbations by macro drop (most negative first)
    ranked = []
    for p, agg in by_pert.items():
        mu = agg["mean_delta"].get("macro_avg")
        if mu is None:
            continue
        ranked.append((p, mu))
    ranked.sort(key=lambda t: t[1])

    return {
        "n_runs": len(experiments_rows),
        "n_clean": len(runs_by_pert.get("clean", [])),
        "n_non_clean": len(experiments_rows) - len(runs_by_pert.get("clean", [])),
        "missing_baselines": missing_baseline,
        "per_perturbation": by_pert,
        "ranked_by_macro_drop": ranked,
        "raw_deltas": deltas,
    }


def render_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Phase 4 Findings (Auto-Generated)")
    lines.append("")
    lines.append(f"- Runs: **{summary['n_runs']}**")
    lines.append(f"- Clean runs: **{summary['n_clean']}**")
    lines.append(f"- Missing baselines: **{summary['missing_baselines']}**")
    lines.append("")
    lines.append("## Perturbations Ranked by Macro Δ vs Clean (lower = worse)")
    lines.append("")
    lines.append("| Rank | Perturbation | Mean Macro Δ |")
    lines.append("|---:|---|---:|")
    for i, (p, mu) in enumerate(summary["ranked_by_macro_drop"], start=1):
        lines.append(f"| {i} | `{p}` | {mu:.4f} |")
    lines.append("")
    lines.append("## Mean Δ vs Clean by Metric (per perturbation)")
    lines.append("")
    lines.append("| Perturbation | macro_avg | yesno_acc | factoid_f1 | list_f1 | summary_rougeL | n |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for p, agg in sorted(summary["per_perturbation"].items()):
        md = agg.get("mean_delta", {})
        def g(k: str) -> str:
            v = md.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else ""
        lines.append(
            f"| `{p}` | {g('macro_avg')} | {g('yesno_acc')} | {g('factoid_f1')} | {g('list_f1')} | {g('summary_rougeL')} | {agg.get('n',0)} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-csv", default="results/phase4/experiments.csv")
    ap.add_argument("--out-dir", default="results/phase4/analysis")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    exp_path = Path(args.experiments_csv)
    out_dir = Path(args.out_dir)

    rows = read_csv(exp_path)
    summary = summarize_experiments(rows)

    write_json(out_dir / "phase4_findings.json", summary)
    write_text(out_dir / "phase4_findings.md", render_markdown(summary))

    print(f"[ok] wrote → {out_dir}/phase4_findings.json and .md")


if __name__ == "__main__":
    main()