#!/usr/bin/env python3
"""
Aggregate Phase 4 experiment results.

Reads all experiment directories under results/experiments/,
extracts metrics + metadata, and writes a flat table suitable
for statistical analysis, plotting, or paper tables.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

EXPERIMENT_ROOT = Path("results/experiments")
OUT_DIR = Path("results/phase4")
OUT_CSV = OUT_DIR / "experiments.csv"
OUT_JSON = OUT_DIR / "summary.json"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_row(exp_dir: Path) -> Dict[str, Any]:
    manifest = load_json(exp_dir / "manifest.json")
    metrics = load_json(exp_dir / "metrics.json")
    inference_manifest_path = exp_dir / "inference_manifest.json"
    inference_manifest = (
        load_json(inference_manifest_path) if inference_manifest_path.exists() else {}
    )

    row = {
        "experiment": manifest["experiment"],
        "dataset": manifest.get("dataset"),
        "model": manifest.get("model"),
        "model_id": manifest.get("model_id") or inference_manifest.get("model_id"),
        "adapter_path": manifest.get("adapter_path"),
        "seed": manifest.get("seed"),
        "perturbation": manifest.get("perturbation"),
        "runtime": manifest.get("runtime"),
        "requested_device": manifest.get("requested_device"),
        "requested_dtype": manifest.get("requested_dtype"),
        "resolved_device": manifest.get("resolved_device") or inference_manifest.get("device"),
        "resolved_dtype": manifest.get("resolved_dtype") or inference_manifest.get("dtype"),
        "inference_manifest": manifest.get("inference_manifest")
        or (str(inference_manifest_path) if inference_manifest_path.exists() else ""),
        "start_time": manifest.get("start_time_utc"),
        "end_time": manifest.get("end_time_utc"),
    }

    # Core metrics (safe defaults)
    row.update(
        {
            "macro_avg": metrics.get("macro_avg"),
            "yesno_acc": metrics.get("yesno", {}).get("accuracy"),
            "factoid_f1": metrics.get("factoid", {}).get("f1"),
            "list_f1": metrics.get("list", {}).get("f1"),
            "summary_rougeL": metrics.get("summary", {}).get("rougeL"),
        }
    )

    counts = metrics.get("counts", {})
    for k, v in counts.items():
        row[f"count_{k}"] = v

    return row


def main() -> None:
    rows: List[Dict[str, Any]] = []

    for exp_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not exp_dir.is_dir():
            continue
        if not (exp_dir / "manifest.json").exists():
            continue
        if not (exp_dir / "metrics.json").exists():
            continue

        rows.append(extract_row(exp_dir))

    if not rows:
        raise SystemExit("No experiments found to aggregate.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write CSV. Use a union header so older runs without newer metadata do not
    # prevent newer runtime or adapter fields from being written.
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Write JSON summary (grouped by perturbation)
    summary: Dict[str, Any] = {}
    for r in rows:
        p = r["perturbation"]
        summary.setdefault(p, []).append(r)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] aggregated {len(rows)} experiments")
    print(f"[csv]  {OUT_CSV.resolve()}")
    print(f"[json] {OUT_JSON.resolve()}")


if __name__ == "__main__":
    main()
