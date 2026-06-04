#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNTIME_FIELDS = [
    "experiment",
    "dataset",
    "model",
    "model_id",
    "adapter_path",
    "seed",
    "perturbation",
    "runtime",
    "requested_device",
    "requested_dtype",
    "resolved_device",
    "resolved_dtype",
    "inference_manifest",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize resolved runtime state from experiments.csv."
    )
    parser.add_argument(
        "--experiments-csv",
        default="results/phase4/experiments.csv",
        help="Aggregated experiment table.",
    )
    parser.add_argument(
        "--out",
        default="results/phase4/runtime/runtime_summary.json",
        help="Runtime summary output.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _key(row: dict[str, str], *fields: str) -> str:
    return " | ".join((row.get(field) or "").strip() or "unrecorded" for field in fields)


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.experiments_csv))
    runtime_rows: list[dict[str, Any]] = [
        {field: row.get(field, "") for field in RUNTIME_FIELDS} for row in rows
    ]

    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiments_csv": args.experiments_csv,
        "n_rows": len(rows),
        "resolved_runtime_counts": dict(
            Counter(_key(row, "resolved_device", "resolved_dtype") for row in rows)
        ),
        "requested_vs_resolved_counts": dict(
            Counter(
                _key(
                    row,
                    "requested_device",
                    "requested_dtype",
                    "resolved_device",
                    "resolved_dtype",
                )
                for row in rows
            )
        ),
        "adapter_rows": [row for row in runtime_rows if (row.get("adapter_path") or "").strip()],
        "rows": runtime_rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[runtime] wrote {out}")


if __name__ == "__main__":
    main()
