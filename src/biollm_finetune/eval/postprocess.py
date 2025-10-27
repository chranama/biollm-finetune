#!/usr/bin/env python3
"""
Post-processing utilities for generated BioASQ answers.

Faithful to the original postprocessing.py with necessary improvements:
- JSON and JSONL (NDJSON) input support
- Merge multiple sources with de-duplication
- Categorize by question type
- Save per-type outputs
- Simple CLI (no hardcoded filenames)
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


# ---------------------------
# I/O helpers
# ---------------------------

def load_data(file_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load a file that may be:
      - JSON list
      - JSON object with {"questions": [...]}
      - JSONL (one object per line)
    Returns a list[dict].
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")

    # Try JSON first
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "questions" in obj and isinstance(obj["questions"], list):
                return obj["questions"]
            # If it's a dict of results, try extracting a plausible list
            if "data" in obj and isinstance(obj["data"], list):
                return obj["data"]
            # Else, not a supported structure → fall through to JSONL attempt
    except json.JSONDecodeError:
        pass

    # Try NDJSON (JSON Lines)
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, rows: List[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------
# Merge & categorize
# ---------------------------

def _canonical_id(rec: Mapping[str, Any], id_key: str = "id") -> str:
    """
    Prefer a provided id; otherwise build one from question-type+body hash so we can de-dup.
    """
    if id_key in rec and rec[id_key] is not None:
        return str(rec[id_key])

    qtype = (rec.get("type") or "").lower().strip()
    question = (rec.get("body") or rec.get("question") or "").strip()
    raw = f"{qtype}::{question}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def merge_data(files: List[str | Path], id_key: str = "id", prefer: str = "last") -> Dict[str, Dict[str, Any]]:
    """
    Merge multiple input files into a dict keyed by canonical id.

    prefer:
      - "last": later files overwrite earlier ones on id collision (default)
      - "first": keep the first occurrence and ignore later duplicates
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for fp in files:
        rows = load_data(fp)
        for rec in rows:
            cid = _canonical_id(rec, id_key=id_key)
            if cid in merged:
                if prefer == "last":
                    merged[cid] = dict(rec)
                elif prefer == "first":
                    pass
                else:
                    # default to last if unknown policy
                    merged[cid] = dict(rec)
            else:
                merged[cid] = dict(rec)
    return merged


def normalize_type(t: Optional[str]) -> str:
    if not t:
        return "unknown"
    t = t.lower().strip()
    if t in {"yesno", "yes/no", "yn"}:
        return "yesno"
    if t in {"factoid", "fact"}:
        return "factoid"
    if t in {"list", "lists"}:
        return "list"
    if t in {"summary", "summ"}:
        return "summary"
    return t


def categorize_data(merged: Mapping[str, Mapping[str, Any]], type_key: str = "type") -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {"yesno": [], "factoid": [], "list": [], "summary": [], "unknown": []}
    for _cid, rec in merged.items():
        t = normalize_type(rec.get(type_key))
        if t not in buckets:
            buckets[t] = []
        buckets[t].append(dict(rec))
    return buckets


# ---------------------------
# Saving
# ---------------------------

def save_data_by_category(
    categorized: Mapping[str, List[Mapping[str, Any]]],
    out_dir: str | Path,
    fmt: str = "jsonl",
) -> Dict[str, Path]:
    """
    Save each category to a file in out_dir with the chosen format (jsonl/json).
    Returns a dict of category -> path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths: Dict[str, Path] = {}
    for qtype, entries in categorized.items():
        if not entries:
            continue
        out_path = out_dir / f"{qtype}.{fmt}"
        if fmt == "jsonl":
            write_jsonl(out_path, entries)
        else:
            write_json(out_path, entries)
        out_paths[qtype] = out_path
        print(f"✅ Saved {len(entries):5d} “{qtype}” records → {out_path}")
    return out_paths


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge, categorize, and save generated BioASQ answers.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=False,
        default=[
            "bioasq_generated_answers.json",
            "goterms_generated_answers.json",
            "drugbank_generated_answers.json",
            "biobiqa_generated_answers.json",
        ],
        help="Input files (JSON/JSONL). Defaults match the original script’s filenames.",
    )
    ap.add_argument("--outdir", default="results/processed", help="Directory to write per-type outputs.")
    ap.add_argument("--format", choices=["jsonl", "json"], default="jsonl", help="Output format per category.")
    ap.add_argument("--id_key", default="id", help="Key to use as unique id (fallback: hashed question).")
    ap.add_argument(
        "--prefer",
        choices=["last", "first"],
        default="last",
        help="On duplicate ids, keep the 'last' (default) or the 'first' seen.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"📥 Inputs: {len(args.inputs)} file(s)")
    merged = merge_data(args.inputs, id_key=args.id_key, prefer=args.prefer)
    print(f"🔗 Merged: {len(merged)} unique records")

    categorized = categorize_data(merged)
    total = sum(len(v) for v in categorized.values())
    print(
        "🗂️  Buckets:" +
        "".join([f"  {k}={len(v)}" for k, v in categorized.items() if len(v) > 0])
    )

    paths = save_data_by_category(categorized, out_dir=args.outdir, fmt=args.format)
    print(f"✅ Done. Wrote {len(paths)} files to: {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()