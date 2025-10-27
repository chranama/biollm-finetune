#!/usr/bin/env python3
"""
Preprocessing pipeline to build training JSON/JSONL from multiple BioASQ-style sources.

Faithful to preprocess_data_files.py:
- Merge multiple source files
- Normalize schema and (optionally) filter/clean
- Write consolidated JSONL or JSON

Only necessary alterations:
- Pure functions + small CLI
- No hardcoded paths
- Uses loaders.py for robust I/O and normalization
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .loaders import load_questions_any, flatten_bioasq, canonical_id


# ---------------------------
# Merge / deduplicate
# ---------------------------

def merge_sources(
    inputs: List[str | Path],
    prefer: str = "last",
) -> List[Dict[str, Any]]:
    """
    Load and merge multiple inputs, de-duplicating by canonical id.
    prefer:
      - "last": later files overwrite earlier ones (default)
      - "first": keep the first seen record
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for path in inputs:
        rows = load_questions_any(path)
        for r in rows:
            cid = canonical_id(r)
            if cid in merged:
                if prefer == "last":
                    merged[cid] = dict(r)
                elif prefer == "first":
                    pass
                else:
                    merged[cid] = dict(r)
            else:
                merged[cid] = dict(r)
    return list(merged.values())


# ---------------------------
# Optional cleaning / filtering
# ---------------------------

def filter_not_usable(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """
    Example filter similar to what you likely had in the original:
    - drop records with missing question text
    - drop empty exact/ideal answers when both are absent (for supervised FT)
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        body = (r.get("body") or r.get("question") or "").strip()
        if not body:
            continue
        exact = r.get("exact_answer", None)
        ideal = r.get("ideal_answer", None)
        if exact in (None, "", []) and ideal in (None, "", []):
            # keep if you're doing instruction tuning without targets?
            # choose policy; default: drop
            continue
        out.append(dict(r))
    return out


def add_snippet_text_field(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure a 'snippets_text' list[str] field exists after flattening.
    (If you already call flatten_bioasq later, this is redundant.)
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        if "snippets_text" not in rr:
            sn = rr.get("snippets") or rr.get("snippets_text") or []
            if isinstance(sn, list):
                txts: List[str] = []
                for s in sn:
                    if isinstance(s, dict) and "text" in s:
                        txts.append(str(s["text"]))
                    elif isinstance(s, str):
                        txts.append(s)
                rr["snippets_text"] = txts
            else:
                rr["snippets_text"] = []
        out.append(rr)
    return out


# ---------------------------
# Write helpers
# ---------------------------

def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------
# Pipeline
# ---------------------------

def preprocess(
    inputs: List[str | Path],
    out_path: str | Path,
    out_format: str = "jsonl",
    drop_unusable: bool = True,
    flatten: bool = True,
    prefer: str = "last",
) -> Path:
    """
    End-to-end:
      inputs → merge → (optional filter) → (optional flatten) → write (json/jsonl)
    """
    merged = merge_sources(inputs, prefer=prefer)
    rows = merged

    if drop_unusable:
        rows = filter_not_usable(rows)

    if flatten:
        rows = flatten_bioasq(rows)

    out_path = Path(out_path)
    if out_format == "jsonl":
        write_jsonl(out_path, rows)
    else:
        write_json(out_path, rows)

    return out_path


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Preprocess BioASQ-style sources into a unified file.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input files (JSON/JSONL or BioASQ JSONs).")
    ap.add_argument("--out", required=True, help="Output file path.")
    ap.add_argument("--format", choices=["jsonl", "json"], default="jsonl", help="Output format (default: jsonl).")
    ap.add_argument("--keep-unlabeled", action="store_true", help="Keep items with no exact/ideal answer.")
    ap.add_argument("--no-flatten", action="store_true", help="Do not flatten to compact schema.")
    ap.add_argument("--prefer", choices=["last", "first"], default="last", help="Duplicate id policy.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = preprocess(
        inputs=args.inputs,
        out_path=args.out,
        out_format=args.format,
        drop_unusable=not args.keep_unlabeled,
        flatten=not args.no_flatten,
        prefer=args.prefer,
    )
    print(f"✅ wrote {out.resolve()}")


if __name__ == "__main__":
    main()