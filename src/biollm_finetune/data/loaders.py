#!/usr/bin/env python3
"""
Lightweight loaders and utilities for BioASQ-style datasets.

Faithful to dataset_stats.py:
- Robust JSON / JSONL readers
- Normalize BioASQ "questions" JSON → list[dict]
- Simple flattening helpers (pull out id/type/body/exact_answer/ideal_answer/snippets)
- Tiny stats (counts per type, missing fields)

Only necessary alterations:
- No hardcoded paths
- Functions are pure/reusable, optional CLI for quick inspection
"""

from __future__ import annotations
import argparse
import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


# ---------------------------
# I/O helpers
# ---------------------------

def load_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_questions_any(path: str | Path) -> List[Dict[str, Any]]:
    """
    Accept:
      - JSON with {"questions":[...]}
      - JSON list [...]
      - JSONL (one item per line)
    Return: list of question dicts.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    # Try JSON first
    try:
        obj = load_json(p)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "questions" in obj and isinstance(obj["questions"], list):
                return obj["questions"]
            if "data" in obj and isinstance(obj["data"], list):
                return obj["data"]
    except json.JSONDecodeError:
        pass

    # Fallback to JSONL
    return load_jsonl(p)


# ---------------------------
# Normalization / flattening
# ---------------------------

def canonical_id(rec: Mapping[str, Any]) -> str:
    """Prefer existing 'id'; otherwise stable hash of (type, body/question)."""
    if "id" in rec and rec["id"] is not None:
        return str(rec["id"])
    qtype = (rec.get("type") or "").lower().strip()
    question = (rec.get("body") or rec.get("question") or "").strip()
    raw = f"{qtype}::{question}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _snippets_to_texts(snips: Any) -> List[str]:
    if not snips:
        return []
    out: List[str] = []
    if isinstance(snips, list):
        for s in snips:
            if isinstance(s, dict):
                t = s.get("text")
                if t:
                    out.append(str(t))
            elif isinstance(s, str):
                out.append(s)
    return out


def flatten_bioasq_item(rec: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Produce a compact, consistent record:
      id, type, body, exact_answer (as-is), ideal_answer (string),
      snippets_text (list[str]) if present.
    """
    qid = canonical_id(rec)
    qtype = (rec.get("type") or "").lower().strip() or "factoid"
    body = (rec.get("body") or rec.get("question") or "").strip()
    exact = rec.get("exact_answer", None)

    ideal = rec.get("ideal_answer", "")
    if isinstance(ideal, list):
        ideal_str = " ".join(str(x) for x in ideal)
    else:
        ideal_str = str(ideal) if ideal is not None else ""

    snips = rec.get("snippets") or rec.get("snippets_text") or []
    snippets_text = _snippets_to_texts(snips)

    return {
        "id": qid,
        "type": qtype,
        "body": body,
        "exact_answer": exact,
        "ideal_answer": ideal_str,
        "snippets_text": snippets_text,
    }


def flatten_bioasq(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [flatten_bioasq_item(r) for r in rows]


# ---------------------------
# Tiny stats (like dataset_stats.py)
# ---------------------------

def basic_stats(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    by_type = Counter((r.get("type") or "").lower().strip() for r in rows)
    missing_body = sum(1 for r in rows if not (r.get("body") or r.get("question")))
    missing_answers = sum(1 for r in rows if (r.get("exact_answer") is None and r.get("ideal_answer") in (None, "", [])))
    with_snips = sum(1 for r in rows if (r.get("snippets") or r.get("snippets_text")))
    return {
        "count": n,
        "by_type": dict(by_type),
        "missing_body": missing_body,
        "missing_answers": missing_answers,
        "with_snippets": with_snips,
    }


# ---------------------------
# CLI (optional quick inspection)
# ---------------------------

def _write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect/flatten BioASQ-style data")
    ap.add_argument("--input", required=True, help="JSON/JSONL or BioASQ JSON with {'questions':[...]}")
    ap.add_argument("--flatten_out", help="Optional path to write flattened JSON")
    ap.add_argument("--stats_out", help="Optional path to write basic stats JSON")
    args = ap.parse_args()

    rows = load_questions_any(args.input)
    flat = flatten_bioasq(rows)
    stats = basic_stats(rows)

    print(f"[load] {args.input}  items={len(rows)}  flat={len(flat)}")
    print(f"[stats] {json.dumps(stats, indent=2)}")

    if args.flatten_out:
        _write_json(args.flatten_out, flat)
        print(f"[save] flattened → {Path(args.flatten_out).resolve()}")
    if args.stats_out:
        _write_json(args.stats_out, stats)
        print(f"[save] stats → {Path(args.stats_out).resolve()}")


if __name__ == "__main__":
    main()