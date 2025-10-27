#!/usr/bin/env python3
"""
Balanced sampler for BioASQ-style QA data.

Examples
--------
# Sample ~5 per type (≈20 total) and write JSONL + gold JSON
python -m bioasq_llm.data.data_sampling \
  --inputs data/BioASQ-train.json \
  --out-questions data/samples/sample_questions.jsonl \
  --out-gold data/samples/sample_gold.json \
  --per-type 5 --seed 42

# Sample exactly 24 examples, auto-balanced across types
python -m bioasq_llm.data.data_sampling \
  --inputs data/BioASQ-train.json \
  --out-questions data/samples/sample24.jsonl \
  --out-gold data/samples/sample24_gold.json \
  --total 24 --seed 123
"""

from __future__ import annotations
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .loaders import load_questions_any, canonical_id


TYPES_DEFAULT = ("yesno", "factoid", "list", "summary")


def _normalize_type(t: Optional[str]) -> str:
    if not t:
        return "unknown"
    t = t.lower().strip()
    if t in {"yes/no", "yn"}:
        return "yesno"
    if t in {"fact"}:
        return "factoid"
    if t in {"lists"}:
        return "list"
    if t in {"summ"}:
        return "summary"
    return t


def _merge_inputs(inputs: List[str | Path]) -> List[Dict[str, Any]]:
    """Load and de-duplicate records across multiple sources."""
    merged: Dict[str, Dict[str, Any]] = {}
    for path in inputs:
        rows = load_questions_any(path)
        for r in rows:
            cid = canonical_id(r)
            merged[cid] = dict(r)  # last write wins (simplest policy)
    return list(merged.values())


def _bucket_by_type(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[_normalize_type(r.get("type"))].append(dict(r))
    return buckets


def _compute_counts_for_total(
    buckets: Dict[str, List[Dict[str, Any]]],
    total: int,
    types_order: Iterable[str],
) -> Dict[str, int]:
    """Proportionally allocate a total across available types (with floor + remainder round-robin)."""
    sizes = {t: len(buckets.get(t, ())) for t in types_order}
    avail = sum(sizes.values())
    if avail == 0:
        return {t: 0 for t in types_order}

    # proportional floors
    counts = {t: min(len(buckets.get(t, ())), math.floor(total * (sizes[t] / avail))) for t in types_order}

    # distribute remainder by round-robin over types with remaining availability
    assigned = sum(counts.values())
    remainder = max(0, total - assigned)
    if remainder:
        # order by (remaining capacity desc) then name for determinism
        def rem_cap(t): return len(buckets.get(t, ())) - counts[t]
        round_robin = [t for t in types_order if rem_cap(t) > 0]
        i = 0
        while remainder > 0 and round_robin:
            t = round_robin[i % len(round_robin)]
            if rem_cap(t) > 0:
                counts[t] += 1
                remainder -= 1
            i += 1
            # refresh list occasionally to avoid spinning
            round_robin = [tt for tt in types_order if (len(buckets.get(tt, ())) - counts[tt]) > 0]
    return counts


def _sample_balanced(
    buckets: Dict[str, List[Dict[str, Any]]],
    per_type: Optional[int],
    total: Optional[int],
    types_order: Iterable[str],
    seed: int,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    selected: List[Dict[str, Any]] = []

    if per_type is not None:
        for t in types_order:
            pool = buckets.get(t, [])
            k = min(per_type, len(pool))
            selected.extend(random.sample(pool, k=k))
        return selected

    # else, total is set
    counts = _compute_counts_for_total(buckets, total or 0, types_order)
    for t in types_order:
        pool = buckets.get(t, [])
        k = min(counts.get(t, 0), len(pool))
        if k > 0:
            selected.extend(random.sample(pool, k=k))
    return selected


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def _write_gold_from_sample(path: str | Path, sample: Iterable[Mapping[str, Any]]) -> Path:
    """
    Write a compact BioASQ-style gold JSON:
    keep id, type, body/question, exact_answer, ideal_answer.
    """
    keep_keys = {"id", "type", "body", "question", "exact_answer", "ideal_answer"}
    out_questions: List[Dict[str, Any]] = []
    for r in sample:
        g: Dict[str, Any] = {}
        for k in keep_keys:
            if k in r:
                g[k] = r[k]
        # prefer 'body' over 'question'
        if "body" not in g and "question" in g:
            g["body"] = g.pop("question")
        out_questions.append(g)

    obj = {"questions": out_questions}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return p


def run_sampling(
    inputs: List[str | Path],
    out_questions: str | Path,
    out_gold: Optional[str | Path] = None,
    per_type: Optional[int] = 5,
    total: Optional[int] = None,
    types: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[Path, Optional[Path], Dict[str, int]]:
    """
    Returns (questions_jsonl_path, gold_json_path_or_None, counts_by_type)
    """
    rows = _merge_inputs(inputs)
    buckets = _bucket_by_type(rows)

    types_order = [t.lower() for t in (types or TYPES_DEFAULT)]
    # filter to requested types only
    buckets = {t: buckets.get(t, []) for t in types_order}

    sample = _sample_balanced(
        buckets=buckets,
        per_type=per_type if total is None else None,
        total=total,
        types_order=types_order,
        seed=seed,
    )

    # Count summary
    cts = defaultdict(int)
    for r in sample:
        cts[_normalize_type(r.get("type"))] += 1
    counts = {t: cts.get(t, 0) for t in types_order}

    qpath = _write_jsonl(out_questions, sample)
    gpath = _write_gold_from_sample(out_gold, sample) if out_gold else None
    return qpath, gpath, counts


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sample a balanced subset of BioASQ-style data.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input files (JSON, {'questions':[...]}, or JSONL).")
    ap.add_argument("--out-questions", required=True, help="Output path for questions JSONL.")
    ap.add_argument("--out-gold", help="Optional output path for a matching gold JSON.")
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument("--per-type", type=int, default=5, help="Number of items per type (default: 5).")
    group.add_argument("--total", type=int, help="Total items across all types (auto-balanced).")
    ap.add_argument("--types", nargs="+", default=list(TYPES_DEFAULT), help="Types to include (default: all).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    qpath, gpath, counts = run_sampling(
        inputs=args.inputs,
        out_questions=args.out_questions,
        out_gold=args.out_gold,
        per_type=args.per_type if args.total is None else None,
        total=args.total,
        types=args.types,
        seed=args.seed,
    )
    print(f"✅ wrote questions → {Path(qpath).resolve()}")
    if gpath:
        print(f"✅ wrote gold      → {Path(gpath).resolve()}")
    print("🧮 counts by type:", counts)


if __name__ == "__main__":
    main()