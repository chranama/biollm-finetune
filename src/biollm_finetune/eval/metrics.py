#!/usr/bin/env python3
"""
Compute metrics for BioASQ-style QA predictions.

Inputs:
- Predictions: JSONL or JSON (list or {"questions":[...]})
  Expected fields per record: {id?, type, question/body?, predicted}
- Gold: JSONL or BioASQ JSON {"questions":[...]}
  Expected gold fields (depending on type):
    yesno:      exact_answer: "yes" | "no"
    factoid:    exact_answer: str | [str] | [[alias1, alias2], ...]
    list:       exact_answer: [str, str, ...]
    summary:    ideal_answer: str | [str, str, ...]

Outputs:
- metrics.json with per-type aggregates and macro averages.

Notes:
- Robust to minor schema variation
- If 'id' missing in either side, falls back to a stable hash of (type, question)
"""

from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# Optional ROUGE (install: pip install rouge-score)
try:
    from rouge_score import rouge_scorer
    _ROUGE_OK = True
except Exception:
    _ROUGE_OK = False


# ---------------------------
# I/O helpers
# ---------------------------

def _read_json_any(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    # Try JSON
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "questions" in obj and isinstance(obj["questions"], list):
                return obj["questions"]
            if "data" in obj and isinstance(obj["data"], list):
                return obj["data"]
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------
# Canonicalization & matching
# ---------------------------

import hashlib

def _canonical_id(rec: Mapping[str, Any]) -> str:
    if "id" in rec and rec["id"] is not None:
        return str(rec["id"])
    qtype = (rec.get("type") or "").lower().strip()
    question = (rec.get("body") or rec.get("question") or "").strip()
    raw = f"{qtype}::{question}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _norm_text(s: str) -> str:
    """
    Light normalization for EM/F1:
    - lowercase
    - strip
    - collapse whitespace
    - strip surrounding quotes and trailing punctuation
    """
    s = s.lower().strip()
    # remove leading 'answer:' or similar prompt artifacts
    s = re.sub(r'^\s*(answer|final answer)\s*[:\-]\s*', '', s)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s)
    # strip trivial surrounding quotes
    s = s.strip(' "\'')
    # remove trailing punctuation like periods/commas
    s = re.sub(r'[.,;:]+$', '', s)
    return s


def _norm_item_list(xs: Iterable[str]) -> List[str]:
    out: List[str] = []
    for x in xs:
        if x is None:
            continue
        s = _norm_text(str(x))
        if s:
            out.append(s)
    return out


def _flatten_exact_answer(ans: Any) -> List[str]:
    """
    BioASQ exact_answer may be:
      - string
      - [alias1, alias2, ...]
      - [[alias1, alias2], [alias3, alias4], ...]
    Return a flat list of strings (aliases).
    """
    if ans is None:
        return []
    if isinstance(ans, str):
        return [ans]
    if isinstance(ans, (list, tuple)):
        flat: List[str] = []
        for e in ans:
            if isinstance(e, (list, tuple)):
                flat.extend([str(x) for x in e])
            else:
                flat.append(str(e))
        return flat
    return [str(ans)]


def _gold_text_for_summary(ideal_answer: Any) -> str:
    if ideal_answer is None:
        return ""
    if isinstance(ideal_answer, str):
        return ideal_answer
    if isinstance(ideal_answer, (list, tuple)):
        return " ".join([str(x) for x in ideal_answer])
    return str(ideal_answer)


# ---------------------------
# Primitive metrics
# ---------------------------

def _token_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    """
    Token-level precision, recall, F1 (like SQuAD-style).
    """
    ps = _norm_text(pred).split()
    gs = _norm_text(gold).split()
    if len(ps) == 0 and len(gs) == 0:
        return 1.0, 1.0, 1.0
    if len(ps) == 0 or len(gs) == 0:
        return 0.0, 0.0, 0.0
    ps_count: Dict[str, int] = {}
    gs_count: Dict[str, int] = {}
    for t in ps:
        ps_count[t] = ps_count.get(t, 0) + 1
    for t in gs:
        gs_count[t] = gs_count.get(t, 0) + 1
    # overlap
    overlap = 0
    for t, c in ps_count.items():
        overlap += min(c, gs_count.get(t, 0))
    prec = overlap / len(ps)
    rec = overlap / len(gs)
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if _norm_text(pred) == _norm_text(gold) else 0.0


def _set_f1(pred_items: Iterable[str], gold_items: Iterable[str]) -> Tuple[float, float, float]:
    """
    Set-based F1 for list questions (duplicates ignored).
    """
    pset = set(_norm_item_list(pred_items))
    gset = set(_norm_item_list(gold_items))
    if not pset and not gset:
        return 1.0, 1.0, 1.0
    if not pset or not gset:
        return 0.0, 0.0, 0.0
    inter = len(pset & gset)
    prec = inter / len(pset)
    rec = inter / len(gset)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


# ---------------------------
# Per-type scoring
# ---------------------------

def score_yesno(pred: str, gold: str) -> float:
    p = _norm_text(pred)
    g = _norm_text(gold)
    # normalize to yes/no
    def yn(x: str) -> str:
        if x in {"yes", "y", "true"}:
            return "yes"
        if x in {"no", "n", "false"}:
            return "no"
        return x
    return 1.0 if yn(p) == yn(g) and yn(g) in {"yes", "no"} else 0.0


def score_factoid(pred: str, gold_aliases: List[str]) -> Tuple[float, float]:
    """
    Returns (best_em, best_f1) across gold aliases.
    """
    if not gold_aliases:
        return (0.0, 0.0)
    best_em, best_f1 = 0.0, 0.0
    for gold in gold_aliases:
        em = _exact_match(pred, gold)
        _, _, f1 = _token_f1(pred, gold)
        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)
    return best_em, best_f1


def score_list(pred: str, gold_items: List[str]) -> Tuple[float, float, float]:
    """
    Many generations provide comma-separated items; split heuristically on commas/semicolons/newlines.
    """
    # split & clean prediction
    raw = re.split(r"[,\n;]", _norm_text(pred))
    pred_items = [s.strip() for s in raw if s.strip()]
    return _set_f1(pred_items, gold_items)


def score_summary(pred: str, gold_text: str) -> float:
    """
    ROUGE-L (F-measure). If rouge-score is unavailable, return 0.0 and warn.
    """
    if not _ROUGE_OK:
        # You can also raise here if you prefer strict behavior.
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(_norm_text(gold_text), _norm_text(pred))
    return float(scores["rougeL"].fmeasure)


# ---------------------------
# Orchestration
# ---------------------------

def build_gold_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return { _canonical_id(r): r for r in rows }


def evaluate(preds: List[Dict[str, Any]], gold_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    # Accumulators
    counts = {"yesno": 0, "factoid": 0, "list": 0, "summary": 0}
    yesno_correct = 0
    factoid_em_sum = 0.0
    factoid_f1_sum = 0.0
    list_prec_sum = 0.0
    list_rec_sum = 0.0
    list_f1_sum = 0.0
    summary_rougel_sum = 0.0

    missing_gold = 0
    missing_pred = 0

    for rec in preds:
        cid = _canonical_id(rec)
        g = gold_index.get(cid)
        if g is None:
            # Try loose match on type+question if id mismatch in input files
            missing_gold += 1
            continue

        qtype = (rec.get("type") or g.get("type") or "").lower().strip()
        predicted = str(rec.get("predicted") or rec.get("answer") or rec.get("prediction") or "")

        if qtype in {"yesno", "yes/no", "yn"}:
            counts["yesno"] += 1
            gold = str(g.get("exact_answer") or g.get("yesno") or "")
            yesno_correct += score_yesno(predicted, gold)

        elif qtype in {"factoid", "fact"}:
            counts["factoid"] += 1
            gold_aliases = _flatten_exact_answer(g.get("exact_answer"))
            em, f1 = score_factoid(predicted, gold_aliases)
            factoid_em_sum += em
            factoid_f1_sum += f1

        elif qtype in {"list", "lists"}:
            counts["list"] += 1
            gold_items = _flatten_exact_answer(g.get("exact_answer"))
            p, r, f1 = score_list(predicted, gold_items)
            list_prec_sum += p
            list_rec_sum += r
            list_f1_sum += f1

        else:  # summary or other treat as summary
            counts["summary"] += 1
            gold_text = _gold_text_for_summary(g.get("ideal_answer"))
            summary_rougel_sum += score_summary(predicted, gold_text)

    # Averages (avoid div-by-zero)
    def _avg(total, n): return (total / n) if n else 0.0

    results = {
        "counts": counts,
        "yesno": {"accuracy": _avg(yesno_correct, counts["yesno"])},
        "factoid": {
            "em": _avg(factoid_em_sum, counts["factoid"]),
            "f1": _avg(factoid_f1_sum, counts["factoid"]),
        },
        "list": {
            "precision": _avg(list_prec_sum, counts["list"]),
            "recall": _avg(list_rec_sum, counts["list"]),
            "f1": _avg(list_f1_sum, counts["list"]),
        },
        "summary": {
            "rougeL": _avg(summary_rougel_sum, counts["summary"]),
            "rougeL_note": "Requires `pip install rouge-score`",
        },
        "missing": {
            "pred_without_gold": missing_gold,
            "gold_without_pred": missing_pred,  # reserved; unused in current join-by-id
        },
    }

    # Macro average over types that have at least one example
    per_type_vals = []
    if counts["yesno"] > 0:
        per_type_vals.append(results["yesno"]["accuracy"])
    if counts["factoid"] > 0:
        per_type_vals.append(results["factoid"]["f1"])
    if counts["list"] > 0:
        per_type_vals.append(results["list"]["f1"])
    if counts["summary"] > 0:
        per_type_vals.append(results["summary"]["rougeL"])
    results["macro_avg"] = sum(per_type_vals) / len(per_type_vals) if per_type_vals else 0.0

    return results


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Score BioASQ-style predictions against gold.")
    ap.add_argument("--pred", required=True, help="Predictions JSON/JSONL")
    ap.add_argument("--gold", required=True, help="Gold JSON/JSONL (or BioASQ JSON with {'questions':[...]})")
    ap.add_argument("--out", default="results/metrics/metrics.json", help="Where to write the metrics JSON")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    preds = _read_json_any(args.pred)
    gold = _read_json_any(args.gold)
    gold_idx = build_gold_index(gold)

    print(f"[load] preds={len(preds)}  gold={len(gold)}  matched_by_id={len(gold_idx)}")
    metrics = evaluate(preds, gold_idx)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    c = metrics["counts"]
    print(
        "[done] "
        f"yesno(acc)={metrics['yesno']['accuracy']:.3f} ({c['yesno']})  "
        f"factoid(f1)={metrics['factoid']['f1']:.3f} ({c['factoid']})  "
        f"list(f1)={metrics['list']['f1']:.3f} ({c['list']})  "
        f"summary(rougeL)={metrics['summary']['rougeL']:.3f} ({c['summary']})  "
        f"macro={metrics['macro_avg']:.3f}"
    )
    print(f"[save] {out_path.resolve()}")


if __name__ == "__main__":
    main()