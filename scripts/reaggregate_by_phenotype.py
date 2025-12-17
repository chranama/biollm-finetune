#!/usr/bin/env python3
"""
Phenotype-stratified reaggregation for BioLLM experiment runs.

Supports phenotypes.json formats:
1) tags: {qid: {phenotype_name: bool, ...}}   (current)
2) tags: {qid: [phenotype_name, ...]}         (legacy/alternative)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from biollm_finetune.data.loaders import load_jsonl
from biollm_finetune.eval.metrics import evaluate_predictions
from biollm_finetune.utils.config import load_config


# ----------------------------
# Helpers
# ----------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _get_id(rec: Dict[str, Any]) -> Optional[str]:
    rid = rec.get("id")
    if rid is None:
        rid = rec.get("_id")
    if rid is None:
        return None
    return str(rid)


def _index_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rid = _get_id(r)
        if rid is None:
            continue
        out[rid] = r
    return out


def _flatten_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    def g(path: str) -> Optional[float]:
        cur: Any = m
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        try:
            return float(cur)
        except Exception:
            return None

    return {
        "macro_avg": g("macro_avg"),
        "yesno_acc": g("yesno.accuracy"),
        "factoid_f1": g("factoid.f1"),
        "factoid_em": g("factoid.em"),
        "list_f1": g("list.f1"),
        "list_precision": g("list.precision"),
        "list_recall": g("list.recall"),
        "summary_rougeL": g("summary.rougeL"),
    }


def _infer_runtime_from_name(run_id: str) -> str:
    # canonical: <dataset>_<runtime>_<perturbation>_seed<seed>
    parts = run_id.split("_")
    if len(parts) < 4:
        return ""
    # common runtime tags are 2 tokens: mps_fp32, cuda_bf16, cpu_fp32
    if len(parts) >= 5 and parts[-4] in {"mps", "cpu", "cuda"} and parts[-3] in {"fp32", "fp16", "bf16"}:
        return f"{parts[-4]}_{parts[-3]}"
    # fallback
    return parts[2] if len(parts) > 2 else ""


def _resolve_task(exp_cfg: Any, gold_examples: List[Dict[str, Any]]) -> str:
    dataset = getattr(exp_cfg, "dataset", None)
    if dataset is not None:
        task = getattr(dataset, "task", None)
        if isinstance(task, str) and task.strip():
            return task.strip()
    if gold_examples and any(isinstance(ex, dict) and ("type" in ex) for ex in gold_examples):
        return "bioasq"
    return "bioasq"


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _normalize_tag_map(raw_tags: Any) -> Tuple[Dict[str, Dict[str, bool]], List[str]]:
    """
    Normalize phenotypes.json["tags"] into:
      norm: { qid: { phenotype: bool, ... } }
    and return observed phenotype names.

    Accepts:
      - {qid: {phenotype: bool}}
      - {qid: [phenotype, ...]}
    """
    norm: Dict[str, Dict[str, bool]] = {}
    observed = set()

    if not isinstance(raw_tags, dict):
        return norm, []

    for qid, v in raw_tags.items():
        qid = str(qid)
        if isinstance(v, dict):
            # already {phenotype: bool}
            vv: Dict[str, bool] = {}
            for k, b in v.items():
                if not isinstance(k, str):
                    continue
                bb = bool(b)
                vv[k] = bb
                if bb:
                    observed.add(k)
            norm[qid] = vv
        elif isinstance(v, list):
            # list[str] => mark True
            vv = {}
            for t in v:
                if isinstance(t, str) and t.strip():
                    vv[t.strip()] = True
                    observed.add(t.strip())
            norm[qid] = vv
        else:
            # unknown => empty
            norm[qid] = {}

    return norm, sorted(observed)


# ----------------------------
# Core
# ----------------------------

@dataclass
class RunInfo:
    run_id: str
    run_dir: Path
    manifest: Dict[str, Any]
    config_path: Path


def discover_runs(experiments_root: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    if not experiments_root.exists():
        return runs

    for run_dir in sorted([p for p in experiments_root.iterdir() if p.is_dir()]):
        manifest_path = run_dir / "manifest.json"
        preds_path = run_dir / "predictions.jsonl"
        phenos_path = run_dir / "phenotypes.json"

        if not (manifest_path.exists() and preds_path.exists() and phenos_path.exists()):
            continue

        try:
            manifest = _read_json(manifest_path)
        except Exception:
            continue

        cfg_str = manifest.get("config")
        if not isinstance(cfg_str, str) or not cfg_str.strip():
            continue

        config_path = Path(cfg_str)
        if not config_path.exists():
            alt = Path.cwd() / cfg_str
            if alt.exists():
                config_path = alt
            else:
                continue

        runs.append(RunInfo(run_id=run_dir.name, run_dir=run_dir, manifest=manifest, config_path=config_path))

    return runs


def compute_rows_for_run(run: RunInfo) -> List[Dict[str, Any]]:
    cfg = load_config(str(run.config_path))

    gold_examples = load_jsonl(cfg.dataset.path)
    gold_by_id = _index_by_id(gold_examples)

    preds = load_jsonl(run.run_dir / "predictions.jsonl")
    preds_by_id = _index_by_id(preds)

    phenos = _read_json(run.run_dir / "phenotypes.json")

    schema = phenos.get("schema")
    schema_tags: List[str] = []
    if isinstance(schema, dict):
        schema_tags = sorted([k for k in schema.keys() if isinstance(k, str) and k.strip()])

    raw_tags = phenos.get("tags")
    tag_map, observed_tags = _normalize_tag_map(raw_tags)

    phenotype_tags = ["ALL"] + sorted(set(schema_tags) | set(observed_tags))

    task = _resolve_task(cfg, gold_examples)

    dataset_name = getattr(cfg.dataset, "name", "") or _safe_str(run.manifest.get("dataset"))
    model_name = getattr(cfg.model, "name", "") or _safe_str(run.manifest.get("model"))
    perturbation = _safe_str(run.manifest.get("perturbation")) or _safe_str(getattr(cfg, "perturbation", ""))

    seed = getattr(cfg, "seed", None)
    if seed is None:
        seed = run.manifest.get("seed")
    seed = int(seed) if isinstance(seed, int) or (isinstance(seed, str) and seed.isdigit()) else seed

    runtime_name = ""
    runtime = getattr(cfg, "runtime", None)
    if runtime is not None:
        runtime_name = _safe_str(getattr(runtime, "name", "")) or _safe_str(getattr(runtime, "device", ""))
    if not runtime_name:
        runtime_name = _infer_runtime_from_name(run.run_id)

    rows: List[Dict[str, Any]] = []
    all_ids = list(gold_by_id.keys())

    # Build an index: phenotype -> set(ids) where True (fast slicing)
    ids_for_tag: Dict[str, set[str]] = {t: set() for t in phenotype_tags if t != "ALL"}
    for qid, flags in tag_map.items():
        if qid not in gold_by_id:
            continue
        for t, b in flags.items():
            if b and t in ids_for_tag:
                ids_for_tag[t].add(qid)

    for pheno in phenotype_tags:
        if pheno == "ALL":
            ids = all_ids
        else:
            ids = [qid for qid in all_ids if qid in ids_for_tag.get(pheno, set())]

        sub_gold: List[Dict[str, Any]] = []
        sub_preds: List[Dict[str, Any]] = []
        for qid in ids:
            g = gold_by_id.get(qid)
            if g is None:
                continue
            sub_gold.append(g)
            p = preds_by_id.get(qid)
            if p is not None:
                sub_preds.append(p)

        if pheno != "ALL" and len(sub_gold) == 0:
            continue

        metrics = evaluate_predictions(predictions=sub_preds, gold=sub_gold, task=task)
        flat = _flatten_metrics(metrics)

        rows.append({
            "run_id": run.run_id,
            "dataset": dataset_name,
            "runtime": runtime_name,
            "model": model_name,
            "perturbation": perturbation,
            "seed": seed,
            "task": task,
            "phenotype": pheno,
            "n_gold": len(sub_gold),
            "n_pred": len(sub_preds),
            "pred_coverage": (len(sub_preds) / len(sub_gold)) if len(sub_gold) else 0.0,
            "n_changed_vs_clean": run.manifest.get("n_changed_vs_clean", None),
            **flat,
        })

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_parent(path)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    preferred = [
        "run_id", "dataset", "runtime", "model", "perturbation", "seed", "task",
        "phenotype", "n_gold", "n_pred", "pred_coverage", "n_changed_vs_clean",
        "macro_avg", "yesno_acc", "factoid_em", "factoid_f1", "list_f1", "summary_rougeL",
        "list_precision", "list_recall",
    ]
    header = preferred + [k for k in keys if k not in preferred]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def compute_deltas_vs_clean(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def is_clean(r: Dict[str, Any]) -> bool:
        return _safe_str(r.get("perturbation")).lower().strip() == "clean"

    idx_clean: Dict[Tuple[str, str, str, Any, str], Dict[str, Any]] = {}
    for r in rows:
        if not is_clean(r):
            continue
        key = (r.get("dataset", ""), r.get("runtime", ""), r.get("model", ""), r.get("seed"), r.get("phenotype", ""))
        idx_clean[key] = r

    metric_cols = [
        "macro_avg", "yesno_acc", "factoid_em", "factoid_f1",
        "list_f1", "list_precision", "list_recall", "summary_rougeL",
    ]

    deltas: List[Dict[str, Any]] = []
    for r in rows:
        if is_clean(r):
            continue
        key = (r.get("dataset", ""), r.get("runtime", ""), r.get("model", ""), r.get("seed"), r.get("phenotype", ""))
        base = idx_clean.get(key)
        if base is None:
            continue

        out = dict(r)
        out["clean_run_id"] = base.get("run_id")

        for col in metric_cols:
            a = r.get(col)
            b = base.get(col)
            if a is None or b is None:
                out[f"delta__{col}"] = None
            else:
                try:
                    out[f"delta__{col}"] = float(a) - float(b)
                except Exception:
                    out[f"delta__{col}"] = None

        deltas.append(out)

    return deltas


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-root", default="results/experiments", help="Root dir containing experiment run folders")
    ap.add_argument("--outdir", default="results/aggregates_re", help="Output directory for reaggregated artifacts")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output CSVs if they exist")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N runs (debug)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_runs = outdir / "phenotype_runs.csv"
    out_deltas = outdir / "phenotype_deltas_vs_clean.csv"

    if (out_runs.exists() or out_deltas.exists()) and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing outputs. Use --overwrite.\n"
            f"  - {out_runs}\n"
            f"  - {out_deltas}"
        )

    runs = discover_runs(experiments_root)
    if args.limit is not None:
        runs = runs[: max(0, int(args.limit))]

    all_rows: List[Dict[str, Any]] = []
    for i, run in enumerate(runs, start=1):
        try:
            rows = compute_rows_for_run(run)
            all_rows.extend(rows)
            print(f"[{i}/{len(runs)}] ok  {run.run_id}  rows={len(rows)}")
        except Exception as e:
            print(f"[{i}/{len(runs)}] skip {run.run_id}  reason={e}")

    write_csv(out_runs, all_rows)

    deltas = compute_deltas_vs_clean(all_rows)
    write_csv(out_deltas, deltas)

    print(f"[done] wrote:")
    print(f"  - {out_runs}")
    print(f"  - {out_deltas}")
    print(f"[stats] runs={len(runs)}  rows={len(all_rows)}  deltas={len(deltas)}")


if __name__ == "__main__":
    main()