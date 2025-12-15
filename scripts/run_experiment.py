"""
Unified experiment runner for BioLLM fine-tuning and robustness analysis.

Phase 3 responsibilities:
- Run clean + perturbed inference
- Persist predictions, metrics, and manifests
- Persist phenotype tags (example-level)
- Persist robustness + stability analysis artifacts

This script assumes:
- inference is already implemented and correct
- perturbations are deterministic via seeding
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime, UTC
from typing import Any, Dict, List

from biollm_finetune.utils.config import load_config
from biollm_finetune.utils.repro import set_seed
from biollm_finetune.analysis.phenotypes import (
    tag_dataset_dict,
    PHENOTYPE_DEFINITIONS,
)
from biollm_finetune.analysis.robustness import (
    compute_robustness_records,
    compute_stability,
    save_json,
)
from biollm_finetune.analysis.run_registry import register_run
from biollm_finetune.data.loaders import load_jsonl
from biollm_finetune.eval.metrics import evaluate_predictions


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Experiment config YAML")
    return p.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _run_inference(
    cfg: Dict[str, Any],
    inputs_path: Path,
    outputs_path: Path,
    adapter_path: Path | None,
) -> None:
    cmd = [
        str(Path(cfg["python_bin"])),
        "-m",
        "biollm_finetune.inference.generate",
        "--config",
        cfg["inference_config"],
        "--input",
        str(inputs_path),
        "--out",
        str(outputs_path),
    ]
    if adapter_path:
        cmd.extend(["--adapter", str(adapter_path)])

    subprocess.check_call(cmd)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    exp_cfg = load_config(args.config)

    # -------------------------
    # Experiment metadata
    # -------------------------

    set_seed(exp_cfg.seed)

    exp_name = exp_cfg.name
    exp_dir = Path(exp_cfg.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": exp_name,
        "config": args.config,
        "dataset": exp_cfg.dataset.name,
        "model": exp_cfg.model.name,
        "seed": exp_cfg.seed,
        "start_time_utc": _now_utc(),
        "perturbation": exp_cfg.perturbation,
    }

    # -------------------------
    # Load inputs
    # -------------------------

    examples = load_jsonl(exp_cfg.dataset.path)
    inputs_path = exp_dir / "inputs.jsonl"
    with inputs_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # -------------------------
    # Phenotype tagging (Phase 3)
    # -------------------------

    phenotype_map = tag_dataset_dict(examples)

    _write_json(
        exp_dir / "phenotypes.json",
        {
            "schema": PHENOTYPE_DEFINITIONS,
            "tags": phenotype_map,
        },
    )

    # -------------------------
    # Inference
    # -------------------------

    preds_path = exp_dir / "predictions.jsonl"

    _run_inference(
        cfg=exp_cfg.runtime,
        inputs_path=inputs_path,
        outputs_path=preds_path,
        adapter_path=Path(exp_cfg.model.adapter) if exp_cfg.model.adapter else None,
    )

    # -------------------------
    # Evaluation
    # -------------------------

    preds = load_jsonl(preds_path)
    metrics = evaluate_predictions(
        predictions=preds,
        gold=examples,
        task=exp_cfg.dataset.task,
    )

    _write_json(exp_dir / "metrics.json", metrics)

    # -------------------------
    # Register run
    # -------------------------

    run_record = {
        "run_id": exp_name,
        "dataset": exp_cfg.dataset.name,
        "model": exp_cfg.model.name,
        "seed": exp_cfg.seed,
        "perturbation": exp_cfg.perturbation,
        "metrics": metrics,
        "paths": {
            "inputs": str(inputs_path),
            "predictions": str(preds_path),
            "metrics": str(exp_dir / "metrics.json"),
            "phenotypes": str(exp_dir / "phenotypes.json"),
        },
    }

    register_run(run_record)

    # -------------------------
    # Optional: robustness + stability
    # -------------------------

    if exp_cfg.robustness.enabled:
        clean_preds = load_jsonl(exp_cfg.robustness.clean_predictions)
        gold = examples

        stability = compute_stability(
            clean_preds=[p["answer"] for p in clean_preds],
            perturbed_preds=[p["answer"] for p in preds],
            gold=[ex.get("exact_answer") for ex in gold],
            run_id=exp_name,
            perturbation=exp_cfg.perturbation,
        )

        save_json([stability], exp_dir / "stability.json")

    manifest["end_time_utc"] = _now_utc()
    _write_json(exp_dir / "manifest.json", manifest)

    print(f"Experiment '{exp_name}' completed → {exp_dir}")


if __name__ == "__main__":
    main()