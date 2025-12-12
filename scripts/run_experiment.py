#!/usr/bin/env python
"""
High-level experiment runner for the biomedical QA robustness suite.

Usage:
    uv run scripts/run_experiment.py \
        --config configs/experiments/bioasq_TINY_clean_seed42.yaml

Responsibilities:
    - Load an experiment config (YAML)
    - Set random seeds
    - Create an experiment output directory
    - Save the resolved config + run metadata
    - Load BioASQ-style questions (gold)
    - Apply perturbations
    - Run model inference via biollm_finetune.inference.generate
    - Compute task metrics + phenotype metrics
    - Optionally compute robustness vs a matching clean run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml  # make sure PyYAML is installed

from biollm_finetune.analysis import (
    phenotypes,
    perturbations,
    robustness,
    run_registry,
)
from biollm_finetune.data.loaders import load_questions_any
from biollm_finetune.eval.metrics import (
    build_gold_index,
    evaluate as evaluate_metrics,
)


# ----------------------------
# Data classes for structure
# ----------------------------

@dataclass
class ModelConfig:
    id: str
    quantization: Optional[str] = None
    adapter_path: Optional[str] = None


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    dataset_split: str          # free-form label, not hard-coded
    perturbation: str           # "clean", "irrelevant_noise", "shuffle_snippets", etc.
    model: ModelConfig
    extra: Dict[str, Any]       # catch-all for anything else (data paths, inference config)


@dataclass
class RunMetadata:
    experiment_name: str
    seed: int
    dataset_split: str
    perturbation: str
    model_id: str
    quantization: Optional[str]
    adapter_path: Optional[str]
    git_commit: Optional[str]
    start_time_utc: str


# ----------------------------
# CLI + config loading
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a robustness experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--experiments-root",
        type=str,
        default="results/experiments",
        help="Root directory where experiment outputs will be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and create dirs, but skip model inference.",
    )
    return parser.parse_args()


def load_experiment_config(path: Path) -> ExperimentConfig:
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    # Accept either {experiment: {...}} or flat top-level keys
    exp_raw = raw.get("experiment", raw)

    model_raw = exp_raw.get("model", {})
    model_cfg = ModelConfig(
        id=model_raw.get("id", "") or model_raw.get("path", "") or model_raw.get("base_model", ""),
        quantization=model_raw.get("quantization"),
        adapter_path=model_raw.get("adapter_path") or model_raw.get("adapter_output_dir"),
    )

    extra_keys = {"name", "seed", "dataset_split", "perturbation", "model"}
    extra = {k: v for k, v in exp_raw.items() if k not in extra_keys}

    return ExperimentConfig(
        name=exp_raw["name"],
        seed=int(exp_raw.get("seed", 42)),
        dataset_split=exp_raw.get("dataset_split", "unknown"),
        perturbation=exp_raw["perturbation"],
        model=model_cfg,
        extra=extra,
    )


def get_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def set_random_seed(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_experiment_dir(root: Path, name: str) -> Path:
    exp_dir = root / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_yaml(data: Any, path: Path) -> None:
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(data: Any, path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------
# Core hooks (wired to your code)
# ----------------------------

def _resolve_questions_path(cfg: ExperimentConfig) -> Path:
    """
    Decide where to load the gold questions from.

    Expected options:
      - experiment.extra["questions_path"]
      - or experiment.extra["data"]["questions_path"]
    """
    data_block = cfg.extra.get("data", {}) if isinstance(cfg.extra.get("data", {}), dict) else {}
    questions_path = data_block.get("questions_path")
    questions_path = questions_path or cfg.extra.get("questions_path")

    if not questions_path:
        raise ValueError(
            "Experiment config must include a 'questions_path' under "
            "'experiment.data.questions_path' or 'experiment.questions_path'."
        )

    return Path(questions_path)


def load_dataset(cfg: ExperimentConfig) -> List[Dict[str, Any]]:
    """
    Load the evaluation dataset for this experiment.

    Uses biollm_finetune.data.loaders.load_questions_any to read:
      - JSON with {'questions': [...]}
      - JSON list [...]
      - JSONL (one per line)

    Returns the list of gold question dicts in BioASQ-style schema.
    """
    path = _resolve_questions_path(cfg)
    rows = load_questions_any(path)
    return rows


def _write_input_jsonl_for_inference(
    examples: Iterable[Dict[str, Any]],
    path: Path,
) -> None:
    """
    Prepare a JSONL file to feed into biollm_finetune.inference.generate.
    """
    save_jsonl(list(examples), path)


def _resolve_inference_config_path(cfg: ExperimentConfig) -> Path:
    """
    Decide which inference YAML config to use.

    Priority:
      1. experiment.extra["inference_config"]
      2. env var BIOLLM_INFERENCE_CONFIG
      3. default 'configs/inference.yaml'
    """
    explicit = cfg.extra.get("inference_config")
    env = os.getenv("BIOLLM_INFERENCE_CONFIG")
    path = explicit or env or "configs/inference.yaml"
    return Path(path)


def run_model_inference(
    examples: List[Dict[str, Any]],
    cfg: ExperimentConfig,
    exp_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Run model inference by calling the existing CLI:

        python -m biollm_finetune.inference.generate \
            --config <inference_config> \
            --input  <input_jsonl> \
            --out    <preds_jsonl> \
            [--adapter <adapter_path>]

    Returns a list of prediction records loaded from the JSONL output.
    """
    input_path = exp_dir / "inputs.jsonl"
    preds_path = exp_dir / "predictions_raw.jsonl"

    _write_input_jsonl_for_inference(examples, input_path)

    inf_cfg_path = _resolve_inference_config_path(cfg)
    cmd = [
        sys.executable,
        "-m",
        "biollm_finetune.inference.generate",
        "--config",
        str(inf_cfg_path),
        "--input",
        str(input_path),
        "--out",
        str(preds_path),
    ]

    if cfg.model.adapter_path:
        cmd.extend(["--adapter", cfg.model.adapter_path])

    print(f"[run] Inference via: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    # Read predictions back in
    preds: List[Dict[str, Any]] = []
    with preds_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Normalize field name for metrics.evaluate
            if "prediction" in rec and "predicted" not in rec:
                rec["predicted"] = rec["prediction"]
            preds.append(rec)

    return preds


def compute_task_metrics(
    predictions: List[Dict[str, Any]],
    gold_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute BioASQ-style metrics using your existing eval.metrics module.

    - Builds a gold index: id -> gold record
    - Calls evaluate(preds, gold_index)
    """
    gold_index = build_gold_index(gold_rows)
    results = evaluate_metrics(predictions, gold_index)
    return results


def compute_phenotype_metrics(
    predictions: List[Dict[str, Any]],
    phenotype_map: Dict[str, List[str]],
    gold_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute metrics broken down by phenotype.

    phenotype_map maps question_id -> list of phenotype tags.
    Returns a dict: phenotype -> metric_name -> value.
    """
    gold_index = build_gold_index(gold_rows)
    by_pheno: Dict[str, Dict[str, Any]] = {}

    # Build inverse index: phenotype -> set of ids
    pheno_to_ids: Dict[str, set] = {}
    for qid, tags in phenotype_map.items():
        for tag in tags:
            pheno_to_ids.setdefault(tag, set()).add(str(qid))

    for pheno, id_set in pheno_to_ids.items():
        subset = [p for p in predictions if str(p.get("id") or p.get("_id")) in id_set]
        if not subset:
            continue
        by_pheno[pheno] = evaluate_metrics(subset, gold_index)

    return by_pheno


# ----------------------------
# Main experiment flow
# ----------------------------

def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    experiments_root = Path(args.experiments_root)

    exp_cfg = load_experiment_config(config_path)
    set_random_seed(exp_cfg.seed)

    exp_dir = ensure_experiment_dir(experiments_root, exp_cfg.name)

    # Save a copy of the resolved config used for this run
    resolved_cfg_dict = {
        "experiment": {
            "name": exp_cfg.name,
            "seed": exp_cfg.seed,
            "dataset_split": exp_cfg.dataset_split,
            "perturbation": exp_cfg.perturbation,
            "model": asdict(exp_cfg.model),
            "extra": exp_cfg.extra,
        }
    }
    save_yaml(resolved_cfg_dict, exp_dir / "config.yaml")

    # Build and save run metadata
    meta = RunMetadata(
        experiment_name=exp_cfg.name,
        seed=exp_cfg.seed,
        dataset_split=exp_cfg.dataset_split,
        perturbation=exp_cfg.perturbation,
        model_id=exp_cfg.model.id,
        quantization=exp_cfg.model.quantization,
        adapter_path=exp_cfg.model.adapter_path,
        git_commit=get_git_commit(),
        start_time_utc=datetime.now(timezone.utc).isoformat(),
    )
    save_json(asdict(meta), exp_dir / "run_metadata.json")

    if args.dry_run:
        print(f"[DRY RUN] Initialized experiment directory at {exp_dir}")
        return

    # 1) Load gold dataset
    gold_rows = load_dataset(exp_cfg)

    # 2) Apply perturbations
    perturbed_examples: List[Dict[str, Any]] = []
    for ex in gold_rows:
        perturbed_examples.append(
            perturbations.apply_perturbation(ex, exp_cfg.perturbation)
        )

    # 3) Tag phenotypes (now using dataset-level dict keyed by id)
    phenotype_map: Dict[str, List[str]] = phenotypes.tag_dataset(gold_rows)

    # 4) Run inference
    predictions = run_model_inference(perturbed_examples, exp_cfg, exp_dir)

    # 5) Compute metrics
    metrics = compute_task_metrics(predictions, gold_rows)
    phenotype_metrics = compute_phenotype_metrics(predictions, phenotype_map, gold_rows)

    # 6) Optionally compute robustness if a corresponding clean run exists
    robustness_summary: Dict[str, Any] = {}
    if exp_cfg.perturbation != "clean":
        clean_name = exp_cfg.name.replace(exp_cfg.perturbation, "clean")
        clean_dir = experiments_root / clean_name
        clean_metrics_path = clean_dir / "metrics.json"
        if clean_metrics_path.exists():
            with clean_metrics_path.open("r") as f:
                clean_metrics = json.load(f)
            robustness_summary = robustness.compute_robustness(
                clean_metrics, metrics
            )
        else:
            # Fallback: try registry-based discovery if available
            clean_runs = [
                r for r in run_registry.iter_experiments(experiments_root)
                if r.path.name.startswith(clean_name)
            ]
            if clean_runs:
                candidate = clean_runs[0].path / "metrics.json"
                if candidate.exists():
                    with candidate.open("r") as f:
                        clean_metrics = json.load(f)
                    robustness_summary = robustness.compute_robustness(
                        clean_metrics, metrics
                    )

    # 7) Save artifacts
    save_jsonl(predictions, exp_dir / "predictions.jsonl")
    save_json(metrics, exp_dir / "metrics.json")
    save_json(phenotype_metrics, exp_dir / "phenotype_metrics.json")
    save_json(robustness_summary, exp_dir / "robustness.json")

    print(f"Experiment '{exp_cfg.name}' completed. Outputs written to {exp_dir}")


if __name__ == "__main__":
    main()