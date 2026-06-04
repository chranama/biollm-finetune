#!/usr/bin/env python3
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

Important notes:
- We ALWAYS write the exact inputs used for inference to results/.../inputs.jsonl
- We tag phenotypes on the exact inputs used (clean or perturbed)
- We evaluate predictions against the clean gold labels (original dataset)
- We propagate the experiment seed into inference (if generate.py supports --seed)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from biollm_finetune.analysis.phenotypes import PHENOTYPE_DEFINITIONS, tag_dataset_dict
from biollm_finetune.analysis.robustness import compute_stability, save_json
from biollm_finetune.analysis.run_registry import register_run
from biollm_finetune.data.loaders import load_jsonl
from biollm_finetune.eval.metrics import evaluate_predictions
from biollm_finetune.utils.config import RuntimeConfig, load_config
from biollm_finetune.utils.repro import set_seed

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
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _run_inference(
    runtime: RuntimeConfig,
    inputs_path: Path,
    outputs_path: Path,
    manifest_path: Path,
    adapter_path: Optional[Path],
    seed: int,
) -> None:
    """
    Invoke biollm_finetune.inference.generate as a module.

    If generate.py supports --seed, we pass it. If it doesn't, we silently
    fall back to not passing it (to avoid breaking older versions).
    """
    base_cmd = [
        sys.executable,
        "-m",
        "biollm_finetune.inference.generate",
        "--config",
        runtime.inference_config,
        "--input",
        str(inputs_path),
        "--out",
        str(outputs_path),
        "--manifest-out",
        str(manifest_path),
    ]
    if adapter_path is not None:
        base_cmd.extend(["--adapter", str(adapter_path)])

    # Try with --seed first (new behavior); fall back if unsupported.
    cmd_with_seed = base_cmd + ["--seed", str(seed)]
    try:
        subprocess.check_call(cmd_with_seed)
        return
    except subprocess.CalledProcessError:
        raise
    except Exception as e:
        # Most commonly: argparse error in generate.py for unknown --seed.
        # Fall back to the old invocation.
        subprocess.check_call(base_cmd)


def _resolve_task(exp_cfg: Any, examples: list[dict[str, Any]]) -> str:
    dataset = getattr(exp_cfg, "dataset", None)
    if dataset is not None:
        task = getattr(dataset, "task", None)
        if isinstance(task, str) and task.strip():
            return task.strip()

    if examples and any(isinstance(ex, dict) and ("type" in ex) for ex in examples):
        return "bioasq"

    return "bioasq"


def _apply_perturbation(
    perturbation: str,
    examples: list[dict[str, Any]],
    seed: int,
    exp_cfg: Any,
) -> list[dict[str, Any]]:
    """
    Apply a named perturbation to a dataset (list of examples), deterministically.

    NOTE: biollm_finetune.analysis.perturbations.apply_perturbation operates on a
    single example, so we map over the dataset here.
    """
    p = (perturbation or "clean").strip().lower()
    if p == "clean":
        return examples

    # Make perturbations deterministic across the dataset.
    set_seed(seed)

    # Optional per-perturbation configuration (paths, budgets, etc.)
    cfg: dict[str, Any] = {}
    pert_cfg = getattr(exp_cfg, "perturbation_config", None)
    if isinstance(pert_cfg, dict):
        cfg.update(pert_cfg)

    # Lazily import to avoid overhead on clean runs
    from biollm_finetune.analysis.perturbations import apply_perturbation as _apply_one

    out: list[dict[str, Any]] = []
    for ex in examples:
        # Never mutate the clean dataset in-place
        ex_copy = json.loads(json.dumps(ex))
        out.append(_apply_one(ex_copy, p, cfg))

    return out


def _count_changed(orig: list[dict[str, Any]], pert: list[dict[str, Any]]) -> int:
    """
    Conservative change detector across common fields.
    """
    n = min(len(orig), len(pert))
    changed = 0
    for i in range(n):
        a = orig[i]
        b = pert[i]
        if (
            (a.get("body") != b.get("body"))
            or (a.get("question") != b.get("question"))
            or (a.get("snippets") != b.get("snippets"))
        ):
            changed += 1
    return changed


def _resolve_adapter_path(model_cfg: Any) -> Optional[Path]:
    """
    Resolve an optional adapter path from experiment model config.

    `model.adapter` is used for existing adapters that should be validated by
    config loading. `model.adapter_output_dir` is useful for workflows where a
    prior fine-tuning step writes the adapter before this runner is invoked.
    """
    for attr in ("adapter", "adapter_output_dir"):
        value = getattr(model_cfg, attr, None)
        if isinstance(value, str) and value.strip():
            return Path(value)
    return None


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

    manifest: dict[str, Any] = {
        "experiment": exp_name,
        "config": args.config,
        "dataset": exp_cfg.dataset.name,
        "model": exp_cfg.model.name,
        "adapter_path": str(_resolve_adapter_path(exp_cfg.model) or ""),
        "seed": exp_cfg.seed,
        "start_time_utc": _now_utc(),
        "perturbation": exp_cfg.perturbation,
        "runtime": exp_cfg.runtime.name,
        "requested_device": exp_cfg.runtime.device,
        "requested_dtype": exp_cfg.runtime.dtype,
    }

    # -------------------------
    # Load inputs (clean)
    # -------------------------
    clean_examples = load_jsonl(exp_cfg.dataset.path)

    # -------------------------
    # Apply perturbations
    # -------------------------
    used_examples = _apply_perturbation(
        perturbation=exp_cfg.perturbation,
        examples=clean_examples,
        seed=exp_cfg.seed,
        exp_cfg=exp_cfg,
    )

    changed = _count_changed(clean_examples, used_examples)
    manifest["n_examples"] = len(used_examples)
    manifest["n_changed_vs_clean"] = int(changed)

    # -------------------------
    # Write the exact inputs used for inference
    # -------------------------
    inputs_path = exp_dir / "inputs.jsonl"
    _write_jsonl(inputs_path, used_examples)

    # -------------------------
    # Phenotype tagging (tag what you actually ran)
    # -------------------------
    phenotype_map = tag_dataset_dict(used_examples)
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
    inference_manifest_path = exp_dir / "inference_manifest.json"
    adapter_path = _resolve_adapter_path(exp_cfg.model)

    _run_inference(
        runtime=exp_cfg.runtime,
        inputs_path=inputs_path,
        outputs_path=preds_path,
        manifest_path=inference_manifest_path,
        adapter_path=adapter_path,
        seed=exp_cfg.seed,
    )
    inference_manifest = (
        _read_json(inference_manifest_path) if inference_manifest_path.exists() else {}
    )
    manifest["inference_manifest"] = str(inference_manifest_path)
    manifest["resolved_device"] = inference_manifest.get("device")
    manifest["resolved_dtype"] = inference_manifest.get("dtype")
    manifest["model_id"] = inference_manifest.get("model_id")

    # -------------------------
    # Evaluation
    # -------------------------
    preds = load_jsonl(preds_path)
    task = _resolve_task(exp_cfg, clean_examples)

    metrics = evaluate_predictions(
        predictions=preds,
        gold=clean_examples,
        task=task,
    )
    _write_json(exp_dir / "metrics.json", metrics)

    # -------------------------
    # Register run
    # -------------------------
    run_record = {
        "run_id": exp_name,
        "dataset": exp_cfg.dataset.name,
        "model": exp_cfg.model.name,
        "adapter_path": str(adapter_path or ""),
        "seed": exp_cfg.seed,
        "perturbation": exp_cfg.perturbation,
        "task": task,
        "runtime": exp_cfg.runtime.name,
        "requested_device": exp_cfg.runtime.device,
        "requested_dtype": exp_cfg.runtime.dtype,
        "resolved_device": inference_manifest.get("device"),
        "resolved_dtype": inference_manifest.get("dtype"),
        "metrics": metrics,
        "paths": {
            "inputs": str(inputs_path),
            "predictions": str(preds_path),
            "metrics": str(exp_dir / "metrics.json"),
            "phenotypes": str(exp_dir / "phenotypes.json"),
            "inference_manifest": str(inference_manifest_path),
        },
    }
    register_run(run_record)

    # -------------------------
    # Optional: robustness + stability
    # -------------------------
    if hasattr(exp_cfg, "robustness") and exp_cfg.robustness and exp_cfg.robustness.enabled:
        clean_preds = load_jsonl(exp_cfg.robustness.clean_predictions)

        stability = compute_stability(
            clean_preds=[p.get("answer") or p.get("prediction") or "" for p in clean_preds],
            perturbed_preds=[p.get("answer") or p.get("prediction") or "" for p in preds],
            gold=[ex.get("exact_answer") for ex in clean_examples],
            run_id=exp_name,
            perturbation=exp_cfg.perturbation,
        )
        save_json([stability], exp_dir / "stability.json")

    manifest["task"] = task
    manifest["end_time_utc"] = _now_utc()
    _write_json(exp_dir / "manifest.json", manifest)

    print(
        f"Experiment '{exp_name}' completed → {exp_dir} "
        f"(perturbation={exp_cfg.perturbation}, changed_vs_clean={changed}/{len(used_examples)})"
    )


if __name__ == "__main__":
    main()
