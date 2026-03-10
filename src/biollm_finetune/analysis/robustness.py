"""
Robustness analysis utilities.

Phase 3 scope:
- Clean vs perturbed robustness records
- Perturbation type + intensity parsing
- Prediction stability and flip analysis
- Phenotype-conditioned robustness aggregation

No plotting. No experiment execution.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class RobustnessRecord:
    run_id: str
    dataset: str
    model: str
    seed: int
    perturbation: str
    perturbation_type: str
    perturbation_intensity: Optional[str]
    metric: str
    clean_score: float
    perturbed_score: float
    delta: float
    relative_drop: Optional[float]


@dataclass
class StabilityRecord:
    run_id: str
    perturbation: str
    total: int
    stable: int
    flipped_correct_to_incorrect: int
    flipped_incorrect_to_correct: int
    stability_rate: float


@dataclass
class PhenotypeRobustnessRecord:
    run_id: str
    phenotype: str
    perturbation: str
    metric: str
    clean_score: float
    perturbed_score: float
    delta: float
    relative_drop: Optional[float]
    n_examples: int


# ---------------------------------------------------------------------
# Perturbation parsing
# ---------------------------------------------------------------------


def parse_perturbation(name: str) -> Tuple[str, Optional[str]]:
    """
    Splits perturbation name into (base_type, intensity).

    Examples:
      lexical_noise_medium -> (lexical_noise, medium)
      irrelevant_noise_heavy -> (irrelevant_noise, heavy)
      contradiction -> (contradiction, None)
    """
    parts = name.split("_")
    if len(parts) <= 2:
        return name, None

    base = "_".join(parts[:-1])
    intensity = parts[-1]
    return base, intensity


# ---------------------------------------------------------------------
# Metric robustness (run-level)
# ---------------------------------------------------------------------


def compute_robustness_records(
    runs: Iterable[Dict[str, Any]],
) -> List[RobustnessRecord]:
    """
    Build clean-vs-perturbed robustness records.

    Each run dict must contain:
      - run_id
      - dataset
      - model
      - seed
      - perturbation
      - metrics (dict[str, float])
    """
    grouped: Dict[Tuple[str, str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for run in runs:
        key = (run["dataset"], run["model"], run["seed"])
        grouped[key][run["perturbation"]] = run

    records: List[RobustnessRecord] = []

    for (dataset, model, seed), run_group in grouped.items():
        if "clean" not in run_group:
            raise ValueError(
                f"Missing clean baseline for dataset={dataset}, model={model}, seed={seed}"
            )

        clean_run = run_group["clean"]
        clean_metrics = clean_run.get("metrics", {})

        for perturbation, pert_run in run_group.items():
            if perturbation == "clean":
                continue

            pert_metrics = pert_run.get("metrics", {})
            p_type, p_intensity = parse_perturbation(perturbation)

            for metric, clean_value in clean_metrics.items():
                if metric not in pert_metrics:
                    continue

                pert_value = pert_metrics[metric]
                delta = pert_value - clean_value
                rel = None if clean_value == 0 else (delta / clean_value) * 100.0

                records.append(
                    RobustnessRecord(
                        run_id=pert_run["run_id"],
                        dataset=dataset,
                        model=model,
                        seed=seed,
                        perturbation=perturbation,
                        perturbation_type=p_type,
                        perturbation_intensity=p_intensity,
                        metric=metric,
                        clean_score=clean_value,
                        perturbed_score=pert_value,
                        delta=delta,
                        relative_drop=rel,
                    )
                )

    return records


# ---------------------------------------------------------------------
# Prediction stability & flips (example-level)
# ---------------------------------------------------------------------


def compute_stability(
    clean_preds: List[Any],
    perturbed_preds: List[Any],
    gold: List[Any],
    run_id: str,
    perturbation: str,
) -> StabilityRecord:
    if not (len(clean_preds) == len(perturbed_preds) == len(gold)):
        raise ValueError("Prediction and gold lengths must match")

    total = len(gold)
    stable = 0
    c_to_i = 0
    i_to_c = 0

    for c_pred, p_pred, g in zip(clean_preds, perturbed_preds, gold):
        clean_correct = c_pred == g
        pert_correct = p_pred == g

        if c_pred == p_pred:
            stable += 1

        if clean_correct and not pert_correct:
            c_to_i += 1
        elif not clean_correct and pert_correct:
            i_to_c += 1

    return StabilityRecord(
        run_id=run_id,
        perturbation=perturbation,
        total=total,
        stable=stable,
        flipped_correct_to_incorrect=c_to_i,
        flipped_incorrect_to_correct=i_to_c,
        stability_rate=stable / total if total > 0 else 0.0,
    )


# ---------------------------------------------------------------------
# Phenotype-conditioned robustness
# ---------------------------------------------------------------------


def compute_phenotype_robustness(
    clean_scores: Dict[str, Dict[str, float]],
    pert_scores: Dict[str, Dict[str, float]],
    phenotype_map: Dict[str, Dict[str, bool]],
    run_id: str,
    perturbation: str,
) -> List[PhenotypeRobustnessRecord]:
    """
    Compute robustness metrics conditioned on phenotype membership.

    clean_scores / pert_scores:
      question_id -> {metric -> score}

    phenotype_map:
      question_id -> {phenotype_key -> bool}
    """
    records: List[PhenotypeRobustnessRecord] = []

    phenotypes = next(iter(phenotype_map.values())).keys()

    for phenotype in phenotypes:
        # restrict to examples with phenotype == True
        ids = [
            qid
            for qid, tags in phenotype_map.items()
            if tags.get(phenotype, False) and qid in clean_scores and qid in pert_scores
        ]

        if not ids:
            continue

        for metric in clean_scores[ids[0]].keys():
            clean_vals = [clean_scores[qid][metric] for qid in ids]
            pert_vals = [pert_scores[qid][metric] for qid in ids]

            clean_mean = sum(clean_vals) / len(clean_vals)
            pert_mean = sum(pert_vals) / len(pert_vals)
            delta = pert_mean - clean_mean
            rel = None if clean_mean == 0 else (delta / clean_mean) * 100.0

            records.append(
                PhenotypeRobustnessRecord(
                    run_id=run_id,
                    phenotype=phenotype,
                    perturbation=perturbation,
                    metric=metric,
                    clean_score=clean_mean,
                    perturbed_score=pert_mean,
                    delta=delta,
                    relative_drop=rel,
                    n_examples=len(ids),
                )
            )

    return records


# ---------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------


def aggregate_by_perturbation(
    records: List[RobustnessRecord],
) -> Dict[str, List[RobustnessRecord]]:
    out: Dict[str, List[RobustnessRecord]] = defaultdict(list)
    for r in records:
        out[r.perturbation].append(r)
    return out


def aggregate_by_type_and_intensity(
    records: List[RobustnessRecord],
) -> Dict[Tuple[str, Optional[str]], List[RobustnessRecord]]:
    out: Dict[Tuple[str, Optional[str]], List[RobustnessRecord]] = defaultdict(list)
    for r in records:
        out[(r.perturbation_type, r.perturbation_intensity)].append(r)
    return out


# ---------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------


def save_json(records: List[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
