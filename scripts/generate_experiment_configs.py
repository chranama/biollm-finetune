#!/usr/bin/env python3
"""
Schema-enforced experiment config generator.

This script generates experiment YAMLs by *constructing FullConfig objects*
directly and serializing them. All configs are validated at generation time.

Phase 4 invariant:
- Every generated YAML must be loadable by load_config(...)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import yaml

from biollm_finetune.utils.config import (
    DataArgs,
    DatasetConfig,
    FullConfig,
    InferenceArgs,
    ModelConfig,
    RuntimeConfig,
    SystemArgs,
)

# ---------------------------------------------------------------------
# Canonical base configuration (single source of truth)
# ---------------------------------------------------------------------

DATASET_NAME = "bioasq_TINY"
RUNTIME_TAG = "mps_fp32"

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE = "data/samples/integration_questions.jsonl"
GOLD_FILE = "data/samples/integration_gold.json"

BASE_SEED = 42
RELIABILITY_SEEDS = [13, 42, 97]

BLOCK0_PERTURBATIONS = [
    "clean",
    "lexical_noise_medium",
    "contradiction",
]

BLOCK1_PERTURBATIONS = [
    "clean",
    "shuffle_snippets",
    "lexical_noise",
    "lexical_noise_medium",
    "lexical_noise_heavy",
    "irrelevant_noise",
    "irrelevant_noise_heavy",
    "contradiction",
]

BLOCK2_PERTURBATIONS = [
    "clean",
    "lexical_noise_heavy",
    "irrelevant_noise_heavy",
    "contradiction",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def canonical_name(dataset: str, runtime: str, perturbation: str, seed: int) -> str:
    return f"{dataset}_{runtime}_{perturbation}_seed{seed}"


def build_base_config(seed: int) -> FullConfig:
    return FullConfig(
        name=None,
        seed=seed,
        perturbation="clean",
        output_dir="results/experiments",
        dataset=DatasetConfig(
            name="bioasq_TINY",
            path="data/samples/integration_questions.jsonl",
            gold_file="data/samples/integration_gold.json",
        ),
        model=ModelConfig(
            name="tinyllama-1.1b-chat",
            path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            adapter=None,
            load_4bit=False,
            load_8bit=False,
            torch_dtype="float32",
        ),
        runtime=RuntimeConfig(
            name="mps_fp32",
            device="mps",
            dtype="float32",
        ),
        data=DataArgs(
            include_snippets=True,
        ),
        inference=InferenceArgs(
            batch_size=1,
            max_new_tokens=128,
            do_sample=False,
        ),
        system=SystemArgs(
            device_map="mps",
        ),
    )


def write_yaml(path: Path, cfg: FullConfig) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(), f, sort_keys=False)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="configs/experiments")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only-block0", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plans = []

    perts_01 = BLOCK0_PERTURBATIONS if args.only_block0 else BLOCK1_PERTURBATIONS
    for p in perts_01:
        plans.append((BASE_SEED, p))

    if not args.only_block0:
        for s in RELIABILITY_SEEDS:
            for p in BLOCK2_PERTURBATIONS:
                plans.append((s, p))

    written = 0
    for seed, perturbation in plans:
        name = canonical_name(DATASET_NAME, RUNTIME_TAG, perturbation, seed)

        cfg = build_base_config(seed)

        # Inject experiment-specific fields
        cfg_dict = cfg.model_dump()
        cfg_dict["name"] = name
        cfg_dict["seed"] = seed
        cfg_dict["perturbation"] = perturbation

        # Re-validate after injection
        cfg = FullConfig(**cfg_dict)

        out_path = outdir / f"{name}.yaml"
        if out_path.exists() and not args.overwrite:
            continue

        write_yaml(out_path, cfg)
        written += 1

    print(f"[done] wrote {written} configs → {outdir}")


if __name__ == "__main__":
    main()
