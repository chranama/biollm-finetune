# Phase 4 — Robustness Experimentation & Aggregation

## Objective

Phase 4 operationalizes robustness evaluation for biomedical question answering models by executing a controlled experiment grid, applying deterministic input perturbations, and aggregating results in a reproducible and phenotype-aware manner.

The central goal of this phase is to quantify performance degradation and stability under targeted perturbations rather than to optimize raw task accuracy.

## Key Contributions

Phase 4 delivers a complete robustness experimentation pipeline.

This includes deterministic perturbation application, reproducible experiment execution, seed-controlled robustness analysis, phenotype-aware aggregation, and clean versus perturbed performance comparison.

All evaluations are inference-only and do not modify model parameters.

## Experiment Design

### Dataset

The dataset used is bioasq_TINY.

A fixed question set is used across all runs, and gold answers remain unchanged for all perturbations.

Each perturbation modifies only the input representation and never the gold answers.

### Model

The model used is TinyLlama/TinyLlama-1.1B-Chat-v1.0.

All experiments are inference-only, and no fine-tuning is performed in Phase 4.

### Runtime Environment

Experiments are run on Apple Silicon using the MPS backend with FP32 precision.

Deterministic execution is enforced via explicit seeding.

## Perturbation Grid

Perturbations are applied prior to inference and are deterministic with respect to the experiment seed.

The perturbation categories include:

Baseline: clean

Structural: shuffle_snippets

Lexical: lexical_noise, lexical_noise_medium, lexical_noise_heavy

Contextual: irrelevant_noise, irrelevant_noise_heavy

Logical: contradiction

## Seed Strategy

Three seeds are used for robustness analysis.

The seed set is:

[13, 42, 97]

Seeds control perturbation randomness, inference determinism, and reproducibility across runs.

Clean baselines are executed with the same seeds to enable valid delta comparisons.

## Experiment Execution

The primary execution entrypoint is scripts/run_experiment.py.

This script loads the experiment configuration, applies perturbations deterministically, persists exact inference inputs, runs inference, evaluates predictions against clean gold labels, tags phenotypes, and registers run metadata.

Batch execution is handled by scripts/run_experiments.py, which supports selector-based execution such as running all non-clean perturbations in a single command.

## Outputs

Each experiment produces an isolated directory containing the exact inputs used for inference, predictions, metrics, phenotype tags, a run manifest, and optional robustness and stability artifacts.

Aggregated results are written to experiments.csv and summary.json for downstream analysis.

## Phase 4 Completion Criteria

Phase 4 is considered complete when all perturbation-seed combinations have been executed, aggregated, and verified for determinism and correctness.

All robustness conclusions must be traceable to concrete experiment artifacts and reproducible via configuration alone.