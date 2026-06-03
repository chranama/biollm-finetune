# Methods — Robustness Experimentation & Aggregation

## Overview

This section describes the methodology used to evaluate the robustness of a biomedical question answering (QA) model under controlled input perturbations. The goal of this methodology is to **systematically quantify performance degradation and stability** when the input representation is modified, while holding model parameters and gold answers fixed.

All experiments are **inference-only**, deterministic, and fully reproducible from configuration files.

---

## Experimental Objective

The robustness evaluation framework is designed to answer the following methodological questions:

- How sensitive is a biomedical QA model to structured perturbations of its input?
- Does robustness degradation scale predictably with perturbation severity?
- Are robustness effects stable across random seeds?
- Do different perturbations disproportionately affect specific linguistic or semantic question phenotypes?

Rather than optimizing task accuracy, the focus is on **measuring relative performance changes** between clean and perturbed conditions.

---

## Dataset

All experiments are conducted on **bioasq_TINY**, a controlled subset of the BioASQ dataset.

Key properties:
- A fixed set of questions is used across all runs.
- Gold answers are identical for clean and perturbed runs.
- Perturbations modify only the *input representation* (question text, snippets, or context), never the gold labels.

This design ensures that any performance changes can be attributed exclusively to input perturbations.

---

## Model

The evaluated model is **TinyLlama/TinyLlama-1.1B-Chat-v1.0**, a lightweight, general-purpose instruction-following language model.

Experimental constraints:
- Inference-only evaluation.
- No fine-tuning, continued pretraining, or parameter updates.
- Identical model weights across all runs.

This isolates robustness effects from confounding training dynamics.

---

## Runtime Environment

Experiments are executed on Apple Silicon using the **MPS backend** with **FP32 precision**.

Determinism is enforced through:
- Explicit random seed control.
- Fixed model configuration.
- Deterministic perturbation application.

This ensures that observed variability reflects true robustness behavior rather than stochastic execution effects.

---

## Perturbation Design

Perturbations are applied *prior to inference* and are deterministic with respect to the experiment seed.

The perturbation grid includes the following categories:

- **Baseline**
  - `clean`

- **Structural**
  - `shuffle_snippets`

- **Lexical**
  - `lexical_noise`
  - `lexical_noise_medium`
  - `lexical_noise_heavy`

- **Contextual**
  - `irrelevant_noise`
  - `irrelevant_noise_heavy`

- **Logical**
  - `contradiction`

Perturbation families with graded intensity are designed to assess monotonic robustness degradation as noise severity increases.

---

## Seed Strategy

Robustness is evaluated across three fixed seeds:

```text
[13, 42, 97]
```

Seeds control:
- Perturbation randomness,
- Inference determinism,
- Reproducibility across runs.

For every perturbed run, a corresponding **clean baseline with the same seed** is executed. This enables valid delta-based comparisons that are not confounded by random variation.

---

## Experiment Execution

### Single-Run Execution

The primary execution entrypoint is:

```text
scripts/run_experiment.py
```

For each experiment configuration, this script performs the following steps:

1. Loads the experiment configuration.
2. Applies the specified perturbation deterministically.
3. Persists the exact inputs used for inference.
4. Runs model inference.
5. Evaluates predictions against clean gold labels.
6. Tags each example with predefined phenotypes.
7. Records metadata and artifacts in an isolated run directory.

---

### Batch Execution

Batch execution is handled by:

```text
scripts/run_experiments.py
```

This script supports selector-based execution, enabling commands such as:
- running all non-clean perturbations,
- rerunning only failed or missing experiments,
- restricting execution by perturbation type or seed.

This allows scalable and reproducible execution of the full experiment grid.

---

## Phenotype Tagging

Each example is annotated with linguistic and semantic phenotypes prior to aggregation.

Phenotypes capture intrinsic properties of the input, such as:
- question length,
- context length,
- answer structure (e.g., multi-answer lists).

Phenotype tags are persisted per run and used to enable **phenotype-conditioned aggregation**, allowing robustness effects to be analyzed beyond global averages.

---

## Evaluation Protocol

Predictions from both clean and perturbed runs are evaluated against the **same clean gold answers**.

Key principles:
- Evaluation metrics are identical across conditions.
- Perturbed predictions are always compared to their seed-matched clean baseline.
- All reported robustness metrics are expressed as **deltas relative to clean performance**.

This protocol ensures that robustness measurements reflect sensitivity to perturbations rather than changes in evaluation criteria.

---

## Aggregation and Reproducibility

After all runs are completed, results are aggregated using deterministic analysis scripts that:

- Merge experiment-level metrics into unified tables.
- Compute deltas versus clean baselines.
- Reaggregate results by phenotype.
- Validate experiment integrity and baseline coverage.

All aggregation artifacts are derived exclusively from persisted run outputs, ensuring full traceability and reproducibility.

---

## Methodological Guarantees

This methodology ensures that:

- Robustness effects are isolated from training dynamics.
- Perturbations are deterministic and auditable.
- All comparisons are seed-matched and baseline-controlled.
- Phenotype-aware analysis is supported without re-running experiments.

The framework is designed to be extensible to larger datasets, alternative models, and additional perturbation families while preserving methodological consistency.