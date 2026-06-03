# Experimental Design

This document specifies the **frozen experimental design** for the BioLLM fine-tuning and robustness analysis project. The design is finalized prior to large-scale experimentation and is explicitly aligned with the frozen research questions defined in `research_goals.md`.

The goal of this experimental design is to ensure that all empirical results are **interpretable, reproducible, and directly attributable to controlled factors**, rather than artifacts of ad hoc exploration.

---

## Design Principles

The experimental design follows five core principles:

1. **Alignment with research questions**  
   Every experiment is mapped to at least one frozen research question.

2. **Controlled perturbations**  
   Perturbations are realistic, parameterized, and applied in isolation unless explicitly stated otherwise.

3. **Minimal but sufficient grid**  
   The experiment grid is intentionally small to avoid combinatorial explosion while remaining expressive enough to reveal systematic behavior.

4. **Hardware realism**  
   All experiments are feasible on a personal computer using a macOS MPS backend.

5. **Pre-registration mindset**  
   The experiment grid is frozen before execution to minimize post-hoc bias.

---

## Dataset Selection

### Primary Dataset

- **BioASQ (TINY subset)**

Rationale:
- Representative mix of biomedical question types
- Supports yes/no, factoid, list, and summary questions
- Small enough for rapid iteration and reproducibility on local hardware

No additional datasets are included at this stage to preserve focus and interpretability.

---

## Model and Runtime Configuration

- Single model configuration
- macOS MPS backend
- Full-precision inference
- No 4-bit or 8-bit quantization (hardware constraint)

The objective is behavioral characterization rather than model comparison.

---

## Random Seeds

Two seed regimes are used:

- **Core experiments:** seed = 42  
- **Reliability experiments:** seeds = {13, 42, 97}

Multi-seed evaluation is limited to selected perturbations to balance rigor and computational cost.

---

## Perturbation Families

### Baseline

- `clean`

### Control Perturbation

- `shuffle_snippets`  
  Randomizes snippet order to control for context length effects without introducing new content.

---

### Lexical Noise (RQ1)

Surface-form perturbations that preserve semantic content:

- `lexical_noise` (low intensity)
- `lexical_noise_medium`
- `lexical_noise_heavy`

These perturbations test graceful versus brittle degradation under increasing surface corruption.

---

### Irrelevant Biomedical Noise (RQ2)

In-domain distractor perturbations using a balanced PubMed-derived corpus:

- `irrelevant_noise`
- `irrelevant_noise_heavy`

These perturbations test the model’s ability to filter irrelevant biomedical evidence beyond simple context-length effects.

---

### Contradiction (RQ4)

- `contradiction`

This perturbation injects explicit evidence contradicting the gold answer, primarily targeting yes/no questions. The analysis is behavioral rather than logical.

---

## Experiment Blocks

### Block 0 — Smoke Validation

Purpose:
- Validate end-to-end execution
- Confirm artifact generation
- Sanity-check robustness and stability outputs

Configuration:
- Dataset: BioASQ TINY
- Seed: 42
- Perturbations:
  - clean
  - lexical_noise_medium
  - contradiction

---

### Block 1 — Robustness Characterization

Purpose:
- Address RQ1, RQ2, and RQ4
- Measure clean versus perturbed degradation
- Enable robustness curves across perturbation intensity

Configuration:
- Dataset: BioASQ TINY
- Seed: 42
- Perturbations:
  - clean
  - shuffle_snippets
  - lexical_noise
  - lexical_noise_medium
  - lexical_noise_heavy
  - irrelevant_noise
  - irrelevant_noise_heavy
  - contradiction

---

### Block 2 — Reliability Analysis

Purpose:
- Address RQ3
- Measure stability and correctness flips across seeds

Configuration:
- Dataset: BioASQ TINY
- Seeds: {13, 42, 97}
- Perturbations:
  - clean
  - lexical_noise_heavy
  - irrelevant_noise_heavy
  - contradiction

---

## Measurements and Outputs

For each experiment run, the following artifacts are produced:

- Predictions
- Task-appropriate evaluation metrics
- Clean versus perturbed deltas
- Relative percentage drops
- Prediction stability statistics
- Correctness flip counts
- Phenotype-conditioned aggregates

All artifacts are persisted per experiment to ensure reproducibility and auditability.

---

## Phenotype Conditioning

All robustness and reliability analyses are additionally conditioned on interpretable input phenotypes:

- long questions
- long contexts
- multi-answer list questions

This enables structured qualitative error analysis and supports explainability-focused interpretation of failures.

---

## Scope and Limitations

- The design characterizes a single model configuration.
- No cross-model or cross-dataset generalization claims are made.
- Contradiction analysis is empirical and behavioral.
- Interaction effects between perturbations are out of scope for this phase.

---

## Status

- Research questions: frozen  
- Experiment grid: frozen  
- Perturbations, phenotypes, and robustness machinery: complete  

The project is ready to proceed to Phase 4 execution without further changes to experimental design.