# Experiment Design  
### Biomedical QA Robustness, Reliability & Failure Analysis Suite

This document defines the experimental framework used in this research suite.  
It specifies what constitutes an experiment, what variations are tested, how the evaluation is structured, and which artifacts are produced.

The goal is to implement reproducible, controlled, scientifically meaningful experiments that investigate the robustness, reliability, and explainability of biomedical question-answering models.

---

## 1. What Is an Experiment?

In this repository, one experiment is defined as:

A single combination of model variant, dataset split, and perturbation condition, executed with a fixed random seed and full configuration snapshot, producing predictions, metrics, and error analysis artifacts.

Each experiment must capture:

- Model variant (architecture, fine-tuning, quantization)  
- Dataset condition (clean or perturbed)  
- Perturbation type (noise, contradiction, shuffle, lexical variation, etc.)  
- Phenotype annotations for the evaluated questions  
- Random seed and configuration state  
- All outputs needed for analysis and reproducibility  

A typical experiment produces an output directory of the form:

results/experiments/
  bioasq_M3-4bit_noise-irrel_seed42/
    config.yaml
    run_metadata.json
    predictions.jsonl
    metrics.json
    phenotype_metrics.json
    robustness.json
    error_samples.jsonl

---

## 2. Datasets and Splits

### 2.1 BioASQ Task B

BioASQ Phase B is the primary evaluation dataset. It includes:

- Yes/No questions  
- Factoid questions  
- List questions  
- Summary questions  

Evaluation runs use:

- Development sets for iteration  
- Held-out test batches (or equivalent custom splits) for final reporting  

### 2.2 Supporting Datasets

BiQA, GO Terms, and DrugBank datasets may be used for:

- Fine-tuning  
- Creating perturbation sources  
- Auxiliary robustness checks  

These do not replace BioASQ as the evaluation benchmark.

---

## 3. Model Variants

A compact and meaningful model lineup:

### 3.1 Core Models

- M1: General LLM (example: Mistral-7B-Instruct)  
- M2: Biomedical pretrained LLM (example: BioMistral-7B)  
- M3: Fine-tuned biomedical LLM using QLoRA  

### 3.2 Quantization Variants

- M3-4bit: QLoRA, 4-bit NF4  
- M3-8bit: Optional 8-bit variant  
- M3-fp16-CPU: Optional small-scale CPU inference variant  

---

## 4. Experimental Factors

Experiments systematically vary perturbations, phenotypes, and model types.

---

### 4.1 Factor A — Perturbation Type

Each BioASQ question is evaluated under these conditions:

- A0: Clean (original snippets)  
- A1: Irrelevant noise injection  
- A2: Contradictory sentence injection  
- A3: Snippet reordering  
- A4: Lexical noise or token substitutions  

Minimal core subset for compute efficiency: A0, A1, A3.

---

### 4.2 Factor B — Phenotype Tags

Each question is labeled with one or more difficulty phenotypes:

- B1: Long question  
- B2: Long context  
- B3: Synonym-heavy answer  
- B4: Multi-hop reasoning  
- B5: Temporal or causal structure  
- B6: Entity-dense answer  
- B7: Multi-answer list  

These tags support failure-mode analysis.

---

### 4.3 Factor C — Model Variant

- C1: M1 general model  
- C2: M2 biomedical pretrained model  
- C3: M3-4bit fine-tuned model  
- Optional: C4 (8-bit), C5 (fp16 CPU subset)  

Most robustness experiments use C3.

---

## 5. Metrics

### 5.1 BioASQ Task Metrics

- Yes/No: accuracy, precision, recall, F1  
- Factoid: exact match, lenient match, MRR  
- List: F1 and exact match  
- Summary: ROUGE-L, ROUGE-2  

Metrics are computed globally, by question type, by phenotype, and by perturbation.

### 5.2 Robustness Metrics

For any metric M:

- M_clean = score on clean input  
- M_pert = score on perturbed input  

Derived measures:

- Robustness ratio = M_pert divided by M_clean  
- Absolute drop = M_pert minus M_clean  

Both are computed per question, per phenotype, per perturbation, and per model.

### 5.3 Statistical Confidence

Bootstrap resampling (for example, one thousand resamples) is recommended for:

- Differences between clean and perturbed performance  
- Differences between model variants  
- Robustness estimates  

---

## 6. Experimental Matrix

### Phase A — Baseline Characterization

Experiments:

- (C1, A0)  
- (C2, A0)  
- (C3, A0)  

Outputs:

- Baseline metrics  
- Phenotype-level performance  
- Seed error catalog  

Supports research question on phenotype-based failures.

---

### Phase B — Robustness to Noise

Experiments for the fine-tuned model:

- (C3, A0)  
- (C3, A1)  
- (C3, A3)  

Optional comparison:

- (C2, A1)  

Supports robustness and phenotype × perturbation interaction analysis.

---

### Phase C — Quantization vs Robustness

Experiments:

- (M3-4bit, A0 A1 A3)  
- (M3-8bit, A0 A1 A3)  
- (M3-fp16 CPU-small, A0 A1)  

Supports analysis of reliability under compression.

---

### Phase D — Advanced Perturbations

Optional deeper experiments:

- (C3, A2) contradiction  
- (C3, A4) lexical noise  

---

## 7. Error Analysis and Explainability

### 7.1 Error Bucketing

Each incorrect prediction is annotated with:

- Question type  
- Phenotype tags  
- Perturbation condition  
- Error type (hallucination, polarity error, omission, misreading, etc.)

### 7.2 Error Sampling

For each (model, perturbation, phenotype):

- Select worst examples  
- Select borderline examples  
- Select representative random examples  

Stored in:

results/error_catalog/<experiment_id>.jsonl

### 7.3 Error Browser

A notebook or lightweight application may later allow interactive exploration of:

- Question  
- Snippets  
- Prediction  
- Gold answer  
- Notes on reasoning failures  

---

## 8. Reproducibility

Each experiment directory contains:

- Fully resolved configuration file  
- Metadata including git commit, seed, model identifiers  
- Raw predictions  
- Core metrics  
- Phenotype-level metrics  
- Robustness metrics  
- Selected error samples  

This ensures complete reproducibility and traceability.

---

## 9. Summary

This experiment design establishes:

- A robust framework for perturbation-based testing  
- A phenotype taxonomy for discovering systematic failures  
- A clear model comparison structure  
- A reproducible experiment-output layout  
- Integrated qualitative and quantitative error analysis  

Together, these practices form the backbone of a rigorous research suite for biomedical LLM robustness.