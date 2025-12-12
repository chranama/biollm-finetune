# Research Goals  
### Biomedical Question Answering — Robustness, Reliability & Failure Analysis Suite

This project extends the user's master's thesis work into a full research suite designed to study **robustness**, **reliability**, **explainability**, and **error behavior** in biomedical question-answering systems built with open-weight large language models (LLMs).  

Instead of focusing on productionization, this repository is structured as a **mini research laboratory**: experiments are controlled, reproducible, and centered around understanding *how* and *why* biomedical LLMs fail under various conditions.

The overarching goal is to showcase the ability to design, implement, analyze, and communicate a rigorous applied-LLM research program using limited hardware but strong methodology.


---

# 1. Research Orientation

Modern biomedical QA systems must operate in environments full of:
- noisy evidence,
- contradictory scientific findings,
- synonym-rich terminology,
- multi-hop reasoning demands,
- varied question types,
- and domain-specific factual constraints.

This project evaluates how well fine-tuned LLMs survive these challenges.  
The aim is not just to measure performance, but to **map the model’s vulnerabilities**, **diagnose reasoning failures**, and **analyze error patterns** both quantitatively and qualitatively.

The research suite is organized around:
- controlled perturbation experiments,
- phenotype-based difficulty analysis,
- interpretability-oriented diagnostics,
- qualitative “error galleries,”
- and comparisons across quantization or fine-tuning variants.


---

# 2. Research Questions

## **RQ1 — How robust are biomedical LLMs to noisy, irrelevant, or contradictory evidence snippets?**

Biomedical text often contains extraneous or inconsistent information.  
This question evaluates how performance degrades when the input is perturbed.

### Experimental manipulations:
- Adding irrelevant biomedical sentences  
- Injecting synthetic contradictions  
- Shuffling snippet order  
- Injecting token-level noise  

### Metrics:
- **Robustness Ratio:** performance_with_noise / performance_clean  
- Absolute metric drops per question type  
- Change in token-probability entropy  
- Error severity index (custom)  


---

## **RQ2 — What categories of questions systematically cause reasoning or factual failures?**

LLM errors are not random — they cluster around certain linguistic or structural features.  
This question identifies “phenotypes” of hard questions.

### Question phenotypes:
- Long questions  
- Long contexts  
- Synonym-heavy answers  
- Multi-hop reasoning  
- Temporal / causal questions  
- Entity-dense answers  
- Multi-answer list questions  

### Metrics:
- Per-phenotype accuracy / F1 / MRR / ROUGE  
- Error clustering and heatmaps  
- Failure rates across BioASQ question types  


---

## **RQ3 — How do robustness failures and phenotype failures interact?**

Some types of questions may be disproportionately vulnerable to noise.  
This question analyzes how perturbations from **RQ1** combine with the difficulty buckets from **RQ2**.

### Metrics:
- Robustness drop per phenotype  
- Phenotype × perturbation interaction plots  
- Fragility ranking of question types and structures  


---

## **RQ4 — How do quantization and fine-tuning choices influence robustness and error profiles?**

Model compression and efficient training are common in biomedical NLP.  
This question evaluates how these choices impact reliability.

### Conditions:
- Base models vs QLoRA-fine-tuned models  
- 4-bit NF4 vs 8-bit vs CPU inference  
- Small-rank vs moderate-rank LoRA adapters  

### Metrics:
- Change in robustness ratio  
- Change in error distributions  
- Latency and memory usage (secondary)  


---

## **RQ5 — What qualitative reasoning failures underlie incorrect predictions?**

This question focuses on explainability through structured error inspection.

### Components:
- Curated error datasets for each question type  
- Side-by-side gold vs predicted answers  
- Perturbation condition metadata  
- Phenotype labels  
- Optional token-level attribution or attention heuristics  

### Deliverables:
- A qualitative “model pathology report”  
- An interactive or notebook-based error browser  


---

# 3. Expected Contributions of This Research Suite

This repository aims to contribute:

1. **A unified framework for robustness testing**  
   Small-scale but rigorous experiments showing how biomedical LLMs fail under noise and contradiction.

2. **A dataset-driven analysis of difficulty phenotypes**  
   Clear evidence of which question families are intrinsically hard for LLMs.

3. **A cross-model comparison of reliability under quantization and efficient fine-tuning**  
   Linking systems decisions to downstream reliability outcomes.

4. **A qualitative catalog of reasoning failures**  
   Human-interpretable examples demonstrating how and why models produce incorrect answers.

5. **Reproducible experiment design**  
   Every experiment is logged, traceable, and understandable in terms of configuration, metrics, and artifacts.

  
---

# 4. What This Enables Next

These research questions and goals establish the foundation for:

- A structured experiment registry  
- A robustness analysis module  
- Phenotype-based evaluation scripts  
- Ablation studies on perturbation strength or LoRA settings  
- Error visualization notebooks  
- A publishable short research paper or blog post  
- Strong portfolio evidence of analysis-driven LLM engineering  


---

# 5. Summary

This research suite is designed to show depth, rigor, and scientific maturity.  
It transforms a finetuning project into a complete analytical workflow that investigates the boundaries of model reliability in biomedical question answering.

The following phases of the project will implement:
- experiment architecture,  
- analysis utilities,  
- perturbation engines,  
- and evaluation dashboards.

Together, they form a compact but powerful **research lab** for biomedical LLM robustness.