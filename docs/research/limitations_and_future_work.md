## Limitations

While Phase 4 establishes a rigorous and reproducible robustness evaluation framework, several limitations should be acknowledged.

### Dataset Scope

All robustness experiments were conducted on **bioasq_TINY**, a deliberately small and controlled subset of BioASQ.  
While this choice enables rapid iteration, deterministic debugging, and fine-grained inspection of perturbation effects, it limits the statistical power and generalizability of the results.

In particular:
- Small sample sizes amplify variance in certain metrics (e.g., list questions).
- Some perturbations may appear benign or beneficial due to dataset idiosyncrasies rather than genuine robustness.

As a result, conclusions should be interpreted as **diagnostic rather than definitive**.

---

### Model Scale and Training Regime

The evaluated model, *TinyLlama-1.1B-Chat*, represents a lightweight, general-purpose LLM rather than a domain-specialized biomedical model.

Additionally:
- All experiments are **inference-only**.
- No fine-tuning, instruction adaptation, or retrieval augmentation is applied.

This isolates robustness effects cleanly, but it does not capture how robustness might change under:
- domain-specific fine-tuning,
- larger model scales,
- or knowledge-augmented architectures.

---

### Perturbation Design

The perturbations implemented in Phase 4 are **synthetic and targeted**, designed to probe specific failure modes:
lexical noise, irrelevant context, structural disruption, and logical contradiction.

While these perturbations are deterministic and interpretable, they do not cover:
- real-world biomedical noise (OCR artifacts, clinical shorthand),
- adversarial paraphrasing,
- or distributional shifts across datasets or time.

Thus, robustness is evaluated along **controlled axes**, not under full naturalistic conditions.

---

### Metric Sensitivity

Aggregate metrics such as macro averages and ROUGE scores may obscure finer-grained behavior.

For example:
- Small positive deltas in certain perturbations may reflect answer format changes rather than genuine reasoning improvements.
- Phenotype-specific effects may be masked when averaged across question types.

Although phenotype-aware reaggregation mitigates this to some extent, deeper qualitative analysis would be required to fully characterize failure modes.

---

## Future Work

Phase 4 lays a strong foundation for several natural extensions, both technical and analytical.

### Scaling to Full BioASQ and Additional Datasets

The most immediate extension is to scale the robustness framework to:
- the full BioASQ training and test sets,
- or complementary biomedical QA datasets (e.g., PubMedQA, MedMCQA).

This would allow:
- statistically robust effect estimation,
- cross-dataset robustness comparison,
- and validation of observed trends.

---

### Model Comparisons and Training Effects

Future experiments could systematically compare:
- different model scales,
- biomedical-specialized models (e.g., BioMistral, BioGPT),
- and fine-tuned versus inference-only variants.

This would enable analysis of:
- whether robustness improves with domain adaptation,
- which perturbations are mitigated by fine-tuning,
- and which remain fundamental weaknesses.

---

### Knowledge-Infused and Retrieval-Augmented QA

Given the biomedical setting, a natural next step is to evaluate robustness under:
- retrieval-augmented generation (RAG),
- structured knowledge injection,
- or ontology-aware prompting.

The existing pipeline is already compatible with such extensions, as perturbations operate strictly at the input level and evaluation remains unchanged.

---

### Phenotype Expansion and Error Taxonomy

The current phenotype schema focuses on input length and answer structure.

Future work could expand this to include:
- reasoning complexity,
- numerical reasoning,
- multi-entity linking requirements,
- or answer ambiguity.

Combined with qualitative error analysis, this would enable a richer taxonomy of robustness failures.

---

### Longitudinal and Stability Analysis

Finally, robustness could be studied longitudinally:
- across multiple random initializations,
- across model updates,
- or across time-shifted datasets.

This would move robustness analysis from a static snapshot to a **stability-aware evaluation paradigm**.

---

## Summary

Phase 4 shows that robustness evaluation can be:
- deterministic,
- reproducible,
- phenotype-aware,
- and analytically meaningful without requiring additional training.

The framework developed here is intentionally modular, enabling future extensions in scale, model complexity, and evaluation depth.

Rather than providing final answers, this phase establishes a **methodologically sound foundation** for robustness-focused biomedical QA research.
