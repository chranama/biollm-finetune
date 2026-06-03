# Results and Discussion

This section presents the results of Phase 4 robustness experiments, analyzing the behavior of a biomedical question answering model under controlled input perturbations. The goal is not to maximize task performance, but to characterize robustness, sensitivity, and stability across perturbation types, evaluation metrics, and question phenotypes.

All results are derived from inference-only experiments using fixed gold answers. Performance changes are reported as deltas relative to clean baselines executed with identical seeds.

---

## 4.1 Overall Robustness to Input Perturbations

Table 1 summarizes the mean change in macro-averaged performance (Δ Macro Avg) for each perturbation type relative to clean baselines, aggregated across seeds. Figure 1 provides a visual ranking of perturbations by their average impact.

Structural perturbations that disrupt input organization, particularly **shuffle_snippets**, produce the largest performance degradation. Heavy contextual noise (**irrelevant_noise_heavy**) also results in consistent negative deltas, indicating sensitivity to the presence of unrelated context.

In contrast, **contradiction** perturbations exhibit near-zero impact on macro performance, suggesting that the model often disregards logically conflicting information when answering questions. Notably, **lexical noise** perturbations yield slightly positive average deltas. These gains are small and should not be interpreted as true improvements, but they indicate a degree of robustness to surface-level lexical variation.

Overall, these results suggest that the model is more sensitive to structural and contextual disruptions than to lexical or logical inconsistencies.

---

## 4.2 Metric-Specific Sensitivity Patterns

While macro-averaged performance provides a high-level summary, robustness effects are not uniform across evaluation metrics. Figure 2 illustrates mean performance deltas across individual metrics for the most impactful perturbations.

Yes/No accuracy remains largely invariant across perturbations, reflecting the relative simplicity and binary nature of these questions. In contrast, **factoid** and **list** questions show greater sensitivity, particularly under contextual noise and structural perturbations. Summary questions, evaluated using ROUGE-L, display distinct sensitivity patterns that differ from extractive metrics.

These results demonstrate that robustness cannot be fully characterized by a single aggregate score. Different question types and evaluation metrics respond differently to the same perturbation, underscoring the importance of metric-level analysis in robustness evaluation.

---

## 4.3 Phenotype-Aware Robustness Effects

To further refine robustness analysis, experiments were stratified by question phenotype. Table 2 reports mean macro-averaged deltas by phenotype and perturbation, while Figure 3 presents a heatmap visualization of these effects.

Phenotype-aware analysis reveals systematic interactions between input characteristics and perturbation types. Questions with **multi-answer lists** are particularly sensitive to irrelevant contextual noise, showing substantial degradation under both standard and heavy noise conditions. Conversely, questions with **long context** exhibit greater tolerance to lexical perturbations, suggesting that broader contextual grounding may buffer surface-level noise.

Importantly, some perturbations exhibit opposing effects depending on phenotype, an effect that is masked in aggregate analyses. These findings demonstrate that robustness is not a global property of the model, but rather a conditional one that depends on input structure and informational demands.

---

## 4.4 Stability Across Random Seeds

All perturbation experiments were repeated across three random seeds to assess stability. Aggregated seed-level analysis indicates that variance across seeds is low relative to mean perturbation effects.

Observed robustness trends are consistent across seeds, supporting the conclusion that the reported effects reflect systematic model behavior rather than stochastic variability introduced by perturbation sampling or inference nondeterminism.

---

## 4.5 Summary of Robustness Findings

The Phase 4 experiments support several key conclusions:

- Structural and heavy contextual perturbations consistently degrade performance.
- Lexical perturbations are largely tolerated and may even produce marginal positive deltas.
- Logical contradictions have minimal impact on overall performance.
- Robustness effects vary substantially across evaluation metrics.
- Question phenotype strongly mediates sensitivity to perturbations.
- Observed robustness patterns are stable across random seeds.

Together, these results demonstrate the value of phenotype-aware and metric-specific robustness evaluation. They also highlight limitations of aggregate metrics for capturing nuanced model behavior under input perturbations.

These findings motivate future work on robustness-aware model design, targeted data augmentation, and more granular evaluation frameworks for biomedical question answering systems.

---