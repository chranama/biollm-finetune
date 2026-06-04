# Evaluation Method

The current evaluation workflow compares clean BioASQ-style inputs with
deterministically perturbed versions of the same inputs. The goal is to inspect
how answer quality changes when the input text is modified in controlled ways.

## Inputs

The active experiment configs use small BioASQ-style JSONL inputs. Each example
contains a question, optional snippets, a question type, and gold answer fields.

Supported question types include:

- `yesno`
- `factoid`
- `list`
- `summary`

## Perturbations

The perturbation layer supports:

- `clean`
- `shuffle_snippets`
- `irrelevant_noise`
- `irrelevant_noise_heavy`
- `lexical_noise`
- `lexical_noise_medium`
- `lexical_noise_heavy`
- `contradiction`
- `contradiction_prepend`

Perturbations are deterministic when the run seed is fixed. The experiment runner
writes the exact inputs used for inference so clean and perturbed examples can be
inspected after the run.

## Concrete Example

A clean yes/no input can ask:

```text
Is daridorexant effective for insomnia?
```

The clean run passes the original question and snippets to inference. A
perturbed run keeps the same gold answer but changes the input condition, such as
by shuffling snippets, adding irrelevant biomedical text, applying lexical noise,
or appending a contradiction-style statement.

The comparison asks whether the model still returns an answer that scores
correctly against the clean gold label. At the run level, the project compares
the clean score with the perturbed score for the same dataset, model, runtime,
and seed.

## Metrics

Metrics are selected by question type:

- yes/no: accuracy
- factoid: exact match and F1
- list: precision, recall, and F1
- summary: ROUGE-L

The aggregate `macro_avg` is the mean of the available per-type scores for the
run. It is useful for comparing perturbation effects across a small mixed
question set, but it should not be read as a broad biomedical benchmark score.

## Clean-Vs-Perturbed Comparison

The Phase 4 artifact set compares perturbed runs against clean baselines matched
by dataset, runtime, model, and seed. Delta tables are written under
`results/phase4/deltas/`.

The saved Phase 4 configs use seeds `13`, `42`, and `97` for the main configured
comparison set.

## Phenotype Analysis

The analysis layer tags examples with linguistic and semantic phenotypes, then
aggregates deltas by phenotype. These outputs are intended to help inspect where
perturbations affect model behavior most strongly.

Primary phenotype outputs:

- `results/phase4/phenotypes/phenotype_runs.csv`
- `results/phase4/phenotypes/phenotype_deltas_vs_clean.csv`
- `results/phase4/analysis/phenotype_findings.md`

## Interpretation Limits

The saved outputs are based on a small reproducible sample and a lightweight open
model configuration. They are useful for checking the evaluation workflow and
for inspecting robustness behavior, but they are not a comprehensive comparison
of biomedical models.
