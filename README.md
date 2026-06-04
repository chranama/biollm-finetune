# BioLLM-Finetune

BioLLM-Finetune evaluates how biomedical question-answering models behave when
question text is changed in controlled ways. It runs config-defined experiments,
applies deterministic perturbations, scores BioASQ-style answers, and saves
artifacts that make clean-vs-perturbed behavior inspectable.

Biomedical QA asks a model to answer domain-specific medical or biological
questions from structured examples. This project focuses on robustness
evaluation: holding the model and experiment settings fixed while changing
inputs, then measuring whether answer quality changes.

The current artifact set uses a small BioASQ-style sample, fixed seeds, and
deterministic perturbations for local research evaluation. The repository also
includes PEFT adapter fine-tuning support, but the saved public outputs
emphasize inference-only robustness analysis.

## Workflow

```text
Optional training config
  -> BioASQ-style prompt/answer rows
  -> LoRA or CUDA QLoRA adapter
  -> adapter-aware experiment config
  -> clean or perturbed inference
  -> BioASQ-style metrics and reviewable artifacts
```

Saved outputs are listed below so reviewers can inspect the experiment inputs,
metrics, tables, figures, and validation metadata directly.

## Responsibilities

- Load and preprocess BioASQ-style question data
- Run local inference with Hugging Face causal language models
- Fine-tune PEFT adapters with local LoRA and CUDA QLoRA configurations
- Apply deterministic perturbations to questions and snippets
- Score yes/no, factoid, list, and summary answers
- Aggregate clean-vs-perturbed deltas across seeds
- Write run manifests, inputs, predictions, metrics, tables, and figures
- Validate saved artifact references through an evidence manifest

## Repository Layout

```text
configs/              Experiment, inference, and fine-tuning YAML files
data/                 Sample datasets and small reproducible inputs
src/biollm_finetune/  Package code for data, training, inference, eval, and analysis
scripts/              Experiment, aggregation, validation, and reporting entry points
tests/                Behavior-oriented unit and smoke tests
results/              Saved experiment outputs and Phase 4 analysis artifacts
proof/                Evidence manifest validation scripts and metadata
docs/                 Active system documentation
archive/              Historical research notes and earlier code snapshots
```

## Run Locally

The local runbook provides the step-by-step guide for setup, tests, fine-tuning,
single experiment runs, adapter evaluation, aggregation, evidence validation,
output inspection, and cleanup:

- [Runbook](docs/runbook.md)

Some experiment runs can download model weights from Hugging Face if the model
is not already cached.

## Current Outputs

The current robustness artifact set is saved under `results/phase4/`.

Useful entry points:

- `results/phase4/summary.json`
- `results/phase4/experiments.csv`
- `results/phase4/analysis/phase4_findings.md`
- `results/phase4/analysis/phenotype_findings.md`
- `results/phase4/report_artifacts/tables/perturbation_ranking_macro_avg.md`
- `results/phase4/report_artifacts/figures/`
- `proof/evidence_manifest.latest.json`

## Documentation

- [Architecture](docs/architecture.md)
- [Workflow Interface](docs/interface.md)
- [Evaluation Method](docs/evaluation.md)
- [Artifacts](docs/artifacts.md)
- [Testing](docs/testing.md)
- [Runbook](docs/runbook.md)
- [Scope](docs/scope.md)
- [Research Notes](docs/research/)

Research notes from the thesis-oriented documentation set are kept in
`docs/research/`. Older planning notes are kept in `archive/docs/`.

## License

Released under the MIT License.

## Citation

Christopher Anaya (2025)  
LLM Fine-Tuning With Biomedical Open-Source Data  
Master's Thesis in Data Science  
Faculty of Science, University of Lisbon  
Repository: https://github.com/chranama/biollm-finetune
