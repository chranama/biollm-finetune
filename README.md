# BioLLM-Finetune

BioLLM-Finetune is a Python evaluation workflow for biomedical question answering.
It runs config-defined experiments, applies deterministic input perturbations,
scores BioASQ-style outputs, and writes artifacts for comparing clean and
perturbed model behavior.

Biomedical question answering evaluates whether a model can answer biomedical
questions from structured examples. Fine-tuning adapts a model with
task-specific training data, while inference-only evaluation runs a configured
model without changing its weights. This workflow changes question text in
controlled ways and compares whether answer quality changes under those
perturbations.

The project is intended for local research and evaluation workflows. It is not a
production inference service, a clinical decision system, or a broad benchmark of
biomedical language models.

## Workflow

```text
YAML config
  -> BioASQ-style input data
  -> optional deterministic perturbation
  -> Hugging Face model inference or adapter-based fine-tuning
  -> BioASQ-style metrics
  -> run artifacts, aggregate tables, and validation metadata
```

The current saved artifact set focuses on inference-only robustness evaluation
for a small BioASQ sample using fixed seeds and deterministic perturbations.

## Responsibilities

- Load and preprocess BioASQ-style question data
- Run local inference with Hugging Face causal language models
- Support LoRA and QLoRA adapter fine-tuning workflows
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

The local runbook provides the step-by-step guide for setup, tests, a single
experiment run, aggregation, evidence validation, output inspection, and cleanup:

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
