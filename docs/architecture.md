# Architecture

BioLLM-Finetune is organized as a local experiment workflow. The main boundary is
the experiment configuration: a YAML file specifies the dataset, model, runtime,
seed, perturbation, and output directory for a run.

## Component Flow

```text
configs/*.yaml
  -> scripts/run_experiment.py
      -> biollm_finetune.data
      -> biollm_finetune.analysis.perturbations
      -> biollm_finetune.inference.generate
      -> biollm_finetune.eval.metrics
      -> biollm_finetune.analysis.phenotypes
      -> results/experiments/<run>/
  -> aggregation and reporting scripts
      -> results/phase4/
      -> proof/evidence_manifest.latest.json
```

## Main Modules

- `biollm_finetune.data`: JSON and JSONL loading, preprocessing, and sampling.
- `biollm_finetune.training`: LoRA and QLoRA fine-tuning entry point.
- `biollm_finetune.inference`: Hugging Face generation from config and input
  files.
- `biollm_finetune.eval`: BioASQ-style scoring and postprocessing.
- `biollm_finetune.analysis`: perturbations, phenotype tagging, robustness
  aggregation, plotting, and run registration.
- `biollm_finetune.utils`: config parsing, device resolution, logging, and seed
  setup.

## Runtime Boundaries

Model execution is isolated behind `biollm_finetune.inference.generate`. The
experiment runner calls inference as a subprocess so each run writes explicit
inputs, outputs, metrics, and metadata to disk.

Configuration and runtime setup are handled separately from scoring and
aggregation:

- YAML files define experiment inputs and runtime settings.
- `scripts/run_experiment.py` executes a single configured run.
- Aggregation scripts read saved run directories instead of rerunning inference.
- Validation scripts check artifact presence and run consistency.

## Artifact Boundary

The system treats generated files as reviewable experiment state. A run directory
contains the exact inference inputs, predictions, metrics, phenotype tags, and
manifest for that run. Phase-level scripts then produce aggregate CSV, JSON,
Markdown, and figure outputs under `results/phase4/`.

## Design Tradeoffs

The repository favors explicit filesystem artifacts over a database or service
layer. That keeps local experiments inspectable and reproducible, but it also
means this is not a multi-user experiment tracking service.

The perturbation implementation is centralized in
`src/biollm_finetune/analysis/perturbations.py`. That makes the currently
supported perturbations easy to find, but the file is a natural candidate for
splitting by perturbation family if the set grows.
