# Workflow Interface

BioLLM-Finetune is a file-and-config workflow rather than a web API. The public
interface is the experiment config, the BioASQ-style input files, and the output
directories written under `results/`.

The runbook contains the runnable command sequences. This document describes
what those commands consume and what they produce.

## Experiment Config

Each experiment is driven by a YAML config. The config names the dataset, model,
runtime, seed, perturbation, inference settings, and output location for one
run.

Concrete example:

- `configs/experiments/bioasq_TINY_mps_fp32_clean_seed42.yaml`

Important fields:

| Field | Purpose |
|---|---|
| `name` | Run id and output directory name under `results/experiments/`. |
| `seed` | Seed used for deterministic perturbation and inference setup. |
| `perturbation` | Input condition, such as `clean`, `shuffle_snippets`, `lexical_noise_heavy`, or `contradiction`. |
| `output_dir` | Root directory for per-run outputs. |
| `dataset.name` | Dataset label used in manifests and aggregate tables. |
| `dataset.path` | BioASQ-style JSONL input file. |
| `dataset.gold_file` | Gold answer file used for scoring. |
| `dataset.task` | Evaluation task, currently `bioasq` for the active configs. |
| `runtime.inference_config` | Inference config consumed by the generation entry point. |
| `runtime.device` and `runtime.dtype` | Local execution target and numeric type. |
| `model.name` and `model.path` | Model label and Hugging Face model path. |
| `model.adapter` or `model.adapter_output_dir` | Optional adapter path for adapter-based inference. |
| `data.include_snippets` | Whether snippets are included in the model input. |
| `inference.*` | Batch size and generation controls. |
| `training` | Not used by the experiment runner; training uses its own config shape. |

Minimal config shape:

```yaml
name: bioasq_TINY_mps_fp32_clean_seed42
seed: 42
perturbation: clean
output_dir: results/experiments
dataset:
  name: bioasq_TINY
  path: data/samples/integration_questions.jsonl
  gold_file: data/samples/integration_gold.json
  task: bioasq
runtime:
  device: mps
  dtype: float32
  inference_config: configs/inference_tiny.yaml
model:
  name: tinyllama-1.1b-chat
  path: TinyLlama/TinyLlama-1.1B-Chat-v1.0
inference:
  batch_size: 1
  max_new_tokens: 128
```

## Training Config

Fine-tuning is driven by a separate YAML shape consumed by
`biollm_finetune.training.finetune`. It trains a Hugging Face causal language
model on BioASQ-style prompt/answer rows and writes a final model or PEFT
adapter directory.

Concrete examples:

- `configs/finetune_tiny.yaml`: local LoRA smoke configuration.
- `configs/finetune.yaml`: CUDA QLoRA configuration for a larger model and
  private training data.

Important fields:

| Field | Purpose |
|---|---|
| `model.base_model` | Hugging Face model id or local model path used for training. |
| `model.adapter_output_dir` | Final adapter/model directory used later by inference. |
| `model.use_peft` | Enables PEFT adapter training when true. |
| `model.load_4bit` and `model.load_8bit` | CUDA-only quantized loading flags. |
| `model.lora_*` | LoRA rank, alpha, dropout, and target modules. |
| `data.train_file` | BioASQ-style JSONL prompt/answer training rows. |
| `data.include_snippets` | Whether snippets are included in the supervised prompt. |
| `data.question_field` and `data.answer_field` | Input and target fields read from each row. |
| `training.output_dir` | Trainer checkpoint and run manifest directory. |
| `training.max_steps`, `num_train_epochs`, and batch fields | Training length and batch sizing. |
| `system.device_map` | Local execution target, such as `auto`, `mps`, `cpu`, or `cuda`. |

Minimal local shape:

```yaml
model:
  base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  adapter_output_dir: results/ckpts/tiny_adapter
  use_peft: true
  load_4bit: false
  load_8bit: false
data:
  train_file: data/samples/smoke_train.jsonl
  include_snippets: true
  question_field: body
  answer_field: ideal_answer
training:
  output_dir: results/ckpts/tiny_run
  max_steps: 30
  per_device_train_batch_size: 1
system:
  device_map: auto
```

After training, set an experiment config's `model.adapter_output_dir` to the
saved adapter path and run `scripts/run_experiment.py` to evaluate the adapter
with the same metric and artifact workflow used for inference-only runs.

## Input Data

The active experiment configs use BioASQ-style JSONL. Each line is one question
object.

Concrete example:

- `data/samples/integration_questions.jsonl`

Common fields:

| Field | Required | Purpose |
|---|---:|---|
| `id` | yes | Stable question id used for matching and inspection. |
| `body` | yes | Biomedical question text. |
| `type` | yes | Question type: `yesno`, `factoid`, `list`, or `summary`. |
| `ideal_answer` | yes | Reference natural-language answer. |
| `exact_answer` | sometimes | Exact label or answer used by yes/no, factoid, and list scoring. |
| `documents` | no | Source document references. |
| `snippets` | no | Evidence snippets included in prompts when configured. |
| `concepts` | no | Optional biomedical concept references. |

Example input row:

```json
{
  "id": "64041e97201352f04a00001e",
  "body": "Is daridorexant effective for insomnia?",
  "type": "yesno",
  "ideal_answer": [
    "Yes. Daridorexant ... being developed for the treatment of insomnia."
  ],
  "exact_answer": "yes",
  "snippets": [
    {
      "text": "Daridorexant ... is ... being developed for the treatment of insomnia.",
      "document": "http://www.ncbi.nlm.nih.gov/pubmed/35298826"
    }
  ]
}
```

The gold file keeps BioASQ-style reference answers. The active sample gold file
is `data/samples/integration_gold.json`.

## Runner Boundary

The single-run entry point is `scripts/run_experiment.py`. It consumes one YAML
config, loads clean input examples, applies the configured perturbation, writes
the exact inference inputs, runs generation, scores predictions against the
clean gold labels, and writes run metadata.

The runner intentionally persists intermediate files so a reviewer can inspect
what the model actually saw and how the run was scored.

## Per-Run Outputs

Each run writes a directory under:

```text
results/experiments/<experiment-name>/
```

Current run files include:

| File | Purpose |
|---|---|
| `inputs.jsonl` | Exact examples passed to model inference after perturbation. |
| `predictions.jsonl` | Model outputs written by the generation step. |
| `metrics.json` | BioASQ-style scores for the run. |
| `phenotypes.json` | Example-level phenotype tags and phenotype definitions. |
| `run_metadata.json` | Registry metadata used by aggregation and inspection helpers. |
| `manifest.json` | Run metadata including config path, dataset, model, seed, perturbation, counts, task, and timestamps. |
| `stability.json` | Optional stability output when robustness settings are enabled. |

## Phase-Level Outputs

Aggregation scripts read saved run directories and write phase-level outputs
under:

```text
results/phase4/
```

Useful outputs:

| File or directory | Purpose |
|---|---|
| `summary.json` | Grouped summary of the current artifact set. |
| `experiments.csv` | Flat table of run metadata and metrics. |
| `deltas/deltas_long.csv` | Clean-vs-perturbed deltas in long format. |
| `deltas/deltas_wide.csv` | Clean-vs-perturbed deltas in wide format. |
| `deltas/deltas_summary.json` | Aggregate delta summary. |
| `analysis/phase4_findings.md` | Human-readable robustness findings. |
| `analysis/phenotype_findings.md` | Human-readable phenotype findings. |
| `report_artifacts/tables/` | Markdown and CSV tables for reporting. |
| `report_artifacts/figures/` | Generated figures for delta and phenotype views. |

## Validation Outputs

The evidence manifest records the saved artifact set that should exist for the
current public repo state:

- `proof/evidence_manifest.latest.json`
- `proof/evidence_contract.schema.json`
- `proof/proof_points.latest.md`

The runbook describes how to validate the manifest locally.

## Interface Limits

The workflow interface is local and filesystem-based. It is not a network API,
an experiment tracking service, or a clinical QA product. The saved artifacts
show a bounded, reproducible evaluation workflow on a small sample.
