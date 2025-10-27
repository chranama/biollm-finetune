# BioLLM-Finetune

**BioLLM-Finetune** is a modular system for fine-tuning and evaluating Large Language Models (LLMs) on **Biomedical Question Answering (QA)** tasks.  
It was developed as part of a Master’s thesis in Data Science and builds upon experiments conducted with the BioASQ Challenge (Task B) and related biomedical datasets.

---

## 🧠 Overview

This project provides an end-to-end pipeline for **Biomedical QA** using open-weight LLMs such as Mistral and TinyLlama.  
It includes all stages of the lifecycle: data preparation, fine-tuning, inference, postprocessing, and evaluation.

| Component | Description |
|------------|--------------|
| `biollm_finetune.training.finetune` | Fine-tunes an instruction-tuned LLM using LoRA/QLoRA adapters on biomedical QA data. |
| `biollm_finetune.inference.generate` | Generates answers to biomedical questions from a model or adapter. |
| `biollm_finetune.eval.postprocess` | Cleans and formats raw model outputs into BioASQ-style answers. |
| `biollm_finetune.eval.metrics` | Computes BioASQ-style evaluation metrics (e.g., accuracy, MRR, ROUGE). |
| `biollm_finetune.data.preprocess` | Processes and normalizes multiple biomedical datasets (BioASQ, BiQA, GO, DrugBank). |
| `biollm_finetune.data.data_sampling` | Produces small, balanced subsets for testing and CI. |

---

## 🗃️ Archival Provenance

The repository preserves its development history for transparency and reproducibility.  
Two subdirectories under `archive/` contain the **original local and server codebases** used during early experimentation.

| Subdirectory | Description |
|---------------|-------------|
| `archive/local_2024/` | Original local development environment used for initial prototype experiments and debugging. |
| `archive/server_2024/` | Original remote GPU server code used for BioASQ fine-tuning and model evaluation. |

These directories are retained **for historical and academic documentation only** and are not part of the current executable system.  
All active modules are under `src/bioasq_llm/`.

---

## 🧩 Repository structure

~~~text
biollm-finetune/
├── archive/                # Historical local/server directories (non-active)
│   ├── local_2024/
│   ├── server_2024/
├── configs/                # YAML configs for fine-tuning and inference
├── data/
│   └── samples/            # Small, reproducible datasets for testing
├── src/biollm_finetune/    # Core Python package
├── tests/                  # Unit and integration tests
├── scripts/                # Docker entrypoint scripts
├── results/                # Model checkpoints and generated outputs
├── pyproject.toml          # Build metadata, console scripts
├── requirements.txt        # Direct dependencies
└── README.md               # You are here
~~~

---

## ⚙️ Installation

### 1️⃣ Local setup (Mac or Linux)
~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
~~~

If you prefer requirements:
~~~bash
pip install -r requirements.txt
~~~

---

## 🍏 Local Mac Flow (MPS / CPU)

This project can run entirely **locally on macOS** using the **Metal Performance Shaders (MPS)** backend.  
The `finetune_tiny.yaml` and `inference_tiny.yaml` configs are specifically designed for this environment — using **TinyLlama (1.1B)** with small datasets and float32 precision.

> 💡 These “tiny” configs are for functional validation (10–30 steps), not benchmarking.

### ⚙️ Environment setup
~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export PYTORCH_ENABLE_MPS_FALLBACK=1
~~~

### 🧠 Inference (tiny model)
~~~bash
bioasq-generate \
  --config configs/inference_tiny.yaml \
  --input data/samples/integration_questions.jsonl \
  --out results/generated/integration_answers.jsonl

bioasq-postprocess \
  --inputs results/generated/integration_answers.jsonl \
  --outdir results/processed --format jsonl

bioasq-metrics \
  --pred results/generated/integration_answers.jsonl \
  --gold data/samples/integration_gold.json \
  --out results/metrics/metrics.json
~~~

### 🔥 Tiny smoke fine-tuning
~~~bash
bioasq-finetune --config configs/finetune_tiny.yaml
~~~

This runs a short (~10-step) fine-tuning loop with **TinyLlama**, verifying end-to-end functionality.

### 📂 Result artifacts

| Directory | Description |
|------------|--------------|
| `results/ckpts/tiny_run/` | LoRA adapter weights and tokenizer |
| `results/generated/` | Model predictions |
| `results/metrics/` | Evaluation metrics (optional) |

### ⚠️ Notes for Mac users

| Issue | Why | Mitigation |
|------|-----|------------|
| `grad_norm: nan` | MPS instability | Safe loss guard + gradient clipping already enabled |
| NaNs at sampling | Sampling on tiny adapters | Use `do_sample: false` (greedy) |
| Slow throughput | No fused CUDA kernels | Keep `max_length ≤ 384`, small batch size |
| `pin_memory` warning | Not on MPS | Ignore |
| Truncated outputs | Short decode | Increase `max_new_tokens` |

#### Optional: CPU-only fallback
~~~yaml
# in your config
system:
  device_map: "cpu"
training:
  max_steps: 30
  learning_rate: 5e-5
~~~
~~~bash
python -m bioasq_llm.training.finetune --config configs/finetune_tiny.yaml
~~~

---

## 🧮 Testing

Run all unit and integration tests:
~~~bash
pytest -q
~~~

Fast tests run on CPU; model-dependent tests are marked `@pytest.mark.slow`.

---

## 🧷 Reproducibility and Data Sample Generation

All test/smoke datasets are balanced subsets of BioASQ (`training12b_new.json`), generated via `bioasq-sample`.

Stored under:
~~~text
data/samples/
~~~

Each dataset exports:
- `*_questions.jsonl` — model input questions  
- `*_gold.json` — reference answers

### 🧪 1. Unit Test Sample
~~~bash
bioasq-sample \
  --inputs data/training12b_new.json \
  --out-questions data/samples/unit_questions.jsonl \
  --out-gold data/samples/unit_gold.json \
  --per-type 1 \
  --seed 7
~~~

### 🔗 2. Integration Test Sample
~~~bash
bioasq-sample \
  --inputs data/training12b_new.json \
  --out-questions data/samples/integration_questions.jsonl \
  --out-gold data/samples/integration_gold.json \
  --per-type 6 \
  --seed 42
~~~

### 🔥 3. Smoke Fine-Tuning Sample
~~~bash
bioasq-sample \
  --inputs data/training12b_new.json \
  --out-questions data/samples/smoke_train.jsonl \
  --out-gold data/samples/smoke_gold.json \
  --total 100 \
  --seed 123
~~~

#### 📊 Summary
| Dataset | Purpose | Mode | Seed | Size | Used In |
|--------|---------|------|------|------|--------|
| Unit | Schema / loaders | `--per-type 1` | 7 | 4 | `test_loaders.py`, `test_preprocess.py` |
| Integration | E2E pipeline | `--per-type 6` | 42 | ~24 | `test_generate_cli.py` |
| Smoke | Short fine-tune | `--total 100` | 123 | ~100 | sanity FT |

To regenerate all three:
~~~bash
make regen-samples
~~~

---

## 🐳 Docker usage

### GPU build
~~~bash
docker build -t bioasq-llm-qa -f Dockerfile .
~~~

### CPU build (for Mac)
~~~bash
docker build -t bioasq-llm-qa-cpu -f Dockerfile.cpu .
~~~

Run:
~~~bash
bash scripts/docker_infer.sh
~~~

---

## 🧰 Development utilities

~~~bash
make dev            # create venv + install in editable mode
make test           # run pytest
make regen-samples  # regenerate sample datasets
make run-tiny       # run tiny inference config
~~~

---

## 🧮 Configuration

Configs live in `configs/`:

- **Tiny inference:** `configs/inference_tiny.yaml`  
- **Tiny fine-tune:** `configs/finetune_tiny.yaml`  
- **Full configs:** `configs/inference.yaml`, `configs/finetune.yaml`

---

## 📄 License

Released under the MIT License (see `LICENSE`).

---

## 📚 Citation

If you use this code or build upon it, please cite:

> **Christopher Anaya** (2025).  
> *LLM Fine-Tuning With Biomedical Open-Source Data.*  
> Master’s Thesis in Data Science, Faculty of Science of the University of Lisbon (Faculdade de Ciências da Universidade de Lisboa), Lisbon, Portugal.  
> URL: *forthcoming institutional repository link*
>
> [https://github.com/chrisanaya/bioasq-llm-qa](https://github.com/chrisanaya/bioasq-llm-qa)

### BibTeX
```bibtex
@mastersthesis{anaya2025biollm,
  title={LLM Fine-Tuning With Biomedical Open-Source Data},
  author={Anaya, Christopher},
  year={2025},
  school={Faculty of Science of the University of Lisbon (Faculdade de Ciências da Universidade de Lisboa)},
  address={Lisbon, Portugal},
  note={Master’s Thesis in Data Science},
  url={https://github.com/chrisanaya/bioasq-llm-qa}
}

---

## ✨ Acknowledgments

This work was conducted as part of the BioASQ Challenge and inspired by open-source contributions from the Hugging Face community and biomedical NLP research.