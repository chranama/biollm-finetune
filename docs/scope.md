# Scope

BioLLM-Finetune is a local biomedical QA experiment and evaluation workflow.

## In Scope

- BioASQ-style data loading and preprocessing
- Local Hugging Face model inference
- PEFT adapter fine-tuning on BioASQ-style prompt/answer rows
- Local LoRA training and adapter-aware evaluation
- CUDA QLoRA configuration support
- Deterministic perturbation of input questions and snippets
- BioASQ-style answer scoring
- Clean-vs-perturbed robustness comparison
- Phenotype-conditioned aggregation
- Saved run artifacts and validation metadata
- CI checks for linting, formatting, tests, and manifest validation

## Out Of Scope

- Clinical decision support
- Production model serving
- Multi-user experiment tracking
- Online monitoring or alerting
- Security hardening
- High-availability deployment
- Full biomedical model benchmarking
- Claims about clinical safety or medical correctness
- Claims that a fine-tuned adapter is clinically useful or broadly superior

## Current Artifact Scope

The current saved outputs use a small reproducible BioASQ sample and a
lightweight open model configuration. They demonstrate local LoRA adapter
training and adapter-aware perturbation evaluation. They do not demonstrate CUDA
QLoRA execution, MPS execution, or a broadly superior fine-tuned model.

## Known Limits

- Some experiment commands can require model downloads from Hugging Face.
- The active artifact set is intentionally small.
- CUDA QLoRA is configured for a CUDA environment but is not locally
  demonstrated by the current artifacts.
- Current local configs can request MPS, but the resolved runtime depends on the
  installed PyTorch build and may be CPU.
- Phenotype-conditioned outputs are optional research context unless explicitly
  refreshed and included in the evidence manifest.
- Perturbations are synthetic and targeted.
- The perturbation implementation is concentrated in one module and may need to
  be split if the perturbation set grows.
- Filesystem artifacts are simple and inspectable, but they are not a substitute
  for a full experiment tracking platform.
