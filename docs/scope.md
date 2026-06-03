# Scope

BioLLM-Finetune is a local biomedical QA experiment and evaluation workflow.

## In Scope

- BioASQ-style data loading and preprocessing
- Local Hugging Face model inference
- LoRA and QLoRA adapter fine-tuning entry points
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

## Current Artifact Scope

The current saved outputs use a small reproducible BioASQ sample and a
lightweight open model configuration. The artifacts are intended to make the
evaluation workflow inspectable, not to rank biomedical language models.

## Known Limits

- Some experiment commands can require model downloads from Hugging Face.
- The active artifact set is intentionally small.
- Perturbations are synthetic and targeted.
- The perturbation implementation is concentrated in one module and may need to
  be split if the perturbation set grows.
- Filesystem artifacts are simple and inspectable, but they are not a substitute
  for a full experiment tracking platform.
