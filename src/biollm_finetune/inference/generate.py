#!/usr/bin/env python3
"""
Generate answers for BioASQ-style questions using a HF causal LM.

- Validates YAML via utils.config (fast, human-friendly errors)
- Resolves device/dtype via utils.device (cuda/mps/cpu safe)
- Optionally loads a PEFT/LoRA adapter if configured
- Reads JSONL questions and writes JSONL predictions
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from biollm_finetune.utils.config import load_inference_config
from biollm_finetune.utils.device import resolve_device
from biollm_finetune.utils.logging import get_logger
from biollm_finetune.utils.repro import set_seed, start_manifest, write_manifest

console = Console()

DEFAULT_TEMPLATES = {
    "with_context": (
        "You are a biomedical domain expert. Answer using ONLY the provided context.\n\n"
        "### Context:\n{context}\n\n"
        "### Question:\n{question}\n\n"
        "### Answer:"
    ),
    "no_context": (
        "You are a biomedical domain expert. Answer concisely.\n\n"
        "### Question:\n{question}\n\n"
        "### Answer:"
    ),
}


# ---------- Helpers ----------


def read_yaml(path: str | Path) -> Dict[str, Any]:
    """Backward-compatible helper used by legacy tests."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def _extract_context(sample: Dict[str, Any]) -> str:
    # Supports snippets as list[str] or list[dict{text:...}]
    sn = sample.get("snippets") or []
    context_lines: List[str] = []
    if isinstance(sn, list):
        for s in sn:
            if isinstance(s, str):
                context_lines.append(s)
            elif isinstance(s, dict):
                # BioASQ often uses {"text": "..."} or {"snippet": "..."}
                if "text" in s:
                    context_lines.append(s["text"])
                elif "snippet" in s:
                    context_lines.append(s["snippet"])
    return "\n".join(context_lines).strip()


def build_prompt(
    sample: Dict[str, Any],
    templates: Dict[str, str] | None = None,
    include_snippets: bool = True,
) -> str:
    """Backward-compatible prompt builder used by legacy tests."""
    t = templates or DEFAULT_TEMPLATES
    question = sample.get("body") or sample.get("question") or ""
    ctx = _extract_context(sample) if include_snippets else ""
    if ctx:
        return t["with_context"].format(context=ctx, question=question)
    return t["no_context"].format(question=question)


def _build_prompt(sample: Dict[str, Any], include_snippets: bool = True) -> str:
    return build_prompt(sample, templates=DEFAULT_TEMPLATES, include_snippets=include_snippets)


def _postcut(generated: str) -> str:
    # Basic cleanup of common stop tokens; customize as needed.
    return generated.strip().replace("</s>", "").strip()


class SanitizeLogitsProcessor(LogitsProcessor):
    """
    Guards against NaN/Inf logits during sampling by:
      - replacing NaN with 0
      - replacing ±Inf with large finite values
      - clamping logits to a safe range
    Keeps behavior as close as possible to original sampling while avoiding runtime errors.
    """

    def __init__(self, min_val: float = -1e4, max_val: float = 1e4):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nan_to_num(scores, nan=0.0, posinf=self.max_val, neginf=self.min_val)
        scores = scores.clamp_(min=self.min_val, max=self.max_val)
        return scores


def _load_causal_lm(model_id: str, dtype: torch.dtype):
    """
    Future-proof dtype handling across Transformers versions.

    Newer Transformers favors `dtype=...`.
    Older versions use `torch_dtype=...`.

    We detect which parameter is supported and pass only that one.
    """
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    kwargs: Dict[str, Any] = {}

    if "dtype" in sig.parameters:
        kwargs["dtype"] = dtype
    elif "torch_dtype" in sig.parameters:
        kwargs["torch_dtype"] = dtype
    else:
        # Very old / unexpected API; fall back to no dtype override
        kwargs = {}

    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


# ---------- Main ----------


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate answers for BioASQ-style questions.")
    ap.add_argument("--config", required=True, help="YAML config (inference_tiny.yaml, etc.)")
    ap.add_argument("--input", required=True, help="JSONL questions file")
    ap.add_argument("--out", required=True, help="Output JSONL predictions file")
    ap.add_argument("--adapter", help="Optional path to a PEFT adapter (overrides config)")
    ap.add_argument("--seed", type=int, default=None, help="Override inference seed")
    args = ap.parse_args()

    # 1) Parse & validate config
    try:
        cfg = load_inference_config(args.config)
    except Exception as e:
        raise SystemExit(f"[ConfigError] {e}")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    # 2) Resolve device + dtype
    device, resolved_dtype = resolve_device(
        requested=(cfg.system.device_map if cfg.system else "auto"),
        prefer_bf16=bool(getattr(cfg.model, "bf16", False)),
        prefer_fp16=bool(getattr(cfg.model, "fp16", False)),
    )
    console.print(
        f"[bold green]Device:[/bold green] {device} | [bold]dtype:[/bold] {resolved_dtype}"
    )

    log = get_logger("biollm_finetune.generate")

    # Seed (optional: add 'seed' under inference in YAML; fallback to 42)
    seed_val = args.seed if args.seed is not None else getattr(cfg.inference, "seed", 42)
    seed_info = set_seed(seed_val)
    log.info(f"Seed: {seed_info['seed']} (deterministic={seed_info['deterministic']})")

    # 3) Resolve model id before creating manifest
    model_id = getattr(cfg.model, "path", None) or getattr(cfg.model, "base_model", None)
    if not model_id:
        raise SystemExit(
            "Model id/path missing: set model.path (inference) or model.base_model (training)."
        )

    # Adapter path (optional)
    adapter_path = (
        args.adapter
        or getattr(cfg.model, "adapter_output_dir", None)
        or getattr(cfg.model, "adapter", None)
    )

    # Guard quantization on non-CUDA
    if (
        getattr(cfg.model, "load_4bit", False) or getattr(cfg.model, "load_8bit", False)
    ) and device != "cuda":
        raise SystemExit("4/8-bit quantization requires CUDA. Disable these on macOS/CPU/MPS.")

    # 4) Write manifest *after* we know model_id and adapter_path
    manifest = start_manifest(
        entrypoint="inference.generate",
        config_path=args.config,
        device=str(device),
        dtype=str(resolved_dtype),
        model_id=model_id,
        adapter_path=adapter_path,
        seed_info=seed_info,
        extra={"input": str(args.input), "out": str(args.out)},
    )
    manifest_path = write_manifest(manifest, out_dir="results/runs/inference")
    log.info(f"[bold green]Run manifest →[/bold green] {manifest_path.resolve()}")

    # 5) Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = _load_causal_lm(model_id, dtype=resolved_dtype)

    # Optional: PEFT adapter
    if adapter_path:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            console.print(f"[bold cyan]Loaded adapter:[/bold cyan] {adapter_path}")
        except Exception as e:
            raise SystemExit(f"Failed to load adapter from {adapter_path}: {e}")

    # Move to device for cpu/mps; for cuda, accelerate could use device_map
    if device in {"cpu", "mps"}:
        model.to(torch.device(device))

    model.eval()

    # 6) Generation args
    infer = cfg.inference
    if infer is None:
        raise SystemExit("Inference section missing in config.")

    gen_kwargs = dict(
        max_new_tokens=infer.max_new_tokens,
        do_sample=infer.do_sample,
        num_beams=infer.num_beams,
        temperature=infer.temperature or 1.0,
        top_p=infer.top_p or 1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    include_snippets = bool(cfg.data.include_snippets)

    # 7) Read inputs, generate, write outputs
    results = []
    for sample in _read_jsonl(args.input):
        qid = sample.get("id") or sample.get("_id")
        prompt = _build_prompt(sample, include_snippets=include_snippets)
        toks = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=infer.max_input_length,
        )
        toks = {k: v.to(model.device) for k, v in toks.items()}

        processors = LogitsProcessorList()
        sanitize = getattr(infer, "sanitize_logits", True)
        if sanitize and bool(infer.do_sample):
            processors.append(SanitizeLogitsProcessor())

        with console.status("[bold]Generating...[/bold]", spinner="dots"):
            output_ids = model.generate(
                **toks,
                **gen_kwargs,
                logits_processor=processors if len(processors) > 0 else None,
            )

        completion_ids = output_ids[0][toks["input_ids"].shape[1] :]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        text = _postcut(text)

        results.append(
            {
                "id": qid,
                "type": sample.get("type"),
                "question": sample.get("body") or sample.get("question"),
                "prompt": prompt,
                "prediction": text,
                "predicted": text,
            }
        )

    outp = _write_jsonl(args.out, results)
    console.print(f"[bold green]Wrote predictions →[/bold green] {outp.resolve()}")


if __name__ == "__main__":
    main()
