#!/usr/bin/env python3
"""
Tiny, faithful fine-tune script for BioASQ-style QA using HF Trainer.

- Validates YAML via utils.config
- Resolves device/dtype via utils.device
- Optional PEFT/LoRA support (no bitsandbytes on macOS/CPU)
- Masks prompt tokens so loss is computed only on the answer span
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import torch
from rich.console import Console
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from bioasq_llm.utils.config import load_config
from bioasq_llm.utils.device import resolve_device

# >>> Minimal additions for logging + reproducibility manifest
from bioasq_llm.utils.logging import get_logger
from bioasq_llm.utils.repro import set_seed, start_manifest, write_manifest
# <<<

console = Console()


# ---------- Data helpers ----------

def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_context(item: Dict[str, Any]) -> str:
    sn = item.get("snippets") or []
    lines: List[str] = []
    if isinstance(sn, list):
        for s in sn:
            if isinstance(s, str):
                lines.append(s)
            elif isinstance(s, dict):
                if "text" in s:
                    lines.append(s["text"])
                elif "snippet" in s:
                    lines.append(s["snippet"])
    return "\n".join(lines).strip()


def _build_prompt(item: Dict[str, Any], include_snippets: bool = True, q_field: str = "body") -> str:
    q = item.get(q_field) or item.get("question") or ""
    ctx = _extract_context(item) if include_snippets else ""
    if ctx:
        return (
            "You are a biomedical domain expert. Answer the question using ONLY the provided context.\n"
            "If the answer cannot be determined from the context, say 'Unknown'.\n\n"
            f"Context:\n{ctx}\n\n"
            f"Question: {q}\nAnswer:"
        )
    else:
        return (
            "You are a biomedical domain expert. Answer the question concisely.\n\n"
            f"Question: {q}\nAnswer:"
        )


def _extract_answer(item: Dict[str, Any], a_field: str = "ideal_answer") -> str:
    # BioASQ often: ideal_answer is list[str]; exact_answer can be str/list
    ans = item.get(a_field)
    if isinstance(ans, list):
        if len(ans) == 0:
            return "Unknown"
        return ans[0] if isinstance(ans[0], str) else "Unknown"
    if isinstance(ans, str):
        return ans
    exact = item.get("exact_answer")
    if isinstance(exact, list):
        return ", ".join(map(str, exact))
    if isinstance(exact, str):
        return exact
    return "Unknown"


# ---------- Dataset ----------

@dataclass
class SupervisedExample:
    prompt: str
    answer: str


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[Dict[str, Any]], include_snippets: bool, q_field: str, a_field: str):
        self.samples: List[SupervisedExample] = []
        for r in rows:
            prompt = _build_prompt(r, include_snippets=include_snippets, q_field=q_field)
            answer = _extract_answer(r, a_field=a_field)
            self.samples.append(SupervisedExample(prompt=prompt, answer=answer))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SupervisedExample:
        return self.samples[idx]


class DataCollatorSFT:
    """
    Collate that tokenizes prompt and answer, concatenates them,
    and masks prompt tokens with label -100 so loss is only on the answer.
    """

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[SupervisedExample]) -> Dict[str, torch.Tensor]:
        input_ids_list, labels_list, attention_list = [], [], []

        for ex in batch:
            tok_prompt = self.tok(
                ex.prompt, add_special_tokens=False, truncation=True, max_length=self.max_length
            )
            tok_answer = self.tok(
                " " + ex.answer.strip(), add_special_tokens=False, truncation=True, max_length=self.max_length
            )

            ids = tok_prompt["input_ids"] + tok_answer["input_ids"]
            if self.tok.eos_token_id is not None:
                ids = ids + [self.tok.eos_token_id]

            ids = ids[: self.max_length]
            attn = [1] * len(ids)

            labels = [-100] * len(tok_prompt["input_ids"]) + tok_answer["input_ids"]
            if self.tok.eos_token_id is not None:
                labels = labels + [self.tok.eos_token_id]
            labels = labels[: self.max_length]

            input_ids_list.append(ids)
            labels_list.append(labels)
            attention_list.append(attn)

        maxlen = max(len(x) for x in input_ids_list)

        def pad(seq, pad_id, to_len):
            return seq + [pad_id] * (to_len - len(seq))

        padded_ids = [pad(x, self.tok.pad_token_id or self.tok.eos_token_id, maxlen) for x in input_ids_list]
        padded_attn = [pad(x, 0, maxlen) for x in attention_list]
        padded_lbls = [pad(x, -100, maxlen) for x in labels_list]

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
            "labels": torch.tensor(padded_lbls, dtype=torch.long),
        }


# ---------- Main ----------

def main() -> None:
    import argparse
    from pathlib import Path
    import inspect
    import torch  # ensure torch is in local scope for device/mps guards

    ap = argparse.ArgumentParser(description="Fine-tune a causal LM on BioASQ-style QA (LoRA optional).")
    ap.add_argument("--config", required=True, help="YAML config (finetune_tiny.yaml, etc.)")
    args = ap.parse_args()

    # 1) Parse & validate config
    try:
        cfg = load_config(args.config)
    except Exception as e:
        raise SystemExit(f"[ConfigError] {e}")
    if not cfg.training:
        raise SystemExit("Training section missing in config.")

    # 2) Resolve device + dtype
    device, dtype = resolve_device(
        requested=(cfg.system.device_map if cfg.system else "auto"),
        prefer_bf16=bool(cfg.model.bf16),
        prefer_fp16=bool(cfg.model.fp16),
    )
    console.print(f"[bold green]Device:[/bold green] {device} | [bold]dtype:[/bold] {dtype}")

    # --- Minimal logger + seed ---
    log = get_logger("bioasq_llm.finetune")
    seed_info = set_seed(cfg.training.seed if cfg.training and cfg.training.seed else 42)
    log.info(f"[bold]Seed:[/bold] {seed_info['seed']} (deterministic={seed_info['deterministic']})")
    # -----------------------------

    # Guard quantization on non-CUDA
    if (cfg.model.load_4bit or cfg.model.load_8bit) and device != "cuda":
        raise SystemExit("4/8-bit quantization requires CUDA. On macOS/CPU/MPS, set load_4bit=false and load_8bit=false.")

    # 3) Model + tokenizer
    model_id = cfg.model.base_model or cfg.model.path
    if not model_id:
        raise SystemExit("Model id/path missing: set model.base_model (training) or model.path (inference).")

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    # Make training stable on smaller devices
    tok.padding_side = "right"                # avoid odd attention masks
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False        # must be False for training (esp. with PEFT)
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"  # avoid fused kernels on MPS

    # 4) Optional PEFT/LoRA
    if cfg.model.use_peft:
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as e:
            raise SystemExit(f"PEFT is requested (model.use_peft=true) but peft is not installed: {e}")

        target_modules = cfg.model.target_modules or ["q_proj", "v_proj"]
        lconf = LoraConfig(
            r=cfg.model.lora_r or 8,
            lora_alpha=cfg.model.lora_alpha or 16,
            lora_dropout=cfg.model.lora_dropout or 0.05,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lconf)
        console.print(f"[bold cyan]Enabled LoRA[/bold cyan] on target modules: {target_modules}")

    # Move to device for cpu/mps
    if device in {"cpu", "mps"}:
        model.to(torch.device(device))

    # 5) Data
    train_file = cfg.data.train_file
    if not train_file:
        raise SystemExit("data.train_file must be set in the config.")
    include_snippets = bool(cfg.data.include_snippets)
    q_field = cfg.data.question_field or "body"
    a_field = cfg.data.answer_field or "ideal_answer"

    rows = _read_jsonl(train_file)
    if len(rows) == 0:
        raise SystemExit(f"No rows found in {train_file}")

    # Split
    val_ratio = cfg.data.validation_split or 0.1
    n_val = int(len(rows) * val_ratio)
    val_rows = rows[:n_val] if n_val > 0 else []
    train_rows = rows[n_val:] if n_val > 0 else rows

    train_ds = SupervisedDataset(train_rows, include_snippets=include_snippets, q_field=q_field, a_field=a_field)
    eval_ds = SupervisedDataset(val_rows, include_snippets=include_snippets, q_field=q_field, a_field=a_field) if val_rows else None

    max_len = cfg.data.max_length or 384
    collator = DataCollatorSFT(tokenizer=tok, max_length=max_len)

    # 6) TrainingArguments (from cfg.training)
    t = cfg.training

    # Some HF versions expect `evaluation_strategy`, some older forks used `eval_strategy`.
    # Choose the right kw name dynamically.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_kw = "evaluation_strategy" if "evaluation_strategy" in ta_params else "eval_strategy"
    ta_kwargs = {
        "output_dir": t.output_dir,
        "num_train_epochs": t.num_train_epochs or 1,
        "max_steps": t.max_steps or -1,
        "per_device_train_batch_size": t.per_device_train_batch_size,
        "per_device_eval_batch_size": t.per_device_eval_batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "learning_rate": t.learning_rate,
        "weight_decay": t.weight_decay,
        "warmup_ratio": t.warmup_ratio,
        "lr_scheduler_type": t.lr_scheduler_type,
        "logging_steps": t.logging_steps,
        "save_steps": t.save_steps,
        eval_kw: (t.evaluation_strategy if getattr(t, "evaluation_strategy", None) else "no"),
        "eval_steps": t.eval_steps,
        "save_total_limit": t.save_total_limit,
        "report_to": (cfg.system.report_to if cfg.system and cfg.system.report_to else "none"),
        "fp16": (dtype == torch.float16),
        "bf16": (dtype == torch.bfloat16),
    }
    targs = TrainingArguments(**ta_kwargs)

    # --- Minimal MANIFEST block (after we know all paths) ---
    manifest = start_manifest(
        entrypoint="training.finetune",
        config_path=args.config,
        device=str(device),
        dtype=str(dtype),
        model_id=model_id,
        adapter_path=getattr(cfg.model, "adapter_output_dir", None),
        seed_info=seed_info,
        extra={
            "output_dir": t.output_dir,
            "train_file": train_file,
            "val_split": cfg.data.validation_split,
            "max_length": max_len,
            "include_snippets": include_snippets,
        },
    )
    manifest_path = write_manifest(manifest, out_dir=t.output_dir)
    log.info(f"[bold green]Run manifest →[/bold green] {manifest_path.resolve()}")
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds if eval_ds and len(eval_ds) > 0 else None,
        data_collator=collator,
        tokenizer=tok,
    )

    # ---- MPS/CPU stability guards ----
    trainer.args.fp16 = False
    trainer.args.bf16 = False
    trainer.args.dataloader_pin_memory = False
    # stronger clipping helps avoid NaNs on MPS
    if getattr(trainer.args, "max_grad_norm", None) is None or trainer.args.max_grad_norm > 0.5:
        trainer.args.max_grad_norm = 0.5

    # Skip updates on non-finite loss (defensive)
    _orig_compute_loss = trainer.compute_loss
    def _safe_compute_loss(model, inputs, num_items_in_batch=None):
        loss = _orig_compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        base = loss[0] if isinstance(loss, tuple) else loss
        if not torch.isfinite(base):
            model.zero_grad(set_to_none=True)
            return base * 0.0 if isinstance(loss, torch.Tensor) else (base * 0.0, {})
        return loss
    trainer.compute_loss = _safe_compute_loss
    # ---- end guards ----

    console.print(f"[bold]Train examples:[/bold] {len(train_ds)} | [bold]Val examples:[/bold] {len(eval_ds) if eval_ds else 0}")
    trainer.train()
    trainer.save_model(t.output_dir)  # saves adapter weights if PEFT, otherwise full model head
    tok.save_pretrained(t.output_dir)

    console.print(f"[bold green]Saved model/tokenizer →[/bold green] {Path(t.output_dir).resolve()}")


if __name__ == "__main__":
    main()