from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _count_jsonl(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_adapter_manifest(
    adapter_dir: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    adapter_path = Path(adapter_dir)
    run_manifest = _read_json(adapter_path / "run.json")
    resolved_config = config_path or run_manifest.get("config_path")
    config = _read_yaml(Path(resolved_config)) if resolved_config else {}
    adapter_config = _read_json(adapter_path / "adapter_config.json")

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    system_cfg = config.get("system", {})
    extra = run_manifest.get("extra", {}) if isinstance(run_manifest.get("extra"), dict) else {}

    train_file = data_cfg.get("train_file") or extra.get("train_file")
    train_path = Path(train_file) if isinstance(train_file, str) else None
    weights_path = adapter_path / "adapter_model.safetensors"

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_path": str(adapter_path),
        "base_model": adapter_config.get("base_model_name_or_path")
        or model_cfg.get("base_model")
        or model_cfg.get("path")
        or run_manifest.get("model_id"),
        "training_run": {
            "entrypoint": run_manifest.get("entrypoint"),
            "config_path": str(resolved_config) if resolved_config else None,
            "started_at_utc": run_manifest.get("started_at_utc"),
            "git_commit": run_manifest.get("git_commit"),
            "python": run_manifest.get("python"),
            "platform": run_manifest.get("platform"),
            "torch": run_manifest.get("torch"),
            "requested_device": system_cfg.get("device_map"),
            "resolved_device": run_manifest.get("device"),
            "dtype": run_manifest.get("dtype"),
            "seed_info": run_manifest.get("seed_info", {}),
        },
        "peft": {
            "enabled": bool(model_cfg.get("use_peft") or adapter_config.get("peft_type")),
            "peft_type": adapter_config.get("peft_type"),
            "task_type": adapter_config.get("task_type"),
            "lora_r": adapter_config.get("r") or model_cfg.get("lora_r"),
            "lora_alpha": adapter_config.get("lora_alpha") or model_cfg.get("lora_alpha"),
            "lora_dropout": adapter_config.get("lora_dropout") or model_cfg.get("lora_dropout"),
            "target_modules": adapter_config.get("target_modules")
            or model_cfg.get("target_modules"),
        },
        "quantization": {
            "load_4bit": bool(model_cfg.get("load_4bit") or extra.get("load_4bit")),
            "load_8bit": bool(model_cfg.get("load_8bit") or extra.get("load_8bit")),
            "bnb_4bit_quant_type": model_cfg.get("bnb_4bit_quant_type"),
            "bnb_4bit_use_double_quant": model_cfg.get("bnb_4bit_use_double_quant"),
            "locally_demonstrated": not bool(
                model_cfg.get("load_4bit")
                or model_cfg.get("load_8bit")
                or extra.get("load_4bit")
                or extra.get("load_8bit")
            ),
        },
        "data": {
            "train_file": train_file,
            "train_rows": _count_jsonl(train_path),
            "validation_split": data_cfg.get("validation_split") or extra.get("val_split"),
            "include_snippets": data_cfg.get("include_snippets")
            if "include_snippets" in data_cfg
            else extra.get("include_snippets"),
            "max_length": data_cfg.get("max_length") or extra.get("max_length"),
            "question_field": data_cfg.get("question_field"),
            "answer_field": data_cfg.get("answer_field"),
        },
        "training": {
            "output_dir": training_cfg.get("output_dir") or extra.get("trainer_output_dir"),
            "max_steps": training_cfg.get("max_steps"),
            "num_train_epochs": training_cfg.get("num_train_epochs"),
            "per_device_train_batch_size": training_cfg.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps"),
            "learning_rate": training_cfg.get("learning_rate"),
            "weight_decay": training_cfg.get("weight_decay"),
            "max_grad_norm": training_cfg.get("max_grad_norm"),
        },
        "artifacts": {
            "adapter_config": str(adapter_path / "adapter_config.json"),
            "adapter_model": str(weights_path),
            "adapter_model_sha256": _sha256(weights_path),
            "adapter_weights_tracked_by_git": False,
            "tokenizer_present": (adapter_path / "tokenizer.json").exists()
            or (adapter_path / "tokenizer.model").exists(),
            "training_args_present": (adapter_path / "training_args.bin").exists(),
        },
        "interpretation": {
            "local_lora_demonstrated": True,
            "cuda_qlora_demonstrated": False,
            "quality_lift_claimed": False,
        },
    }


def write_adapter_manifest(
    adapter_dir: str | Path,
    out_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    adapter_path = Path(adapter_dir)
    target = Path(out_path) if out_path is not None else adapter_path / "adapter_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_adapter_manifest(adapter_path, config_path=config_path)
    with target.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return target
