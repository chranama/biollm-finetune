# src/bioasq_llm/utils/config.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    # Common fields for inference & training (names match your YAMLs)
    path: Optional[str] = Field(None, description="HF model id or local path (inference)")
    base_model: Optional[str] = Field(None, description="HF model id or local path (training)")
    load_4bit: Optional[bool] = False
    load_8bit: Optional[bool] = False
    bf16: Optional[bool] = False
    fp16: Optional[bool] = False
    max_length: Optional[int] = 2048
    adapter_output_dir: Optional[str] = None
    gradient_checkpointing: Optional[bool] = False
    use_peft: Optional[bool] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    target_modules: Optional[List[str]] = None
    torch_dtype: Optional[Literal["float32", "float16", "bfloat16"]] = None

    @model_validator(mode="after")
    def _check_precision_exclusivity(self) -> "ModelConfig":
        if self.bf16 and self.fp16:
            raise ValueError("Set at most one of bf16/fp16.")
        return self


class InferenceArgs(BaseModel):
    batch_size: int = 1
    max_input_length: int = 2048
    max_new_tokens: int = 128
    do_sample: bool = False
    num_beams: int = 1
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class DataArgs(BaseModel):
    include_snippets: bool = True
    # Training-only fields also live here to keep schema minimal:
    train_file: Optional[str] = None
    validation_split: Optional[float] = 0.1
    max_length: Optional[int] = None  # training max seq len
    question_field: Optional[str] = "body"
    answer_field: Optional[str] = "ideal_answer"

    @field_validator("train_file")
    @classmethod
    def _train_file_exists(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not Path(v).exists():
            raise ValueError(f"Train file not found: {v}")
        return v


class TrainingArgs(BaseModel):
    output_dir: str
    num_train_epochs: Optional[int] = 1
    max_steps: Optional[int] = None
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"
    logging_steps: int = 10
    save_steps: int = 100
    evaluation_strategy: Literal["no", "steps", "epoch"] = "steps"
    eval_steps: Optional[int] = None
    save_total_limit: Optional[int] = 1
    seed: int = 42

    @field_validator("output_dir")
    @classmethod
    def _ensure_output_parent(cls, v: str) -> str:
        Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v


class SystemArgs(BaseModel):
    device_map: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    use_mps: Optional[bool] = None
    report_to: Optional[str] = "none"
    disable_tqdm: bool = False


class FullConfig(BaseModel):
    # Match your YAML sections: model, inference, data, training, system
    model: ModelConfig
    data: DataArgs
    inference: Optional[InferenceArgs] = None
    training: Optional[TrainingArgs] = None
    system: Optional[SystemArgs] = SystemArgs()

    @model_validator(mode="after")
    def _mac_sanity(self) -> "FullConfig":
        import platform
        is_mac = platform.system() == "Darwin"
        if is_mac and (self.model.load_4bit or self.model.load_8bit):
            raise ValueError("4-bit/8-bit quantization is not supported on macOS CPU/MPS. "
                             "Set load_4bit=false, load_8bit=false.")
        # Allow bf16/fp16 flags; dtype fallback handled in device resolver.
        return self


def read_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str) -> FullConfig:
    raw = read_yaml(path)
    # Permit minimal inference-only or training-only files
    if "model" not in raw or "data" not in raw:
        raise ValueError("Config must contain at least 'model' and 'data' sections.")
    return FullConfig(**raw)