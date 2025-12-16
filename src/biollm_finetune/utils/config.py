from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Literal, Dict, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    # Identity (for manifests/plots)
    name: Optional[str] = None
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
    dtype: Optional[Literal["float32", "float16", "bfloat16"]] = None
    torch_dtype: Optional[Literal["float32", "float16", "bfloat16"]] = None
    # Optional adapter (PEFT)
    adapter: Optional[str] = Field(None, description="Path to a PEFT adapter directory for inference")
    adapter_output_dir: Optional[str] = None

    @model_validator(mode="after")
    def _check_precision_exclusivity(self) -> "ModelConfig":
        if self.bf16 and self.fp16:
            raise ValueError("Set at most one of bf16/fp16.")
        return self
    
    @model_validator(mode="after")
    def _default_name(self) -> "ModelConfig":
        # If name not provided, default to path or base_model for readability
        if not self.name:
            if self.path:
                self.name = str(self.path)
            elif self.base_model:
                self.name = str(self.base_model)
            else:
                self.name = "unknown_model"
        return self
    
    @field_validator("adapter")
    @classmethod
    def _adapter_exists_if_set(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not Path(v).exists():
            raise ValueError(f"Adapter path not found: {v}")
        return v
    
    @model_validator(mode="after")
    def _dtype_alias(self) -> "ModelConfig":
        # If dtype isn't set, fall back to torch_dtype (legacy)
        if self.dtype is None and self.torch_dtype is not None:
            self.dtype = self.torch_dtype
        return self

class DatasetConfig(BaseModel):
    """
    Dataset identity and file locations.
    """

    name: str
    path: str
    gold_file: Optional[str] = None
    task: str = "bioasq" 

    @field_validator("path")
    @classmethod
    def _path_exists(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"Dataset file not found: {v}")
        return v

    @field_validator("gold_file")
    @classmethod
    def _gold_file_exists(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not Path(v).exists():
            raise ValueError(f"Gold file not found: {v}")
        return v
    
class RuntimeConfig(BaseModel):
    """
    Runtime identity for experiment tracking and manifests.
    """

    name: str
    device: Literal["cpu", "cuda", "mps"]
    dtype: Literal["float32", "float16", "bfloat16"]
    inference_config: str = "configs/inference_tiny.yaml"


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
    """
    Unified config for training, inference, and experiments.

    Experiment metadata fields (Phase 4):
      - name: experiment/run id used for output directory naming
      - seed: RNG seed used by run_experiment orchestration
      - perturbation: perturbation key applied in run_experiment
    """

    # --- Experiment metadata ---
    name: Optional[str] = None
    seed: int = 42
    perturbation: str = "clean"
    output_dir: str = "results/experiments"

    # --- Identity blocks ---
    dataset: DatasetConfig
    runtime: RuntimeConfig 

    # --- Core sections ---
    model: ModelConfig
    data: DataArgs
    inference: Optional[InferenceArgs] = None
    training: Optional[TrainingArgs] = None
    system: Optional[SystemArgs] = SystemArgs()

    @field_validator("output_dir")
    @classmethod
    def _ensure_output_dir(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("seed")
    @classmethod
    def _seed_nonnegative(cls, v: int) -> int:
        if v is None:
            return 42
        if v < 0:
            raise ValueError("seed must be >= 0")
        return v

    @field_validator("perturbation")
    @classmethod
    def _perturbation_nonempty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            return "clean"
        return v

    @model_validator(mode="after")
    def _mac_sanity(self) -> "FullConfig":
        import platform

        is_mac = platform.system() == "Darwin"
        if is_mac and (self.model.load_4bit or self.model.load_8bit):
            raise ValueError(
                "4-bit/8-bit quantization is not supported on macOS CPU/MPS. "
                "Set load_4bit=false, load_8bit=false."
            )
        return self


def read_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str) -> FullConfig:
    raw = read_yaml(path)
    if "model" not in raw or "data" not in raw:
        raise ValueError("Config must contain at least 'model' and 'data' sections.")
    return FullConfig(**raw)

class InferenceOnlyConfig(BaseModel):
    model: ModelConfig
    data: DataArgs
    inference: InferenceArgs
    system: Optional[SystemArgs] = SystemArgs()


def load_inference_config(path: str) -> InferenceOnlyConfig:
    raw = read_yaml(path)
    if "model" not in raw or "inference" not in raw or "data" not in raw:
        raise ValueError("Inference config must contain 'model', 'data', and 'inference'.")
    return InferenceOnlyConfig(**raw)