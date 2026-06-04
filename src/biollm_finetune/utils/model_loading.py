from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM


def build_causal_lm_load_kwargs(model_cfg: Any, dtype: torch.dtype, device: str) -> dict[str, Any]:
    """
    Build keyword arguments for Hugging Face causal LM loading.

    LoRA runs use regular full-precision model loading. QLoRA-style runs request
    4-bit or 8-bit bitsandbytes loading and are only supported on CUDA.
    """
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    load_4bit = bool(getattr(model_cfg, "load_4bit", False))
    load_8bit = bool(getattr(model_cfg, "load_8bit", False))

    if not (load_4bit or load_8bit):
        return kwargs

    if device != "cuda":
        raise ValueError("4/8-bit quantization requires CUDA.")

    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover - depends on optional GPU stack
        raise RuntimeError(
            "Quantized loading requires the optional GPU dependencies. "
            "Install with `pip install -e '.[gpu]'` in a CUDA environment."
        ) from exc

    compute_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else torch.float16
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=load_4bit,
        load_in_8bit=load_8bit,
        bnb_4bit_quant_type=getattr(model_cfg, "bnb_4bit_quant_type", "nf4") or "nf4",
        bnb_4bit_use_double_quant=bool(getattr(model_cfg, "bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    kwargs["device_map"] = "auto"
    return kwargs


def load_causal_lm(model_id: str, model_cfg: Any, dtype: torch.dtype, device: str):
    kwargs = build_causal_lm_load_kwargs(model_cfg=model_cfg, dtype=dtype, device=device)
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
