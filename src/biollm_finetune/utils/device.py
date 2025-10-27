# src/bioasq_llm/utils/device.py
from __future__ import annotations
import platform
import torch
from typing import Literal, Tuple, Optional

ResolvedDevice = Tuple[str, torch.dtype]

def resolve_device(
    requested: Literal["auto", "cuda", "mps", "cpu"] = "auto",
    prefer_bf16: bool = False,
    prefer_fp16: bool = False,
) -> ResolvedDevice:
    """
    Returns (device_str, torch_dtype) with safe fallbacks.
    - On CUDA: bfloat16 if available (A100/H100/etc), else float16 if requested, else float32.
    - On macOS MPS: always float32 for training stability.
    - On CPU: float32.
    """
    sys = platform.system()

    def has_cuda():
        try:
            return torch.cuda.is_available()
        except Exception:
            return False

    def has_mps():
        try:
            return torch.backends.mps.is_available()
        except Exception:
            return False

    # Resolve device
    if requested == "cuda" or (requested == "auto" and has_cuda()):
        device = "cuda"
    elif requested == "mps" or (requested == "auto" and sys == "Darwin" and has_mps()):
        device = "mps"
    elif requested == "cpu" or requested == "auto":
        device = "cpu"
    else:
        device = "cpu"

    # Resolve dtype
    if device == "cuda":
        bf16_ok = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else False
        if prefer_bf16 and bf16_ok:
            return device, torch.bfloat16
        if prefer_fp16:
            return device, torch.float16
        return device, torch.float32
    elif device == "mps":
        # MPS training generally safer in float32
        return device, torch.float32
    else:
        return device, torch.float32