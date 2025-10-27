# src/bioasq_llm/utils/repro.py
from __future__ import annotations
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def set_seed(seed: int = 42, deterministic: bool = True) -> Dict[str, Any]:
    """
    Set seeds across Python, NumPy, and Torch.
    Returns a small dict you can embed in the run manifest.
    """
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover (not in CI)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            try:
                import torch.backends.cudnn as cudnn
                cudnn.deterministic = True
                cudnn.benchmark = False
            except Exception:
                pass
    return {"seed": seed, "deterministic": deterministic}


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return None


@dataclass
class RunManifest:
    entrypoint: str                     # "inference.generate" or "training.finetune"
    config_path: str
    started_at_utc: str
    python: str
    platform: str
    torch: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    model_id: Optional[str] = None
    adapter_path: Optional[str] = None
    seed_info: Dict[str, Any] = field(default_factory=dict)
    git_commit: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def start_manifest(
    entrypoint: str,
    config_path: str,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    model_id: Optional[str] = None,
    adapter_path: Optional[str] = None,
    seed_info: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> RunManifest:
    return RunManifest(
        entrypoint=entrypoint,
        config_path=str(config_path),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()}",
        torch=getattr(sys.modules.get("torch"), "__version__", None),
        device=device,
        dtype=str(dtype) if dtype is not None else None,
        model_id=model_id,
        adapter_path=adapter_path,
        seed_info=seed_info or {},
        git_commit=_git_commit(),
        extra=extra or {},
    )


def write_manifest(manifest: RunManifest, out_dir: str | Path, filename: str = "run.json") -> Path:
    out = Path(out_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
    return out