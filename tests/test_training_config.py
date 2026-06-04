import importlib.util
from types import SimpleNamespace

import pytest
import torch
from biollm_finetune.utils.config import load_config, load_training_config
from biollm_finetune.utils.model_loading import build_causal_lm_load_kwargs


def test_tiny_training_config_loads():
    cfg = load_training_config("configs/finetune_tiny.yaml")

    assert cfg.model.base_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert cfg.model.use_peft is True
    assert cfg.model.adapter_output_dir == "results/ckpts/tiny_adapter"
    assert cfg.data.train_file == "data/samples/smoke_train.jsonl"
    assert cfg.training.output_dir == "results/ckpts/tiny_run"
    assert cfg.training.max_grad_norm == 0.5


def test_adapter_experiment_config_loads():
    cfg = load_config("configs/experiments/bioasq_TINY_mps_fp32_lora_clean_seed42.yaml")

    assert cfg.model.name == "tinyllama-1.1b-chat-lora"
    assert cfg.model.adapter_output_dir == "results/ckpts/tiny_adapter"
    assert cfg.runtime.inference_config == "configs/inference_tiny.yaml"


def test_quantized_loading_rejects_non_cuda():
    model_cfg = SimpleNamespace(load_4bit=True, load_8bit=False)

    with pytest.raises(ValueError, match="requires CUDA"):
        build_causal_lm_load_kwargs(model_cfg=model_cfg, dtype=torch.float32, device="cpu")


def test_regular_loading_kwargs_include_dtype():
    model_cfg = SimpleNamespace(load_4bit=False, load_8bit=False)

    kwargs = build_causal_lm_load_kwargs(
        model_cfg=model_cfg,
        dtype=torch.float32,
        device="cpu",
    )

    assert kwargs == {"torch_dtype": torch.float32}


def test_run_experiment_resolves_adapter_output_dir(repo_root):
    script_path = repo_root / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    model_cfg = SimpleNamespace(adapter=None, adapter_output_dir="results/ckpts/tiny_adapter")

    assert str(module._resolve_adapter_path(model_cfg)) == "results/ckpts/tiny_adapter"


def test_run_experiment_passes_inference_manifest_path(repo_root, monkeypatch, tmp_path):
    script_path = repo_root / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    calls = []

    def fake_check_call(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(module.subprocess, "check_call", fake_check_call)
    runtime = SimpleNamespace(inference_config="configs/inference_tiny.yaml")
    manifest_path = tmp_path / "inference_manifest.json"

    module._run_inference(
        runtime=runtime,
        inputs_path=tmp_path / "inputs.jsonl",
        outputs_path=tmp_path / "predictions.jsonl",
        manifest_path=manifest_path,
        adapter_path=None,
        seed=42,
    )

    assert calls
    assert "--manifest-out" in calls[0]
    assert str(manifest_path) in calls[0]
