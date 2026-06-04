import json

from biollm_finetune.training.adapter_manifest import write_adapter_manifest


def test_write_adapter_manifest_records_peft_state(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    train_file = tmp_path / "train.jsonl"
    run_dir = tmp_path / "run"
    train_file.write_text('{"body": "Q", "ideal_answer": "A"}\n', encoding="utf-8")
    cfg = tmp_path / "finetune.yaml"
    cfg.write_text(
        f"""
model:
  base_model: "base-model"
  adapter_output_dir: "{adapter_dir}"
  use_peft: true
  load_4bit: false
  load_8bit: false
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj"]
data:
  train_file: "{train_file}"
  validation_split: 0.1
  include_snippets: true
  max_length: 384
training:
  output_dir: "{run_dir}"
  max_steps: 30
system:
  device_map: "auto"
""",
        encoding="utf-8",
    )
    (adapter_dir / "run.json").write_text(
        json.dumps(
            {
                "entrypoint": "training.finetune",
                "config_path": str(cfg),
                "device": "cpu",
                "dtype": "torch.float32",
                "model_id": "base-model",
                "seed_info": {"seed": 42, "deterministic": True},
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "base-model",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")

    out = write_adapter_manifest(adapter_dir)
    manifest = json.loads(out.read_text(encoding="utf-8"))

    assert manifest["base_model"] == "base-model"
    assert manifest["peft"]["peft_type"] == "LORA"
    assert manifest["peft"]["lora_r"] == 8
    assert manifest["quantization"]["load_4bit"] is False
    assert manifest["training_run"]["resolved_device"] == "cpu"
    assert manifest["data"]["train_rows"] == 1
    assert manifest["artifacts"]["adapter_model_sha256"]
