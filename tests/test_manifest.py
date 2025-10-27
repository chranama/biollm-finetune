from pathlib import Path
import json
from bioasq_llm.utils.repro import start_manifest, write_manifest

def test_manifest_written(tmp_path):
    man = start_manifest(
        entrypoint="inference.generate",
        config_path="configs/inference_tiny.yaml",
        device="cpu", dtype="float32",
        model_id="sshleifer/tiny-gpt2",
        seed_info={"seed": 123, "deterministic": True},
        extra={"input": "x.jsonl", "out": "y.jsonl"},
    )
    out = write_manifest(man, tmp_path/"runs")
    assert out.exists()
    obj = json.loads(out.read_text(encoding="utf-8"))
    assert obj["entrypoint"] == "inference.generate"
    assert obj["seed_info"]["seed"] == 123
    assert "python" in obj and "platform" in obj