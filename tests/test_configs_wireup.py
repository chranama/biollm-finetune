import yaml
from bioasq_llm.inference.generate import read_yaml

def test_read_yaml(samples_dir):
    cfg = read_yaml(samples_dir / "tiny_prompts.yaml")
    assert "model" in cfg and "inference" in cfg
    assert cfg["model"]["path"]