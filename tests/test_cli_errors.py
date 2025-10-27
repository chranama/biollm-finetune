import argparse
import json
import os
from pathlib import Path
import pytest

from bioasq_llm.utils.config import load_config
from bioasq_llm.inference.generate import main as gen_main
from bioasq_llm.training.finetune import main as ft_main

CFG_MINIMAL = """
model:
  path: "sshleifer/tiny-gpt2"
data:
  include_snippets: true
inference:
  batch_size: 1
  max_input_length: 64
  max_new_tokens: 8
"""

def _write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return p

def test_load_config_missing_sections():
    with pytest.raises(Exception):
        load_config.__wrapped__  # silence type checkers
    # more meaningful: missing 'data'
    with pytest.raises(Exception):
        load_config({"model": {"path": "sshleifer/tiny-gpt2"}})  # type: ignore

def test_generate_missing_input_raises(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "cfg.yaml", CFG_MINIMAL)
    out = tmp_path / "out.jsonl"

    # Simulate CLI args: --config cfg --input (nonexistent) --out out
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import sys
    argv = ["prog", "--config", str(cfg), "--input", str(tmp_path/"nope.jsonl"), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        gen_main()

def test_finetune_missing_train_file_raises(tmp_path, monkeypatch):
    cfg_text = """
model:
  base_model: "sshleifer/tiny-gpt2"
data:
  include_snippets: true
  train_file: "does_not_exist.jsonl"
training:
  output_dir: "{}"
""".format(tmp_path/"out")
    cfg = _write(tmp_path, "cfg.yaml", cfg_text)

    import sys
    argv = ["prog", "--config", str(cfg)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        ft_main()