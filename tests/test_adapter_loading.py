import sys
from pathlib import Path
import pytest
from bioasq_llm.inference.generate import main as gen_main

CFG = """
model:
  path: "sshleifer/tiny-gpt2"
  adapter_output_dir: "{adapter}"
data:
  include_snippets: true
inference:
  batch_size: 1
  max_input_length: 32
  max_new_tokens: 4
"""

def test_adapter_missing_path_fails_cleanly(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    # point to a non-existent adapter dir
    cfg.write_text(CFG.format(adapter=tmp_path/"no_adapter"), encoding="utf-8")
    inp = tmp_path / "q.jsonl"
    inp.write_text('{"id": "1", "body": "What is insulin?", "snippets": []}\n', encoding="utf-8")
    out = tmp_path / "pred.jsonl"

    argv = ["prog", "--config", str(cfg), "--input", str(inp), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        gen_main()
    assert "adapter" in str(exc.value).lower()