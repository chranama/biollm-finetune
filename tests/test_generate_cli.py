import json
import subprocess
import sys
from pathlib import Path

def test_generate_tiny(tmp_path, repo_root, sample_questions_path, samples_dir):
    cfg = samples_dir / "tiny_prompts.yaml"
    outp = tmp_path / "preds.jsonl"
    cmd = [
        sys.executable, "-m", "bioasq_llm.inference.generate",
        "--config", str(cfg),
        "--input", str(sample_questions_path),
        "--out", str(outp),
    ]
    r = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert outp.exists()
    # ensure JSONL and expected fields
    line = outp.read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(line)
    assert {"id", "type", "question", "predicted"} <= set(rec.keys())