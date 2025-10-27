import json
from bioasq_llm.data.preprocess import preprocess

def test_preprocess_to_jsonl(tmp_path, sample_questions_path):
    outp = tmp_path / "train.jsonl"
    res = preprocess(
        inputs=[sample_questions_path],
        out_path=outp,
        out_format="jsonl",
        drop_unusable=False,
        flatten=True,
        prefer="last",
    )
    assert res.exists()
    # quick sanity: read first two lines
    lines = res.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 4
    first = json.loads(lines[0])
    assert "id" in first and "type" in first and "body" in first