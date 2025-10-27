import json
from bioasq_llm.eval.postprocess import merge_data, categorize_data

def test_merge_and_categorize(sample_questions_path, tmp_path):
    # Re-use questions as "preds" shape; add minimal predicted field
    rows = []
    with open(sample_questions_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            obj["predicted"] = "dummy"
            rows.append(obj)
    p = tmp_path / "preds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))

    merged = merge_data([p])
    cats = categorize_data(merged)
    assert sum(len(v) for v in cats.values()) == len(rows)
    assert "yesno" in cats and len(cats["yesno"]) >= 1