import json
from bioasq_llm.eval.metrics import evaluate, build_gold_index

def test_metrics_end_to_end(sample_questions_path, sample_gold_path, tmp_path):
    # Build fake preds aligned to gold types
    preds = [
        {"id": "yn-1", "type": "yesno", "predicted": "Yes"},
        {"id": "fact-1", "type": "factoid", "predicted": "reverse transcriptase"},
        {"id": "list-1", "type": "list", "predicted": "warfarin, heparin"},
        {"id": "sum-1", "type": "summary", "predicted": "BRCA1 participates in HR repair."},
    ]
    gold = json.load(open(sample_gold_path, "r", encoding="utf-8"))["questions"]
    gold_idx = build_gold_index(gold)
    m = evaluate(preds, gold_idx)
    # Sanity: counts match and core metrics are reasonable
    assert m["counts"]["yesno"] == 1
    assert m["yesno"]["accuracy"] in (0.0, 1.0)
    assert 0.0 <= m["factoid"]["f1"] <= 1.0
    assert 0.0 <= m["list"]["f1"] <= 1.0
    assert "rougeL" in m["summary"]