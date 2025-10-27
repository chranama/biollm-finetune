from bioasq_llm.data.loaders import load_questions_any, flatten_bioasq, basic_stats

def test_load_and_flatten(sample_questions_path):
    rows = load_questions_any(sample_questions_path)
    assert len(rows) >= 4
    flat = flatten_bioasq(rows)
    assert all("id" in r and "type" in r and "body" in r for r in flat)
    # snippets consolidated
    has_snips = sum(1 for r in flat if r.get("snippets_text"))
    assert has_snips >= 1
    stats = basic_stats(rows)
    assert stats["count"] == len(rows)
    assert "by_type" in stats