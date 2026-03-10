from biollm_finetune.inference.generate import DEFAULT_TEMPLATES, build_prompt


def test_prompt_contains_headers(sample_questions_path):
    import json

    row = json.loads(open(sample_questions_path, "r", encoding="utf-8").readline())
    prompt = build_prompt(row, templates=DEFAULT_TEMPLATES, include_snippets=True)
    assert "### Question:" in prompt
    assert "### Context:" in prompt
    assert "### Answer" in prompt
