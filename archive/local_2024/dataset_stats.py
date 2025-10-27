import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from collections import Counter

# Set this to the model you're using (e.g., Mistral or BioMistral)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# File paths — update with your actual paths
dataset_files = {
    "BioASQ": "training12b_flattened.json",
    "BiQA": "biobiqa_all.json",
    "GO Terms": "go-basic_2024-03-25_summary.jsonl",
    "DrugBank": "drugbank_fulldb_summary.jsonl",
}

def analyze_dataset(name, filepath):
    num_qa_pairs = 0
    total_tokens = 0
    question_types = Counter()

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            num_qa_pairs += 1

            # Customize these keys to match your JSON structure
            question = data.get("question", "")
            answer = data.get("answer", "") or data.get("ideal_answer", "") or ""
            q_type = data.get("type", "unknown")

            question_types[q_type] += 1

            combined_text = f"{question} {answer}"
            tokens = tokenizer.tokenize(combined_text)
            total_tokens += len(tokens)

    avg_tokens = total_tokens / num_qa_pairs if num_qa_pairs > 0 else 0
    common_types = ", ".join(t for t, _ in question_types.most_common(2))

    return {
        "Dataset": name,
        "QA Pairs": num_qa_pairs,
        "Avg. Tokens": round(avg_tokens),
        "Primary Type": common_types,
    }

# Analyze all datasets
summary = [analyze_dataset(name, path) for name, path in dataset_files.items()]

# Display as a table
from tabulate import tabulate
print(tabulate(summary, headers="keys", tablefmt="github"))