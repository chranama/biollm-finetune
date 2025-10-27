import json
from collections import defaultdict

# ✅ Load JSON Data
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)  # Try standard JSON first
        except json.JSONDecodeError:
            # If JSON fails, try NDJSON (each line is a JSON object)
            f.seek(0)  # Reset file read pointer
            return [json.loads(line) for line in f if line.strip()]  # Read as JSON Lines

# ✅ Merge Data from Multiple Sources
def merge_data(files):
    merged_data = {}
    
    for file in files:
        data = load_data(file)
        for entry in data:
            entry_id = entry.get("id")
            if entry_id:
                if entry_id not in merged_data:
                    merged_data[entry_id] = entry  # Add new entry
                else:
                    merged_data[entry_id].update(entry)  # Merge fields if ID already exists

    return list(merged_data.values())  # Convert back to list

# ✅ Categorize Data by Question Type
def categorize_data(data):
    categorized_data = defaultdict(list)

    for entry in data:
        question_type = entry.get("question type", "").strip().lower()
        categorized_data[question_type].append(entry)

    return categorized_data

# ✅ Save Data to Different Files Based on Question Type
def save_data_by_category(categorized_data):
    for question_type, entries in categorized_data.items():
        output_file = f"{question_type}_questions.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=4)
        print(f"✅ Saved {len(entries)} '{question_type}' questions to '{output_file}'.")

# ✅ Main Execution
if __name__ == "__main__":
    input_files = [
        "bioasq_generated_answers.json",
        "goterms_generated_answers.json",
        "drugbank_generated_answers.json",
        "biobiqa_generated_answers.json"
    ]

    merged_data = merge_data(input_files)
    categorized_data = categorize_data(merged_data)  # Categorize merged data
    save_data_by_category(categorized_data)  # Save categorized data

    print(f"✅ Processed {len(merged_data)} entries and saved them to separate files by question type.")