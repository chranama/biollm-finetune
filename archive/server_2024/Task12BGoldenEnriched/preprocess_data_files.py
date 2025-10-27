import json
from glob import glob

# Define input JSON files (modify this if needed)
input_files = glob("*.json")

# Output JSONL file
output_file = "12B_golden_processed.jsonl"

# Initialize an empty list to store all questions
merged_questions = []

# Loop through each file and merge "questions" lists
for file in input_files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "questions" in data:
                merged_questions.extend(data["questions"])  # Append list contents
            else:
                print(f"Warning: questions field missing in {file}")
    except Exception as e:
        print(f"Error reading {file}: {e}")

processed_questions = []
for entry in merged_questions:
    if entry["type"] == "yesno":
        entry["exact_answer"] = [entry.get("exact_answer", "")]
    elif entry["type"] == "summary":
        entry["exact_answer"] = []
    elif entry["type"] == "factoid":
        entry["exact_answer"] = [item for sublist in entry.get("exact_answer", []) for item in sublist]
    elif entry["type"] == "list":
        for index, item in enumerate(entry.get("exact_answer", [])):
            if len(item) > 1:
                entry["exact_answer"][index] = str(item)
            else:
                entry["exact_answer"][index] = item[0]

    # Extract only necessary fields
    filtered_entry = {
        "body": entry.get("body", ""),
        "type": entry.get("type", ""),
        "id": entry.get("id", ""),
        "exact_answer": entry.get("exact_answer", []),
        "ideal_answer": entry.get("ideal_answer", [])
    }

    processed_questions.append(filtered_entry)

# Save processed data in JSON Lines format
with open(output_file, "w", encoding="utf-8") as f:
    for entry in processed_questions:
        f.write(json.dumps(entry) + "\n")  # Write each JSON object as a new line

print(f"Merged {len(input_files)} JSON files into '{output_file}' (JSONL format).")
print(f"Total questions processed: {len(merged_questions)}")