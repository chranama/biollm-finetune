import json

file_path = "12B_golden_cleaned.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Check the structure of the "questions" field
for i, entry in enumerate(data["questions"][:2]):  # Inspect first 10 entries
    print(f"Entry {i} type:", type(entry))
    print(entry)