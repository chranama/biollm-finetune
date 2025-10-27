import json
import os

def validate_json(file_path, output_file=None):
    """
    Validates a JSON file to ensure it is compatible with `load_dataset("json")`.
    If issues are found, a cleaned version is saved.
    """
    print(f"🔍 Checking JSON file: {file_path}")

    # Load JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return False

    # Ensure root is a dictionary
    if not isinstance(data, dict):
        print("❌ Root must be a dictionary. Found:", type(data))
        return False

    # Ensure there is at least one valid list
    list_keys = [key for key, value in data.items() if isinstance(value, list)]
    if not list_keys:
        print("❌ No list-type fields found in the JSON file. Expected at least one list.")
        return False

    print(f"✅ Found list-type fields: {list_keys}")

    # Validate each list entry
    cleaned_data = {}
    for key in list_keys:
        print(f"\n🔹 Checking `{key}` field...")
        cleaned_data[key] = []
        
        for i, entry in enumerate(data[key]):
            if isinstance(entry, dict):
                cleaned_data[key].append(entry)  # Keep as is
            elif isinstance(entry, list):
                # Extract the first valid dictionary from the list
                valid_entry = next((item for item in entry if isinstance(item, dict)), {})
                print(f"⚠ Warning: Entry at index {i} in `{key}` is a list. Using first valid dictionary.")
                cleaned_data[key].append(valid_entry)
            elif isinstance(entry, str):
                # Convert string to a dictionary with "text" key
                print(f"⚠ Warning: Entry at index {i} in `{key}` is a string. Wrapping in a dictionary.")
                cleaned_data[key].append({"text": entry})
            elif entry is None:
                print(f"⚠ Warning: Entry at index {i} in `{key}` is None. Converting to empty dictionary.")
                cleaned_data[key].append({})
            else:
                print(f"❌ Error: Entry at index {i} in `{key}` has an unsupported type: {type(entry)}")
                return False

    # Save cleaned version if requested
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=4)
        print(f"✅ Cleaned JSON saved as: {output_file}")

    print("\n🎯 JSON is valid for `load_dataset`!")
    return True

# Example Usage:
json_file = "12B_golden_cleaned.json"
output_json = "12B_golden_checked.json"

validate_json(json_file, output_json)