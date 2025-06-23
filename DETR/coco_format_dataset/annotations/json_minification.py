import json

# Load the JSON data from your file
keys = ['train','test', 'val']
for key in keys:
    with open(f"instances_{key}2017.json", 'r') as file:
        data = json.load(file)
    # Write the JSON data to a new file without any whitespace
    with open(f"instances_{key}2017.json", 'w') as file:
        json.dump(data, file, separators=(',', ':'))

print("JSON minification completed successfully.")
