import json
import random
import os

def select_k_desc(filename, nos_desc):
    # Read the JSON file
    with open(f'{filename}.json', 'r') as file:
        data = json.load(file)

    # Function to randomly select two descriptions for each category
    def select_descriptions(categories):
        selected_descriptions = {}
        for category, descriptions in categories.items():
            selected_descriptions[category] = random.sample(descriptions, k=nos_desc)
        return selected_descriptions

    # Select descriptions for each category
    selected_data = select_descriptions(data)

    # Create the directory if it doesn't exist
    output_dir = 'k_desc'
    os.makedirs(output_dir, exist_ok=True)

    # Write selected descriptions to a new JSON file
    new_file_name = f'{filename}_{nos_desc}.json'
    output_path = os.path.join(output_dir, new_file_name)
    with open(output_path, 'w') as file:
        json.dump(selected_data, file, indent=4)

    print('Completed!')

# Example usage
nos_desc = [50]
filenames = ['UCM_60ish']
for filename in filenames:
    for nos in nos_desc:
        select_k_desc(filename, nos)