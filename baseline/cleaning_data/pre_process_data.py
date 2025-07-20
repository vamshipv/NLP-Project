import csv
import json
import os
import re

csv_file = os.path.join('..', 'data', 'reviews.csv')
json_file = os.path.join('..', 'data', 'reviews.json')

cleaned_data = []

"""
This script cleans smartphone review data from a CSV file and saves it as a JSON file.
It processes the 'stars' and 'Model' fields, ensuring that:
- 'stars' contains only numeric values.
- 'Model' names are cleaned of unnecessary parentheses unless they contain specific keywords like RAM, Storage, or ROM.
- Extra spaces and newlines are removed from all fields.
"""
def should_keep_parentheses(content):
    # Keep if it contains RAM, Storage, ROM or has multiple items (like commas)
    keywords = ['RAM', 'ROM', 'Storage']
    return any(kw in content for kw in keywords) or ',' in content

"""
Cleans the model name by removing unnecessary parentheses unless they contain keep-worthy content.
Args:
    model (str): The raw model name string.
    Returns:
    str: The cleaned model name with unnecessary parentheses removed.
"""
def clean_model_name(model):
    # Remove parentheses if they don't match keep-worthy criteria
    return re.sub(
        r'\(([^)]+)\)', 
        lambda m: f"({m.group(1)})" if should_keep_parentheses(m.group(1)) else '', 
        model
    ).strip()

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, skipinitialspace=True)
    for row in reader:
        # Clean 'stars' field
        stars_raw = row.get('stars', '').strip()
        stars_match = re.match(r'(\d+(\.\d+)?)', stars_raw)
        stars_clean = stars_match.group(1) if stars_match else ''
        
        # Clean 'Model' field
        model_raw = row.get('Model', '').strip()
        cleaned_model = clean_model_name(model_raw)
        
        cleaned_row = {
            key.strip(): value.strip().replace('\n', ' ') 
            for key, value in row.items() 
            if key.strip() in ['Brand', 'Model', 'stars', 'comment']
        }
        cleaned_row['stars'] = stars_clean
        cleaned_row['Model'] = cleaned_model
        cleaned_data.append(cleaned_row)

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

print("data is cleaned")
