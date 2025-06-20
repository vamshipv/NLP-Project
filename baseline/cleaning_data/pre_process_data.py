import csv
import json
import os
import re

csv_file = os.path.join('..', 'data', 'reviews.csv')
json_file = os.path.join('..', 'data', 'reviews.json')

cleaned_data = []

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, skipinitialspace=True)
    for row in reader:
        # Remove unnamed keys like """, and clean newlines/spaces from each field
        stars_raw = row.get('stars', '').strip()
        stars_match = re.match(r'(\d+(\.\d+)?)', stars_raw)
        stars_clean = stars_match.group(1) if stars_match else ''
        cleaned_row = {
            key.strip(): value.strip().replace('\n', ' ') 
            for key, value in row.items() 
            if key.strip() in ['Brand', 'Model', 'stars', 'comment']
        }
        cleaned_row['stars'] = stars_clean
        cleaned_data.append(cleaned_row)

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

print("data is cleaned")
