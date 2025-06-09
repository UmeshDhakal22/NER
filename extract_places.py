import json
from langdetect import detect, LangDetectException

def is_english(text):
    """Check if the given text is in English."""
    try:
        return detect(text) == 'en'
    except (LangDetectException, Exception):
        return False

def extract_places(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    places = set()
    for feature in data.get('features', []):
        if 'properties' in feature:
            properties = feature['properties']
            name = properties.get('name:en') or properties.get('name')
            if name:
                places.add(name.strip())
    
    places = sorted(list(places))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(places, f, ensure_ascii=False, indent=2)
    
    print(f"Extracted {len(places)} unique English place names to {output_file}")

if __name__ == "__main__":
    input_file = 'export (1).geojson'
    output_file = 'places.json'
    extract_places(input_file, output_file)
