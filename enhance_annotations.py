import json
import pandas as pd
from pathlib import Path

def load_json_list(file_path):
    """Load a JSON file containing a list of items."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Remove any surrounding whitespace and filter out empty strings
            return [str(item).strip() for item in data if item and str(item).strip()]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def main():
    # File paths
    base_dir = Path(__file__).parent
    annotations_file = base_dir / "ner_annotations.csv"
    places_file = base_dir / "places.json"
    types_file = base_dir / "type.json"
    
    # Load existing annotations
    if annotations_file.exists():
        try:
            df = pd.read_csv(annotations_file)
            print(f"Loaded {len(df)} existing annotations")
        except Exception as e:
            print(f"Error loading annotations: {e}")
            df = pd.DataFrame(columns=['text', 'entities'])
    else:
        df = pd.DataFrame(columns=['text', 'entities'])
    
    # Track existing texts to avoid duplicates
    existing_texts = set(df['text'].astype(str).str.strip())
    new_entries = []
    
    # Add places as LOCATION entities
    places = load_json_list(places_file)
    for place in places:
        if place and place not in existing_texts:
            new_entries.append({
                'text': place,
                'entities': f"[[0, {len(place)}, 'LOCATION']]"
            })
            existing_texts.add(place)
    
    # Add types as TYPE entities
    types = load_json_list(types_file)
    for type_name in types:
        if type_name and type_name not in existing_texts:
            new_entries.append({
                'text': type_name,
                'entities': f"[[0, {len(type_name)}, 'TYPE']]"
            })
            existing_texts.add(type_name)
    
    # Add new entries to the DataFrame
    if new_entries:
        new_df = pd.DataFrame(new_entries)
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Save back to CSV
        df.to_csv(annotations_file, index=False)
        print(f"Added {len(new_entries)} new entries to {annotations_file}")
        print(f"Total annotations: {len(df)}")
    else:
        print("No new entries to add")

if __name__ == "__main__":
    main()
