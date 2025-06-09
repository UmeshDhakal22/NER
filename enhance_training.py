import json
import random
from pathlib import Path

def load_json_list(file_path):
    """Load a JSON file containing a list of items."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Remove any surrounding whitespace and filter out empty strings
            return [item.strip() for item in data if item and str(item).strip()]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def create_ner_examples(texts, entity_type):
    """Create NER training examples from a list of texts."""
    examples = []
    for text in texts:
        # Skip empty or very short texts
        if not text or len(text.strip()) < 2:
            continue
            
        # Create a training example
        example = {
            "text": text,
            "entities": [[0, len(text), entity_type.upper()]]
        }
        examples.append(example)
    return examples

def enhance_training_data(original_data_file, places_file, types_file, output_file):
    """Enhance the training data with additional examples from JSON files."""
    # Load original data
    try:
        with open(original_data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading original data: {e}")
        return
    
    # Load places and types
    places = load_json_list(places_file)
    types = load_json_list(types_file)
    
    print(f"Loaded {len(places)} places and {len(types)} types")
    
    # Create additional examples
    place_examples = create_ner_examples(places[:1000], "LOC")  # Limit to first 1000 places
    type_examples = create_ner_examples(types, "TYPE")
    
    # Combine with original data
    enhanced_data = data + place_examples + type_examples
    
    # Shuffle the data
    random.shuffle(enhanced_data)
    
    # Save enhanced data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"Enhanced data saved to {output_file}")
    print(f"Original examples: {len(data)}")
    print(f"Added place examples: {len(place_examples)}")
    print(f"Added type examples: {len(type_examples)}")
    print(f"Total examples: {len(enhanced_data)}")

if __name__ == "__main__":
    # Define file paths
    base_dir = Path(__file__).parent
    original_data = base_dir / "ner_annotations.json"  # You need to export your current annotations to this format
    places_file = base_dir / "places.json"
    types_file = base_dir / "type.json"
    output_file = base_dir / "enhanced_ner_data.json"
    
    # Convert CSV annotations to JSON format if needed
    if not original_data.exists() and (base_dir / "ner_annotations.csv").exists():
        import pandas as pd
        df = pd.read_csv(base_dir / "ner_annotations.csv")
        df.to_json(original_data, orient='records', force_ascii=False, indent=2)
    
    enhance_training_data(original_data, places_file, types_file, output_file)
