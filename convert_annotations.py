import pandas as pd
import json
from pathlib import Path

def convert_csv_to_json(csv_file, json_file):
    """Convert CSV annotations to spaCy's JSON format for training."""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Initialize list to store training examples
        training_data = []
        
        # Process each row
        for _, row in df.iterrows():
            if pd.isna(row.get('text')) or pd.isna(row.get('entities')):
                continue
                
            text = str(row['text']).strip()
            if not text:
                continue
                
            try:
                # Parse entities
                entities = []
                if isinstance(row.get('entities'), str) and row['entities'].strip():
                    # Convert string representation of list to actual list
                    ent_list = eval(row['entities'])
                    for ent in ent_list:
                        if len(ent) >= 3:  # Ensure we have (start, end, label)
                            start = int(ent[0])
                            end = int(ent[1])
                            label = str(ent[2])
                            
                            # Validate entity span
                            if 0 <= start <= end <= len(text):
                                entities.append([start, end, label.upper()])
                
                # Add to training data
                training_data.append({
                    'text': text,
                    'entities': entities
                })
                
            except Exception as e:
                print(f"Error processing row {_}: {e}")
                continue
        
        # Save to JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
            
        print(f"Converted {len(training_data)} examples to {json_file}")
        return training_data
        
    except Exception as e:
        print(f"Error converting CSV to JSON: {e}")
        return []

if __name__ == "__main__":
    # Define file paths
    base_dir = Path(__file__).parent
    csv_file = base_dir / "ner_annotations.csv"
    json_file = base_dir / "ner_annotations.json"
    
    # Convert CSV to JSON
    training_data = convert_csv_to_json(csv_file, json_file)
    
    if training_data:
        print(f"\nFirst example:")
        print(json.dumps(training_data[0], indent=2, ensure_ascii=False))
