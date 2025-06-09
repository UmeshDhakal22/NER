import pandas as pd
import ast

def clean_annotations(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file, keep_default_na=False)
    
    print(f"Original shape: {df.shape}")
    
    # Clean the data
    cleaned_rows = []
    
    for idx, row in df.iterrows():
        try:
            # Clean text
            text = str(row['text']).strip()
            if not text:
                continue
                
            # Clean entities
            entities_str = str(row['entities']).strip()
            if not entities_str:
                continue
                
            # Try to parse entities
            try:
                entities = ast.literal_eval(entities_str)
                if not isinstance(entities, (list, tuple)):
                    print(f"Skipping row {idx}: entities is not a list")
                    continue
                    
                # Validate each entity
                valid_entities = []
                for ent in entities:
                    try:
                        if len(ent) >= 3:  # At least [start, end, label]
                            start, end, label = ent[0], ent[1], ent[2]
                            # Convert to proper types
                            start = int(start)
                            end = int(end)
                            label = str(label).strip().upper()
                            
                            # Basic validation
                            if start < 0 or end > len(text) or start >= end:
                                print(f"Skipping invalid entity {ent} in text: {text}")
                                continue
                                
                            valid_entities.append([start, end, label])
                    except (ValueError, IndexError, TypeError) as e:
                        print(f"Skipping invalid entity {ent} in text: {text}")
                        continue
                
                if valid_entities:
                    cleaned_rows.append({
                        'text': text,
                        'entities': str(valid_entities)
                    })
                else:
                    print(f"No valid entities for text: {text}")
                    
            except (ValueError, SyntaxError) as e:
                print(f"Could not parse entities for text: {text}")
                continue
                
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    # Create new DataFrame
    cleaned_df = pd.DataFrame(cleaned_rows)
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    # Save to new file
    cleaned_df.to_csv(output_file, index=False)
    print(f"Saved cleaned data to {output_file}")

if __name__ == "__main__":
    input_file = "ner_annotations.csv"
    output_file = "cleaned_ner_annotations.csv"
    clean_annotations(input_file, output_file)
