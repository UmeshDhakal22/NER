import spacy
import argparse
from pathlib import Path

def load_model(model_path):
    """Load the trained NER model."""
    try:
        nlp = spacy.load(model_path)
        print(f"Loaded model from {model_path}")
        return nlp
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def test_single_text(nlp, text):
    """Test the model on a single text input with detailed output."""
    doc = nlp(text)
    
    print(f"\nText: {text}")
    if doc.ents:
        print("\nEntities found:")
        for ent in doc.ents:
            # Get the surrounding context (3 words before and after)
            start = max(0, ent.start_char - 20)
            end = min(len(text), ent.end_char + 20)
            context = text[start:end]
            context = context.replace('\n', ' ').strip()
            
            print(f"  - Entity: {ent.text}")
            print(f"    Type: {ent.label_}")
            print(f"    Position: {ent.start_char}-{ent.end_char}")
            print(f"    Context: ...{context}...\n")
    else:
        print("No entities found.")
    
    # Print tokens and their predictions
    print("\nToken-level predictions:")
    for token in doc:
        ent_type = token.ent_type_ if token.ent_type_ else "-"
        print(f"  '{token.text}': {ent_type}")
    
    return doc
    
    return doc

def test_from_file(nlp, file_path):
    """Test the model on a file with one text per line or JSON format."""
    try:
        if str(file_path).endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Testing on {len(data)} examples from {file_path}")
            for i, item in enumerate(data, 1):
                if isinstance(item, dict):
                    text = item.get('text', '')
                    true_entities = item.get('entities', [])
                    print(f"\n\n=== Example {i} ===")
                    doc = test_single_text(nlp, text)
                    
                    # Compare with true entities if available
                    if true_entities:
                        print("\nTrue entities:")
                        for ent in true_entities:
                            if len(ent) >= 3:  # Ensure we have [start, end, label]
                                start, end, label = ent[0], ent[1], ent[2]
                                entity_text = text[start:end]
                                print(f"  - {entity_text} ({label}): {start}-{end}")
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"Testing on {len(texts)} texts from {file_path}")
            for i, text in enumerate(texts, 1):
                print(f"\n\n=== Example {i} ===")
                test_single_text(nlp, text)
            
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Test a trained NER model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the trained model directory')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to analyze (if not using --file)')
    parser.add_argument('--file', type=str, default=None,
                       help='File containing one text per line to analyze')
    
    args = parser.parse_args()
    
    # Load the model
    nlp = load_model(args.model)
    if nlp is None:
        return
    
    # Print model information
    print("\nModel information:")
    print(f"Pipeline components: {nlp.pipe_names}")
    if 'ner' in nlp.pipe_names:
        ner = nlp.get_pipe('ner')
        print(f"NER labels: {ner.labels}")
    
    # Test on the provided input
    if args.text:
        test_single_text(nlp, args.text)
    elif args.file:
        test_from_file(nlp, args.file)
    else:
        # Interactive mode
        print("\nEnter text to analyze (or 'quit' to exit):")
        while True:
            text = input("> ")
            if text.lower() in ('quit', 'exit', 'q'):
                break
            if text.strip():
                test_single_text(nlp, text)

if __name__ == "__main__":
    main()
