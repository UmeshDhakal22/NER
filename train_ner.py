import pandas as pd
import spacy
import random
import json
from pathlib import Path
from spacy.training import Example
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer
from spacy.tokens import DocBin
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class NERDataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.nlp = spacy.blank('en')
        self.label_set = set()
        
    def load_data(self):
        """Load and process the dataset."""
        print("Loading data...")
        # Read CSV and handle potential parsing issues
        df = pd.read_csv(self.data_path, keep_default_na=False)
        
        # Drop rows with missing text
        df = df.dropna(subset=['text'])
        
        # Convert empty strings to empty lists for entities
        df['entities'] = df['entities'].fillna('[]')
        
        # Safely convert string representation of entities to list of tuples
        def safe_eval(x):
            if not x or not isinstance(x, str):
                return []
            try:
                return eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
            except (SyntaxError, ValueError):
                return []
        
        df['entities'] = df['entities'].apply(safe_eval)
        
        # Extract all unique entity types
        for ents in df['entities']:
            if not isinstance(ents, list):
                continue
                
            for ent in ents:
                if isinstance(ent, (list, tuple)) and len(ent) >= 3:  # Ensure valid entity format [start, end, label]
                    self.label_set.add(ent[2])
        
        print(f"Loaded {len(df)} examples with {len(self.label_set)} entity types")
        return df
    
    def create_training_data(self, df):
        """Convert data to spaCy training format."""
        train_data = []
        skipped = 0
        
        for _, row in df.iterrows():
            try:
                text = str(row['text']).strip()
                if not text:  # Skip empty text
                    skipped += 1
                    continue
                    
                entities = row.get('entities', [])
                if not isinstance(entities, list):
                    entities = []
                
                # Convert character offsets to token offsets
                doc = self.nlp.make_doc(text)
                spacy_ents = []
                
                for ent in entities:
                    # Ensure entity has at least 3 elements [start, end, label]
                    if not isinstance(ent, (list, tuple)) or len(ent) < 3:
                        continue
                        
                    try:
                        start, end, label = int(ent[0]), int(ent[1]), str(ent[2])
                        # Ensure valid numeric indices
                        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                            continue
                            
                        # Ensure the entity span is within text bounds
                        start = max(0, min(int(start), len(text)))
                        end = max(start, min(int(end), len(text)))
                        
                        # Skip empty or invalid entities
                        if start >= end or not label:
                            continue
                            
                        span = doc.char_span(start, end, label=label, alignment_mode='contract')
                        if span is not None:
                            spacy_ents.append(span)
                    except (ValueError, TypeError, IndexError) as e:
                        # Skip malformed entity annotations
                        continue
                
                # Create entity dictionary in spaCy format
                entities_dict = {
                    'entities': [(ent.start_char, ent.end_char, ent.label_) for ent in spacy_ents]
                }
                train_data.append((text, entities_dict))
                
            except Exception as e:
                print(f"Error processing row {_}: {str(e)}")
                skipped += 1
                continue
        
        if skipped > 0:
            print(f"Skipped {skipped} examples due to formatting issues")
        print(f"Successfully processed {len(train_data)} examples")
        return train_data

class NERTrainer:
    def __init__(self, model_name=None):
        self.model_name = model_name
        # Use a blank model with only the tokenizer and NER components
        if model_name is None:
            self.nlp = spacy.blank('en')
            # Disable other components for faster training
            for pipe in self.nlp.pipe_names:
                if pipe != 'ner':
                    self.nlp.remove_pipe(pipe)
        else:
            self.nlp = spacy.load(model_name)
        
    def create_model(self, labels):
        """Create a new NER model with the given labels."""
        # Add NER component if it doesn't exist
        if 'ner' not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe('ner')
        else:
            self.ner = self.nlp.get_pipe('ner')
            
        # Add labels to the NER component
        for label in labels:
            self.ner.add_label(label)
            
        # Configure the model to be more sensitive to short spans
        if 'ner' in self.nlp.pipe_factories:
            self.nlp.config['components']['ner']['factory'] = 'ner'
            self.nlp.config['nlp']['tokenizer'] = {'@tokenizers': 'spacy.Tokenizer.v1'}
            
        # Configure the model to be more sensitive to short spans
        if 'ner' in self.nlp.pipe_factories:
            cfg = {
                'moves': None,
                'update_with_oracle_cut_size': 100,
                'model': {
                    '@architectures': ['spacy.TransitionBasedParser.v2'],
                    'tok2vec': {'@architectures': ['spacy.Tok2Vec.v2']},
                    'state_type': 'ner',
                    'extra_state_tokens': False,
                    'hidden_width': 64,
                    'maxout_pieces': 2,
                    'use_upper': False,
                    'nO': None,
                    'learn_tokens': False,
                    'min_action_freq': 30,
                    'update_with_oracle_cut_size': 100,
                }
            }
            self.nlp.get_pipe('ner').cfg.update(cfg)
    
    def train(self, train_data, val_data, output_dir, n_iter=20, dropout=0.3):
        """Train the NER model with enhanced configuration for short spans."""
        # Configure training with more frequent updates
        nlp = self.nlp
        
        # Disable other pipeline components during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):
            # Initialize with random weights
            if self.model_name is None:
                optimizer = nlp.initialize()
                print("Initialized model with weights")
            else:
                optimizer = nlp.create_optimizer()
                
            print("Training the model...")
            best_f1 = 0.0
            
            # Early stopping configuration
            patience = 3  # Stop if no improvement for 3 epochs
            patience_counter = 0
            
            for itn in range(n_iter):
                random.shuffle(train_data)
                losses = {}
                
                # Use smaller batch sizes for more frequent updates
                batch_sizes = compounding(1.0, 16.0, 1.001)
                batches = minibatch(train_data, size=batch_sizes)
                
                # Process batches with progress bar
                for batch in batches:
                    examples = []
                    for text, annot in batch:
                        doc = nlp.make_doc(text)
                        example = Example.from_dict(doc, annot)
                        examples.append(example)
                    
                    # Update with smaller learning rate
                    try:
                        # Try with the most recent API first
                        nlp.update(
                            examples,
                            drop=dropout,
                            losses=losses,
                            sgd=optimizer
                        )
                    except TypeError:
                        # Fall back to simpler update call for older spaCy versions
                        nlp.update(
                            examples,
                            drop=dropout,
                            losses=losses
                        )
                
                # Reduce learning rate over time
                optimizer.learn_rate = 0.001 * (0.9 ** (itn // 2))
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"\nEarly stopping after {itn+1} iterations")
                    break
                
                # Evaluate on validation set
                if val_data:
                    scores = self.evaluate(val_data)
                    f1 = scores.get('f1', 0.0)
                    print(f"\n--- Epoch {itn+1} ---")
                    print(f"Loss: {losses.get('ner', 0.0):.2f}")
                    print(f"Precision: {scores.get('p', 0.0):.4f}")
                    print(f"Recall: {scores.get('r', 0.0):.4f}")
                    print(f"F1 Score: {f1:.4f}")
                    
                    # Print per-type scores if available
                    per_type = scores.get('per_type', {})
                    if per_type:
                        print("\nPer-type scores:")
                        for label, metrics in per_type.items():
                            print(f"  {label}:")
                            print(f"    Precision: {metrics.get('p', 0.0):.4f}")
                            print(f"    Recall: {metrics.get('r', 0.0):.4f}")
                            print(f"    F1: {metrics.get('f', 0.0):.4f}")
                    
                    # Save the best model
                    if f1 > best_f1:
                        best_f1 = f1
                        if output_dir:
                            self.save_model(output_dir)
                            print(f"\n🔥 New best model saved with F1: {best_f1:.4f}")
                else:
                    print(f"Iteration {itn+1}, Loss: {losses.get('ner', 0.0):.2f}")
    
    def evaluate(self, examples, batch_size=32):
        """Evaluate the model on the given examples with detailed metrics."""
        if not examples:
            return {'p': 0.0, 'r': 0.0, 'f1': 0.0, 'per_type': {}}
            
        # Initialize counters
        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives
        
        # Initialize per-type counters
        type_counts = {}
        
        # Process in batches to handle memory for large datasets
        for batch in minibatch(examples, size=batch_size):
            for text, annot in batch:
                # Create doc and example
                doc = self.nlp.make_doc(text)
                example = Example.from_dict(doc, annot)
                
                # Get predicted entities
                pred_doc = self.nlp(text)
                pred_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents}
                
                # Get true entities from the annotation
                true_ents = set()
                if 'entities' in annot:
                    for start, end, label in annot['entities']:
                        true_ents.add((int(start), int(end), str(label)))
                        # Update type counts
                        if label not in type_counts:
                            type_counts[label] = {'tp': 0, 'fp': 0, 'fn': 0}
                
                # Calculate true positives, false positives, false negatives
                batch_tp = 0
                for pred_ent in pred_ents:
                    if pred_ent in true_ents:
                        batch_tp += 1
                        type_counts[pred_ent[2]]['tp'] += 1
                    else:
                        fp += 1
                        if pred_ent[2] in type_counts:
                            type_counts[pred_ent[2]]['fp'] += 1
                
                # Calculate false negatives
                for true_ent in true_ents:
                    if true_ent not in pred_ents:
                        fn += 1
                        label = true_ent[2]
                        if label in type_counts:
                            type_counts[label]['fn'] += 1
                
                tp += batch_tp
        
        # Calculate overall metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate per-type metrics
        per_type_metrics = {}
        for label, counts in type_counts.items():
            tp_type = counts['tp']
            fp_type = counts['fp']
            fn_type = counts['fn']
            
            p = tp_type / (tp_type + fp_type) if (tp_type + fp_type) > 0 else 0.0
            r = tp_type / (tp_type + fn_type) if (tp_type + fn_type) > 0 else 0.0
            f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
            
            per_type_metrics[label] = {
                'p': p,
                'r': r,
                'f': f
            }
        
        return {
            'p': precision,
            'r': recall,
            'f1': f1,
            'per_type': per_type_metrics
        }
    
    def save_model(self, output_dir):
        """Save the model to disk."""
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        
        self.nlp.to_disk(output_dir)
        print(f"Model saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train a spaCy NER model')
    parser.add_argument('--data', type=str, required=True, help='Path to the CSV file with annotations')
    parser.add_argument('--output', type=str, required=True, help='Output directory to save the model')
    parser.add_argument('--model', type=str, default=None, help='Pre-trained model to fine-tune (optional)')
    parser.add_argument('--iterations', type=int, default=20, help='Number of training iterations')
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test set (0-1)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (0-1)')
    
    args = parser.parse_args()
    
    # Process the data
    print("Processing data...")
    processor = NERDataProcessor(args.data)
    df = processor.load_data()
    train_data = processor.create_training_data(df)
    
    # Split into train and validation sets
    train_data, val_data = train_test_split(
        train_data, 
        test_size=args.test_size,
        random_state=42
    )
    
    print(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples")
    print(f"Entity labels: {processor.label_set}")
    
    # Train the model
    trainer = NERTrainer(model_name=args.model)
    trainer.create_model(processor.label_set)
    trainer.train(train_data, val_data, args.output, args.iterations, args.dropout)
    
    # Evaluate on test set
    if val_data:
        print("\nFinal evaluation on validation set:")
        scores = trainer.evaluate(val_data)
        print(f"Precision: {scores.get('p', 0.0):.2f}")
        print(f"Recall: {scores.get('r', 0.0):.2f}")
        print(f"F1: {scores.get('f1', 0.0):.2f}")

if __name__ == "__main__":
    main()
