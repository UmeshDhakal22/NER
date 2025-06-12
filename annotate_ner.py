import csv
import os
from typing import List, Dict, Tuple

# Common entity types
ENTITY_TYPES = ["name", "type", "location", "stop_word"]

class NERAnnotator:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.places = self._load_places()
        self.annotations = self._load_existing_annotations()
        self.current_index = 0

    def _load_places(self) -> List[Tuple[int, str]]:
        """Load places from the input CSV file and assign indices."""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header if exists
            try:
                next(reader)  # Skip header
            except StopIteration:
                pass
            # Return list of (index, place) tuples
            return [(i, row[0].strip()) for i, row in enumerate(reader) if row and row[0].strip()]

    def _load_existing_annotations(self) -> Dict[int, Dict[str, List[Tuple[int, int, str]]]]:
        """Load existing annotations if the output file exists."""
        if not os.path.exists(self.output_file):
            return {}
        
        annotations = {}
        with open(self.output_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                next(reader)  # Skip header
            except StopIteration:
                return annotations
                
            for row in reader:
                if not row or len(row) < 2:  # Skip empty rows or rows without enough columns
                    continue
                try:
                    # Convert to float first to handle both '0' and '0.0' formats, then to int
                    idx = int(float(row[0]))
                    text = row[1]
                    entities = eval(row[2]) if len(row) > 2 and row[2].strip() else []
                    annotations[idx] = {
                        'text': text,
                        'entities': entities
                    }
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse row: {row}. Error: {e}")
                    continue
        return annotations

    def save_annotations(self):
        """Save annotations to the output file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'text', 'entities'])
            for idx, data in sorted(self.annotations.items()):
                writer.writerow([idx, data['text'], data['entities']])

    def _get_char_offsets(self, text, words):
        """Get character offsets for each word in the text."""
        offsets = []
        current_pos = 0
        
        for word in words:
            try:
                start = text.index(word, current_pos)
                end = start + len(word)
                offsets.append((start, end))
                current_pos = end
            except ValueError:
                # Fallback for words not found (shouldn't happen with proper input)
                end = current_pos + len(word)
                offsets.append((current_pos, end))
                current_pos = end
                
        return offsets
    
    def _is_overlapping(self, new_entity, existing_entities, offsets):
        """Check if the new entity overlaps with any existing ones."""
        new_start, new_end = offsets[new_entity[0]][0], offsets[new_entity[1]-1][1]
        
        for entity in existing_entities:
            ent_start_idx, ent_end_idx, _ = entity
            ent_start = offsets[ent_start_idx][0]
            ent_end = offsets[ent_end_idx-1][1]
            
            # Check for overlap
            if not (new_end <= ent_start or new_start >= ent_end):
                return True
                
        return False
    
    def annotate(self):
        """Start the annotation process."""
        print("NER Annotation Tool")
        print("-------------------")
        print("Instructions:")
        print("1. For each word, enter the entity type (name, type, location, stop_word)")
        print("   - 'name': Main name of the place (e.g., 'Nepal rastra')")
        print("   - 'type': Type of place (e.g., 'bank', 'cafe')")
        print("   - 'location': Location or area (e.g., 'chabahil')")
        print("   - 'stop_word': Common words like 'and', 'by', 'the'")
        print("2. Enter 's' to skip to the next place")
        print("3. Enter 'q' to quit and save")
        print("4. Enter 'p' to go to previous word")
        print("5. Enter 'r' to restart current place")
        print()

        while self.current_index < len(self.places):
            place_idx, place_text = self.places[self.current_index]
            
            # Skip if already annotated
            if place_idx in self.annotations:
                self.current_index += 1
                continue

            print(f"\nPlace {self.current_index + 1}/{len(self.places)}")
            print(f"Text: {place_text}")
            
            # Split into words and get character offsets
            words = place_text.split()
            char_offsets = self._get_char_offsets(place_text, words)
            entities = []
            i = 0
            
            while i < len(words):
                word = words[i]
                print(f"\nCurrent word: {word} (position {i+1}/{len(words)})")
                
                # Show current annotations
                if entities:
                    print("\nCurrent annotations:")
                    for ent in entities:
                        start_idx, end_idx, label = ent
                        entity_text = ' '.join(words[start_idx:end_idx])
                        print(f"  - {entity_text} ({label})")
                
                print("\nOptions:")
                print("1. name")
                print("2. type")
                print("3. location")
                print("4. stop_word")
                print("s. Skip this place")
                print("p. Previous word")
                print("r. Restart this place")
                print("q. Quit and save")
                
                choice = input("\nEnter your choice: ").strip().lower()
                
                if choice == 'q':
                    self.save_annotations()
                    print(f"Annotations saved to {self.output_file}")
                    return
                elif choice == 's':
                    break
                elif choice == 'p':
                    i = max(0, i - 1)
                    continue
                elif choice == 'r':
                    entities = []
                    i = 0
                    continue
                elif choice in ['1', '2', '3', '4']:
                    entity_type = ENTITY_TYPES[int(choice) - 1]
                    
                    # For name, type, location - try to capture multi-word entities
                    if entity_type != 'stop_word':
                        # Ask how many words to include
                        max_words = min(5, len(words) - i)  # Limit to 5 words max
                        if max_words > 1:
                            print(f"\nHow many words to include in this {entity_type}? (1-{max_words}, default=1): ", end='')
                            num_words = input().strip()
                            try:
                                num_words = min(max(1, int(num_words)), max_words) if num_words else 1
                            except ValueError:
                                num_words = 1
                        else:
                            num_words = 1
                            
                        end_idx = min(i + num_words, len(words))
                        entity_text = ' '.join(words[i:end_idx])
                        print(f"Marking as {entity_type}: {entity_text}")
                        
                        # Add the entity
                        entities.append((i, end_idx, entity_type))
                        i = end_idx
                    else:
                        # For stop words, just mark the current word
                        entities.append((i, i+1, entity_type))
                        i += 1
                else:
                    print("Invalid choice. Please try again.")
            
            # Save the annotations for this place
            if entities:
                # Convert word indices to character offsets
                char_entities = []
                for start_idx, end_idx, label in entities:
                    start = char_offsets[start_idx][0]
                    end = char_offsets[end_idx-1][1]
                    char_entities.append((start, end, label))
                
                self.annotations[place_idx] = {
                    'text': place_text,
                    'entities': char_entities
                }
            
            self.current_index += 1
            self.save_annotations()
        
        print("\nAll places have been processed!")
        self.save_annotations()

if __name__ == "__main__":
    input_file = "ner_test.csv"
    output_file = "ner_annotations.csv"
    
    annotator = NERAnnotator(input_file, output_file)
    annotator.annotate()
