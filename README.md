# NER Annotation and Training Pipeline

This project provides tools for annotating place names, training a custom NER (Named Entity Recognition) model, and testing its performance. The system recognizes the following entity types:
- `name`: The main name of the place (e.g., "Nepal rastra", "Three by four")
- `type`: The type of place (e.g., "bank", "cafe", "bar")
- `location`: The location or area (e.g., "chabahil", "bishalnagar")
- `stop_word`: Common words like "and", "by", "the", etc.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** This will install spaCy and other necessary dependencies.

3. Download the English language model for spaCy:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Workflow

### 1. Annotation

Annotate your place names using the interactive tool:
```bash
python annotate_ner.py
```

**Annotation Instructions:**
1. For each word in the place name, assign an entity type using the number keys.
2. Navigation:
   - `1-4`: Select entity type (name, type, location, stop_word)
   - `s`: Skip to next place
   - `p`: Previous word
   - `q`: Quit and save

3. Annotations are automatically saved to `ner_annotations.csv`.

### 2. Training

Train a custom NER model on your annotated data:
```bash
python train_ner.py --data ner_annotations.csv --output models/ner_model --iterations 20 
```

**Training Options:**
- `--data`: Path to annotated CSV file (default: `ner_annotations.csv`)
- `--output`: Directory to save the trained model (default: `models/ner_model`)
- `--dropout`: Dropout rate (default: 0.3)
- `--iterations`: Number of training iterations (default: 20)
- `--test_size`: Size of the test set (default: 0.2)

### 3. Testing

Test your trained model on sample text or a file:

**Test on a single text:**
```bash
python test_ner.py --model models/ner_model --text "Three by four cafe in bishalnagar"
```

**Test on a file (one text per line):**
```bash
python test_ner.py --model models/ner_model --file test_data.txt
```

**Test on a JSON file with ground truth (for evaluation):**
```bash
python test_ner.py --model models/ner_model --file test_data.json
```

## File Formats

### Input CSV Format
```csv
index,text,entities
0,Three by four cafe and bar bishalnagar,"[[0, 15, 'name'], [16, 20, 'type'], [21, 24, 'stop_word'], [25, 28, 'type'], [29, 40, 'location']]"
```

### Test JSON Format
```json
[
  {
    "text": "Three by four cafe in bishalnagar",
    "entities": [[0, 15, "name"], [29, 40, "location"]]
  }
]
```

## Example

For the place "Three by four cafe and bar bishalnagar":
- "Three by four" → `name` (0-15)
- "cafe" → `type` (16-20)
- "and" → `stop_word` (21-24)
- "bar" → `type` (25-28)
- "bishalnagar" → `location` (29-40)

## Best Practices

1. **Consistency**: Be consistent with your annotations (e.g., always use "cafe" not sometimes "restaurant")
2. **Coverage**: Include diverse examples for each entity type
3. **Validation**: Split your data into training and validation sets
4. **Iteration**: Train and test your model regularly to identify areas for improvement

## Troubleshooting

- **Model not learning well?** Try increasing the training data or adjusting hyperparameters
- **Getting errors during training?** Check your annotations for consistency
- **Performance issues?** Try reducing the model size or batch size
