# NER Annotation Tool

This tool helps you annotate place names with the following entity types:
- `name`: The main name of the place (e.g., "Nepal rastra", "Three by four")
- `type`: The type of place (e.g., "bank", "cafe", "bar")
- `location`: The location or area (e.g., "chabahil", "bishalnagar")
- `stop_word`: Common words like "and", "by", "the", etc.

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Annotation Tool

Run the annotation script:
```bash
python annotate_ner.py
```

## Annotation Instructions

1. For each word in the place name, you'll be prompted to assign an entity type.
2. Use the following options:
   - `1` for `name`
   - `2` for `type`
   - `3` for `location`
   - `4` for `stop_word`
   - `s` to skip to the next place
   - `p` to go back to the previous word
   - `q` to quit and save your progress

3. Your annotations will be automatically saved to `ner_annotations.csv`.

## Example

For the place "Three by four cafe and bar bishalnagar":
- "Three by four" → `name`
- "cafe" → `type`
- "and" → `stop_word`
- "bar" → `type`
- "bishalnagar" → `location`

## Next Steps

After annotating your data, you can use the `ner_annotations.csv` file to train a NER model using libraries like spaCy or Hugging Face Transformers.
