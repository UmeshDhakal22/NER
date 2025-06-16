import json
import pandas as pd
import re
from tqdm import tqdm

def compile_patterns(words):
    if not words:
        return re.compile(r'(?!x)x')
    pattern = r'\b(?:' + '|'.join(map(re.escape, words)) + r')\b'
    return re.compile(pattern, flags=re.IGNORECASE)

def main():
    with open('places.json', 'r', encoding='utf-8') as f:
        places = json.load(f)
    with open('type.json', 'r', encoding='utf-8') as f:
        types = json.load(f)
    
    place_pattern = compile_patterns(places)
    type_pattern = compile_patterns(types)
    
    df = pd.read_csv('./ner_test.csv')
    names = []
    
    for text in tqdm(df['Places'], total=len(df)):
        text = place_pattern.sub('', text)
        text = type_pattern.sub('', text)
        cleaned = ' '.join(text.split())
        if cleaned:
            names.append(cleaned)
    
    with open('names.json', 'w', encoding='utf-8') as f:
        json.dump(names, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
