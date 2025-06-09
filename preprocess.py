import pandas as pd

df = pd.read_csv('ner_test.csv')
df['Places'] = df['Places'].str.split(',').str[0]
df = df.drop_duplicates()
df.to_csv('ner_test.csv', index=False)