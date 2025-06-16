import pandas as pd

df = pd.read_csv('baato_requests_places.csv')
df = df[['name']]
df = df.rename(columns={'name': 'Places'})
df['Places'] = df['Places'].str.split(',').str[0]
df = df.drop_duplicates()
df.to_csv('ner_test.csv', index=False)