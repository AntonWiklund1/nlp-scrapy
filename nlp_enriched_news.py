import spacy
import pandas as pd
from tqdm import tqdm

print()

nlp = spacy.load('en_core_web_lg')

data = pd.read_csv('./data/news.csv')

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'ORG']

# Create a progress bar
tqdm.pandas()

# Apply the extract_entities function with progress bar
print("Extracting entities from headlines...")
data['entities'] = data['headline'].progress_apply(extract_entities)

print("Extracting entities from bodies...")
data['entities'] += data['body'].progress_apply(extract_entities)

# put the entities to the left of the DataFrame
data = data[['entities', 'headline', 'link', 'date', 'body']]

# Export the DataFrame to a .csv file
try:
    data.to_csv('./data/news_enriched.csv', sep=",", index=True)
    print("File saved successfully.")
except Exception as e:
    print(f"Failed to save the file: {str(e)}")
