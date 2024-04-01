import spacy
import pandas as pd
from tqdm import tqdm
import joblib

print()

nlp = spacy.load('en_core_web_lg')

classifier = joblib.load('./models/news_classifier.pkl')
vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')


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

# Assuming you want to predict the category based on the 'body'. Adjust if necessary.
body_text = data['body'].tolist()
texts_vect = vectorizer.transform(body_text)  # Transform the texts to TF-IDF vector
predicted_categories = classifier.predict(texts_vect)  # Predict the categories

# Add the predicted categories to your DataFrame
data['predicted_category'] = predicted_categories

# put the entities to the left of the DataFrame
data = data[['predicted_category','entities', 'headline', 'link', 'date', 'body']]

# Export the DataFrame to a .csv file
try:
    data.to_csv('./data/news_enriched.csv', sep=",", index=True)
    print("File saved successfully.")
except Exception as e:
    print(f"Failed to save the file: {str(e)}")
