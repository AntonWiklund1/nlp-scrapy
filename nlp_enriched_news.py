import spacy
import pandas as pd
import joblib
from tqdm.auto import tqdm
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from colorama import Fore, Style, init
init()

# No need to manually download the VADER lexicon every time if already downloaded
# nltk.download('vader_lexicon')  # Should be commented out after first use

# Initialize NLP models
nlp = spacy.load('en_core_web_lg')
sia = SentimentIntensityAnalyzer()

# Load the models
classifier = joblib.load('./models/news_classifier.pkl')
vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')

# Load the data
data = pd.read_csv('./data/news.csv')

# Define functions
def extract_entities(text):
    """Extract ORG entities from the text using SpaCy."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'ORG']

def classify_compound_score(compound_score, threshold=0.05):
    """Classify the sentiment based on the compound score from VADER."""
    if compound_score > threshold:
        return 'Positive'
    elif compound_score < -threshold:
        return 'Negative'
    else:
        return 'Neutral'

# Main processing
def process_data(df):
    """Process the data to add entities, predicted categories, and sentiment."""
    tqdm.pandas()  # Initialize tqdm for pandas apply
    print("Extracting entities from headlines and bodies...")
    df['entities'] = df['headline'].progress_apply(extract_entities) + df['body'].progress_apply(extract_entities)
    
    print("Predicting article categories...")
    texts_vect = vectorizer.transform(df['body'])  # Transform to TF-IDF vector
    df['predicted_category'] = classifier.predict(texts_vect)  # Predict categories
    
    print("Analyzing sentiment of articles...")
    df['sentiment'] = df['body'].progress_apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment'] = df['sentiment'].apply(classify_compound_score)
    
    # Reorder columns
    df = df[['predicted_category', 'sentiment', 'entities', 'headline', 'link', 'date', 'body']]
    return df

# Apply the processing to the data
processed_data = process_data(data)

# Save the processed data
output_file = Path('./results/news_enriched.csv')
try:
    processed_data.to_csv(output_file, sep=",", index=False)
    print(f"{Fore.GREEN}File saved successfully to {output_file}{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}Failed to save the file: {e}{Style.RESET_ALL}")