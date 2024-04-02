
from pathlib import Path
from sentence_transformers import SentenceTransformer
import scipy.spatial
import spacy
import pandas as pd
import joblib
from tqdm.auto import tqdm
from transformers import pipeline, RobertaTokenizer, RobertaForSequenceClassification
import warnings
warnings.filterwarnings('ignore')
from colorama import Fore, Style, init
init()  # Initialize colorama for Windows


# Initialize the models for embeddings 
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


# Load the RoBERTa model for sentiment analysis
roberta_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_sentiment = pipeline("sentiment-analysis", model=roberta_model, tokenizer=roberta_tokenizer, return_all_scores=False)

# Load other necessary libraries and models
nlp = spacy.load('en_core_web_lg')  # For entity extraction

# Load the models for topic classification
classifier = joblib.load('./models/news_classifier.pkl')
vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')

# Load the data and keywords
data = pd.read_csv('./data/news.csv')
keywords = pd.read_csv('./data/environment_keywords.txt', header=None)[0].tolist()

# Compute embeddings for the keywords
keyword_embeddings = sentence_model.encode(keywords)

# Define functions
def extract_entities(text):
    """Extract ORG entities from the text using SpaCy."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'ORG']

def classify_compound_score(text):
    """Classify the sentiment based on RoBERTa's analysis."""
    result = roberta_sentiment(text)
    sentiment = result[0]['label']
    if sentiment == 'LABEL_0':
        return 'Negative'
    elif sentiment == 'LABEL_2':
        return 'Positive'
    else:
        return 'Neutral'

def compute_similarity_to_keywords(text, keyword_embeddings):
    """Compute similarity of a text to environmental keywords."""
    # Ensure the text is not empty
    if text:
        text_embedding = sentence_model.encode([text])  # Ensure text is in a list for consistent output shape
        if text_embedding.ndim == 1:
            # If the embedding is 1D (happens if a single sentence is passed), reshape it to be 2D
            text_embedding = text_embedding.reshape(1, -1)
        similarities = scipy.spatial.distance.cdist(text_embedding, keyword_embeddings, "cosine")[0]
        return 1 - min(similarities)  # Since 'cosine' gives distance, convert to similarity
    else:
        # If text is empty, return a default similarity that indicates no similarity.
        # Adjust this behavior as needed for your application.
        return 0
    
def find_scandals(keyword_similarity, entities, sentiment):
    """
        Find potential scandals in the text.
        based on the keyword_similarity and entities.
    """

    if keyword_similarity > 0.4 and len(entities) > 0 and sentiment == 'Negative':
        return 'Scandal'
    elif keyword_similarity > 0.1 and len(entities) > 0 and sentiment == 'Negative':
        return 'Scandal'
    else:
        return 'Normal'

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
    # Use RoBERTa for sentiment analysis
    df['sentiment'] = df['headline'].progress_apply(classify_compound_score)

    print("Computing similarity to keywords...")
    df['keyword_similarity'] = df['headline'].progress_apply(lambda x: compute_similarity_to_keywords(x, keyword_embeddings))

    print("Finding potential scandals...")
    df['scandal'] = df.progress_apply(lambda x: find_scandals(x['keyword_similarity'], x['entities'], x['sentiment']), axis=1)

    # Reorder columns for better readability
    df = df[['predicted_category', 'sentiment', 'entities', 'headline', 'link', 'date', 'body', 'keyword_similarity','scandal']]
    
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
