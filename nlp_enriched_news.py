
from pathlib import Path
from sentence_transformers import SentenceTransformer
import scipy.spatial
import spacy
import pandas as pd
import joblib
from tqdm.auto import tqdm
from transformers import pipeline, RobertaTokenizer, RobertaForSequenceClassification
import warnings
from torchtext.data.utils import get_tokenizer


from training_model import TextClassifier
warnings.filterwarnings('ignore')
from colorama import Fore, Style, init
init()  # Initialize colorama for Windows
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence 
import constants

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

import pickle

# Initialize the models for embeddings 
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Assume these are defined or loaded appropriately
 
embed_dim = constants.emded_dim
num_class = constants.num_class

def yield_tokens(data_iter):
    tokenizer = get_tokenizer("basic_english")
    for text in data_iter:
        yield tokenizer(text)
# Loading the dictionary back

with open('vocab.pkl', 'rb') as vocab_file:
    vocab = pickle.load(vocab_file)

with open('category_to_int.pkl', 'rb') as handle:
    category_to_int = pickle.load(handle)



vocab_size = len(vocab)

# This assumes you also have the tokenizer and vocab ready
def text_pipeline(x, vocab):
    tokenizer = get_tokenizer("basic_english")
    return [vocab[token] for token in tokenizer(x)]

# Recreate the model structure
model = TextClassifier(len(vocab), constants.emded_dim, constants.num_class, num_heads=4)
model.load_state_dict(torch.load("topic_classifier.pth"))
model.eval()  # Set the model to evaluation mode

# Load the RoBERTa model for sentiment analysis
roberta_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_sentiment = pipeline("sentiment-analysis", model=roberta_model, tokenizer=roberta_tokenizer, return_all_scores=False)

# Load other necessary libraries and models
nlp = spacy.load('en_core_web_lg')  # For entity extraction

# Load the models for topic classification
vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')

# Load the data and keywords
data = pd.read_csv('./data/news.csv')
keywords = pd.read_csv('./data/environment_keywords.txt', header=None)[0].tolist()

# Compute embeddings for the keywords
keyword_embeddings = sentence_model.encode(keywords)

def extract_entities(text):
    """Extract ORG entities from the text using SpaCy."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'ORG']

def classify_sentiment(text):
    """Classify the sentiment based on RoBERTa's analysis."""
    result = roberta_sentiment(text)
    sentiment = result[0]['label']
    if sentiment == 'LABEL_0':
        return 'Negative'
    elif sentiment == 'LABEL_2':
        return 'Positive'
    else:
        return 'Neutral'
    
def pre_process_data(df):
    """Preprocess the data by removing duplicates, missing values, links, and images."""
    # Remove missing values
    df = df.dropna(subset=['headline', 'body'])
    # Remove duplicates
    df = df.drop_duplicates(subset=['headline', 'body'])

    #remove links from the body
    df['body'] = df['body'].str.replace(r'http\S+', '', case=False)

    #Remove images from the body
    df['body'] = df['body'].str.replace(r'\w*\.(?:jpg|gif|png)', '', case=False)

    return df

def compute_similarity_and_keyword(text, keyword_embeddings, keywords):
    """Compute similarity of text to environmental keywords and return the most similar keyword."""
    if text:
        text_embedding = sentence_model.encode([text])
        if text_embedding.ndim == 1:
            text_embedding = text_embedding.reshape(1, -1)
        similarities = scipy.spatial.distance.cdist(text_embedding, keyword_embeddings, "cosine")[0]
        max_similarity = 1 - min(similarities)
        most_similar_keyword = keywords[similarities.argmin()]  # Get the most similar keyword
        return max_similarity, most_similar_keyword
    else:
        return 0, None
    
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

# Predict categories with confidence threshold
def predict_categories(df, threshold=0.6):
    int_to_category = {v: k for k, v in category_to_int.items()}
    print("Predicting article categories with confidence scores...")
    results = []
    for text in df['body']:
        text_tensor = torch.tensor(text_pipeline(text, vocab), dtype=torch.int64).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            probabilities = torch.softmax(model(text_tensor), dim=1)
            max_prob, predicted_category = torch.max(probabilities, dim=1)
            if max_prob.item() >= threshold:
                results.append((int_to_category[predicted_category.item()], max_prob.item()))  # Convert to category name here
            else:
                results.append((None, max_prob.item()))  # Use None or a placeholder if confidence is below threshold
    # Assuming df has an index that's aligned with the iteration order
    df['predicted_category'] = [res[0] for res in results]  # This will now hold the category names
    df['confidence'] = [res[1] for res in results]
    return df[df['predicted_category'].notnull()]  # Return only confident predictions

# Main processing
def process_data(df):
    """Process the data to add entities, predicted categories, and sentiment."""
    tqdm.pandas()  # Initialize tqdm for pandas apply

    df = pre_process_data(df)
    print("Extracting entities from headlines and bodies...")
    df['entities'] = df['headline'].progress_apply(extract_entities) + df['body'].progress_apply(extract_entities)
    
    # Process data with confidence threshold
    df = predict_categories(df)
    
    print("Analyzing sentiment of articles...")
    # Use RoBERTa for sentiment analysis
    df['sentiment'] = df['headline'].progress_apply(classify_sentiment)
    
    print("Computing similarity to keywords...")
    similarity_and_keyword = df['body'].progress_apply(lambda x: compute_similarity_and_keyword(x, keyword_embeddings, keywords))
    df['keyword_similarity'], df['most_similar_keyword'] = zip(*similarity_and_keyword)

    print("Finding potential scandals...")
    df['scandal'] = df.progress_apply(lambda x: find_scandals(x['keyword_similarity'], x['entities'], x['sentiment']), axis=1)

    # Reorder columns for better readability
    df = df[['predicted_category', 'sentiment', 'entities', 'headline', 'link', 'date', 'body', 'keyword_similarity','scandal','most_similar_keyword']]
    
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
