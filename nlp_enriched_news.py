
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

from model import TextClassifier
warnings.filterwarnings('ignore')
from colorama import Fore, Style, init
init()  # Initialize colorama for Windows
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence 
import constants

from torchtext.data.utils import get_tokenizer

import pickle
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataset.preprocessing import get_vocab_size, prepare_data, collate_batch
from dataset.news_dataset import NewsDataset

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import TextClassifier

#nltk.download('stopwords')
#nltk.download('wordnet')


# Initialize the models for embeddings 
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

with open('category_to_int.pkl', 'rb') as handle:
    category_to_int = pickle.load(handle)


# Load the RoBERTa model for sentiment analysis
roberta_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_sentiment = pipeline("sentiment-analysis", model=roberta_model, tokenizer=roberta_tokenizer, return_all_scores=False)

# Load other necessary libraries and models
nlp = spacy.load('en_core_web_lg')  # For entity extraction

keywords = pd.read_csv('./data/environment_keywords.txt', header=None)[0].tolist()

# Compute embeddings for the keywords
keyword_embeddings = sentence_model.encode(keywords)

def load_and_predict_categories():
    int_to_category = {v: k for k, v in category_to_int.items()}
    
    df = prepare_data("./data/scraped/bbc_articles.csv", text='body')

    dataset = NewsDataset(df['body'].reset_index(drop=True), df['Category'].reset_index(drop=True))
    loader = DataLoader(dataset, batch_size=constants.batch_size, collate_fn=collate_batch)


    vocab_size = get_vocab_size()
    model = TextClassifier(vocab_size, constants.embed_dim, constants.num_class, constants.num_heads, constants.dropout_rate, constants.layer_size, constants.number_of_layers)
    model.load_state_dict(torch.load("./results/fine_tuned_model.pth", map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    predictions = []
    with torch.no_grad():
        for texts, labels in loader:
            output = model(texts)
            predictions.extend(output.argmax(1).tolist())
            total_accuracy += (output.argmax(1) == labels).sum().item()
            total_count += len(labels)

    df['predicted_category'] = [int_to_category[pred] for pred in predictions]
    df['confidence'] = [output[idx].max().item() for idx, output in zip(predictions, output)]

    # Sort by confidence in descending order and select the top 300
    top_df = df.sort_values('confidence', ascending=False).head(300)
    
    return top_df

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
    # Lowercase conversion
    df['body'] = df['body'].apply(lambda x: x.lower())
    
    #remove links
    df['body'] = df['body'].apply(lambda x: re.sub(r'http\S+', '', x))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['body'] = df['body'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    df['body'] = df['body'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    df['body'] = df['body'].apply(lambda x: re.sub(r'\s+', ' ', x))

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



# Main processing
def process_data(df):
    """Process the data to add entities, predicted categories, and sentiment."""

    df = load_and_predict_categories()

    tqdm.pandas()  # Initialize tqdm for pandas apply

    df = pre_process_data(df)

    print("Extracting entities from headlines and bodies...")
    df['entities'] = df['headline'].progress_apply(extract_entities) + df['body'].progress_apply(extract_entities)

    
    print("Analyzing sentiment of articles...")
    # Use RoBERTa for sentiment analysis
    df['sentiment'] = df['headline'].progress_apply(classify_sentiment)
    
    print("Computing similarity to keywords...")
    similarity_and_keyword = df['body'].progress_apply(lambda x: compute_similarity_and_keyword(x, keyword_embeddings, keywords))
    df['keyword_similarity'], df['most_similar_keyword'] = zip(*similarity_and_keyword)

    print("Finding potential scandals...")
    df['scandal'] = df.progress_apply(lambda x: find_scandals(x['keyword_similarity'], x['entities'], x['sentiment']), axis=1)

    actual_categories = df['category'].tolist()  # Fill None with 'Unknown'
    predicted_categories = df['predicted_category'].tolist()  # Fill None with 'Unknown'

    # Check if there are any None values left
    print("Actual categories contain None:", None in actual_categories)
    print("Predicted categories contain None:", None in predicted_categories)

    #print how many None in the predicted categories
    print("Number of None values in predicted categories:", predicted_categories.count(None))

    # Metrics calculation
    accuracy = accuracy_score(actual_categories, predicted_categories)
    precision, recall, f1, _ = precision_recall_fscore_support(actual_categories, predicted_categories, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    
    # Reorder columns for better readability
    try:
        return df[['predicted_category', 'sentiment', 'entities', 'headline', 'link', 'date', 'body', 'keyword_similarity','scandal','most_similar_keyword']]
    except KeyError:
        return df

# Apply the processing to the data
processed_data = process_data()

# Save the processed data
output_file = Path('./results/news_enriched.csv')
try:
    processed_data.to_csv(output_file, sep=",", index=False)
    print(f"{Fore.GREEN}File saved successfully to {output_file}{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}Failed to save the file: {e}{Style.RESET_ALL}")
