import pandas as pd
import torch
import torch.nn as nn
import pickle
import re
import scipy.spatial
from pathlib import Path
from tqdm.auto import tqdm
from transformers import pipeline, RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
import spacy
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import constants
from model import TextClassifier
from dataset.preprocessing import get_vocab_size, prepare_data, collate_batch
from dataset.news_dataset import NewsDataset

# Initialize and configure necessary modules
nlp = spacy.load('en_core_web_lg')  # For entity extraction
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
roberta_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_sentiment = pipeline("sentiment-analysis", model=roberta_model, tokenizer=roberta_tokenizer)

# Load category mapping
with open('category_to_int.pkl', 'rb') as handle:
    category_to_int = pickle.load(handle)

# Load keywords and compute embeddings
keywords = pd.read_csv('./data/environment_keywords.txt', header=None)[0].tolist()
keyword_embeddings = sentence_model.encode(keywords)

def load_and_predict_categories():
    int_to_category = {v: k for k, v in category_to_int.items()}
    df = prepare_data("./data/scraped/bbc_articles.csv", text='body', augment=False)
    dataset = NewsDataset(df['body'].reset_index(drop=True), df['Category'].reset_index(drop=True))
    loader = DataLoader(dataset, batch_size=constants.batch_size, collate_fn=collate_batch)
    with open('./results/topic_classifier.pkl', "rb") as f:
        model = pickle.load(f)
    model.eval()

    predictions, confidences = [], []
    with torch.no_grad():
        for texts, labels in loader:
            output = model(texts)
            predictions.extend(output.argmax(dim=1).tolist())
            confidences.extend(output.max(dim=1).values.tolist())

    df['predicted_category'] = [int_to_category[pred] for pred in predictions]
    df['confidence'] = confidences
    df['Category'] = df['Category'].apply(lambda x: int_to_category[x])
    return df.sort_values('confidence', ascending=False).head(300)

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'ORG']

def classify_sentiment(text):
    result = roberta_sentiment(text)[0]['label']
    return {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}.get(result, 'Neutral')

def pre_process_data(df):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        return re.sub(r'\s+', ' ', text).strip()

    df['body'] = df['body'].apply(clean_text)
    return df

def find_scandals(keyword_similarity, entities, sentiment):
    if keyword_similarity > 0.1 and entities and sentiment == 'Negative':
        return 'Scandal'
    return 'Normal'

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

def process_data():

    #initialize tqdm
    tqdm.pandas()

    print("Loading and predicting categories...")
    df = load_and_predict_categories()
    print("extracting entities...")
    df['entities'] = df['headline'].progress_apply(extract_entities) + df['body'].apply(extract_entities)

    print("sentiment analysis")
    df['sentiment'] = df['headline'].progress_apply(classify_sentiment)

    print("computing similarity and keyword...")
    df['keyword_similarity'], df['most_similar_keyword'] = zip(*df['body'].progress_apply(lambda x: compute_similarity_and_keyword(x, keyword_embeddings, keywords)))
    
    print("finding scandals...")
    df['scandal'] = df.progress_apply(lambda x: find_scandals(x['keyword_similarity'], x['entities'], x['sentiment']), axis=1)

    accuracy = accuracy_score(df['Category'], df['predicted_category'])
    precision, recall, f1, _ = precision_recall_fscore_support(df['Category'], df['predicted_category'], average='weighted')

    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}")
    return df[['predicted_category', 'Category', 'sentiment', 'entities', 'headline', 'url', 'time', 'body', 'keyword_similarity', 'scandal', 'most_similar_keyword']]

# Apply the processing to the data
processed_data = process_data()

# Save the processed data
output_file = Path('./results/news_enriched.csv')
try:
    processed_data.to_csv(output_file, sep=",", index=False)
    print("File saved successfully to", output_file)
except Exception as e:
    print("Failed to save the file:", e)
