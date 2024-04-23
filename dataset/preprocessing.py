import pickle
import re
import pandas as pd
import torch
from transformers import RobertaTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nlpaug.augmenter.word as naw
from torch.nn.utils.rnn import pad_sequence


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
torch.manual_seed(42)

def get_vocab_size():
    return tokenizer.vocab_size


def prepare_data(file_path, text, augment=True, rows=None, categories=None, stratified_sampling = False):

    # Load the data
    df = pd.read_csv(file_path)

    df = df[df['Category'] != 'sport']
    #df = df.sample(frac=1).reset_index(drop=True)

    if rows:
        df = df.head(rows)
    
    if categories:
        #convert the category to int
        categories = df['Category'].unique()
        category_to_int = {category: i for i, category in enumerate(categories)}

        with open('./category_to_int.pkl', 'wb') as handle:
            pickle.dump(category_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if stratified_sampling:
        balanced_subsets = []

        for category in df['Category'].unique():
            category_subset = df[df['Category'] == category]

            # Sample 75 rows from each category, with or without replacement depending on the count
            if len(category_subset) < 75:
                sampled_subset = category_subset.sample(75, replace=True)  # Oversampling if less than 75 rows
            else:
                sampled_subset = category_subset.sample(75, random_state=42)  # Sample 75 rows without replacement

            balanced_subsets.append(sampled_subset)

        # Concatenate all the balanced subsets to form a new DataFrame
        df = pd.concat(balanced_subsets, ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the resulting DataFrame

    with open('./category_to_int.pkl', 'rb') as handle:
        category_to_int = pickle.load(handle)

    df['Category'] = df['Category'].map(category_to_int)
    # Pre-process data
    df = preprocess_data(df, text=text)

    if augment:
        augmenters = [
            naw.SynonymAug(aug_src='wordnet', aug_p=0.1),
            #naw.RandomWordAug(action="insert"),
            naw.RandomWordAug(action="delete"),
            naw.RandomWordAug(action="swap")
        ]
        augmented_texts = augment_text(df, augmenters, num_augments=1)
        augmented_df = pd.DataFrame({'Text': augmented_texts, 'Category': df['Category']})
        df = augmented_df


    return df


def bpe_tokenizer(text):
    return tokenizer.tokenize(text)

def yield_tokens(data_iter, tokenizer=bpe_tokenizer):
    for text in data_iter:
        yield tokenizer(text)

def augment_text(dataframe, augmenters, num_augments=1):
    augmented_texts = []
    for text in dataframe['Text']:
        augmented_text = text
        for augmenter in augmenters:
            augmented_text = augmenter.augment(augmented_text, n=num_augments)
        augmented_texts.append(augmented_text)
    return augmented_texts



def preprocess_data(df, text):
    """Pre-process the data."""
    print("Pre-processing data...")
    # Lowercase conversion
    df[f'{text}'] = df[f'{text}'].apply(lambda x: x.lower())
    
    # Remove links
    df[f'{text}'] = df[f'{text}'].apply(lambda x: re.sub(r'http\S+', '', x))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df[f'{text}'] = df[f'{text}'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    df[f'{text}'] = df[f'{text}'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    # Remove more than one space
    df[f'{text}'] = df[f'{text}'].apply(lambda x: re.sub(r'\s+', ' ', x))


    #export the preprocessed data
    df.to_csv('./data/temp/pre_processed.csv', index=False)
    
    return df

def text_pipeline(x):
    """bpe tokenizes the input text and returns the input_ids tensor."""
    return tokenizer(x, 
                     padding='max_length',  # Adds padding
                     truncation=True,       # Truncates
                     max_length=512,        # Maximum sequence length
                     return_tensors='pt'    # PyTorch tensors
                    )['input_ids'].squeeze()  # Ensure it's a single tensor, not a batch


def collate_batch(batch):
    label_list, text_list = [], []
    for _text, _label in batch:
        text_list.append(_text)
        label_list.append(_label)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=tokenizer.pad_token_id)  # Pad using tokenizer's pad token
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return text_list, label_list

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")