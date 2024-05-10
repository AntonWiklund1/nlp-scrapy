import pickle
import re
import pandas as pd
import torch
from transformers import RobertaTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nlpaug.augmenter.word as naw
from torch.nn.utils.rnn import pad_sequence
import constants
import random
import numpy as np


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pd.set_option('mode.chained_assignment', None)

def get_vocab_size():
    return tokenizer.vocab_size


def prepare_data(file_path, text, augment=False, rows=None, categories=None, stratified_sampling=False, return_og=False):
    # Load the data
    set_seed()
    df = pd.read_csv(file_path)
    df = df[df['Category'] != 'sport']

    temp_df = df.copy()
    #shuffle the data
    #temp_df = temp_df.sample(frac=1, random_state=42).reset_index(drop=True)

    if rows:
        # Ensure even distribution of categories if rows is specified
        num_categories = len(temp_df['Category'].unique())
        samples_per_category = rows // num_categories
        subsets = []

        for category in temp_df['Category'].unique():
            category_subset = temp_df[temp_df['Category'] == category]
            if len(category_subset) < samples_per_category:
                sampled_subset = category_subset.sample(samples_per_category, replace=True)
            else:
                sampled_subset = category_subset.sample(samples_per_category, random_state=42)
            subsets.append(sampled_subset)

        temp_df = pd.concat(subsets).sample(frac=1, random_state=42).reset_index(drop=True)

    if categories:
        # Convert the category to int
        categories = temp_df['Category'].unique()
        category_to_int = {category: i for i, category in enumerate(categories)}
        with open('./category_to_int.pkl', 'wb') as handle:
            pickle.dump(category_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if stratified_sampling:
        balanced_subsets = []
        for category in temp_df['Category'].unique():
            category_subset = temp_df[temp_df['Category'] == category]
            if len(category_subset) < 75:
                sampled_subset = category_subset.sample(75, replace=True)
            else:
                sampled_subset = category_subset.sample(75, random_state=42)
            balanced_subsets.append(sampled_subset)
        temp_df = pd.concat(balanced_subsets, ignore_index=True)
        temp_df = temp_df.sample(frac=1, random_state=42).reset_index(drop=True)

    with open('./category_to_int.pkl', 'rb') as handle:
        category_to_int = pickle.load(handle)


    temp_df['Category'] = temp_df['Category'].map(category_to_int)
    temp_df = preprocess_data(temp_df, text=text)

    if augment:
        augmenters = [
            naw.SynonymAug(aug_src='wordnet', aug_p=0.1),
            naw.RandomWordAug(action="delete"),
            naw.RandomWordAug(action="swap")
        ]
        augmented_texts = augment_text(temp_df, augmenters, num_augments=1)
        augmented_temp_df = pd.DataFrame({'Text': augmented_texts, 'Category': temp_df['Category']})
        temp_df = augmented_temp_df

    if return_og:
        return temp_df, df
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
    """Pre-process the data using optimized methods."""
    # Precompile Regex and reuse objects
    regex_links = re.compile(r'http\S+')
    regex_spaces = re.compile(r'\s+')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Lowercase conversion and remove links
    df[text] = df[text].str.lower().replace(regex_links, '')

    # Remove stopwords and lemmatize in one pass
    def process_text(text):
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        return ' '.join(lemmatized_words)

    df[text] = df[text].apply(process_text)

    # Remove more than one space
    df[text] = df[text].replace(regex_spaces, ' ', regex=True)

    # Export the preprocessed data
    df.to_csv('./data/temp/pre_processed.csv', index=False)

    return df


def text_pipeline(x):
    """bpe tokenizes the input text and returns the input_ids tensor."""
    return tokenizer(x, 
                     padding='max_length',  # Adds padding
                     truncation=True,       # Truncates
                     max_length=constants.sequence_length,        # Maximum sequence length
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