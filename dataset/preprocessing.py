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


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
torch.manual_seed(42)

def get_vocab_size():
    return tokenizer.vocab_size


def prepare_data(file_path, text, augment=True, rows=None, categories=None, stratified_sampling=False):
    # Load the data
    df = pd.read_csv(file_path)
    df = df[df['Category'] != 'sport']

    if rows:
        # Ensure even distribution of categories if rows is specified
        num_categories = len(df['Category'].unique())
        samples_per_category = rows // num_categories
        subsets = []

        for category in df['Category'].unique():
            category_subset = df[df['Category'] == category]
            if len(category_subset) < samples_per_category:
                sampled_subset = category_subset.sample(samples_per_category, replace=True)
            else:
                sampled_subset = category_subset.sample(samples_per_category, random_state=42)
            subsets.append(sampled_subset)

        df = pd.concat(subsets).sample(frac=1, random_state=42).reset_index(drop=True)

    if categories:
        # Convert the category to int
        categories = df['Category'].unique()
        category_to_int = {category: i for i, category in enumerate(categories)}
        with open('./category_to_int.pkl', 'wb') as handle:
            pickle.dump(category_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if stratified_sampling:
        balanced_subsets = []
        for category in df['Category'].unique():
            category_subset = df[df['Category'] == category]
            if len(category_subset) < 75:
                sampled_subset = category_subset.sample(75, replace=True)
            else:
                sampled_subset = category_subset.sample(75, random_state=42)
            balanced_subsets.append(sampled_subset)
        df = pd.concat(balanced_subsets, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    with open('./category_to_int.pkl', 'rb') as handle:
        category_to_int = pickle.load(handle)


    df['Category'] = df['Category'].map(category_to_int)
    df = preprocess_data(df, text=text)

    if augment:
        augmenters = [
            naw.SynonymAug(aug_src='wordnet', aug_p=0.1),
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