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

def get_vocab_size():
    return tokenizer.vocab_size


def prepare_data(file_path, text, augment=True, rows=None):

    if rows:
        df = pd.read_csv(f'{file_path}')[:rows]
    else:
        df = pd.read_csv(f'{file_path}')

    df = df[df['Category'] != 'sport']
    #df = df.sample(frac=1).reset_index(drop=True)

    #convert the category to int
    categories = df['Category'].unique()
    category_to_int = {category: i for i, category in enumerate(categories)}

    with open('./category_to_int.pkl', 'wb') as handle:
        pickle.dump(category_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    df['Category'] = df['Category'].map(category_to_int)
    # Pre-process data
    df = preprocess_data(df, text=text)

    if augment:
        augmenter = naw.SynonymAug(aug_src='wordnet', aug_p=0.1)  # 10% probability of synonym replacement
        augmented_texts = augment_text(df, augmenter, num_augments=2)  # Each text is augmented twice
        augmented_df = pd.DataFrame({'Text': augmented_texts, 'Category': df['Category'].repeat(2)})  # Repeat labels for augmented texts
        df = pd.concat([df, augmented_df]).sample(frac=1).reset_index(drop=True)  # Shuffle the dataset

    return df

def bpe_tokenizer(text):
    return tokenizer.tokenize(text)

def yield_tokens(data_iter, tokenizer=bpe_tokenizer):
    for text in data_iter:
        yield tokenizer(text)

def augment_text(dataframe, augmenter, num_augments=1):
    augmented_texts = []
    for text in dataframe['Text']:
        augmented_texts.extend(augmenter.augment(text, n=num_augments))
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


    #exprot the preprocessed data
    df.to_csv('./data/temp/pre_prosseced.csv', index=False)
    
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