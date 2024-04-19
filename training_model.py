import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import constants
import pickle
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from colorama import Fore, Style, init
init()  # Initialize colorama for Windows
from warnings import filterwarnings
filterwarnings('ignore')
from datetime import datetime
import time
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


#set manuel seed
torch.manual_seed(42)

def plot_confusion_matrix(actuals, predictions, classes, title='Confusion Matrix'):
    cm = confusion_matrix(actuals, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{title}')
    plt.savefig('./results/confusion_matrix.png')

# nltk.download('stopwords')
# nltk.download('wordnet')

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
def bpe_tokenizer(text):
    return tokenizer.tokenize(text)

def yield_tokens(data_iter, tokenizer=bpe_tokenizer):
    for text in data_iter:
        yield tokenizer(text)

def pre_process_data(df, text='Text'):
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

# Load data and prepare vocab
def prepare_data_and_vocab():

    train_df = pd.read_csv('./data/bbc_news_train.csv')
    #drop the rows where category is sport
    train_df = train_df[train_df['Category'] != 'sport']
    #train_df = train_df.sample(frac=1).reset_index(drop=True)
    #tokenizer = get_tokenizer('basic_english')
    
    # Pre-process data
    train_df = pre_process_data(train_df)
    
    # Build vocabulary
    vocab = build_vocab_from_iterator(yield_tokens(train_df['Text']), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])


    with open('vocab.pkl', 'wb') as vocab_file:
        pickle.dump(vocab, vocab_file)

    return train_df


def text_pipeline(x):
    # Adjust this function to truncate and pad the inputs to a fixed length
    return tokenizer(x, 
                     padding='max_length',  # Adds padding
                     truncation=True,       # Truncates
                     max_length=512,        # Maximum sequence length
                     return_tensors='pt'    # PyTorch tensors
                    )['input_ids'].squeeze()  # Ensure it's a single tensor, not a batch


vocab_size = tokenizer.vocab_size

class SparseMultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=10):
        super(SparseMultiHeadAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.window_size = window_size

    def forward(self, embeddings):
        batch_size, seq_length, _ = embeddings.size() # (batch_size, seq_length, embed_dim)
        # Applying windowed attention:
        # Only allow attention within a window around each token.
        attn_mask = torch.full((seq_length, seq_length), float('-inf')).to(embeddings.device)
        for i in range(seq_length):
            left = max(0, i - self.window_size)
            right = min(seq_length, i + self.window_size + 1)
            attn_mask[i, left:right] = 0

        embeddings = embeddings.permute(1, 0, 2)  # permute for nn.MultiheadAttention (seq_len, batch_size, embed_dim)
        attn_output, attn_output_weights = self.multihead_attn(embeddings, embeddings, embeddings, attn_mask=attn_mask)
        attn_output = attn_output.permute(1, 0, 2)  # permute back to (batch_size, seq_len, embed_dim)
        max_pooled_output = torch.max(attn_output, dim=1)[0]
        return max_pooled_output, attn_output_weights


class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Directly use the text_pipeline which now utilizes the tokenizer
        text = text_pipeline(self.texts.iloc[idx])
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.int64)
        return text, label

def collate_batch(batch):
    label_list, text_list = [], []
    for _text, _label in batch:
        text_list.append(_text)
        label_list.append(_label)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=tokenizer.pad_token_id)  # Pad using tokenizer's pad token
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return text_list, label_list


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        
    def forward(self, embeddings):
        embeddings = embeddings.permute(1, 0, 2)
        attn_output, attn_output_weights = self.multihead_attn(embeddings, embeddings, embeddings)
        attn_output = attn_output.permute(1, 0, 2)
        max_pooled_output = torch.max(attn_output, dim=1)[0]  # Use max pooling along the sequence dimension
        return max_pooled_output, attn_output_weights


# Define the modelclass TextClassifier(nn.Module):
class TextClassifier(nn.Module):
    """A text classifier model with an attention mechanism and multiple hidden layers."""
    def __init__(self, vocab_size, embed_dim, num_class, num_heads = 4, window_size=10):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Embedding layer
        self.dropout = nn.Dropout(0.6)  # First dropout layer
        
        print(f"Vocabulary Size: {vocab_size}")
        print(f"Embedding Dimension: {embed_dim}")
        print(f"Total Weights in Embedding Matrix: {vocab_size * embed_dim:,}")

        # MultiHead Attention Layer
        #self.attention = MultiHeadAttentionLayer(embed_dim, num_heads)
        self.attention = SparseMultiHeadAttentionLayer(embed_dim, num_heads, window_size)

        # Additional hidden layers
        self.fc1 = nn.Linear(embed_dim, 512)  # First hidden layer
        self.fc2 = nn.Linear(512, 256)  # Second hidden layerr
        self.fc3 = nn.Linear(256, num_class)  # Output layer

    def forward(self, text):
        # Embedding and initial dropout
        embedded = self.embedding(text)

        embedded = self.dropout(embedded)
        
        # Apply attention
        context_vector, attention_weights = self.attention(embedded)
        
        # Passing through the hidden layers with activation functions and dropout
        hidden1 = F.leaky_relu(self.fc1(context_vector))
        hidden2 = F.leaky_relu(self.fc2(hidden1))
        output = self.fc3(hidden2)

        return output

def train(dataloader, model, loss_fn, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss, total_count = 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        total_count += X.size(0)
    avg_loss = total_loss / total_count
    print(f"Training Loss: {loss:>8f}")
    return avg_loss  # Return the average loss over the epoch

def evaluate(dataloader, model, loss_fn, device):
    """Evaluate the model on the validation set."""
    model.eval()
    total_acc, total_count, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * X.size(0)
            total_acc += (pred.argmax(1) == y).sum().item()
            total_count += y.size(0)
    accuracy = total_acc / total_count
    avg_loss = total_loss / total_count
    print(f'Validation Accuracy: {(accuracy * 100):>0.1f}%, Avg loss: {avg_loss:>8f}')
    return accuracy, avg_loss

def test(model,device):
    """Test the model on the test set and print the accuracy."""
    test_df = pd.read_csv('./data/bbc_news_tests.csv')
    test_df = test_df[test_df['Category'] != 'sport']
    test_df = pre_process_data(test_df, text='Text')

    with open('category_to_int.pkl', 'rb') as handle:
        category_to_int = pickle.load(handle)

    test_df['Category'] = test_df['Category'].map(category_to_int)
    test_dataset = NewsDataset(test_df['Text'].reset_index(drop=True), test_df['Category'].reset_index(drop=True))
    test_loader = DataLoader(test_dataset, batch_size=constants.batch_size, collate_fn=collate_batch)

    model.eval()
    total_acc, total_count = 0, 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            total_acc += (pred.argmax(1) == y).sum().item()
            total_count += y.size(0)
    accuracy = total_acc / total_count
    print(f'Test Accuracy: {(accuracy * 100):>0.1f}%')
    return accuracy, all_preds, all_labels

def test_single_sample(model, device):
    model.eval()  # Set the model to evaluation mode

    test_df = pd.read_csv('./data/bbc_news_tests.csv')
    test_df = test_df[test_df['Category'] != 'sport']
    test_df = pre_process_data(test_df)

    with open('category_to_int.pkl', 'rb') as handle:
        category_to_int = pickle.load(handle)


    test_df['Category'] = test_df['Category'].map(category_to_int)
    dataset = NewsDataset(test_df['Text'].reset_index(drop=True), test_df['Category'].reset_index(drop=True))

    with torch.no_grad():
        for i in range(min(len(dataset), 5)):  # Test with 5 samples
            sample = dataset[i]
            text, label = sample
            text = text.unsqueeze(0).to(device)  # Add batch dimension
            label = label.unsqueeze(0).to(device)
            output = model(text)
            prediction = output.argmax(1)
            print(f"Test Sample {i}: True Label: {label.item()}, Predicted Label: {prediction.item()}")

def test_scraped_data():

    with open('category_to_int.pkl', 'rb') as handle:
        category_to_int = pickle.load(handle)

    int_to_category = {index: category for category, index in category_to_int.items()}


    model = TextClassifier(vocab_size, constants.emded_dim, constants.num_class)
    model.load_state_dict(torch.load("./models/topic_classifier.pth", map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    # Load the data and keywords
    df = pd.read_csv('./data/scraped/bbc_articles.csv')

    # Pre-process data
    df = pre_process_data(df, text='body')

    # Compute embeddings for the keywords
    results = []
    
    for text in tqdm(df['body'], desc="Predicting categories", total=df.shape[0]):
        text_tensor = torch.tensor(text_pipeline(text), dtype=torch.int64).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            probabilities = torch.softmax(model(text_tensor), dim=1)
            max_prob, predicted_category = torch.max(probabilities, dim=1)
            category_name = int_to_category[predicted_category.item()]
            results.append((category_name, max_prob.item()))
    
    df['predicted_category'] = [res[0] for res in results]
    df['confidence'] = [res[1] for res in results]
    
    # Sort by confidence in descending order and select the top 300
    df = df.sort_values(by='confidence', ascending=False).head(300)
    
    actual_categories = df['Category'].tolist()  
    predicted_categories = df['predicted_category'].tolist()  

    # Check if there are any None values left
    print("Actual categories contain None:", None in actual_categories)
    print("Predicted categories contain None:", None in predicted_categories)

    #print how many None in the predicted categories
    print("Number of None values in predicted categories:", predicted_categories.count(None))

    # Metrics calculation
    accuracy = accuracy_score(actual_categories, predicted_categories)
    precision, recall, f1, _ = precision_recall_fscore_support(actual_categories, predicted_categories, average='weighted')

    plot_confusion_matrix(actual_categories, predicted_categories, list(category_to_int.keys()), title='Confusion Matrix Scrape Data')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    return accuracy


def main():
    start_time = time.time()

    torch.cuda.empty_cache()
  
    # Load data and prepare vocab
    train_df = prepare_data_and_vocab()

    # Convert categories to integer labels
    unique_categories = train_df['Category'].unique()
    category_to_int = {category: index for index, category in enumerate(unique_categories)}
    with open('category_to_int.pkl', 'wb') as handle:
        pickle.dump(category_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_df['Category'] = train_df['Category'].map(category_to_int)

    # Prepare the full dataset
    full_dataset = NewsDataset(train_df['Text'].reset_index(drop=True), 
                               train_df['Category'].reset_index(drop=True))

    # K-Fold Cross-Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    best_accuracy = 0.0
    best_fold = -1
    epochs = 100
    best_val_loss_global = float('inf')
    best_model_state_global = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"FOLD {fold}")

        # Initialize the early stopping criteria and fold-specific best validation loss
        early_stopping_patience = 5
        early_stopping_counter = 0
        best_val_loss_fold = float('inf')

        # Initialize training and validation loss history for plotting
        train_losses, val_losses = [], []

        # Set up the DataLoaders for the training and validation sets
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(full_dataset, batch_size=constants.batch_size, sampler=train_subsampler, collate_fn=collate_batch)
        val_loader = DataLoader(full_dataset, batch_size=constants.batch_size, sampler=val_subsampler, collate_fn=collate_batch)
        
        # Define the model, loss function, and optimizer

        model = TextClassifier(vocab_size, constants.emded_dim, constants.num_class)
        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        scheduler = OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=epochs)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

        # Training and validation for the current fold
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs} in fold {fold}")

            # Training phase
            train_loss = train(train_loader, model, loss_fn, optimizer, device)
            train_losses.append(train_loss)

            # Validation phase
            accuracy, val_loss = evaluate(val_loader, model, loss_fn, device)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            ## Early stopping and checkpointing logic
            # If the val loss is decreasing then we save the model
            if val_loss < best_val_loss_fold:
                best_val_loss_fold = val_loss
                if val_loss < best_val_loss_global:
                    best_val_loss_global = val_loss
                    best_model_state_global = model.state_dict()
                    best_fold = fold
                    best_accuracy = accuracy
                    torch.save(best_model_state_global, "./models/topic_classifier.pth")
                    print(f"{Fore.GREEN}New best model found for fold {fold} with validation loss {best_val_loss_global:.4f}{Style.RESET_ALL}")
                early_stopping_counter = 0
            # If the val loss is not decreasing then we increase the early stopping counter
            else:
                early_stopping_counter += 1
                print(f"{Fore.YELLOW}EarlyStopping counter: {early_stopping_counter} out of {early_stopping_patience}{Style.RESET_ALL}")
                # If the early stopping counter is greater than the patience then we break the loop
                if early_stopping_counter >= early_stopping_patience:
                    print(f"{Fore.RED}Early stopping triggered.{Style.RESET_ALL}")
                    break

        print(f"Training complete for fold {fold} with best validation loss {best_val_loss_fold:.4f}")

    print("Training complete for all folds!")
    print(f"Best accuracy: {(best_accuracy * 100):.2f}% in fold {best_fold}")

    # Save the best model
    if best_model_state_global is not None:
        torch.save(best_model_state_global, "./models/topic_classifier.pth")
        print(f"Best model saved with validation loss {best_val_loss_global:.4f} from fold {best_fold}")


    # Test the best model
    model = TextClassifier(vocab_size, constants.emded_dim, constants.num_class)
    model.load_state_dict(best_model_state_global)
    model.to(device)
    test_accuracy, all_preds, all_labels = test(model, device)

    category_names = [cat for cat, index in sorted(category_to_int.items(), key=lambda x: x[1])]

    plot_confusion_matrix(all_labels, all_preds, category_names, title='Confusion Matrix Test Data')

    test_single_sample(model, device)

    accuracy_on_scrape_data = test_scraped_data()

    # Optionally, plot the learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig('./results/learning_curve.png')
    plt.show()


    try:
        with open('./results/best_accuracy_log.txt', 'a') as f:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{current_time} - Best accuracy: {(best_accuracy * 100):.2f}% in fold {best_fold}. Test accuracy: {(test_accuracy * 100):.2f}%. Scrape data accuracy: {accuracy_on_scrape_data}. Time taken: {((time.time() - start_time) / 60 ):.2f} minutes, using device: {device}\n")
            print(f"Logged to file")
    except Exception as e:
        print(f"Error writing to log file: {e}")

    print(f"{Fore.GREEN}Complete!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
