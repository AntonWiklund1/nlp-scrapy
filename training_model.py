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

def yield_tokens(data_iter):
    tokenizer = get_tokenizer("basic_english")
    for text in data_iter:
        yield tokenizer(text)

# Load data and prepare vocab
def prepare_data_and_vocab():
    train_df = pd.read_csv('./data/bbc_news_train.csv')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_df['Text']), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return train_df, vocab, tokenizer

def text_pipeline(x, vocab):
    tokenizer = get_tokenizer("basic_english")
    return [vocab[token] for token in tokenizer(x)]

class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = torch.tensor(text_pipeline(self.texts.iloc[idx], self.vocab), dtype=torch.int64)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.int64)
        return text, label

def collate_batch(batch):
    label_list, text_list = [], []
    for _text, _label in batch:
        # Ensure text is not empty
        if len(_text) > 0:
            text_list.append(_text)
            label_list.append(_label)
    # Safety check for empty batch or text_list
    if len(text_list) == 0:
        raise ValueError("No valid text found in batch to pad.")
    # Padding text sequences
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return text_list, label_list

# Define the model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_dim, 200)
        self.fc2 = nn.Linear(200, num_class)
    
    def forward(self, text):
        embedded = self.embedding(text).mean(dim=1)
        hidden = F.leaky_relu(self.fc1(embedded))
        return self.fc2(hidden)

def train(dataloader, model, loss_fn, optimizer, device):
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

def main():
    # Load data and prepare vocab
    train_df, vocab, tokenizer = prepare_data_and_vocab()

    # Convert categories to integer labels
    unique_categories = train_df['Category'].unique()
    category_to_int = {category: index for index, category in enumerate(unique_categories)}
    with open('category_to_int.pkl', 'wb') as handle:
        pickle.dump(category_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_df['Category'] = train_df['Category'].map(category_to_int)

    # Prepare the full dataset
    full_dataset = NewsDataset(train_df['Text'].reset_index(drop=True), 
                               train_df['Category'].reset_index(drop=True), vocab)

    # K-Fold Cross-Validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
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
        early_stopping_patience = 10
        early_stopping_counter = 0
        best_val_loss_fold = float('inf')

        # Initialize training and validation loss history for plotting
        train_losses, val_losses = [], []

        # Set up the DataLoaders for the training and validation sets
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(full_dataset, batch_size=16, sampler=train_subsampler, collate_fn=collate_batch)
        val_loader = DataLoader(full_dataset, batch_size=16, sampler=val_subsampler, collate_fn=collate_batch)
        
        # Define the model, loss function, and optimizer
        model = TextClassifier(len(vocab), constants.emded_dim, constants.num_class)
        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
        scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=epochs)
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

            # Early stopping and checkpointing logic
            if val_loss < best_val_loss_fold:
                best_val_loss_fold = val_loss
                if val_loss < best_val_loss_global:
                    best_val_loss_global = val_loss
                    best_model_state_global = model.state_dict()
                    best_fold = fold
                    best_accuracy = accuracy
                    torch.save(best_model_state_global, "best_text_classifier_global.pth")
                    print(f"{Fore.GREEN}New best model found for fold {fold} with validation loss {best_val_loss_global:.4f}{Style.RESET_ALL}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"{Fore.YELLOW}EarlyStopping counter: {early_stopping_counter} out of {early_stopping_patience}{Style.RESET_ALL}")
                if early_stopping_counter >= early_stopping_patience:
                    print(f"{Fore.RED}Early stopping triggered.{Style.RESET_ALL}")
                    break

        print(f"Training complete for fold {fold} with best validation loss {best_val_loss_fold:.4f}")

    print("Training complete for all folds!")
    print(f"Best accuracy: {(best_accuracy * 100):.2f}% in fold {best_fold}")
    if best_model_state_global is not None:
        torch.save(best_model_state_global, "best_text_classifier_global.pth")
        print(f"Best model saved with validation loss {best_val_loss_global:.4f} from fold {best_fold}")

    # Optionally, plot the learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()
    plt.savefig('./results/learning_curve.png')

    # Save vocabulary and category mapping
    with open('vocab.pkl', 'wb') as vocab_file:
        pickle.dump(vocab, vocab_file)

if __name__ == "__main__":
    main()
