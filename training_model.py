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
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, num_class)
    
    def forward(self, text):
        embedded = self.embedding(text).mean(dim=1)
        hidden = F.leaky_relu(self.fc1(embedded))
        return self.fc2(hidden)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss = loss.item()
            print(f"Train loss: {loss:>7f}")


def evaluate(dataloader, model, loss_fn):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            total_acc += (pred.argmax(1) == y).sum().item()
            total_count += y.size(0)
    accuracy = total_acc / total_count
    print(f'Validation Accuracy: {(accuracy * 100):>0.1f}%, Avg loss: {loss:>8f}')
    return accuracy

def main():
    # Load data and prepare vocab
    train_df, vocab, tokenizer = prepare_data_and_vocab()
    
    # Convert categories to integer labels
    unique_categories = train_df['Category'].unique()
    category_to_int = {category: index for index, category in enumerate(unique_categories)}
    train_df['Category'] = train_df['Category'].map(category_to_int)

    # Prepare the full dataset
    full_dataset = NewsDataset(train_df['Text'].reset_index(drop=True), 
                               train_df['Category'].reset_index(drop=True), vocab)

    # K-Fold Cross-Validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    best_accuracy = 0.0
    best_fold = 0

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"FOLD {fold}")        
        # Split dataset into training and validation
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(full_dataset, batch_size=16, sampler=train_subsampler, collate_fn=collate_batch)
        val_loader = DataLoader(full_dataset, batch_size=16, sampler=val_subsampler, collate_fn=collate_batch)
        
        # Define the model, loss function, and optimizer
        model = TextClassifier(len(vocab), embed_dim=100, num_class=len(set(train_df['Category'])))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)  # Example weight_decay
        
        # Training and Validation for the current fold
        for epoch in range(100):  # Adjust the number of epochs if necessary
            print(f" \n--------------------------\nEpoch {epoch + 1} in fold: {fold}")
            train(train_loader, model, loss_fn, optimizer)
            accuracy = evaluate(val_loader, model, loss_fn)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_fold = fold
                print(f"New best model found at epoch {epoch + 1} of fold {fold} with accuracy: {(best_accuracy * 100):>0.1f}%")
                # Save the model
                torch.save(model.state_dict(), "best_text_classifier.pth")

        
        print("Training complete for fold!")

    print("Training complete!")
    print(f"Best accuracy: {(best_accuracy * 100):>0.1f}% in fold {best_fold}")

if __name__ == "__main__":
    main()
