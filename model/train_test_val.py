import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import constants
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import TextClassifier
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from dataset.news_dataset import NewsDataset
from dataset.preprocessing import preprocess_data, collate_batch, get_vocab_size

from vissualize.plot import plot_learning_curve

from colorama import Fore, Style, init
init()  # Initialize colorama for Windows compatibility
torch.manual_seed(42)

"""globbal variables"""
vocab_size = get_vocab_size()
embed_dim = constants.embed_dim
num_class = constants.num_class
num_heads = constants.num_heads
early_stopping_patience = constants.early_stopping_patience

def full_training_cycle(train_loader, val_loader, model, loss_fn, optimizer, scheduler, device, epochs, early_stopping_patience, fold, best_global_val_loss, best_global_model):
    """
    Performs one fold of training and validation.
    Returns the validation accuracy of the best model.
    """
    model = model.to(device)
    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_count = 0.0, 0
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip the gradients to prevent exploding gradients
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            total_count += X.size(0)
        train_losses.append(total_loss / total_count)

        print(f'\nEpoch {epoch+1}/{epochs} Fold {fold+1}: Training Loss = {train_losses[-1]:.6f}')

        # Validation phase
        val_accuracy, val_loss = evaluate(val_loader, model, loss_fn, device)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)  # Adjust learning rate based on validation loss

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_val_loss < best_global_val_loss:
                best_global_val_loss = best_val_loss
                best_global_model = model.state_dict()
                print(f"{Fore.GREEN}New best global model with improved validation loss: {best_val_loss:.6f}{Style.RESET_ALL}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"{Fore.YELLOW}Early stopping counter: {early_stopping_counter}/{early_stopping_patience}{Style.RESET_ALL}")
            if early_stopping_counter >= early_stopping_patience:
                print(f"{Fore.RED}Early stopping triggered in fold {fold+1} after {epoch+1} epochs without improvement. Best Validation Loss for this fold: {best_val_loss:.6f}{Style.RESET_ALL}")
                break
    
    
    return train_losses,best_val_loss,val_losses,val_accuracy, best_global_val_loss, best_global_model

def train(df, device, k_folds, epochs):
    """Perform k-fold cross-validation training on the full dataset."""
    kfold = KFold(n_splits=k_folds, shuffle=True)
    val_loss_and_accuracy = {}
    texts = df['Text'].values
    labels = df['Category'].values

    all_train_losses = []
    all_val_losses = []

    full_dataset = NewsDataset(texts, labels)
    best_global_val_loss = float('inf')
    best_global_model = None
    

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f'Starting training for Fold {fold+1}')

        # Set up the data loaders
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        train_loader = DataLoader(full_dataset, batch_size=constants.batch_size, sampler=train_subsampler, collate_fn=collate_batch)
        val_loader = DataLoader(full_dataset, batch_size=constants.batch_size, sampler=val_subsampler, collate_fn=collate_batch)

        model = TextClassifier(vocab_size, embed_dim, num_class, num_heads=num_heads)  

        optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        scheduler = OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=epochs)
        loss_fn = nn.CrossEntropyLoss()

        train_losses, best_val_loss, val_losses, val_accuracy, best_global_val_loss, best_global_model = full_training_cycle(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            fold=fold,
            best_global_val_loss=best_global_val_loss,
            best_global_model=best_global_model
        )
        val_loss_and_accuracy[best_val_loss] = val_accuracy

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

    num_epochs = min(len(fold_losses) for fold_losses in all_train_losses)  # Taking the shortest length due to possible early stopping
    average_train_losses = [sum(fold_losses[i] for fold_losses in all_train_losses) / len(all_train_losses) for i in range(num_epochs)]
    average_val_losses = [sum(fold_val_losses) / len(all_val_losses) for fold_val_losses in zip(*all_val_losses)]

    # Plot the learning curve
    plot_learning_curve(average_train_losses, average_val_losses, 'Average Learning Curve', './results/average_learning_curve.png')

    best_val_loss = min(val_loss_and_accuracy.keys())
    best_accuracy = val_loss_and_accuracy[best_val_loss]

    torch.save(best_global_model, './results/best_model.pth')
    return best_val_loss, best_accuracy

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
    print(f'{Fore.BLUE}Validation Completed: Accuracy = {accuracy * 100:.2f}%, Average Loss = {avg_loss:.6f}{Style.RESET_ALL}')
    return accuracy, avg_loss


def test(df, device, model_path, feature_col='Text', label_col='Category'):
    """Test the model on the test set and print the accuracy."""

    test_dataset = NewsDataset(df[f'{feature_col}'].reset_index(drop=True), df[f'{label_col}'].reset_index(drop=True))
    test_loader = DataLoader(test_dataset, batch_size=constants.batch_size, collate_fn=collate_batch)

    model = TextClassifier(get_vocab_size(), embed_dim, num_class, num_heads=num_heads)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    model.eval()
    total_acc, total_count = 0, 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_acc += (pred.argmax(1) == y).sum().item()
            total_count += y.size(0)
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = total_acc / total_count

    #load the category to int mapping
    with open('category_to_int.pkl', 'rb') as f:
        category_to_int = pickle.load(f)

    #invert the mapping
    int_to_category = {v: k for k, v in category_to_int.items()}

    #convert the predictions and labels to their respective categories

    pred_categories = [int_to_category[pred] for pred in all_preds]
    label_categories = [int_to_category[label] for label in all_labels]

    return accuracy, pred_categories, label_categories

def fine_tune(df, device, epochs=20, percent_to_train=0.5):
    """Fine-tune the model on the scraped data."""
    print(f"{Fore.YELLOW}Fine-tuning the model on {device}...{Style.RESET_ALL}")
    # Load the model
    model = TextClassifier(get_vocab_size(), embed_dim, num_class, num_heads=num_heads)
    model.load_state_dict(torch.load('./results/best_model.pth'))
    model = model.to(device)

    
    # Split the data to percent_to_train for training
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle the data
    train_size = int(percent_to_train * len(df))

    df = df[:train_size]

    fine_tune_dataset = NewsDataset(df['body'].reset_index(drop=True), df['Category'].reset_index(drop=True))
    fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=constants.batch_size, collate_fn=collate_batch)
    
    optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(fine_tune_loader), epochs=epochs)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    total_loss, total_count = 0.0, 0
    for epoch in range(epochs):
        for X, y in fine_tune_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            total_count += X.size(0)
        print(f'Epoch {epoch+1}/{epochs}: Training Loss = {total_loss / total_count:.6f}')
        scheduler.step() 
    
    torch.save(model.state_dict(), './results/fine_tuned_model.pth')

    
