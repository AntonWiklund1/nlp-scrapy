import optuna
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import TextClassifier
import constants
from dataset.preprocessing import get_vocab_size, prepare_data, collate_batch
from sklearn.model_selection import KFold
from dataset.news_dataset import NewsDataset
from optuna.pruners import MedianPruner
from torch.cuda.amp import autocast, GradScaler

import colorama
from colorama import Fore, Style, init
init()  # Initialize colorama for Windows

# Assuming constants.py contains all the required constants
vocab_size = get_vocab_size()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial):
    pruner = MedianPruner()

    scaler = GradScaler()
    # Hyperparameters to be tuned by Optuna
    lr = trial.suggest_loguniform('lr', 3e-5, 3e-3) # Learning rate first is lower bound and second is upper bound 0.00001 to 0.0003
    embed_dim = trial.suggest_categorical('embed_dim', [128, 256, 512])
    num_heads = 8 if embed_dim == 128 else trial.suggest_categorical('num_heads', [8, 16])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.6)
    layer_size = trial.suggest_categorical('layer_size', [128, 256, 512])
    number_of_layers = trial.suggest_int('number_of_layers', 1, 5)

    print(f"Learning rate: {lr}, Embedding dimension: {embed_dim}, Number of heads: {num_heads}, Dropout rate: {dropout_rate}, Layer size: {layer_size}, Number of layers: {number_of_layers}")

    best_global_val_loss = float('inf')

    # Model setup
    
    # Prepare the data
    df = prepare_data("./data/bbc_news_train_2.csv", text='Text', augment=True, categories=True)
    print(df['Category'].value_counts())
    texts = df['Text'].values
    labels = df['Category'].values
    full_dataset = NewsDataset(texts, labels)
    kfold = KFold(n_splits=constants.folds, shuffle=True)
    
   
    # Perform K-fold Cross Validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        best_validation_loss = float('inf')
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        train_loader = DataLoader(full_dataset, batch_size=constants.batch_size, sampler=train_subsampler, collate_fn=collate_batch)
        val_loader = DataLoader(full_dataset, batch_size=constants.batch_size, sampler=val_subsampler, collate_fn=collate_batch)

        model = TextClassifier(vocab_size, embed_dim, constants.num_class, num_heads, dropout_rate, layer_size, number_of_layers)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        loss_fn = torch.nn.CrossEntropyLoss()

        early_stopping_counter = 0

        # Training and Validation loop
        for epoch in range(constants.epochs):

            # Training phase
            model.train()
            for texts, labels in train_loader:
                texts, labels = texts.to(device), labels.to(device)
                optimizer.zero_grad()

                with autocast():  # Enable automatic mixed precision
                    output = model(texts)
                    loss = loss_fn(output, labels)

                scaler.scale(loss).backward()  # Scale the loss before backward
                scaler.step(optimizer)  # Update optimizer
                scaler.update()  # Update the scale for next iteration

            
            # Validation phase
            model.eval()
            fold_validation_loss = 0
            fold_validation_count = 0
            total_accuracy = 0
            with torch.no_grad():
                for texts, labels in val_loader:
                    texts, labels = texts.to(device), labels.to(device)
                    output = model(texts)
                    loss = loss_fn(output, labels)
                    fold_validation_loss += loss.item() * len(labels)
                    fold_validation_count += len(labels)
                    total_accuracy += (output.argmax(1) == labels).sum().item()

            epoch_validation_loss = fold_validation_loss / fold_validation_count
            epoch_accuracy = total_accuracy / fold_validation_count    

            print(f"Fold {fold + 1}, Epoch {epoch + 1}: Training Loss: {loss.item():.6f}, Validation Loss: {epoch_validation_loss:.6f}, Validation Accuracy: {epoch_accuracy * 100:.2f}%")
            
            # Check for early stopping
            if epoch_validation_loss < best_validation_loss:
                best_validation_loss = epoch_validation_loss
                early_stopping_counter = 0  # Reset counter on improvement
                if epoch_validation_loss < best_global_val_loss:
                    best_global_val_loss = epoch_validation_loss
                    print(f"{Fore.GREEN}Best validation loss improved to {best_global_val_loss:.6f}{Style.RESET_ALL}")
            else:
                early_stopping_counter += 1  # Increment counter when no improvement
                print(f"{Fore.YELLOW}Early stopping counter: {early_stopping_counter}/{constants.early_stopping_patience}{Style.RESET_ALL}")

            if early_stopping_counter == constants.early_stopping_patience:
                print(f"{Fore.RED}Early stopping at epoch {epoch + 1}{Style.RESET_ALL}")
                break  # Stop if no improvement for 'early_stopping_patience' consecutive epochs


             # Report the current loss to Optuna and check if the trial should be pruned
            trial.report(epoch_validation_loss, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_global_val_loss

def main():
    # Define the path to the SQLite database file
    storage_path = "sqlite:///nlp_study.db"
    
    # Create or load a study
    study = optuna.create_study(study_name='bbc_text_classification',
                                storage=storage_path,
                                load_if_exists=True,
                                direction='minimize',
                                pruner=MedianPruner())
    
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print(f" Value: {trial.value}")
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
