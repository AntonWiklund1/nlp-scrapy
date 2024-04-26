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

# Assuming constants.py contains all the required constants
vocab_size = get_vocab_size()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial):
    pruner = MedianPruner()
    # Hyperparameters to be tuned by Optuna
    lr = trial.suggest_loguniform('lr', 3e-5, 3e-3) # Learning rate first is lower bound and second is upper bound 0.00001 to 0.0003
    embed_dim = trial.suggest_categorical('embed_dim', [128, 256, 512])
    num_heads = 8 if embed_dim == 128 else trial.suggest_categorical('num_heads', [8, 16])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)

    # Model setup
    model = TextClassifier(vocab_size, embed_dim, constants.num_class, num_heads, dropout_rate)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Prepare the data
    df = prepare_data("./data/bbc_news_train_2.csv", text='Text', augment=True, categories=True, rows=5000)
    print(df['Category'].value_counts())
    texts = df['Text'].values
    labels = df['Category'].values
    full_dataset = NewsDataset(texts, labels)
    kfold = KFold(n_splits=constants.folds, shuffle=True)
    
    best_validation_loss = float('inf')


    # Perform K-fold Cross Validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        train_loader = DataLoader(full_dataset, batch_size=constants.batch_size, sampler=train_subsampler, collate_fn=collate_batch)
        val_loader = DataLoader(full_dataset, batch_size=constants.batch_size, sampler=val_subsampler, collate_fn=collate_batch)

        # Training and Validation loop
        for epoch in range(constants.epochs):

            # Training phase
            model.train()
            for texts, labels in train_loader:
                texts, labels = texts.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(texts)
                loss = loss_fn(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
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

             # Report the current loss to Optuna and check if the trial should be pruned
            trial.report(epoch_validation_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_validation_loss  # Optuna minimizes this

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial
    print(f" Value: {trial.value}")
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
