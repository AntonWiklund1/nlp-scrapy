import torch
from torch.utils.data import DataLoader
from model import TextClassifier
from dataset.preprocessing import prepare_data, collate_batch
from dataset.news_dataset import NewsDataset
import constants
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from colorama import Fore, Style, init
init()

from warnings import filterwarnings
filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(df, model_path, feature_col='Text', label_col='Category'):
    """Test the model on the test set and compute detailed evaluation metrics."""

    # Load and prepare data
    test_dataset = NewsDataset(df[feature_col].reset_index(drop=True), df[label_col].reset_index(drop=True))
    test_loader = DataLoader(test_dataset, batch_size=constants.batch_size, collate_fn=collate_batch)

    # Load the complete model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # Model evaluation
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Metrics calculation
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    # Load the category to int mapping
    with open('category_to_int.pkl', 'rb') as f:
        category_to_int = pickle.load(f)

    # Convert predictions and labels to their respective categories
    int_to_category = {v: k for k, v in category_to_int.items()}
    pred_categories = [int_to_category[pred] for pred in all_preds]
    label_categories = [int_to_category[label] for label in all_labels]

    return accuracy, precision, recall, f1, pred_categories, label_categories


def main():
    model_path = './results/topic_classifier.pkl'
    test_df = prepare_data("./data/bbc_news_tests.csv", text='Text', augment=False)
    accuracy, precision, recall, f1, _, _ = test(test_df, model_path, feature_col='Text', label_col='Category')
    print(f"{Fore.WHITE}Test Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}{Style.RESET_ALL}")

if __name__ == '__main__':
    main()
