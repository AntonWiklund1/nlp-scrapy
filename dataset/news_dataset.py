import torch
from torch.utils.data import Dataset
from .preprocessing import text_pipeline


class NewsDataset(Dataset):
    """Custom dataset for news classification task."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = text_pipeline(self.texts[idx]) # Pre-process text
        label = torch.tensor(self.labels[idx], dtype=torch.int64) # Convert label to tensor
        return text, label