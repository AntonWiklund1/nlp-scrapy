import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from colorama import Fore, Style, init
init()  # Initialize colorama for Windows
from warnings import filterwarnings
filterwarnings('ignore')
torch.manual_seed(42)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, num_heads=8, window_size=10):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.dropout = nn.Dropout(0.6)

        self.attention = TransformerBlock(embed_dim, num_heads, dropout=0.6, forward_expansion=4)

        self.fc1 = nn.Linear(embed_dim, 512) # first fully connected layer
        self.fc2 = nn.Linear(512, 256) # second fully connected layer
        self.fc3 = nn.Linear(256, num_class) # output layer

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        embedded = embedded.permute(1, 0, 2)  # Adjusting for MultiheadAttention expectation (L, N, E)

        context_vector, attention_weights = self.attention(embedded)
        context_vector = context_vector.permute(1, 0, 2)  # Adjust back (N, L, E)
        # Apply max pooling over the sequence dimension (L)
        pooled_output = torch.max(context_vector, dim=1)[0]

        hidden1 = F.leaky_relu(self.fc1(pooled_output))
        hidden1 = self.dropout(hidden1)
        hidden2 = F.leaky_relu(self.fc2(hidden1))
        hidden2 = self.dropout(hidden2)
        output = self.fc3(hidden2)

        return output
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Assuming x is the query, key, value which are all the same in this context
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.dropout(self.norm1(attn_output + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out, attn_weights
