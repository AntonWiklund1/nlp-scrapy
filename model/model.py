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
    def __init__(self, vocab_size, embed_dim, num_class, num_heads):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.dropout = nn.Dropout(0.6)

        self.attention = TransformerBlock(embed_dim, num_heads, dropout=0.6, forward_expansion=4)

        self.fc1 = nn.Linear(embed_dim, 256) # first fully connected layer
        self.fc2 = nn.Linear(256, 256) # second fully connected layer
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
        # MultiheadAttention for parallel attention processing
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        
        # Normalization layer 1
        self.norm1 = nn.LayerNorm(embed_size)
        # Normalization layer 2
        self.norm2 = nn.LayerNorm(embed_size)
        
        # Feed Forward Network
        # Expands the embedding dimension, applies ReLU, and contracts it back
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input x is the query, key, value for the self-attention, which are all the same here
        attn_output, attn_weights = self.attention(x, x, x)
        
        # Apply dropout and normalization after adding the input (residual connection)
        x = self.dropout(self.norm1(attn_output + x))
        
        # Passing the result through the feed-forward network
        forward = self.feed_forward(x)
        
        # Another dropout and normalization step to form the final output
        out = self.dropout(self.norm2(forward + x))
        
        # Return the transformed output and the attention weights
        return out, attn_weights

