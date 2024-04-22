import torch
import torch.nn as nn
import torch.nn.functional as F

from colorama import Fore, Style, init
init()  # Initialize colorama for Windows
from warnings import filterwarnings
filterwarnings('ignore')

class TextClassifier(nn.Module):
    """A text classifier model with an attention mechanism and multiple hidden layers."""
    def __init__(self, vocab_size, embed_dim, num_class, num_heads = 8, window_size=10):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Embedding layer
        self.dropout = nn.Dropout(0.6)  # First dropout layer

        # MultiHead Attention Layer
        #self.attention = MultiHeadAttentionLayer(embed_dim, num_heads)
        self.attention = SparseMultiHeadAttentionLayer(embed_dim, num_heads, window_size)

        # Additional hidden layers
        self.fc1 = nn.Linear(embed_dim, 512)  # First hidden layer
        self.fc2 = nn.Linear(512, 256)  # Second hidden layerr
        self.fc3 = nn.Linear(256, 128)  # Third hidden layer
        self.fc4 = nn.Linear(128, num_class)  # Output layer

    def forward(self, text):
        # Embedding and initial dropout
        embedded = self.embedding(text)

        embedded = self.dropout(embedded)
        
        # Apply attention
        context_vector, attention_weights = self.attention(embedded)
        
        # Passing through the hidden layers with activation functions and dropout
        hidden1 = F.leaky_relu(self.fc1(context_vector))
        hidden2 = F.leaky_relu(self.fc2(hidden1))
        hidden3 = F.leaky_relu(self.fc3(hidden2))
        output = self.fc4(hidden3)

        return output
    

class SparseMultiHeadAttentionLayer(nn.Module):
    """A sparse multi-head attention layer with a max-pooling operation."""
    def __init__(self, embed_dim, num_heads, window_size):
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