import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from colorama import Fore, Style, init

import constants
init()  # Initialize colorama for Windows
from warnings import filterwarnings
filterwarnings('ignore')
torch.manual_seed(42)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, num_heads, dropout_rate=0.5):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)


        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.attention = TransformerBlock(embed_dim, num_heads, dropout=0.6, forward_expansion=4)
        self.attention_pool = AttentionPooling(embed_dim)

        self.fc1 = nn.Linear(embed_dim, 256) # first fully connected layer
        self.fc2 = nn.Linear(256, 256) # second fully connected layer
        self.fc3 = nn.Linear(256, num_class) # output layer

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = self.pos_encoder(embedded)
        embedded = self.dropout1(embedded)
        embedded = embedded.permute(1, 0, 2)  # Adjusting for MultiheadAttention expectation (L, N, E)

        context_vector, attn_weights = self.attention(embedded)
        context_vector = context_vector.permute(1, 0, 2)  # Adjust back (N, L, E)
        # Apply max pooling over the sequence dimension (L)
        pooled_output = self.attention_pool(context_vector)

        hidden1 = F.leaky_relu(self.fc1(pooled_output))
        hidden1 = self.dropout2(hidden1)
        hidden2 = F.leaky_relu(self.fc2(hidden1))
        hidden2 = self.dropout2(hidden2)
        output = self.fc3(hidden2)
        return output

    

class TransformerBlock(nn.Module):
    """Transformer Block with MultiheadAttention and Feed Forward Network"""
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
        # The attention mechanism uses the same input for all three
        attn_output, attn_weights = self.attention(x, x, x)
        
        # Apply dropout and normalization after adding the input (residual connection)
        x = self.dropout(self.norm1(attn_output + x))
        
        # Passing the result through the feed-forward network
        forward = self.feed_forward(x)
        
        # Another dropout and normalization step to form the final output
        out = self.dropout(self.norm2(forward + x))
        
        # Return the transformed output and the attention weights
        return out, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=constants.sequence_length):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', self.encoding)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),  # Project down for computational efficiency
            nn.Tanh(),
            nn.Linear(128, 1),          # Output a single scalar (attention weight)
            nn.Softmax(dim=0)           # Normalize weights across the sequence dimension
        )

    def forward(self, x):
        # x shape is (batch_size, seq_length, features)
        batch_size, seq_length, features = x.shape
        x = x.reshape(-1, features)    # Flatten to (batch_size * seq_length, features)
        attn_weights = self.attention(x)  # Compute attention weights
        attn_weights = attn_weights.view(batch_size, seq_length, 1)  # Reshape to (batch_size, seq_length, 1)
        # Perform weighted average using attention weights
        pooled = torch.sum(attn_weights * x.view(batch_size, seq_length, features), dim=1)
        return pooled