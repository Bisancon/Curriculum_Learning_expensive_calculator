import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert n_embd % num_heads == 0
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = head_size
        self.attention_head_size = n_embd // num_heads

        self.query_proj = nn.Linear(n_embd, n_embd)
        self.key_proj = nn.Linear(n_embd, n_embd)
        self.value_proj = nn.Linear(n_embd, n_embd)

        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.attention_head_size).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.attention_head_size).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.attention_head_size).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.attention_head_size)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_probs = F.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_embd)
        output = self.out_proj(context)

        return output


class FeedForward(nn.Module):
    def __init__(self, n_embd, hidden_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(n_embd, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, n_embd)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(n_embd, num_heads, n_embd // num_heads, dropout)
        self.feed_forward = FeedForward(n_embd, 4 * n_embd, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        attn_output = self.multihead_attention(x, x, x, attention_mask)
        attn_output = self.dropout(attn_output)
        x = self.layer_norm1(x + attn_output)

        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.layer_norm2(x + ff_output)

        return x


class DecoderTransformer(nn.Module):
    def __init__(self, num_tokens, n_embd, num_classes, num_layers=6, num_heads=8, dropout=0.1, max_len=512):
        super(DecoderTransformer, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_tokens, n_embd)
        self.position_encoding = PositionEncoding(n_embd, dropout, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(n_embd, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(n_embd)
        # Сменим последний слой на классификационный
        self.output_linear = nn.Linear(n_embd, num_classes)

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        embeddings = self.position_encoding(embeddings)

        for layer in self.layers:
            embeddings = layer(embeddings, attention_mask)

        embeddings = self.norm(embeddings)
        # Мы используем только последний токен для классификации
        logits = self.output_linear(embeddings[:, -1, :])  # Используем последний токен из последовательности

        return logits
