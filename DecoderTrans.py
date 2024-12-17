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


class DecoderTransformer(nn.Module):
    
    def __init__(self, num_tokens=4, n_embd=2, max_len=6):
        
        super().__init__()
        
        
        self.we = nn.Embedding(num_embeddings=num_tokens, 
                               embedding_dim=n_embd)     
        self.pe = PositionEncoding(d_model=n_embd, 
                                   max_len=max_len)
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_embd, n_head, head_size)
        self.fc_layer = nn.Linear(in_features=n_embd, out_features=num_tokens)
        
        self.loss = nn.CrossEntropyLoss()
        
        
    def forward(self, token_ids):
                
        word_embeddings = self.we(token_ids)        
        position_encoded = self.pe(word_embeddings)
        
        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0))))
        mask = mask == 0
        
        self_attention_values = self.self_attention(position_encoded, 
                                                    mask)
                
        residual_connection_values = position_encoded + self_attention_values        
        fc_layer_output = self.fc_layer(residual_connection_values)
        
        return fc_layer_output
