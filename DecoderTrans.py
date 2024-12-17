import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEncoding(nn.Module):
    
    def __init__(self, d_model=2, max_len=6):
        
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        
        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)
        

        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        
        self.register_buffer('pe', pe) 

        
    def forward(self, word_embeddings):
        
        return word_embeddings + self.pe[:word_embeddings.size(0), :] 

class Head(nn.Module): 
    
    def __init__(self, n_embd, head_size):
        
        super().__init__()
        
        self.W_q = nn.Linear(in_features=n_embd, out_features=head_size, bias=False)
        self.W_k = nn.Linear(in_features=n_embd, out_features=head_size, bias=False)
        self.W_v = nn.Linear(in_features=n_embd, out_features=head_size, bias=False)
        
        self.row_dim = 0
        self.col_dim = 1

        
    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):

        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
        
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)
        
        return attention_scores

class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = torch.cat([h(x,x,x,mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

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
