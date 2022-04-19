import torch
from torch import nn
from torch import functional as F
from attentions import MSAGatedAttention

import math

# stochastic depth linear decay
def depth_prob(p_lowest: float, layer: int, n_layers: int) -> float:
    #print(layer, p_lowest, n_layers)
    return 1 - layer / n_layers * (1 - p_lowest)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128): # d_model = embed_dim = E, max_len = seq_len = S
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # S x 1
        # torch.arange(max_len) returns tensor([ 0,  1,  2,  3,  4, ..., max_len-1])
        # torch.unsqueeze(x, 1) returns tensor([[ 1],
        #                                       [ 2],
        #                                       [ 3],
        #                                       [ 4],
        #                                       [...]
        #                                       [max_len-1]])
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # torch.arange(start, end, step), math.log() is actually ln()
        pe = torch.zeros(max_len, d_model) # S x E
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, num_reads, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(-2)] # pe S x E, x.size(-2) = seq_len --> self.pe[:seq_len], S x E
        # so self.pe[:x.size(-2)].shape = seq_len x d_model, x.shape = B R S E = B R seq_len d_model
        return self.dropout(x)

class Evoformer(nn.Module):
    def __init__(self, msa_embedding_dim, heads, num_blocks, p_keep_lowest):
        super().__init__()

        prob_fn = lambda i: depth_prob(p_keep_lowest, i, num_blocks)
        self.blocks = nn.Sequential(*[EvoformerBlock(msa_embedding_dim, heads, prob_fn(i+1)) for i in range(num_blocks)])

    def forward(self, msa_repr):
        return self.blocks(msa_repr)


class EvoformerBlock(nn.Module):
    def __init__(self, msa_embedding_dim, heads, p_keep):
        super().__init__()

        self.msa_row_att = MSAGatedAttention('row', msa_embedding_dim, heads)
        self.msa_col_att = MSAGatedAttention('column', msa_embedding_dim, heads)
        self.p_keep = p_keep
        self.msa_transition = Transition(msa_embedding_dim, projection_factor=4)

    def forward(self, msa_repr):
        if not self.training or torch.rand(1) <= self.p_keep:
            #print("evo")
            # MSA track
            msa_repr = msa_repr + self.msa_row_att(msa_repr)
            msa_repr = msa_repr + self.msa_col_att(msa_repr)
            msa_repr = msa_repr + self.msa_transition(msa_repr)
            return msa_repr
        else:
            print("skipped")
            return msa_repr

class Transition(nn.Module):
    def __init__(self, embedding_dim, projection_factor=4):
        super().__init__()

        self.linear1 = nn.Linear(embedding_dim, projection_factor * embedding_dim)
        self.linear2 = nn.Linear(projection_factor * embedding_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.linear2(torch.relu(x))
        return x
