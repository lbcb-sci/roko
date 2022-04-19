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
    def __init__(self, d_model: int, dropout: float = 0.1, seq_len: int = 90, num_reads: int = 30): #d_model = embed_dim = E, seq_len = S, num_reads = R
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.seq_len = seq_len
        self.num_reads = num_reads

        position = torch.arange(max(seq_len, num_reads)).unsqueeze(1) # max(S, R) x 1
        # torch.arange(max_len) returns tensor([ 0,  1,  2,  3,  4, ..., max_len-1])
        # torch.unsqueeze(x, 1) returns tensor([[ 1],
        #                                       [ 2],
        #                                       [ 3],
        #                                       [ 4],
        #                                       [...]
        #                                       [max_len-1]])
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # torch.arange(start, end, step), math.log() is actually ln()
        pe = torch.zeros(max(seq_len, num_reads), d_model) # max(S, R) x E
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, num_reads, seq_len, embedding_dim]
        """
        if mode == 'row':
            x = x + self.pe.unsqueeze(1).expand(-1, self.seq_len, -1)[:self.num_reads] # after expanding: N S E, where N = max(S, R). take the first R entries of pe (R x E)
        else: # if mode == 'col' # take the first S entries in pe (S x E)
            x = x + self.pe[:self.seq_len] # pe S x E

        # if mode is row: self.pe[:self.num_reads].shape = self.pe[:x.size(1)].shape = num_reads x d_model = R E
        # if mode is col: self.pe[:self.seq_len].shape = self.pe[:x.size(-2)].shape = seq_len x d_model = S E
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
