import torch
from torch import nn
from torch import functional as F
import math
from torch.utils.checkpoint import checkpoint


def dot_product_attention(q, k, v, attn_bias=None):
    _, _, c = q.shape
    attn = torch.bmm(q / math.sqrt(c), k.transpose(-1, -2))
    if attn_bias is not None:
        attn = attn + attn_bias.repeat(attn.shape[0] // attn_bias.shape[0], 1, 1)

    attn = attn.softmax(dim=-1)

    output = torch.bmm(attn, v)
    return output, attn

class GatedAxialAttentionUnit(nn.Module):
    def __init__(self, num_dimensions, embedding_dim, heads, attn_dim): #4, 50, heads=8, attn_dim
        super().__init__()

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.attn_dim = attn_dim

        self.head_dim = embedding_dim // self.heads
        self.hidden_dim = self.head_dim * heads

        self.to_q = nn.Linear(embedding_dim, self.hidden_dim, bias=False)
        self.to_k = nn.Linear(embedding_dim, self.hidden_dim, bias=False)
        self.to_v = nn.Linear(embedding_dim, self.hidden_dim, bias=False)

        self.to_g = nn.Linear(embedding_dim, self.hidden_dim)

        self.to_o = nn.Linear(self.hidden_dim, embedding_dim)

        self.dims_permutation = [*[i for i in range(num_dimensions - 1) if i != attn_dim], attn_dim, num_dimensions - 1]
        self.inv_dims_permutation = [self.dims_permutation.index(i) for i in range(num_dimensions)]

    def _merge_heads(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.heads, self.head_dim).transpose(1, 2).reshape(b * self.heads, t, self.head_dim)

    def _unmerge_heads(self, x):
        b, t, _ = x.shape
        return x.reshape(-1, self.heads, t, self.head_dim).transpose(1, 2).reshape(-1, t, self.hidden_dim)

    def forward(self, x, attn_bias=None):
        axial = x.permute(*self.dims_permutation).contiguous()
        permuted_shape = axial.shape
        axial = axial.reshape(-1, axial.shape[-2], axial.shape[-1])

        attn_bias_axial = attn_bias
        if attn_bias is not None:
            attn_bias_axial = attn_bias.permute(0, 3, 1, 2).reshape(-1, x.shape[self.attn_dim], x.shape[self.attn_dim])
            attn_bias_axial = attn_bias_axial.reshape(-1, attn_bias_axial.shape[-2], attn_bias_axial.shape[-1])

        q, k, v = self.to_q(axial), self.to_k(axial), self.to_v(axial)
        q, k, v = self._merge_heads(q), self._merge_heads(k), self._merge_heads(v)

        output, _ = dot_product_attention(q, k, v, attn_bias_axial)

        gate = self._merge_heads(torch.sigmoid(self.to_g(axial)))
        output = gate * output

        output = self._unmerge_heads(output)

        output = self.to_o(output)
        output = output.reshape(axial.shape)

        return output.reshape(permuted_shape).permute(*self.inv_dims_permutation).contiguous()

class MSAGatedAttention(nn.Module):
    def __init__(self, axis, embedding_dim, heads):
        super().__init__()

        if axis == 'row':
            self.ax_attn = GatedAxialAttentionUnit(4, embedding_dim, heads, attn_dim=2)
            self.layer_norm = nn.LayerNorm(embedding_dim)
        elif axis == 'column':
            self.ax_attn = GatedAxialAttentionUnit(4, embedding_dim, heads, attn_dim=1)
            self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, msa_repr):
        msa_repr = self.layer_norm(msa_repr)
        msa_repr = self.ax_attn(msa_repr) #checkpoint(self.ax_attn, msa_repr)
        return msa_repr
