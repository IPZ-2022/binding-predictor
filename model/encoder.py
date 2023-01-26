import torch.nn as nn
import torch

from model.feed_forward import FeedForward
from model.initilizers import linear_init_with_he_normal
from model.multi_head_attention import MultiHeadAttention
from model.layer_norm import LayerNorm
from model.positional_encoder import PositionalEncoder


class Encoder(nn.Module):
    def __init__(self, s_len_1, s_len_2, d_model, n, heads, dropout, device):
        super().__init__()
        self.n = n
        self.device = device
        self.pe_1 = PositionalEncoder(s_len_1, d_model // 2, device)
        self.pe_2 = PositionalEncoder(s_len_2, d_model // 2, device)
        self.embed_1 = linear_init_with_he_normal(nn.Linear(s_len_1, d_model)).to(device)
        self.embed_2 = linear_init_with_he_normal(nn.Linear(s_len_2, d_model)).to(device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout, device) for _ in range(n)])
        self.relu = nn.ReLU()

    def forward(self, x_1, x_2):
        x_1 = self.relu(self.embed_1(self.pe_1(x_1).transpose(1, 2)))
        x_2 = self.relu(self.embed_2(self.pe_2(x_2).transpose(1, 2)))
        x = torch.cat([x_1, x_2], dim=1)
        for i in range(self.n):
            x = self.layers[i](x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout, device):
        super().__init__()
        self.device = device
        self.norm_1 = LayerNorm(d_model, device)
        self.norm_2 = LayerNorm(d_model, device)
        self.attn = MultiHeadAttention(heads, d_model, device)
        self.ff = FeedForward(d_model, device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm_1(x)
        x = x + self.dropout(self.attn(x, x, x))
        x = self.norm_2(x)
        x = x + self.dropout(self.ff(x))
        return x

