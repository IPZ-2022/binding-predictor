import torch
import torch.nn as nn

from model.initilizers import linear_init_with_glorot_uniform, linear_init_with_lecun_normal
from model.attention import attention


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, device):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = linear_init_with_glorot_uniform(nn.Linear(d_model, d_model)).to(self.device)
        self.v_linear = linear_init_with_glorot_uniform(nn.Linear(d_model, d_model)).to(self.device)
        self.k_linear = linear_init_with_glorot_uniform(nn.Linear(d_model, d_model)).to(self.device)
        self.out = linear_init_with_lecun_normal(nn.Linear(d_model, d_model)).to(self.device)
        self.heads = None
        self.scores = None

    def forward(self, q, k, v):
        bs = q.size(0)

        # q,k,v ~ (batch, max_seq_length, d_model)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).to(self.device)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).to(self.device)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).to(self.device)

        # q,k,v ~ (batch, max_seq_length, h, d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # q,k,v ~ (batch, h, max_seq_length, d_k)
        scores, heads = attention(q, k, v, self.d_k, self.device)

        # scores ~ (batch, h, max_seq_length, d_k)
        scores = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        # scores ~ (batch, max_seq_length, d_model)
        output = self.out(scores).to(self.device)
        # output ~ (batch, max_seq_length, d_model)

        if not self.training:
            self.heads = heads.detach()
            self.scores = scores.detach()

        return output
