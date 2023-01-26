import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class PositionalEncoder(nn.Module):
    def __init__(self, max_seq_length, d_model, device):
        super().__init__()
        self.d_model = d_model
        self.device = device
        pe = torch.zeros(max_seq_length, d_model).to(device)
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(self.device)
        self.std = x.std()
        return x
