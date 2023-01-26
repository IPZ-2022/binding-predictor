import math
import torch.nn as nn
import torch.nn.functional as f
from model.encoder import Encoder
from model.initilizers import linear_init_with_he_normal, linear_init_with_lecun_normal


class Model(nn.Module):
    def __init__(self, classes, d_model, s_len_1, s_len_2, n, heads, dropout, device):
        super().__init__()
        self.device = device
        self.n = n
        self.heads = heads
        self.d_model = d_model
        self.in1 = nn.Linear(1, d_model // 2).to(device)
        self.in2 = nn.Linear(1, d_model // 2).to(device)
        self.encoder = Encoder(s_len_1, s_len_2, d_model, n, heads, dropout, device)
        self.out1 = linear_init_with_he_normal(nn.Linear(d_model ** 2, d_model ** 2 // 4)).to(device)
        self.out2 = linear_init_with_he_normal(nn.Linear(d_model ** 2 // 4, d_model // 4 * classes)).to(device)
        self.out3 = linear_init_with_lecun_normal(nn.Linear(d_model // 4 * classes, classes)).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_1, x_2):
        x_1 = x_1.unsqueeze(-1)
        x_2 = x_2.unsqueeze(-1)
        x_1 = self.in1(x_1)
        x_2 = self.in1(x_2)
        x = self.encoder(x_1, x_2)
        x = x.view(-1, self.d_model ** 2)
        x = f.relu(self.out1(x))
        x = f.relu(self.out2(x))
        x = self.dropout(self.out3(x))
        return x
