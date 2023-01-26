import torch.nn as nn
import torch.nn.functional as f

from model.initilizers import linear_init_with_he_normal, linear_init_with_lecun_normal


class FeedForward(nn.Module):
    def __init__(self, d_model, device):
        super().__init__()
        self.device = device
        self.linear_1 = linear_init_with_he_normal(nn.Linear(d_model, 4 * d_model)).to(device)
        self.linear_2 = linear_init_with_lecun_normal(nn.Linear(4 * d_model, d_model)).to(device)

    def forward(self, x):
        x = f.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x
