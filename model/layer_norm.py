import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    def __init__(self, d_model, device, eps=1e-6):
        super().__init__()
        self.device = device
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size).to(device))
        self.bias = nn.Parameter(torch.zeros(self.size).to(device))
        self.eps = eps

    def forward(self, x):
        norm = (self.alpha * (x - x.mean(dim=-1, keepdim=True))
                / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias).to(self.device)
        return norm

