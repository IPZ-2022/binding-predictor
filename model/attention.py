import math
import torch
import torch.nn.functional as f


def attention(q, k, v, d_k, device):
    heads = (torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)).to(device)
    heads = f.softmax(heads, dim=-1).to(device)
    scores = torch.matmul(heads, v).to(device)
    return scores, heads
