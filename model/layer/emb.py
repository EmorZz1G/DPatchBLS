import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=5000, device='cpu'):
        super(PositionalEmbedding, self).__init__()
        self.device = device
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        position.to(self.device)
        div_term.to(self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.to(self.device)
        return x + self.pe[0:1,:]
