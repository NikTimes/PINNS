import torch
import torch.nn as nn


class ScaledMSE(nn.Module):
    def __init__(self, yscale):
        super().__init__()
        self.register_buffer("yscale", torch.tensor(yscale, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor): 
        return torch.mean(((pred - target) / self.yscale) ** 2)