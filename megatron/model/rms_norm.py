import torch
from torch import nn

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, sequence_parallel: bool = False):
        super().__init__()
        self.sequence_parallel = sequence_parallel
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = (self.weight * self._norm(x)).type_as(x)
        # output = self.weight * self._norm(x).type_as(x)

        return output