import torch
from torch import nn

EPS = 1e-6

class QTNormalization(nn.Module):
    def __init__(self, algebra, features, init: float = 0):
        super().__init__()
        self.algebra = algebra
        self.in_features = features
        self.a = nn.Parameter(torch.zeros(self.in_features, 4) + init)

    def forward(self, input):
        assert input.shape[1] == self.in_features
        norms = torch.cat(self.algebra.norms_qt(input), dim=-1)
        s_a = torch.sigmoid(self.a)
        norms = s_a * (norms - 1) + 1  
        norms = norms.repeat_interleave(torch.tensor([self.algebra.dim_0, self.algebra.dim_1, self.algebra.dim_2, self.algebra.dim_3]),
                                               dim=-1)[:, :, self.algebra.weights_permutation]
        normalized = input / (norms + EPS)
        return normalized