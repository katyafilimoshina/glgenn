import math
import torch
from torch import nn


class QTLinear(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        out_features,
    ):
        super().__init__()

        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, 4))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))

    def _forward_subspaces(self, input):
        weight = self.weight.repeat_interleave(torch.tensor([self.algebra.dim_0, self.algebra.dim_1, self.algebra.dim_2, self.algebra.dim_3]),
                                               dim=-1)[:, :, self.algebra.weights_permutation]
        return torch.einsum("bm...i, nmi->bn...i", input, weight)

    def forward(self, input):
        result = self._forward_subspaces(input)
        return result