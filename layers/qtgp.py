import math
import torch
from torch import nn

from .qtlinear import QTLinear
from .qtnorm import QTNormalization


class QTGeometricProduct(nn.Module):
    def __init__(
        self, algebra, features, normalization_init=0
        ):
        super().__init__()

        self.algebra = algebra
        self.features = features
        
        if normalization_init is not None:
            self.normalization = QTNormalization(
                algebra, features, normalization_init
            )
        else:
            self.normalization = nn.Identity()
        
        self.linear_right = QTLinear(algebra, features, features)
        self.qt_product_paths = algebra.qt_geometric_product_paths
        self.weight = nn.Parameter(torch.empty(features, self.qt_product_paths.sum()))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / (math.sqrt(self.algebra.dim + 1)))

    def _get_weight(self):
        weight = torch.zeros(
            self.features,
            *self.qt_product_paths.size(),
            dtype=self.weight.dtype,
            device=self.weight.device,
        ) # [features, 4, 4, 4]

        weight[:, self.qt_product_paths] = self.weight   # [features, 4, 4, 4]

        permutation = self.algebra.weights_permutation
        qt_tensor = torch.tensor([self.algebra.dim_0, self.algebra.dim_1, self.algebra.dim_2, self.algebra.dim_3])
        weight_repeated = weight.repeat_interleave(qt_tensor, dim=-1)[:, :, :, permutation].repeat_interleave(qt_tensor, dim=-2)[:, :, permutation, :].repeat_interleave(qt_tensor, dim=-3)[:, permutation, :, :] # [features, 2**n, 2**n, 2**n]
        
        return self.algebra.cayley * weight_repeated

    def forward(self, input):
        input_right = self.linear_right(input)
        input_right = self.normalization(input_right)
        weight = self._get_weight()
        return torch.einsum("bni, nijk, bnk -> bnj", input, weight, input_right)


