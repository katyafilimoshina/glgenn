import torch.nn.functional as F
from torch import nn

from glgenn.algebra.cliffordalgebraex import CliffordAlgebraQT
from glgenn.engineer.metrics.metrics import Loss, MetricCollection
from glgenn.layers.qtgp import QTGeometricProduct
from glgenn.layers.qtlinear import QTLinear

class ConvexHullGLGMLP(nn.Module):
    def __init__(
        self,
        n,
        in_features=16,
        hidden_features=32,
        out_features=1,
        num_layers=4,
    ):
        super().__init__()

        self.n = n
        self.algebra = CliffordAlgebraQT(
            (1.0,) * self.n
        )

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers

        self.net = nn.Sequential(
            QTLinear(self.algebra, in_features, hidden_features),
            QTGeometricProduct(self.algebra, hidden_features),
            QTGeometricProduct(self.algebra, hidden_features),
            QTGeometricProduct(self.algebra, hidden_features),
            QTGeometricProduct(self.algebra, hidden_features),
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

        self.train_metrics = MetricCollection({"loss": Loss()})
        self.test_metrics = MetricCollection({"loss": Loss()})

    def _forward(self, x):
        return self.net(x)

    def forward(self, batch, step):
        points, products = batch
        input = self.algebra.embed_grade(points, 1)

        y = self._forward(input)

        y = y.norm(dim=-1)
        y = self.mlp(y).squeeze(-1)

        assert y.shape == products.shape, breakpoint()
        loss = F.mse_loss(y, products, reduction="none")

        return loss.mean(), {"loss": loss,}