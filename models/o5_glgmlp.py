import torch.nn.functional as F
from torch import nn

from glgenn.algebra.cliffordalgebraex import CliffordAlgebraQT
from glgenn.engineer.metrics.metrics import Loss, MetricCollection
from glgenn.layers.qtgp import QTGeometricProduct
from glgenn.layers.qtlinear import QTLinear


class O5GLGMLP(nn.Module):
    def __init__(
        self,
        ymean, 
        ystd,
    ):
        super().__init__()
        self.algebra = CliffordAlgebraQT((1.0, 1.0, 1.0, 1.0, 1.0,))

        self.qtgp = nn.Sequential(
            QTLinear(self.algebra, 2, 8),
            QTGeometricProduct(self.algebra, 8),
        )

        self.mlp = nn.Sequential(
            nn.Linear(8, 580),
            nn.ReLU(),
            nn.Linear(580, 580),
            nn.ReLU(),
            nn.Linear(580, 1),
        )

        self.train_metrics = MetricCollection({"loss": Loss()})
        self.test_metrics = MetricCollection({"loss": Loss()})

        self.ymean = ymean
        self.ystd = ystd

    def forward(self, batch, step):
        points, products = batch

        points = points.view(len(points), 2, 5)
        input = self.algebra.embed_grade(points, 1)

        y = self.mlp(self.qtgp(input)[..., 0])
        normalized_y = y * self.ystd + self.ymean
        normalized_products = products * self.ystd + self.ymean

        assert y.shape == products.shape, breakpoint()
        loss = F.mse_loss(normalized_y, normalized_products.float(), reduction="none")
        return loss.mean(), {
            "loss": loss,
        }