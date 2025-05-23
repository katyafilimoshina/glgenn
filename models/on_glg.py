import torch.nn.functional as F
from torch import nn

from glgenn.algebra.cliffordalgebraex import CliffordAlgebraQT
from glgenn.engineer.metrics.metrics import Loss, MetricCollection
from glgenn.layers.qtgp import QTGeometricProduct
from glgenn.layers.qtlinear import QTLinear


class OnGLGMLP(nn.Module):
    def __init__(
        self,
        n,
        ymean, 
        ystd,
        output_qtgp=8,
        hidden_mlp_1=580,
        hidden_mlp_2=580,
        if_mlp=True
    ):
        super().__init__()
        self.n = n
        self.algebra = CliffordAlgebraQT(
            (1.0,) * self.n
        )

        self.qtgp = nn.Sequential(
            QTLinear(self.algebra, 2, output_qtgp),
            QTGeometricProduct(self.algebra, output_qtgp),
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(output_qtgp, hidden_mlp_1),
            nn.ReLU(),
            nn.Linear(hidden_mlp_1, hidden_mlp_2),
            nn.ReLU(),
            nn.Linear(hidden_mlp_2, 1),
        )
        self.no_mlp = nn.Linear(output_qtgp, 1)
        self.if_mlp = if_mlp


        self.train_metrics = MetricCollection({"loss": Loss()})
        self.test_metrics = MetricCollection({"loss": Loss()})

        self.ymean = ymean
        self.ystd = ystd

    def forward(self, batch, step):
        points, products = batch

        points = points.view(len(points), 2, self.n)
        input = self.algebra.embed_grade(points, 1)

        if self.if_mlp:
            y = self.mlp(self.qtgp(input)[..., 0])
        else: 
            y = self.no_mlp(self.qtgp(input)[..., 0])
    
        normalized_y = y * self.ystd + self.ymean
        normalized_products = products * self.ystd + self.ymean

        assert y.shape == products.shape, breakpoint()
        loss = F.mse_loss(normalized_y, normalized_products.float(), reduction="none")
        return loss.mean(), {
            "loss": loss,
        }