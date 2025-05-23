# This file extends the functionality of the original Trainer class
# Original code from: https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks

import torch 
from torch import nn
from torch.utils.data import DataLoader
from typing import Any

from .trainer import Trainer


class ExtendedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_losses = [] # loss on each step
        self.train_metrics_history = [] # metrics for each batch on each step
        self.val_losses = []
        self.val_metrics_history = []
        self.test_losses = []
        self.test_metrics_history = []

    def train_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, batch: Any):
        super().train_step(model, optimizer, batch)
        loss, outputs = model(batch, self.global_step)
        self.train_losses.append(loss.item())
        self.train_metrics_history.append(outputs)

    def test_loop(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        test_loader: DataLoader,
        validation=False,
    ):
        super().test_loop(model, optimizer, test_loader, validation)

        if validation:
            val_metrics = self.train_metrics_history[-1]  
            val_loss = self.train_losses[-1]  
            self.val_losses.append(val_loss)
            self.val_metrics_history.append(val_metrics)
        else:
            test_metrics = self.train_metrics_history[-1] 
            test_loss = self.train_losses[-1] 
            self.test_losses.append(test_loss)
            self.test_metrics_history.append(test_metrics)
