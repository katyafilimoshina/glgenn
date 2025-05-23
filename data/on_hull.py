import os
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from numpy.lib.format import open_memmap


class MemmapDataset(Dataset):
    def __init__(self, *paths, num_samples=None):
        self.paths = paths
        self.memmaps = [open_memmap(path, mode="r") for path in paths]

        assert all([m.shape[0] == self.memmaps[0].shape[0] for m in self.memmaps])

        if num_samples is not None:
            self.memmaps = [m[:num_samples] for m in self.memmaps]
            assert all([m.shape[0] == num_samples for m in self.memmaps])

        self.length = self.memmaps[0].shape[0]

    def __getitem__(self, index):
        return [m[index].astype(np.float32) for m in self.memmaps]

    def __len__(self):
        return self.length



class ConvexHullDataset:
    def __init__(self, num_samples=256, batch_size=32) -> None:
        super().__init__()

        dataroot = os.path.join(os.environ["DATAROOT"], "hulls")

        self.train_dataset = MemmapDataset(
            os.path.join(dataroot, "hulls_train_input.npy"),
            os.path.join(dataroot, "hulls_train_target.npy"),
            num_samples=num_samples,
        )

        self.val_dataset = MemmapDataset(
            os.path.join(dataroot, "hulls_val_input.npy"),
            os.path.join(dataroot, "hulls_val_target.npy"),
        )

        self.test_dataset = MemmapDataset(
            os.path.join(dataroot, "hulls_test_input.npy"),
            os.path.join(dataroot, "hulls_test_target.npy"),
        )

        self.batch_size = batch_size

    def train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_loader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)