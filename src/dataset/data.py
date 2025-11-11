import torch
from torch import Tensor
from torch.utils.data import Dataset


class SequentialDataset(Dataset[float]):
    length: int

    def __init__(self, length: int, x: Tensor, H: int, W: int):
        self.length = length
        self.x = x
        self.h = H
        self.w = W

    def __len__(self):
        return self.x.shape[1] - self.length

    def __getitem__(self, idx: int):  # type: ignore
        x = self.x[:, idx : idx + self.length, :self.h, :self.w]
        y = self.x[:, idx + 1 : idx + self.length + 1, :self.h, :self.w]

        return x, y