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
        x = self.x[:, idx : idx + self.length, : self.h, : self.w]
        y = self.x[:, idx + 1 : idx + self.length + 1, : self.h, : self.w]

        return x, y

class MultiStepSequentialDataset(Dataset):
    """
    x is assumed to have shape (C, T, H, W), same as in your SequentialDataset.

    Each item:
        x: x[:, idx : idx + in_len, :H, :W]          # context sequence
        y: x[:, idx + in_len : idx + in_len + out_len, :H, :W]  # future sequence
    """

    def __init__(self, x: Tensor, in_len: int, out_len: int, H: int, W: int):
        assert in_len >= 1
        assert out_len >= 1
        assert x.ndim >= 3  # (C, T, ...)

        self.x = x
        self.in_len = in_len
        self.out_len = out_len
        self.h = H
        self.w = W

        # total usable time steps = T - (in_len + out_len) + 1
        self._len = x.shape[1] - (in_len + out_len) + 1
        if self._len <= 0:
            raise ValueError(
                f"Not enough time steps ({x.shape[1]}) for in_len={in_len}, out_len={out_len}"
            )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # context sequence
        x_seq = self.x[:, idx : idx + self.in_len, : self.h, : self.w]

        # future sequence (multi-step target)
        y_seq = self.x[
            :,
            idx + self.in_len : idx + self.in_len + self.out_len,
            : self.h,
            : self.w,
        ]

        return x_seq, y_seq

class AugmentedDataset(Dataset):
    """
    Wrap a base Dataset that returns (x, y) tensors.
    Expands dataset length by `n_augment` and injects gaussian noise.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        n_augment: int = 1,
        noise_std: float = 0.0,
        noise_on_inputs: bool = True,
        noise_on_targets: bool = False,
        seed: int | None = None,
    ):
        assert n_augment >= 1
        self.base: Dataset = base_dataset
        self.n_augment = int(n_augment)
        self.noise_std = float(noise_std)
        self.noise_on_inputs = noise_on_inputs
        self.noise_on_targets = noise_on_targets
        self.seed = int(seed) if seed is not None else 0

    def __len__(self) -> int:
        return len(self.base) * self.n_augment  # type: ignore

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        base_idx = idx // self.n_augment
        aug_idx = idx % self.n_augment

        x, y = self.base[base_idx]

        if not torch.is_floating_point(x):
            x = x.float()
        if not torch.is_floating_point(y):
            y = y.float()

        if self.n_augment == 1 or self.noise_std == 0.0:
            return x, y

        device = x.device if hasattr(x, "device") else torch.device("cpu")

        g = torch.Generator(device=device)
        g.manual_seed(self.seed + base_idx * 1000003 + aug_idx)

        if self.noise_on_inputs:
            x = x + torch.randn(x.size(), generator=g) * self.noise_std

        if self.noise_on_targets:
            g2 = torch.Generator(device=device)
            g2.manual_seed(self.seed + base_idx * 1000003 + aug_idx + 1)
            y = y + torch.randn(y.size(), generator=g2) * self.noise_std

        return x, y
