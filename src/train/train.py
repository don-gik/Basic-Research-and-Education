import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from src.model import Model


class BaseTrainer(ABC):
    @abstractmethod
    def run(self) -> None: ...


class Trainer(BaseTrainer):
    def __init__(
        self, config: DictConfig | ListConfig, train_dataset, val_dataset, device: torch.device = torch.device("cpu")
    ):
        self.model = Model(
            in_channels=config.data.var_cnt,
            patch_size=config.model.patch_size,
            time=config.model.time,
            img_H=config.data.H,
            img_W=config.data.W,
            depth=config.model.depth,
        ).to(device)

        self.dataset = train_dataset
        self.validation = val_dataset

        self.logger = logging.getLogger(__name__)

        self.batch_size = config.train.batch_size
        self.subset_size = config.train.subset_size

        self.load = config.train.load
        self.path = config.train.save
        self.epochs = config.train.epochs
        self.log_freq = config.train.log_freq
        self.name = config.model.name
        self.var_cnt = config.data.var_cnt

        self.device = device

    def run(self) -> None:
        try:
            self.model.load_state_dict(torch.load(self.load, weights_only=True))
        except FileNotFoundError as e:
            self.logger.warning(f"Skipping to load model... {e}")
        except IsADirectoryError as e:
            self.logger.warning(f"Skipping to load model... {e}")

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-2)

        dataset_size = len(self.dataset)
        subset_size = self.subset_size

        self.logger.info("-------- Starting training... --------")

        epochbar = tqdm(
            range(self.epochs),
            desc="Epoch...",
            mininterval=0.1,
            file=open(os.devnull, "w"),
        )

        for epoch in epochbar:
            self.logger.info(f"Starting epoch {epoch}...")
            running_loss: float = 0.0

            random_indices: list[int] = np.random.choice(
                np.arange(dataset_size), size=subset_size, replace=False
            ).tolist()
            random_sampler = SubsetRandomSampler(random_indices)
            loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=random_sampler,
                num_workers=2,  # Adjust num_workers as needed
            )

            loaderbar = tqdm(
                enumerate(loader),
                desc="Batches",
                mininterval=0.1,
                file=open(os.devnull, "w"),
                total=subset_size // self.batch_size,
            )

            for i, data in loaderbar:
                inputs, outputs = data
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device, dtype=torch.float)

                inputs = self._ensure_bcthw(inputs)
                outputs = self._ensure_bcthw(outputs)

                optimizer.zero_grad()

                target_steps = outputs.size(2)
                base_pred = self.model(inputs)
                pred = self._rollout_predictions(inputs, base_pred, target_steps)
                pred = pred[:, :, :target_steps, :, :]
                target = outputs[:, :, :target_steps, :, :]

                loss = criterion(pred, target)
                loss.backward()

                optimizer.step()  # type: ignore

                running_loss += loss.item()  # type: ignore

                if i % self.log_freq == self.log_freq - 1:
                    last_loss = running_loss / self.log_freq  # loss per batch
                    self.logger.info("{}  batch {} loss: {}".format(str(loaderbar), i + 1, last_loss))

                    running_loss = 0.0

            self.evaluate()

            self._save(epoch, self.name)

    def evaluate(self):
        val_criterion = nn.MSELoss(reduction="mean")
        loader = DataLoader(self.validation, batch_size=self.batch_size, num_workers=2)
        total_batches = len(loader)
        loaderbar = tqdm(
            enumerate(loader),
            desc="Batches",
            mininterval=0.1,
            file=open(os.devnull, "w"),
            total=total_batches,
        )

        with torch.no_grad():
            full_loss = [0.0 for _ in range(self.var_cnt)]
            for i, batch in loaderbar:
                inputs, outputs = batch
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device, dtype=torch.float)

                inputs = self._ensure_bcthw(inputs)
                outputs = self._ensure_bcthw(outputs)

                target_steps = outputs.size(2)
                base_pred = self.model(inputs)
                pred = self._rollout_predictions(inputs, base_pred, target_steps)
                pred = pred[:, :, :target_steps, :, :]
                target = outputs[:, :, :target_steps, :, :]

                for var in range(pred.size(1)):
                    val_loss = val_criterion(pred[:, var], target[:, var])
                    full_loss[var] += float(val_loss.item())

                if i % self.log_freq == self.log_freq - 1:
                    self.logger.info("{}  {}".format(str(loaderbar), i))
            for var in range(len(full_loss)):
                self.logger.info("final_loss {} : {}".format(var, full_loss[var] / max(total_batches, 1)))

    def _save(self, epoch: int, name: str) -> None:
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(self.model.state_dict(), os.path.join(self.path, name + str(epoch) + ".pt"))
        return

    def _ensure_bcthw(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensor is shaped [B, C, T, H, W]; if input is [B, T, C, H, W] reorder.
        """
        if tensor.ndim != 5:
            return tensor
        b, d1, d2, h, w = tensor.shape
        # Already channels-second (B, C, T, H, W)
        if d1 == self.var_cnt:
            return tensor
        # Channels-third (B, T, C, H, W) -> permute
        if d2 == self.var_cnt:
            return tensor.permute(0, 2, 1, 3, 4).contiguous()
        return tensor

    def _rollout_predictions(self, inputs: torch.Tensor, base_pred: torch.Tensor, horizon: int) -> torch.Tensor:
        if horizon <= 0:
            return base_pred[:, :, :0, :, :]

        time_dim = inputs.size(2)
        if horizon <= base_pred.size(2):
            return base_pred[:, :, :horizon, :, :]

        preds = [base_pred]
        total_steps = base_pred.size(2)
        current_window = torch.cat([inputs, base_pred], dim=2)[:, :, -time_dim:, :, :]

        while total_steps < horizon:
            next_pred = self.model(current_window)
            next_frame = next_pred[:, :, -1:, :, :]
            preds.append(next_frame)
            current_window = torch.cat([current_window[:, :, 1:, :, :], next_frame], dim=2)
            total_steps += 1

        return torch.cat(preds, dim=2)
