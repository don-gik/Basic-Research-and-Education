import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import Model


@dataclass
class EvalSample:
    inp: Tensor
    target: Tensor
    pred: Tensor
    avg_loss: float
    loss: list[float]


@dataclass
class EvalData:
    best_sample: EvalSample
    worst_sample: EvalSample
    random_sample: EvalSample


class BaseEval(ABC):
    @abstractmethod
    def run(self) -> None: ...


class Evaluator(BaseEval):
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
        self.path = config.eval.save
        self.epochs = config.train.epochs
        self.log_freq = config.train.log_freq
        self.name = config.model.name
        self.lead = config.model.time

        self.device = device

    def run(self) -> None:
        try:
            self.model.load_state_dict(torch.load(self.load, weights_only=True))
        except FileNotFoundError as e:
            self.logger.warning(f"Skipping to load model... {e}")
        except IsADirectoryError as e:
            self.logger.warning(f"Skipping to load model... {e}")

        self.logger.info("-------- Starting Evaluating... --------")

        self.evaluate()

    def evaluate(self):
        self.model.eval()
        val_criterion = nn.MSELoss()
        loader = DataLoader(self.validation, batch_size=self.batch_size, num_workers=2)
        loaderbar = tqdm(
            enumerate(loader),
            desc="Batches",
            mininterval=0.1,
            file=open(os.devnull, "w"),
        )

        V = 10
        full_loss = [0.0 for _ in range(V)]
        best_data: EvalSample | None = None
        worst_data: EvalSample | None = None
        random_data: EvalSample | None = None
        cur_loss: list[float] = [0.0 for _ in range(V)]
        best_aloss = float("inf")
        worst_aloss = float("-inf")

        with torch.no_grad():
            full_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for i, batch in loaderbar:
                inputs, outputs = batch
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device, dtype=torch.float32)

                pred = self.model(inputs)

                for var in range(V):
                    val_loss = val_criterion(pred[:, var], outputs[:, var])
                    cur_loss[var] = val_loss
                    full_loss[var] += val_loss

                if random_data is None:
                    j = random.randrange(self.batch_size)
                    random_data = EvalSample(
                        inp=inputs[j].detach().cpu(),
                        target=outputs[j].detach().cpu(),
                        pred=pred[j].detach().cpu(),
                        avg_loss=float(val_criterion(pred[j], outputs[j]).item()),
                        loss=cur_loss,
                    )

                for j in range(self.batch_size):
                    loss = val_criterion(pred[j, :, :, :, :], outputs[j, :, :, :, :])
                    if loss < best_aloss:
                        best_aloss = loss
                        best_data = EvalSample(
                            inp=inputs[j].detach().cpu(),
                            target=outputs[j].detach().cpu(),
                            pred=pred[j].detach().cpu(),
                            avg_loss=float(val_criterion(pred[j], outputs[j]).item()),
                            loss=cur_loss,
                        )
                    if loss > worst_aloss:
                        worst_aloss = loss
                        worst_data = EvalSample(
                            inp=inputs[j].detach().cpu(),
                            target=outputs[j].detach().cpu(),
                            pred=pred[j].detach().cpu(),
                            avg_loss=float(val_criterion(pred[j], outputs[j]).item()),
                            loss=cur_loss,
                        )

                if i % self.log_freq == self.log_freq - 1:
                    self.logger.info("{}  {}".format(str(loaderbar), i))

                    for var in range(10):
                        self.logger.info("loss {} : {}".format(var, full_loss[var] / (i + 1)))

            for var in range(10):
                self.logger.info("final_loss {} : {}".format(var, full_loss[var] / (len(loaderbar) + 1)))

        if best_data is not None and worst_data is not None and random_data is not None:
            data = EvalData(best_sample=best_data, worst_sample=worst_data, random_sample=random_data)
            self._save(data)

    @staticmethod
    def _to_numpy_img(x: Tensor) -> np.ndarray:
        """
        Convert a Tensor to a 2D numpy array suitable for imshow.
        Picks the first indices if there are extra dims.
        """
        arr = x.detach().cpu().numpy()

        # Squeeze batch/time dims etc.
        while arr.ndim > 2:
            arr = arr[0]

        if arr.ndim != 2:
            raise ValueError(f"Cannot convert tensor with shape {x.shape} to 2D image.")

        return arr

    def _plot_sample(self, sample: EvalSample, tag: str) -> None:
        """
        Plot target vs prediction for a single EvalSample and save as PNG.
        """
        save_dir = Path(self.path) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

        target_img = self._to_numpy_img(sample.target)
        pred_img = self._to_numpy_img(sample.pred)

        vmin = min(target_img.min(), pred_img.min())
        vmax = max(target_img.max(), pred_img.max())

        fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0))

        im0 = axes[0].imshow(target_img, origin="lower", vmin=vmin, vmax=vmax)  # noqa
        axes[0].set_title("Ground truth")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        im1 = axes[1].imshow(pred_img, origin="lower", vmin=vmin, vmax=vmax)
        axes[1].set_title("Prediction")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Single colorbar for both
        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Value")

        fig.suptitle(f"{self.name} - {tag} sample (avg loss={sample.avg_loss:.3f})", y=0.95)

        fig.tight_layout()
        out_path = save_dir / f"{self.name}_{tag}_sample.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def _save(self, data: EvalData):
        save_dir = Path(self.path) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update(
            {
                "figure.figsize": (6.0, 4.0),
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 10,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "savefig.dpi": 300,
            }
        )

        fig, ax = plt.subplots()

        leads_best = np.arange(len(data.best_sample.loss))
        leads_worst = np.arange(len(data.worst_sample.loss))
        leads_rand = np.arange(len(data.random_sample.loss))

        ax.plot(
            leads_best,
            data.best_sample.loss,
            marker="o",
            linestyle="-",
            label=f"Best (avg={data.best_sample.avg_loss:.3f})",
        )
        ax.plot(
            leads_worst,
            data.worst_sample.loss,
            marker="^",
            linestyle="--",
            label=f"Worst (avg={data.worst_sample.avg_loss:.3f})",
        )
        ax.plot(
            leads_rand,
            data.random_sample.loss,
            marker="s",
            linestyle="-.",
            label=f"Random (avg={data.random_sample.avg_loss:.3f})",
        )

        ax.set_xlabel("Lead time")
        ax.set_ylabel("Loss")
        ax.set_title("Per-lead validation loss")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend(frameon=False)

        fig.tight_layout()

        # Save in paper-friendly formats
        out_base = save_dir / f"{self.name}_eval_loss"
        fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
        fig.savefig(str(out_base.with_suffix(".png")), bbox_inches="tight")

        plt.close(fig)

        self._plot_sample(data.best_sample, "best")
        self._plot_sample(data.worst_sample, "worst")
        self._plot_sample(data.random_sample, "random")
