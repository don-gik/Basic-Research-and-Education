import json
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import Model


VAR_NAMES = [
    "u100",
    "v100",
    "u10",
    "v10",
    "d2m",
    "t2m",
    "msl",
    "sp",
    "ssrc",
    "sst",
]


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

        self.batch_size = 1
        self.subset_size = config.train.subset_size

        self.load = config.train.load
        self.path = config.eval.save
        self.epochs = config.train.epochs
        self.log_freq = config.train.log_freq
        self.name = config.model.name
        self.lead = config.model.time
        self.var = config.eval.var
        self.var_cnt = config.data.var_cnt
        self.time = config.model.time
        self.H = config.data.H
        self.W = config.data.W

        self.device = device
    
    def _log_sst(self, data: Tensor, time_step: int = 12, n_samples: int = 100) -> None:
        """
        Compute an SST lead-time curve by iteratively predicting and
        comparing to ground truth, then plot RMSE per lead.

        data: Tensor of shape [C, T, H, W]
        time_step: number of forecast leads
        n_samples: number of different start indices to average over
        """
        # Move data and criterion outside loops
        data = data.to(self.device)
        val_criterion = nn.MSELoss()

        C, T_total, _, _ = data.shape

        # Make sure we have enough time steps
        max_start = T_total - (self.time + time_step)
        if max_start <= 0:
            raise ValueError(
                "Not enough time steps in data (T={}) for self.time={} and time_step={}."
                .format(T_total, self.time, time_step)
            )

        # Accumulate MSE per lead (we will take sqrt at the end for RMSE)
        lead_mse_sum = [0.0 for _ in range(time_step)]

        self.model.eval()
        with torch.no_grad():
            # Repeat experiment for several different start indices
            for sample_idx in range(n_samples):
                # You can also use a fixed pattern (e.g. start = sample_idx)
                start = random.randint(0, max_start)

                # Initial window: [C, self.time, H, W] -> [1, C, self.time, H, W]
                test = data[:, start:start + self.time, -self.H:, -self.W:].unsqueeze(0)

                # Iterative 1-step-ahead prediction
                for lead in range(time_step):
                    # Use only the last self.time time steps as input
                    inp = test[:, :, -self.time:, :, :]  # [1, C, self.time, H, W]
                    pred = self.model(inp)               # [1, C, T_out, H, W] (assumed)

                    # Predicted SST at next step: last time index of SST channel
                    pred_sst = pred[0, -1, -1, :, :]     # [H, W]

                    # True SST at (start + self.time + lead)
                    true_sst = data[-1, start + self.time + lead, -self.H:, -self.W:]

                    # Global "point" value: average over H, W -> 0-D
                    point_pred = torch.mean(torch.mean(pred_sst, dim=-1), dim=-1)
                    point_target = torch.mean(torch.mean(true_sst, dim=-1), dim=-1)

                    # MSE between scalar predictions (this is just (pred - target)^2)
                    val_loss = val_criterion(point_pred, point_target)
                    lead_mse_sum[lead] += val_loss.item()

                    # Append the predicted last frame for all channels to the test window
                    next_frame = pred[:, :, -1:, :, :]   # [1, C, 1, H, W]
                    test = torch.cat((test, next_frame), dim=2)

        # Compute RMSE per lead and log
        lead_rmse = [0.0 for _ in range(time_step)]
        for lead in range(time_step):
            mse = lead_mse_sum[lead] / float(n_samples)
            rmse = mse ** 0.5
            lead_rmse[lead] = rmse
            self.logger.info("{} Lead RMSE : {}".format(lead, rmse))

        # Plot RMSE lead curve
        save_dir = Path(self.path) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

        x = np.arange(time_step)

        fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)

        ax.plot(
            x,
            lead_rmse,
            marker="o",
            linestyle="-",
            linewidth=1.0,
            markersize=3,
            label="SST RMSE",
        )

        ax.set_xticks(x)
        ax.set_xlabel("Lead index")
        ax.set_ylabel("RMSE")
        ax.set_title("SST lead-time RMSE")

        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True, which="both", linestyle=":", linewidth=0.4)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.legend(frameon=False, fontsize=8, loc="best")
        fig.tight_layout()

        out_base = save_dir / f"{self.name}_sst_lead_rmse"
        fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
        fig.savefig(str(out_base.with_suffix(".png")), bbox_inches="tight")

        plt.close(fig)

    def _log(self, data: Tensor) -> None:
        """
        Public wrapper used by eval_run to trigger the SST lead-time logging.
        """
        try:
            self._log_sst(data)
        except Exception as exc:  # pragma: no cover - logging helper
            self.logger.warning("Failed to log SST lead curve: %s", exc)

    def run(self) -> None:
        try:
            self.model.load_state_dict(torch.load(self.load, weights_only=True))
        except FileNotFoundError as e:
            self.logger.warning(f"Skipping to load model... {e}")
            return
        except IsADirectoryError as e:
            self.logger.warning(f"Skipping to load model... {e}")
            return

        self.logger.info("-------- Starting Evaluating... --------")

        samples, metrics = self.evaluate()

        if samples is not None:
            self._save(samples, metrics)
        else:
            self.logger.warning("No qualitative samples were collected; skipping sample plots.")

        self._save_metrics(metrics)

    def evaluate(self) -> tuple[EvalData | None, dict[str, Any]]:
        self.model.eval()

        loader = DataLoader(self.validation, batch_size=self.batch_size, num_workers=2)
        total_batches = len(loader)
        if total_batches == 0:
            self.logger.warning("Validation dataloader is empty, skipping evaluation.")
            return None, {}

        loaderbar = tqdm(
            enumerate(loader),
            desc="Batches",
            mininterval=0.1,
        )

        accum = {
            "per_var_sse": torch.zeros(self.var_cnt, dtype=torch.float64, device=self.device),
            "per_var_l1": torch.zeros(self.var_cnt, dtype=torch.float64, device=self.device),
            "per_var_point_sse": torch.zeros(self.var_cnt, dtype=torch.float64, device=self.device),
            "per_var_point_l1": torch.zeros(self.var_cnt, dtype=torch.float64, device=self.device),
            "per_lead_sse": torch.zeros(self.time, dtype=torch.float64, device=self.device),
            "overall_sse": torch.zeros(1, dtype=torch.float64, device=self.device),
            "overall_l1": torch.zeros(1, dtype=torch.float64, device=self.device),
        }
        totals = {
            "pixel_count": 0,
            "point_count": 0,
            "lead_count": 0,
            "numel": 0,
            "samples": 0,
        }

        best_data: EvalSample | None = None
        worst_data: EvalSample | None = None
        random_data: EvalSample | None = None
        best_loss = float("inf")
        worst_loss = float("-inf")

        with torch.no_grad():
            for batch_idx, batch in loaderbar:
                inputs, outputs = batch
                inputs = inputs.to(self.device, non_blocking=True)
                outputs = outputs.to(self.device, dtype=torch.float32, non_blocking=True)

                pred = self.model(inputs)
                diff = pred - outputs
                sq_error = diff.pow(2)
                abs_error = diff.abs()

                batch_size = inputs.size(0)
                var_cnt = diff.shape[1]
                time_dim = diff.shape[2]
                height = diff.shape[3]
                width = diff.shape[4]

                pixels_per_sample = time_dim * height * width
                totals["samples"] += batch_size
                totals["pixel_count"] += batch_size * pixels_per_sample
                totals["lead_count"] += batch_size * var_cnt * height * width
                totals["numel"] += diff.numel()

                accum["per_var_sse"] += sq_error.sum(dim=(0, 2, 3, 4)).to(dtype=torch.float64)
                accum["per_var_l1"] += abs_error.sum(dim=(0, 2, 3, 4)).to(dtype=torch.float64)
                accum["per_lead_sse"] += sq_error.sum(dim=(0, 1, 3, 4)).to(dtype=torch.float64)
                accum["overall_sse"] += sq_error.sum(dtype=torch.float64)
                accum["overall_l1"] += abs_error.sum(dtype=torch.float64)

                point_pred = pred[:, :, -1].mean(dim=(-1, -2))
                point_target = outputs[:, :, -1].mean(dim=(-1, -2))
                point_diff = point_pred - point_target
                accum["per_var_point_sse"] += point_diff.pow(2).sum(dim=0).to(dtype=torch.float64)
                accum["per_var_point_l1"] += point_diff.abs().sum(dim=0).to(dtype=torch.float64)
                totals["point_count"] += batch_size

                sample_var_loss = sq_error.mean(dim=(2, 3, 4))
                sample_avg_loss = sample_var_loss.mean(dim=1)

                if random_data is None:
                    rand_idx = random.randrange(batch_size)
                    random_data = self._make_eval_sample(rand_idx, inputs, outputs, pred, sample_var_loss, sample_avg_loss)

                for j in range(batch_size):
                    loss_value = float(sample_avg_loss[j].item())
                    if loss_value < best_loss:
                        best_loss = loss_value
                        best_data = self._make_eval_sample(j, inputs, outputs, pred, sample_var_loss, sample_avg_loss)
                    if loss_value > worst_loss:
                        worst_loss = loss_value
                        worst_data = self._make_eval_sample(j, inputs, outputs, pred, sample_var_loss, sample_avg_loss)

                if (batch_idx + 1) % self.log_freq == 0 or batch_idx == 0:
                    self._log_running_losses(accum, totals, batch_idx + 1, total_batches)

        metrics = self._build_metrics(accum, totals)
        self._log_metrics(metrics)

        samples = None
        if best_data is not None and worst_data is not None and random_data is not None:
            samples = EvalData(best_sample=best_data, worst_sample=worst_data, random_sample=random_data)

        return samples, metrics

    def _make_eval_sample(
        self,
        idx: int,
        inputs: Tensor,
        targets: Tensor,
        preds: Tensor,
        per_var_loss: Tensor,
        avg_loss: Tensor,
    ) -> EvalSample:
        return EvalSample(
            inp=inputs[idx].detach().cpu(),
            target=targets[idx].detach().cpu(),
            pred=preds[idx].detach().cpu(),
            avg_loss=float(avg_loss[idx].item()),
            loss=per_var_loss[idx].detach().cpu().tolist(),
        )

    def _log_running_losses(
        self, accum: dict[str, torch.Tensor], totals: dict[str, int], step: int, total_steps: int
    ) -> None:
        pixel_count = max(totals["pixel_count"], 1)
        running_mse = (accum["per_var_sse"] / pixel_count).detach().cpu().tolist()
        labels = self._var_labels(len(running_mse))
        summary = ", ".join(f"{label}:{value:.5f}" for label, value in zip(labels, running_mse))
        self.logger.info("Batch %d/%d pixel MSE -> %s", step, total_steps, summary)

    def _build_metrics(self, accum: dict[str, torch.Tensor], totals: dict[str, int]) -> dict[str, Any]:
        if totals["samples"] == 0:
            return {}

        pixel_count = max(totals["pixel_count"], 1)
        point_count = max(totals["point_count"], 1)
        lead_count = max(totals["lead_count"], 1)
        numel_count = max(totals["numel"], 1)

        per_var_mse = (accum["per_var_sse"] / pixel_count).detach().cpu()
        per_var_rmse = torch.sqrt(per_var_mse)
        per_var_mae = (accum["per_var_l1"] / pixel_count).detach().cpu()

        per_var_point_mse = (accum["per_var_point_sse"] / point_count).detach().cpu()
        per_var_point_rmse = torch.sqrt(per_var_point_mse)
        per_var_point_mae = (accum["per_var_point_l1"] / point_count).detach().cpu()

        per_lead_rmse = torch.sqrt((accum["per_lead_sse"] / lead_count).detach().cpu())

        overall_mse = float((accum["overall_sse"] / numel_count).item())
        overall_rmse = overall_mse ** 0.5
        overall_mae = float((accum["overall_l1"] / numel_count).item())

        metrics: dict[str, Any] = {
            "per_variable": {
                "mse": per_var_mse.tolist(),
                "rmse": per_var_rmse.tolist(),
                "mae": per_var_mae.tolist(),
                "point_mse": per_var_point_mse.tolist(),
                "point_rmse": per_var_point_rmse.tolist(),
                "point_mae": per_var_point_mae.tolist(),
            },
            "per_lead": {
                "rmse": per_lead_rmse.tolist(),
            },
            "overall": {
                "mse": overall_mse,
                "rmse": overall_rmse,
                "mae": overall_mae,
            },
            "totals": totals.copy(),
        }

        return metrics

    def _log_metrics(self, metrics: dict[str, Any]) -> None:
        if not metrics:
            self.logger.warning("No metrics were produced during evaluation.")
            return

        per_var = metrics.get("per_variable", {})
        rmse = per_var.get("rmse")
        mse = per_var.get("mse")
        mae = per_var.get("mae")
        point_rmse = per_var.get("point_rmse")

        if rmse and mse and mae and point_rmse:
            labels = self._var_labels(len(rmse))
            for label, mse_val, rmse_val, mae_val, prmse in zip(labels, mse, rmse, mae, point_rmse):
                self.logger.info(
                    "Var %-4s -> pixel MSE: %.6f | RMSE: %.6f | MAE: %.6f | point RMSE: %.6f",
                    label,
                    mse_val,
                    rmse_val,
                    mae_val,
                    prmse,
                )

        overall = metrics.get("overall")
        if overall:
            self.logger.info(
                "Overall -> pixel MSE: %.6f | RMSE: %.6f | MAE: %.6f",
                overall.get("mse", 0.0),
                overall.get("rmse", 0.0),
                overall.get("mae", 0.0),
            )

        lead_metrics = metrics.get("per_lead", {})
        lead_rmse = lead_metrics.get("rmse")
        if lead_rmse:
            summary = ", ".join(f"t{idx}:{value:.5f}" for idx, value in enumerate(lead_rmse))
            self.logger.info("Per-lead RMSE -> %s", summary)

    def _save_metrics(self, metrics: dict[str, Any]) -> None:
        if not metrics:
            return

        save_dir = Path(self.path) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

        out_path = save_dir / f"{self.name}_metrics.json"
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)

    def _var_labels(self, count: int) -> list[str]:
        if count <= len(VAR_NAMES):
            return VAR_NAMES[:count]

        labels = VAR_NAMES[:]
        labels.extend(f"var_{idx}" for idx in range(len(labels), count))
        return labels[:count]

    @staticmethod
    def _to_numpy_img(x: Tensor, var: int) -> np.ndarray:
        """
        Convert a Tensor to a 2D numpy array suitable for imshow.
        Picks the first indices if there are extra dims.
        """
        arr = x.detach().cpu().numpy()

        # Squeeze batch/time dims etc.
        while arr.ndim > 3:
            arr = arr[0]
        
        if arr.ndim == 3:
            arr = arr[var]

        if arr.ndim != 2:
            raise ValueError(f"Cannot convert tensor with shape {x.shape} to 2D image.")

        return arr

    def _plot_sample(self, sample: EvalSample, tag: str) -> None:
        """
        Plot target vs prediction (with error) for a single EvalSample and save as PNG.
        """
        save_dir = Path(self.path) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

        target_img = self._to_numpy_img(sample.target, self.var)
        pred_img   = self._to_numpy_img(sample.pred, self.var)
        error_img  = np.abs(target_img - pred_img)

        target_img = np.asarray(target_img)
        pred_img   = np.asarray(pred_img)
        error_img  = np.asarray(error_img)

        vmin = min(target_img.min(), pred_img.min())
        vmax = max(target_img.max(), pred_img.max())

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(9.0, 3.0),
            dpi=300,
            constrained_layout=True,
        )

        im0 = axes[0].imshow(target_img, origin="lower", vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground truth")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        im1 = axes[1].imshow(pred_img, origin="lower", vmin=vmin, vmax=vmax)
        axes[1].set_title("Prediction")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        im2 = axes[2].imshow(error_img, origin="lower", cmap="magma")
        axes[2].set_title("|Prediction - Truth|")
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        cbar = fig.colorbar(im1, ax=axes[:2].ravel().tolist(), fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Value", fontsize=9)

        err_cbar = fig.colorbar(im2, ax=[axes[2]], fraction=0.046, pad=0.04)
        err_cbar.ax.set_ylabel("|Error|", fontsize=9)

        loss_labels = self._var_labels(len(sample.loss))
        loss_lines = "\n".join(f"{label}: {value:.4f}" for label, value in zip(loss_labels, sample.loss))
        fig.text(0.01, 0.02, f"Per-variable MSE\n{loss_lines}", ha="left", va="bottom", fontsize=8)

        fig.suptitle(f"{self.name} - {tag} sample (avg loss={sample.avg_loss:.3f})")

        out_path = save_dir / f"{self.name}_{tag}_sample.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _save(self, data: EvalData, metrics: dict[str, Any]) -> None:
        save_dir = Path(self.path) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 10,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "savefig.dpi": 300,
            }
        )

        self._plot_loss_breakdown(metrics, save_dir)
        self._plot_lead_breakdown(metrics, save_dir)

        self._plot_sample(data.best_sample, "best")
        self._plot_sample(data.worst_sample, "worst")
        self._plot_sample(data.random_sample, "random")

    def _plot_loss_breakdown(self, metrics: dict[str, Any], save_dir: Path) -> None:
        per_var = metrics.get("per_variable", {})
        rmse = per_var.get("rmse")
        point_rmse = per_var.get("point_rmse")
        if not rmse or not point_rmse:
            return

        labels = self._var_labels(len(rmse))
        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(5.5, 3.0), dpi=300)
        ax.bar(x - width / 2, rmse, width, label="Pixel RMSE")
        ax.bar(x + width / 2, point_rmse, width, label="Point RMSE")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Variable")
        ax.set_title("Per-variable RMSE (pixel vs point)")
        ax.grid(True, linestyle=":", linewidth=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        out_base = save_dir / f"{self.name}_per_var_rmse"
        fig.savefig(str(out_base.with_suffix(".png")), bbox_inches="tight")
        fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
        plt.close(fig)

    def _plot_lead_breakdown(self, metrics: dict[str, Any], save_dir: Path) -> None:
        per_lead = metrics.get("per_lead", {})
        rmse = per_lead.get("rmse")
        if not rmse:
            return

        x = np.arange(len(rmse))
        fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)
        ax.plot(x, rmse, marker="o", linestyle="-", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xlabel("Lead index")
        ax.set_ylabel("RMSE")
        ax.set_title("Lead-wise RMSE (direct)")
        ax.grid(True, linestyle=":", linewidth=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        fig.tight_layout()
        out_base = save_dir / f"{self.name}_lead_rmse"
        fig.savefig(str(out_base.with_suffix(".png")), bbox_inches="tight")
        fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
        plt.close(fig)
