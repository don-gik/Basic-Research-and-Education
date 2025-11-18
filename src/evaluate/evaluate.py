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
import math
from omegaconf import DictConfig, ListConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset as TorchDataset
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


class _EvalSubsetDataset(TorchDataset):
    """
    Thin wrapper that augments a dataset/subset with the original index.
    Used so we can retrieve additional ground-truth frames beyond the
    immediate supervision horizon.
    """

    def __init__(self, base_dataset):
        self.base = base_dataset
        self.has_indices = hasattr(base_dataset, "indices") and hasattr(base_dataset, "dataset")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        if not isinstance(sample, tuple):
            raise TypeError("Evaluation dataset is expected to return (inputs, outputs) tuples.")
        base_idx = self.base.indices[idx] if self.has_indices else idx
        return (*sample, base_idx)


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
        eval_cfg = getattr(config, "eval", None)
        if eval_cfg is None:
            raise AttributeError("Configuration is missing the 'eval' section required for evaluation.")
        self.eval_cfg = eval_cfg
        self.path = eval_cfg.save
        self.epochs = config.train.epochs
        self.log_freq = config.train.log_freq
        self.name = config.model.name
        self.time = config.model.time
        lead_cfg = getattr(eval_cfg, "lead", self.time)
        try:
            parsed_lead = int(lead_cfg)
        except (TypeError, ValueError):
            parsed_lead = self.time
        self.lead = max(1, parsed_lead)
        self.var_cnt = config.data.var_cnt
        raw_var = getattr(eval_cfg, "var", self.var_cnt - 1)
        try:
            raw_var_idx = int(raw_var)
        except (TypeError, ValueError):
            raw_var_idx = self.var_cnt - 1
        self.var = raw_var_idx % self.var_cnt
        self.var_label = self._var_labels(self.var_cnt)[self.var]
        self.H = config.data.H
        self.W = config.data.W

        self.sequence_tensor = self._extract_sequence_tensor(self.validation)
        self.full_sequence_len = int(self.sequence_tensor.shape[1]) if self.sequence_tensor is not None else 0
        self.lead_trunc_warned = False
        if self.sequence_tensor is None:
            self.logger.warning(
                "Sequence tensor unavailable; lead evaluation is limited to the supervised horizon (%d).",
                self.time,
            )

        self.device = device
    
    def _log_sst(self, data: Tensor, time_step: int | None = None, n_samples: int = 100) -> None:
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
        horizon = time_step or self.lead

        C, T_total, _, _ = data.shape

        # Make sure we have enough time steps
        max_start = T_total - (self.time + horizon)
        if max_start <= 0:
            raise ValueError(
                "Not enough time steps in data (T={}) for self.time={} and lead horizon={}."
                .format(T_total, self.time, horizon)
            )

        # Accumulate MSE per lead (we will take sqrt at the end for RMSE)
        lead_mse_sum = [0.0 for _ in range(horizon)]

        self.model.eval()
        with torch.no_grad():
            # Repeat experiment for several different start indices
            for sample_idx in range(n_samples):
                # You can also use a fixed pattern (e.g. start = sample_idx)
                start = random.randint(0, max_start)

                # Initial window: [C, self.time, H, W] -> [1, C, self.time, H, W]
                test = data[:, start:start + self.time, -self.H:, -self.W:].unsqueeze(0)

                # Iterative 1-step-ahead prediction
                for lead in range(horizon):
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
        lead_rmse = [0.0 for _ in range(horizon)]
        for lead in range(horizon):
            mse = lead_mse_sum[lead] / float(n_samples)
            rmse = mse ** 0.5
            lead_rmse[lead] = rmse
            self.logger.info("{} Lead RMSE : {}".format(lead, rmse))

        # Plot RMSE lead curve
        save_dir = Path(self.path) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

        x = np.arange(horizon)

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

        eval_dataset = _EvalSubsetDataset(self.validation)
        loader = DataLoader(eval_dataset, batch_size=self.batch_size, num_workers=2)
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
            "per_lead_sse": torch.zeros(self.lead, dtype=torch.float64, device=self.device),
            "per_lead_point_sse": torch.zeros(self.lead, dtype=torch.float64, device=self.device),
            "per_lead_point_l1": torch.zeros(self.lead, dtype=torch.float64, device=self.device),
            "per_lead_counter": torch.zeros(self.lead, dtype=torch.float64, device=self.device),
            "per_lead_point_counter": torch.zeros(self.lead, dtype=torch.float64, device=self.device),
            "overall_sse": torch.zeros(1, dtype=torch.float64, device=self.device),
            "overall_l1": torch.zeros(1, dtype=torch.float64, device=self.device),
        }
        totals = {
            "pixel_count": 0,
            "point_count": 0,
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
                if len(batch) != 3:
                    raise ValueError("Evaluation loader must return (inputs, outputs, indices).")
                inputs, outputs, base_indices = batch
                sample_indices = self._to_index_list(base_indices)
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
                totals["numel"] += diff.numel()

                accum["per_var_sse"] += sq_error.sum(dim=(0, 2, 3, 4)).to(dtype=torch.float64)
                accum["per_var_l1"] += abs_error.sum(dim=(0, 2, 3, 4)).to(dtype=torch.float64)
                accum["overall_sse"] += sq_error.sum(dtype=torch.float64)
                accum["overall_l1"] += abs_error.sum(dtype=torch.float64)

                point_pred = pred[:, :, -1].mean(dim=(-1, -2))
                point_target = outputs[:, :, -1].mean(dim=(-1, -2))
                point_diff = point_pred - point_target
                accum["per_var_point_sse"] += point_diff.pow(2).sum(dim=0).to(dtype=torch.float64)
                accum["per_var_point_l1"] += point_diff.abs().sum(dim=0).to(dtype=torch.float64)
                totals["point_count"] += batch_size

                lead_targets, lead_horizon = self._build_target_sequence(outputs, sample_indices, self.lead)
                if lead_horizon > 0:
                    lead_preds = self._rollout_predictions(inputs, pred, lead_horizon)
                    lead_preds = lead_preds[:, :, :lead_horizon, :, :]
                    lead_targets = lead_targets[:, :, :lead_horizon, :, :]
                    lead_diff = lead_preds - lead_targets
                    lead_sq_error = lead_diff.pow(2)
                    accum["per_lead_sse"][:lead_horizon] += lead_sq_error.sum(dim=(0, 1, 3, 4)).to(dtype=torch.float64)
                    accum["per_lead_counter"][:lead_horizon] += batch_size * var_cnt * height * width

                    var_lead_pred = lead_preds[:, self.var, :, :, :].mean(dim=(-1, -2))
                    var_lead_target = lead_targets[:, self.var, :, :, :].mean(dim=(-1, -2))
                    var_lead_diff = var_lead_pred - var_lead_target

                    accum["per_lead_point_sse"][:lead_horizon] += var_lead_diff.pow(2).sum(dim=0).to(dtype=torch.float64)
                    accum["per_lead_point_l1"][:lead_horizon] += var_lead_diff.abs().sum(dim=0).to(dtype=torch.float64)
                    accum["per_lead_point_counter"][:lead_horizon] += batch_size

                    if lead_horizon < self.lead and not self.lead_trunc_warned:
                        self.logger.warning(
                            "Lead horizon truncated to %d (requested %d). Extend the sequence tensor for full evaluation.",
                            lead_horizon,
                            self.lead,
                        )
                        self.lead_trunc_warned = True

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
        numel_count = max(totals["numel"], 1)

        per_var_mse = (accum["per_var_sse"] / pixel_count).detach().cpu()
        per_var_rmse = torch.sqrt(per_var_mse)
        per_var_mae = (accum["per_var_l1"] / pixel_count).detach().cpu()

        per_var_point_mse = (accum["per_var_point_sse"] / point_count).detach().cpu()
        per_var_point_rmse = torch.sqrt(per_var_point_mse)
        per_var_point_mae = (accum["per_var_point_l1"] / point_count).detach().cpu()

        lead_counts = accum["per_lead_counter"].detach().cpu()
        lead_point_counts = accum["per_lead_point_counter"].detach().cpu()

        per_lead_mse = torch.full_like(lead_counts, float("nan"))
        per_lead_point_mse = torch.full_like(lead_point_counts, float("nan"))

        valid_lead = lead_counts > 0
        valid_point = lead_point_counts > 0

        per_lead_mse[valid_lead] = (accum["per_lead_sse"].detach().cpu()[valid_lead] / lead_counts[valid_lead])
        per_lead_point_mse[valid_point] = (
            accum["per_lead_point_sse"].detach().cpu()[valid_point] / lead_point_counts[valid_point]
        )
        per_lead_point_mae = torch.full_like(lead_point_counts, float("nan"))
        per_lead_point_mae[valid_point] = (
            accum["per_lead_point_l1"].detach().cpu()[valid_point] / lead_point_counts[valid_point]
        )

        per_lead_rmse = torch.sqrt(per_lead_mse)
        per_lead_point_rmse = torch.sqrt(per_lead_point_mse)

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
                "rmse": self._tensor_to_list(per_lead_rmse),
                "point_rmse": self._tensor_to_list(per_lead_point_rmse),
                "point_mae": self._tensor_to_list(per_lead_point_mae),
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
        if isinstance(lead_rmse, list) and lead_rmse:
            summary = self._format_lead_summary(lead_rmse)
            self.logger.info("Per-lead RMSE -> %s", summary)
        lead_point_rmse = lead_metrics.get("point_rmse")
        if isinstance(lead_point_rmse, list) and lead_point_rmse:
            summary = self._format_lead_summary(lead_point_rmse)
            self.logger.info("Per-lead point RMSE (%s) -> %s", self.var_label, summary)

    def _save_metrics(self, metrics: dict[str, Any]) -> None:
        if not metrics:
            return

        save_dir = Path(self.path) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

        out_path = save_dir / f"{self.name}_metrics.json"
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)

    def _build_target_sequence(
        self, outputs: Tensor, sample_indices: list[int], desired_horizon: int
    ) -> tuple[Tensor, int]:
        """
        Combine direct supervision targets with additional frames from the raw sequence tensor
        so per-lead metrics can extend beyond the immediate training horizon.
        """
        base_len = outputs.size(2)
        initial_len = min(base_len, desired_horizon)
        targets = outputs[:, :, :initial_len, :, :].detach()

        if desired_horizon <= initial_len or self.sequence_tensor is None:
            return targets, initial_len

        available_extra = [
            max(0, self.full_sequence_len - (idx + self.time + 1)) for idx in sample_indices
        ]
        if not available_extra:
            return targets, initial_len

        max_extra = min(available_extra)
        extra_len = min(desired_horizon - initial_len, max_extra)
        if extra_len <= 0:
            return targets, initial_len

        extras: list[Tensor] = []
        for idx in sample_indices:
            start = idx + self.time + 1
            end = min(start + extra_len, self.full_sequence_len)
            slice_tensor = self.sequence_tensor[:, start:end, : self.H, : self.W]
            extras.append(slice_tensor)

        extra_tensor = torch.stack(extras, dim=0).to(outputs.device, dtype=outputs.dtype)
        full_targets = torch.cat([targets, extra_tensor], dim=2)
        return full_targets, full_targets.size(2)

    def _rollout_predictions(self, inputs: Tensor, base_pred: Tensor, horizon: int) -> Tensor:
        """
        Use the model autoregressively to reach the requested forecast horizon.
        Assumes evaluation runs under torch.no_grad.
        """
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

    def _extract_sequence_tensor(self, dataset) -> Tensor | None:
        """
        Try to retrieve the underlying raw tensor (SequentialDataset.x) from nested subsets.
        """
        current = dataset
        depth = 0
        while hasattr(current, "dataset") and depth < 16:
            current = getattr(current, "dataset")
            depth += 1
        return getattr(current, "x", None)

    @staticmethod
    def _to_index_list(indices: Any) -> list[int]:
        if isinstance(indices, torch.Tensor):
            return [int(val) for val in indices.view(-1).tolist()]
        if isinstance(indices, (list, tuple)):
            return [int(val) for val in indices]
        return [int(indices)]

    @staticmethod
    def _tensor_to_list(tensor: torch.Tensor) -> list[float | None]:
        values: list[float | None] = []
        for value in tensor.detach().cpu().tolist():
            if isinstance(value, float) and math.isnan(value):
                values.append(None)
            else:
                values.append(value)
        return values

    @staticmethod
    def _lead_array(values: list[float | None]) -> np.ndarray:
        return np.array([np.nan if value is None else value for value in values], dtype=float)

    @staticmethod
    def _format_lead_summary(values: list[float | None]) -> str:
        formatted = []
        for idx, value in enumerate(values):
            if value is None:
                formatted.append(f"t{idx}:--")
            else:
                formatted.append(f"t{idx}:{value:.5f}")
        return ", ".join(formatted)

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
        self._plot_lead_point_breakdown(metrics, save_dir)

        self._plot_sample(data.best_sample, "best")
        self._plot_sample(data.worst_sample, "worst")
        self._plot_sample(data.random_sample, "random")

    def _plot_loss_breakdown(self, metrics: dict[str, Any], save_dir: Path) -> None:
        per_var = metrics.get("per_variable", {})
        rmse = per_var.get("rmse")
        point_rmse = per_var.get("point_rmse")
        mae = per_var.get("mae")
        point_mae = per_var.get("point_mae")
        if not rmse or not point_rmse:
            return

        rmse_arr = np.asarray(rmse, dtype=float)
        point_rmse_arr = np.asarray(point_rmse, dtype=float)
        labels = np.asarray(self._var_labels(len(rmse_arr)))
        order = np.argsort(point_rmse_arr)[::-1]
        rmse_arr = rmse_arr[order]
        point_rmse_arr = point_rmse_arr[order]
        labels = labels[order]

        show_mae = bool(mae and point_mae and len(mae) == len(rmse_arr))
        mae_arr = np.asarray(mae, dtype=float)[order] if show_mae else None
        point_mae_arr = np.asarray(point_mae, dtype=float)[order] if show_mae else None

        fig_cols = 2 if show_mae else 1
        fig, axes = plt.subplots(1, fig_cols, figsize=(6.0 * fig_cols, 3.2), dpi=300)
        if fig_cols == 1:
            axes = [axes]
        else:
            axes = list(axes)

        width = 0.38
        x = np.arange(len(labels))

        axes[0].bar(x - width / 2, rmse_arr, width, label="Pixel RMSE")
        axes[0].bar(x + width / 2, point_rmse_arr, width, label="Point RMSE")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha="right")
        axes[0].set_ylabel("RMSE")
        axes[0].set_xlabel("Variable")
        axes[0].set_title("Per-variable RMSE (sorted by point error)")
        axes[0].grid(True, linestyle=":", linewidth=0.4)
        axes[0].legend(frameon=False, fontsize=8)

        for idx in range(min(3, len(x))):
            axes[0].annotate(
                f"{point_rmse_arr[idx]:.2f}",
                xy=(x[idx] + width / 2, point_rmse_arr[idx]),
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=7,
                ha="center",
            )

        if show_mae and mae_arr is not None and point_mae_arr is not None and len(axes) > 1:
            axes[1].bar(x - width / 2, mae_arr, width, label="Pixel MAE", color="#7aa0c4")
            axes[1].bar(x + width / 2, point_mae_arr, width, label="Point MAE", color="#c47a8b")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(labels, rotation=45, ha="right")
            axes[1].set_ylabel("MAE")
            axes[1].set_xlabel("Variable")
            axes[1].set_title("Per-variable MAE (pixel vs point)")
            axes[1].grid(True, linestyle=":", linewidth=0.4)
            axes[1].legend(frameon=False, fontsize=8)

        for ax in axes:
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

        overall = metrics.get("overall", {})
        summary_text = (
            f"Overall RMSE: {overall.get('rmse', float('nan')):.3f} | "
            f"MSE: {overall.get('mse', float('nan')):.3f} | "
            f"MAE: {overall.get('mae', float('nan')):.3f}"
        )
        fig.text(0.5, 0.04, summary_text, ha="center", fontsize=9)

        fig.tight_layout(rect=(0, 0.08, 1, 1))
        out_base = save_dir / f"{self.name}_per_var_errors"
        fig.savefig(str(out_base.with_suffix(".png")), bbox_inches="tight")
        fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
        plt.close(fig)

    def _plot_lead_point_breakdown(self, metrics: dict[str, Any], save_dir: Path) -> None:
        per_lead = metrics.get("per_lead", {})
        point_rmse_vals = per_lead.get("point_rmse")
        if not point_rmse_vals:
            return

        point_rmse_arr = self._lead_array(point_rmse_vals)
        if not np.isfinite(point_rmse_arr).any():
            return

        point_mae_vals = per_lead.get("point_mae")
        point_mae_arr = None
        if isinstance(point_mae_vals, list) and len(point_mae_vals) == len(point_rmse_arr):
            point_mae_arr = self._lead_array(point_mae_vals)
            if not np.isfinite(point_mae_arr).any():
                point_mae_arr = None

        x = np.arange(len(point_rmse_arr))
        fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=300)
        ax.plot(x, point_rmse_arr, marker="o", linewidth=1.2, color="#8e44ad", label="Point RMSE")
        if point_mae_arr is not None:
            ax.plot(x, point_mae_arr, marker="s", linewidth=1.0, color="#f39c12", label="Point MAE")

        best_idx = int(np.nanargmin(point_rmse_arr))
        worst_idx = int(np.nanargmax(point_rmse_arr))
        ax.scatter(best_idx, point_rmse_arr[best_idx], color="#2ecc71", s=30, zorder=5)
        ax.scatter(worst_idx, point_rmse_arr[worst_idx], color="#e74c3c", s=30, zorder=5)

        ax.set_xticks(x)
        ax.set_xlabel("Lead index")
        ax.set_ylabel("Error")
        ax.set_title(f"{self.var_label} point error by lead")
        ax.grid(True, linestyle=":", linewidth=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")

        summary_text = (
            f"min RMSE t{best_idx}: {point_rmse_arr[best_idx]:.3f}\n"
            f"max RMSE t{worst_idx}: {point_rmse_arr[worst_idx]:.3f}"
        )
        ax.text(
            0.98,
            0.02,
            summary_text,
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.85"),
        )

        fig.tight_layout()
        safe_label = self.var_label.replace("/", "-")
        out_base = save_dir / f"{self.name}_{safe_label}_lead_point_error"
        fig.savefig(str(out_base.with_suffix(".png")), bbox_inches="tight")
        fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
        plt.close(fig)

    def _plot_lead_breakdown(self, metrics: dict[str, Any], save_dir: Path) -> None:
        per_lead = metrics.get("per_lead", {})
        rmse_vals = per_lead.get("rmse")
        if not rmse_vals:
            return

        rmse_arr = self._lead_array(rmse_vals)
        if not np.isfinite(rmse_arr).any():
            return

        x = np.arange(len(rmse_arr))
        fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=300)
        ax.plot(x, rmse_arr, marker="o", linestyle="-", linewidth=1.2, label="Pixel RMSE", color="#2a4d69")

        best_idx = int(np.nanargmin(rmse_arr))
        worst_idx = int(np.nanargmax(rmse_arr))
        ax.scatter(best_idx, rmse_arr[best_idx], color="#2ecc71", s=30, zorder=5, label="Min RMSE")
        ax.scatter(worst_idx, rmse_arr[worst_idx], color="#e74c3c", s=30, zorder=5, label="Max RMSE")

        mean_rmse = float(np.nanmean(rmse_arr))
        ax.axhline(mean_rmse, color="0.5", linestyle="--", linewidth=0.8, label=f"Mean {mean_rmse:.3f}")

        ax.set_xticks(x)
        ax.set_xlabel("Lead index")
        ax.set_ylabel("RMSE")
        ax.set_title("Lead-wise pixel RMSE")
        ax.grid(True, linestyle=":", linewidth=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

        stats_text = (
            f"min t{best_idx}: {rmse_arr[best_idx]:.3f}\n"
            f"max t{worst_idx}: {rmse_arr[worst_idx]:.3f}\n"
            f"std: {np.nanstd(rmse_arr):.3f}"
        )
        ax.text(
            0.98,
            0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.85"),
        )

        fig.tight_layout()
        out_base = save_dir / f"{self.name}_lead_rmse"
        fig.savefig(str(out_base.with_suffix(".png")), bbox_inches="tight")
        fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
        plt.close(fig)
