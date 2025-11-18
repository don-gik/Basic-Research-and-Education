import json
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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
    target_phys: Tensor
    pred_phys: Tensor
    target_norm: Tensor
    pred_norm: Tensor
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
        self.channel_labels = self._var_labels(self.var_cnt)
        self.var_name_to_idx = {name: idx for idx, name in enumerate(self.channel_labels)}
        raw_var = getattr(eval_cfg, "var", self.var_cnt - 1)
        try:
            raw_var_idx = int(raw_var)
        except (TypeError, ValueError):
            raw_var_idx = self.var_cnt - 1
        self.var = raw_var_idx % self.var_cnt
        self.var_label = self.channel_labels[self.var]
        self.H = config.data.H
        self.W = config.data.W

        stats_path = Path(getattr(config.data, "stats_path", "./data/processed/monthly_tensor_stats.json"))
        self.stats_path = stats_path
        self.channel_stats = self._load_channel_stats(stats_path)
        self.var_std_tensor = self._build_var_std_tensor(self.channel_stats, self.channel_labels)
        self.sst_idx = self.var_name_to_idx.get("sst", min(self.var_cnt - 1, len(self.channel_labels) - 1))

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
        sst_idx: int = 0
        if self.sst_idx is not None:
            sst_idx = self.sst_idx
            
        sst_scale = self._var_std_scalar(sst_idx, self.device, data.dtype)

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
                    pred_sst = pred[0, sst_idx, -1, :, :]     # [H, W]

                    # True SST at (start + self.time + lead)
                    true_sst = data[sst_idx, start + self.time + lead, -self.H:, -self.W:]

                    pred_sst = pred_sst * sst_scale
                    true_sst = true_sst * sst_scale

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

        accum_phys = self._init_accumulator()
        accum_norm = self._init_accumulator()
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
                scaled_outputs: Tensor = self._scale_by_std(outputs)
                scaled_pred: Tensor = self._scale_by_std(pred)
                phys_diff = scaled_pred - scaled_outputs
                phys_sq_error = phys_diff.pow(2)
                phys_abs_error = phys_diff.abs()

                norm_diff = pred - outputs
                norm_sq_error = norm_diff.pow(2)
                norm_abs_error = norm_diff.abs()

                batch_size = inputs.size(0)
                var_cnt = phys_diff.shape[1]
                time_dim = phys_diff.shape[2]
                height = phys_diff.shape[3]
                width = phys_diff.shape[4]

                pixels_per_sample = time_dim * height * width
                totals["samples"] += batch_size
                totals["pixel_count"] += batch_size * pixels_per_sample
                totals["numel"] += phys_diff.numel()

                self._update_batch_metrics(
                    accum_phys,
                    phys_sq_error,
                    phys_abs_error,
                    scaled_pred,
                    scaled_outputs,
                )
                self._update_batch_metrics(
                    accum_norm,
                    norm_sq_error,
                    norm_abs_error,
                    pred,
                    outputs,
                )
                totals["point_count"] += batch_size

                lead_targets_norm, lead_horizon = self._build_target_sequence(outputs, sample_indices, self.lead)
                if lead_horizon > 0:
                    lead_preds_norm = self._rollout_predictions(inputs, pred, lead_horizon)
                    lead_preds_norm = lead_preds_norm[:, :, :lead_horizon, :, :]
                    lead_targets_norm = lead_targets_norm[:, :, :lead_horizon, :, :]

                    lead_preds_phys = self._scale_by_std(lead_preds_norm)
                    lead_targets_phys = self._scale_by_std(lead_targets_norm)

                    self._update_lead_metrics(
                        accum_phys,
                        lead_preds_phys,
                        lead_targets_phys,
                        lead_horizon,
                        batch_size,
                        var_cnt,
                        height,
                        width,
                    )
                    self._update_lead_metrics(
                        accum_norm,
                        lead_preds_norm,
                        lead_targets_norm,
                        lead_horizon,
                        batch_size,
                        var_cnt,
                        height,
                        width,
                    )

                    if lead_horizon < self.lead and not self.lead_trunc_warned:
                        self.logger.warning(
                            "Lead horizon truncated to %d (requested %d). Extend the sequence tensor for full evaluation.",
                            lead_horizon,
                            self.lead,
                        )
                        self.lead_trunc_warned = True

                sample_var_loss = phys_sq_error.mean(dim=(2, 3, 4))
                sample_avg_loss = sample_var_loss.mean(dim=1)

                if random_data is None:
                    rand_idx = random.randrange(batch_size)
                    random_data = self._make_eval_sample(
                        rand_idx,
                        inputs,
                        scaled_outputs,
                        scaled_pred,
                        outputs,
                        pred,
                        sample_var_loss,
                        sample_avg_loss,
                    )

                for j in range(batch_size):
                    loss_value = float(sample_avg_loss[j].item())
                    if loss_value < best_loss:
                        best_loss = loss_value
                        best_data = self._make_eval_sample(
                            j,
                            inputs,
                            scaled_outputs,
                            scaled_pred,
                            outputs,
                            pred,
                            sample_var_loss,
                            sample_avg_loss,
                        )
                    if loss_value > worst_loss:
                        worst_loss = loss_value
                        worst_data = self._make_eval_sample(
                            j,
                            inputs,
                            scaled_outputs,
                            scaled_pred,
                            outputs,
                            pred,
                            sample_var_loss,
                            sample_avg_loss,
                        )

                if (batch_idx + 1) % self.log_freq == 0 or batch_idx == 0:
                    self._log_running_losses(accum_phys, totals, batch_idx + 1, total_batches)

        metrics = self._build_metrics(accum_phys, accum_norm, totals)
        self._log_metrics(metrics)

        samples = None
        if best_data is not None and worst_data is not None and random_data is not None:
            samples = EvalData(best_sample=best_data, worst_sample=worst_data, random_sample=random_data)

        return samples, metrics

    def _make_eval_sample(
        self,
        idx: int,
        inputs: Tensor,
        targets_phys: Tensor,
        preds_phys: Tensor,
        targets_norm: Tensor,
        preds_norm: Tensor,
        per_var_loss: Tensor,
        avg_loss: Tensor,
    ) -> EvalSample:
        return EvalSample(
            inp=inputs[idx].detach().cpu(),
            target_phys=targets_phys[idx].detach().cpu(),
            pred_phys=preds_phys[idx].detach().cpu(),
            target_norm=targets_norm[idx].detach().cpu(),
            pred_norm=preds_norm[idx].detach().cpu(),
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

    def _init_accumulator(self) -> dict[str, torch.Tensor]:
        return {
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

    def _update_batch_metrics(
        self,
        accum: dict[str, torch.Tensor],
        sq_error: Tensor,
        abs_error: Tensor,
        preds: Tensor,
        targets: Tensor,
    ) -> None:
        accum["per_var_sse"] += sq_error.sum(dim=(0, 2, 3, 4)).to(dtype=torch.float64)
        accum["per_var_l1"] += abs_error.sum(dim=(0, 2, 3, 4)).to(dtype=torch.float64)
        accum["overall_sse"] += sq_error.sum(dtype=torch.float64)
        accum["overall_l1"] += abs_error.sum(dtype=torch.float64)

        point_pred = preds[:, :, -1].mean(dim=(-1, -2))
        point_target = targets[:, :, -1].mean(dim=(-1, -2))
        point_diff = point_pred - point_target
        accum["per_var_point_sse"] += point_diff.pow(2).sum(dim=0).to(dtype=torch.float64)
        accum["per_var_point_l1"] += point_diff.abs().sum(dim=0).to(dtype=torch.float64)

    def _update_lead_metrics(
        self,
        accum: dict[str, torch.Tensor],
        preds: Tensor,
        targets: Tensor,
        horizon: int,
        batch_size: int,
        var_cnt: int,
        height: int,
        width: int,
    ) -> None:
        lead_diff = preds - targets
        lead_sq_error = lead_diff.pow(2)
        accum["per_lead_sse"][:horizon] += lead_sq_error.sum(dim=(0, 1, 3, 4)).to(dtype=torch.float64)
        accum["per_lead_counter"][:horizon] += batch_size * var_cnt * height * width

        var_lead_pred = preds[:, self.var, :, :, :].mean(dim=(-1, -2))
        var_lead_target = targets[:, self.var, :, :, :].mean(dim=(-1, -2))
        var_lead_diff = var_lead_pred - var_lead_target
        accum["per_lead_point_sse"][:horizon] += var_lead_diff.pow(2).sum(dim=0).to(dtype=torch.float64)
        accum["per_lead_point_l1"][:horizon] += var_lead_diff.abs().sum(dim=0).to(dtype=torch.float64)
        accum["per_lead_point_counter"][:horizon] += batch_size

    def _build_metrics(
        self,
        accum_phys: dict[str, torch.Tensor],
        accum_norm: dict[str, torch.Tensor],
        totals: dict[str, int],
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {"totals": totals.copy()}
        phys_block = self._compute_metric_block(accum_phys, totals)
        norm_block = self._compute_metric_block(accum_norm, totals)
        if phys_block:
            metrics["physical"] = phys_block
        if norm_block:
            metrics["normalized"] = norm_block
        return metrics

    def _compute_metric_block(self, accum: dict[str, torch.Tensor], totals: dict[str, int]) -> dict[str, Any]:
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
        per_lead_point_mae = torch.full_like(lead_point_counts, float("nan"))

        valid_lead = lead_counts > 0
        valid_point = lead_point_counts > 0

        if valid_lead.any():
            per_lead_mse[valid_lead] = (
                accum["per_lead_sse"].detach().cpu()[valid_lead] / lead_counts[valid_lead]
            )
        if valid_point.any():
            per_lead_point_mse[valid_point] = (
                accum["per_lead_point_sse"].detach().cpu()[valid_point] / lead_point_counts[valid_point]
            )
            per_lead_point_mae[valid_point] = (
                accum["per_lead_point_l1"].detach().cpu()[valid_point] / lead_point_counts[valid_point]
            )

        per_lead_rmse = torch.sqrt(per_lead_mse)
        per_lead_point_rmse = torch.sqrt(per_lead_point_mse)

        overall_mse = float((accum["overall_sse"] / numel_count).item())
        overall_rmse = overall_mse ** 0.5
        overall_mae = float((accum["overall_l1"] / numel_count).item())

        return {
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
        }

    def _log_metrics(self, metrics: dict[str, Any]) -> None:
        if not metrics:
            self.logger.warning("No metrics were produced during evaluation.")
            return

        for label, key in (("Std-scaled", "physical"), ("Normalized", "normalized")):
            block = metrics.get(key)
            if not block:
                continue

            prefix = f"[{label}] "
            per_var = block.get("per_variable", {})
            rmse = per_var.get("rmse")
            mse = per_var.get("mse")
            mae = per_var.get("mae")
            point_rmse = per_var.get("point_rmse")

            if rmse and mse and mae and point_rmse:
                labels = self._var_labels(len(rmse))
                for name, mse_val, rmse_val, mae_val, prmse in zip(labels, mse, rmse, mae, point_rmse):
                    self.logger.info(
                        "%sVar %-4s -> pixel MSE: %.6f | RMSE: %.6f | MAE: %.6f | point RMSE: %.6f",
                        prefix,
                        name,
                        mse_val,
                        rmse_val,
                        mae_val,
                        prmse,
                    )

            overall = block.get("overall")
            if overall:
                self.logger.info(
                    "%sOverall -> pixel MSE: %.6f | RMSE: %.6f | MAE: %.6f",
                    prefix,
                    overall.get("mse", 0.0),
                    overall.get("rmse", 0.0),
                    overall.get("mae", 0.0),
                )

            lead_metrics = block.get("per_lead", {})
            lead_rmse = lead_metrics.get("rmse")
            if isinstance(lead_rmse, list) and lead_rmse:
                summary = self._format_lead_summary(lead_rmse)
                self.logger.info("%sPer-lead RMSE -> %s", prefix, summary)
            lead_point_rmse = lead_metrics.get("point_rmse")
            if isinstance(lead_point_rmse, list) and lead_point_rmse:
                summary = self._format_lead_summary(lead_point_rmse)
                self.logger.info("%sPer-lead point RMSE (%s) -> %s", prefix, self.var_label, summary)

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

    def _load_channel_stats(self, stats_path: Path) -> dict[str, Any]:
        try:
            with stats_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except FileNotFoundError:
            self.logger.warning("Stats file '%s' not found; evaluation will stay normalized.", stats_path)
            return {}
        except json.JSONDecodeError as exc:
            self.logger.warning("Failed to parse stats file '%s': %s", stats_path, exc)
            return {}
        if not isinstance(data, dict):
            self.logger.warning("Stats file '%s' does not contain a dictionary.", stats_path)
            return {}
        return data

    def _build_var_std_tensor(self, stats: dict[str, Any], labels: list[str]) -> torch.Tensor:
        stds: list[float] = []
        for name in labels:
            entry = stats.get(name) or stats.get(name.lower()) or stats.get(name.upper())
            value = 1.0
            if isinstance(entry, dict):
                raw_std = entry.get("std")
                if isinstance(raw_std, (float, int)) and math.isfinite(raw_std) and raw_std != 0:
                    value = float(raw_std)
            stds.append(value)
        if not stds:
            stds = [1.0]
        return torch.tensor(stds, dtype=torch.float32)

    def _scale_by_std(self, tensor: Tensor) -> Tensor:
        if tensor is None or self.var_std_tensor is None:
            return tensor
        dims = tensor.dim()
        std = self.var_std_tensor.to(tensor.device, tensor.dtype)
        if dims == 5:
            scale = std.view(1, -1, 1, 1, 1)
        elif dims == 4:
            scale = std.view(-1, 1, 1, 1)
        elif dims == 3 and tensor.size(0) == std.numel():
            scale = std.view(-1, 1, 1)
        else:
            return tensor
        return tensor * scale

    def _var_std_scalar(self, idx: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        std = self.var_std_tensor[idx]
        return std.to(device=device, dtype=dtype)

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

    def _per_var_order(
        self, phys_metrics: dict[str, Any] | None, norm_metrics: dict[str, Any] | None
    ) -> np.ndarray | None:
        for block in (phys_metrics, norm_metrics):
            if not block:
                continue
            per_var = block.get("per_variable", {})
            point_rmse = per_var.get("point_rmse")
            if point_rmse:
                arr = np.asarray(point_rmse, dtype=float)
                if arr.size:
                    return np.argsort(arr)[::-1]
        return None

    def _draw_per_var_panel(
        self, ax: Axes, metrics: dict[str, Any], title: str, order: np.ndarray | None
    ) -> None:
        per_var = metrics.get("per_variable", {})
        rmse = per_var.get("rmse")
        point_rmse = per_var.get("point_rmse")
        if not rmse or not point_rmse:
            ax.axis("off")
            return

        rmse_arr = np.asarray(rmse, dtype=float)
        point_rmse_arr = np.asarray(point_rmse, dtype=float)
        labels = np.asarray(self._var_labels(len(rmse_arr)))

        if order is not None:
            valid_order = order[order < len(labels)]
            if valid_order.size > 0:
                rmse_arr = rmse_arr[valid_order]
                point_rmse_arr = point_rmse_arr[valid_order]
                labels = labels[valid_order]

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, rmse_arr, width, label="Pixel RMSE")
        ax.bar(x + width / 2, point_rmse_arr, width, label="Point RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Variable")
        ax.set_title(f"{title}: per-variable RMSE")
        ax.grid(True, linestyle=":", linewidth=0.4)
        ax.legend(frameon=False, fontsize=8)

        for idx in range(min(3, len(x))):
            ax.annotate(
                f"{point_rmse_arr[idx]:.2f}",
                xy=(x[idx] + width / 2, point_rmse_arr[idx]),
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=7,
                ha="center",
            )

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    def _var_labels(self, count: int) -> list[str]:
        if count <= len(VAR_NAMES):
            return VAR_NAMES[:count]

        labels = VAR_NAMES[:]
        labels.extend(f"var_{idx}" for idx in range(len(labels), count))
        return labels[:count]

    def _to_numpy_img(self, x: Tensor, var: int) -> np.ndarray:
        """
        Convert a prediction/target tensor into a 2D array for plotting.
        Expects tensors in [C, T, H, W] (standard for this project) but
        gracefully handles squeezed dimensions as well.
        """
        arr = x.detach().cpu()

        while arr.ndim > 4:
            arr = arr[0]

        if arr.ndim == 4:
            var_idx = var % arr.shape[0]
            time_idx = arr.shape[1] - 1 if arr.shape[1] > 1 else 0
            arr = arr[var_idx, time_idx]
        elif arr.ndim == 3:
            first_dim = arr.shape[0]
            if first_dim >= self.var_cnt:
                var_idx = var % first_dim
                arr = arr[var_idx]
                if arr.ndim == 3:  # Remaining dims might still include time
                    arr = arr[-1]
            else:
                time_idx = first_dim - 1 if first_dim > 1 else 0
                arr = arr[time_idx]
        elif arr.ndim != 2:
            raise ValueError(f"Cannot convert tensor with shape {tuple(x.shape)} to 2D image.")

        if arr.ndim != 2:
            raise ValueError(f"Unexpected tensor shape after slicing: {tuple(arr.shape)}")

        return arr.numpy()

    def _plot_sample(self, sample: EvalSample, tag: str) -> None:
        """
        Plot target vs prediction for both physical and normalized variants.
        """
        save_dir = Path(self.path) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

        variants = [
            ("physical", sample.target_phys, sample.pred_phys),
            ("normalized", sample.target_norm, sample.pred_norm),
        ]

        for label, target_tensor, pred_tensor in variants:
            target_img = self._to_numpy_img(target_tensor, self.var)
            pred_img = self._to_numpy_img(pred_tensor, self.var)
            error_img = np.abs(target_img - pred_img)

            target_img = np.asarray(target_img)
            pred_img = np.asarray(pred_img)
            error_img = np.asarray(error_img)

            vmin = min(target_img.min(), pred_img.min())
            vmax = max(target_img.max(), pred_img.max())

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(9.0, 3.0),
                dpi=300,
                constrained_layout=True,
            )

            axes[0].imshow(target_img, origin="lower", vmin=vmin, vmax=vmax)
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

            fig.suptitle(f"{self.name} - ({label}) sample")

            safe_label = label.replace("/", "-")
            out_path = save_dir / f"{self.name}_{tag}_{safe_label}_sample.png"
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

        phys_metrics = metrics.get("physical")
        norm_metrics = metrics.get("normalized")

        self._plot_loss_breakdown(phys_metrics, norm_metrics, save_dir)
        self._plot_lead_breakdown(phys_metrics, norm_metrics, save_dir)
        self._plot_lead_point_breakdown(phys_metrics, norm_metrics, save_dir)

        self._plot_sample(data.best_sample, "best")
        self._plot_sample(data.worst_sample, "worst")
        self._plot_sample(data.random_sample, "random")

    def _plot_loss_breakdown(
        self, phys_metrics: dict[str, Any] | None, norm_metrics: dict[str, Any] | None, save_dir: Path
    ) -> None:
        order = self._per_var_order(phys_metrics, norm_metrics)
        for title, block, suffix in (
            ("Std-scaled", phys_metrics, "physical"),
            ("Normalized", norm_metrics, "normalized"),
        ):
            if not block:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.2), dpi=300)
            self._draw_per_var_panel(ax, block, title, order)
            overall = block.get("overall", {})
            fig.text(
                0.5,
                0.02,
                f"RMSE: {overall.get('rmse', float('nan')):.3f} | MAE: {overall.get('mae', float('nan')):.3f}",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.05, 1, 1))
            out_base = save_dir / f"{self.name}_per_var_errors_{suffix}"
            fig.savefig(str(out_base.with_suffix(".png")), bbox_inches="tight")
            fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
            plt.close(fig)

    def _plot_lead_point_breakdown(
        self, phys_metrics: dict[str, Any] | None, norm_metrics: dict[str, Any] | None, save_dir: Path
    ) -> None:
        variants = [
            ("Std-scaled", phys_metrics, "physical", "#8e44ad"),
            ("Normalized", norm_metrics, "normalized", "#1f77b4"),
        ]
        for label, block, suffix, color in variants:
            if not block:
                continue
            per_lead = block.get("per_lead", {})
            point_rmse_vals = per_lead.get("point_rmse")
            if not point_rmse_vals:
                continue
            point_rmse_arr = self._lead_array(point_rmse_vals)
            if not np.isfinite(point_rmse_arr).any():
                continue

            x = np.arange(len(point_rmse_arr))
            fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=300)
            ax.plot(x, point_rmse_arr, marker="o", linewidth=1.2, color=color, label=f"{label} point RMSE")

            best_idx = int(np.nanargmin(point_rmse_arr))
            worst_idx = int(np.nanargmax(point_rmse_arr))
            ax.scatter(best_idx, point_rmse_arr[best_idx], color="#2ecc71", s=32, zorder=5)
            ax.scatter(worst_idx, point_rmse_arr[worst_idx], color="#e74c3c", s=32, zorder=5)

            ax.set_xticks(x)
            ax.set_xlabel("Lead index")
            ax.set_ylabel("Error")
            ax.set_title(f"{self.var_label} {label.lower()} point error by lead")
            ax.grid(True, linestyle=":", linewidth=0.4)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.legend(frameon=False, fontsize=8, loc="upper left")

            summary_text = (
                f"min t{best_idx}: {point_rmse_arr[best_idx]:.3f}\n"
                f"max t{worst_idx}: {point_rmse_arr[worst_idx]:.3f}"
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
            out_base = save_dir / f"{self.name}_{safe_label}_lead_point_error_{suffix}"
            fig.savefig(str(out_base.with_suffix(".png")), bbox_inches="tight")
            fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
            plt.close(fig)

    def _plot_lead_breakdown(
        self, phys_metrics: dict[str, Any] | None, norm_metrics: dict[str, Any] | None, save_dir: Path
    ) -> None:
        variants = [
            ("Std-scaled", phys_metrics, "physical", "#2a4d69"),
            ("Normalized", norm_metrics, "normalized", "#ff7f0e"),
        ]
        for label, block, suffix, color in variants:
            if not block:
                continue
            per_lead = block.get("per_lead", {})
            rmse_vals = per_lead.get("rmse")
            if not rmse_vals:
                continue
            rmse_arr = self._lead_array(rmse_vals)
            if not np.isfinite(rmse_arr).any():
                continue

            x = np.arange(len(rmse_arr))
            fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=300)
            ax.plot(
                x,
                rmse_arr,
                marker="o",
                linestyle="-",
                linewidth=1.2,
                color=color,
                label=f"{label} RMSE",
            )

            best_idx = int(np.nanargmin(rmse_arr))
            worst_idx = int(np.nanargmax(rmse_arr))
            ax.scatter(best_idx, rmse_arr[best_idx], color="#2ecc71", s=32, zorder=5)
            ax.scatter(worst_idx, rmse_arr[worst_idx], color="#e74c3c", s=32, zorder=5)

            mean_rmse = float(np.nanmean(rmse_arr))
            ax.axhline(mean_rmse, color="0.5", linestyle="--", linewidth=0.8, label=f"Mean {mean_rmse:.3f}")

            ax.set_xticks(x)
            ax.set_xlabel("Lead index")
            ax.set_ylabel("RMSE")
            ax.set_title(f"{label} lead-wise pixel RMSE")
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
            out_base = save_dir / f"{self.name}_lead_rmse_{suffix}"
            fig.savefig(str(out_base.with_suffix(".png")), bbox_inches="tight")
            fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
            plt.close(fig)
