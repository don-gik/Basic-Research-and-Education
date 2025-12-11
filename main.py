# Must be coded in future for easily testing and training.

# import subprocess
import logging
import os
import sys

# import sys
from pathlib import Path

import hydra
import pandas as pd
import torch
from torch.utils.data import Subset

from src.dataset import AugmentedDataset, MultiStepSequentialDataset
from src.evaluate import Evaluator
from src.preprocess import discover_grib_variables, save_normalized_tensor
from src.train import Trainer

config_path = ""
config_name = ""

LOGGING_CONFIG = "logging.conf"


@hydra.main(config_path="configs", version_base=None)
def run(config):
    if config.args.delete_logs:
        if os.path.exists("debug.log"):
            os.remove("debug.log")
        if os.path.exists("error.log"):
            os.remove("error.log")

    if config.args.evaluate:
        eval_run(config)
        sys.exit(0)

    logging.config.fileConfig("logging.conf")  # type: ignore

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info("-------- Starting the program --------")
    logger.info(f"Using config path : {config_path} / {config_name}")

    if config.args.preprocess:
        logger.info("-------- Starting preprocess --------")

        GRIB_PATH = Path("./data/raw/monthly.grib")
        TENSOR_PATH = Path("./data/processed/monthly_tensor.pt")
        STATS_PATH = Path("./data/processed/monthly_tensor_stats.json")

        if os.path.exists(TENSOR_PATH):
            logger.warning("Tensor path exists. Automatically skipping preprocess...")

        else:
            CHANNEL_SHORT_NAMES = {
                "u100": "100u",
                "v100": "100v",
                "u10": "10u",
                "v10": "10v",
                "d2m": "2d",
                "t2m": "2t",
                "msl": "msl",
                "sp": "sp",
                "ssrc": "ssrc",
                "sst": "sst",
            }

            metadata = discover_grib_variables(GRIB_PATH)
            logger.info(
                pd.DataFrame(metadata)
                .drop_duplicates(subset=["short_name"])
                .sort_values("short_name")
                .reset_index(drop=True)
            )

            tensor = save_normalized_tensor(
                GRIB_PATH,
                CHANNEL_SHORT_NAMES,
                TENSOR_PATH,
                STATS_PATH,
            )
            logger.info(f"Tensor shape : {tensor.shape}")

    data = torch.load(config.data.path)
    logger.info(f"Data shape : {data.shape}")

    horizon = getattr(config.data, "horizon", config.model.time)
    dataset = MultiStepSequentialDataset(
        x=data,
        in_len=config.model.time,
        out_len=int(horizon),
        H=config.data.H,
        W=config.data.W,
    )
    train_idx = list(range(0, int(len(dataset) * config.train.train_size)))
    val_idx = list(range(int(len(dataset) * config.train.train_size), len(dataset)))

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_augmented_dataset = AugmentedDataset(
        base_dataset=train_set, n_augment=config.data.augment, noise_std=config.data.std, noise_on_inputs=True
    )
    trainer = Trainer(
        config=config,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        train_dataset=train_augmented_dataset,
        val_dataset=val_set,
    )

    trainer.run()


def eval_run(config):
    logging.config.fileConfig("logging.conf")  # type: ignore

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info("-------- Starting the program --------")
    logger.info(f"Using config path : {config_path} / {config_name}")

    data = torch.load(config.data.path)
    logger.info(f"Data shape : {data.shape}")
    horizon = getattr(config.data, "horizon", config.model.time)
    dataset = MultiStepSequentialDataset(
        x=data,
        in_len=config.model.time,
        out_len=int(horizon),
        H=config.data.H,
        W=config.data.W,
    )
    train_idx = list(range(0, int(len(dataset) * config.train.train_size)))
    val_idx = list(range(int(len(dataset) * config.train.train_size), len(dataset)))

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    evaluator = Evaluator(
        config,
        train_set,
        val_set,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )
    evaluator.run()

    evaluator._log(data)


if __name__ == "__main__":
    run()
