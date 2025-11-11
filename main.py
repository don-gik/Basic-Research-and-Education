# Must be coded in future for easily testing and training.
import argparse
import json

# import subprocess
import logging
import os

# import sys
from pathlib import Path

import hydra
import torch
import pandas as pd

from src.train import Trainer
from src.dataset import SequentialDataset
from src.preprocess import discover_grib_variables, save_normalized_tensor


config_path = ""
config_name = ""

LOGGING_CONFIG = "logging.conf"


def run():
    logging.config.fileConfig("logging.conf")  # type: ignore

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info("-------- Starting the program --------")
    logger.info(f"Using config path : {config_path} / {config_name}")

    # Load config file
    hydra.initialize(config_path=config_path, version_base=None)
    config = hydra.compose(config_name=config_name)

    if config.args.preprocess:
        logger.info("-------- Starting preprocess --------")

        GRIB_PATH = Path("./data/raw/monthly.grib")
        TENSOR_PATH = Path("./data/processed/monthly_tensor.pt")
        STATS_PATH = Path("./data/processed/monthly_tensor_stats.json")

        if(os.path.exists(TENSOR_PATH)):
            logger.warning("Tensor path exists. Automatically skipping preprocess...")

        else:
            CHANNEL_SHORT_NAMES = {
                "u100" : "100u",
                "v100" : "100v",
                "u10"  : "10u",
                "v10"  : "10v",
                "d2m"  : "2d",
                "t2m"  : "2t",
                "msl"  : "msl",
                "sp"   : "sp",
                "ssrc" : "ssrc",
                "sst"  : "sst"
            }
            
            metadata = discover_grib_variables(GRIB_PATH)
            logger.info(pd.DataFrame(metadata).drop_duplicates(subset=["short_name"]).sort_values("short_name").reset_index(drop=True))

            tensor = save_normalized_tensor(
                GRIB_PATH,
                CHANNEL_SHORT_NAMES,
                TENSOR_PATH,
                STATS_PATH,
            )
            logger.info(f'Tensor shape : {tensor.shape}')
    
    data = torch.load(config.data.path)
    logger.info(f'Data shape : {data.shape}')
    dataset = SequentialDataset(config.model.time, data, config.data.H, config.data.W)
    trainer = Trainer(config=config, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), dataset=dataset)

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", dest="config", action="store")
    parser.add_argument("--delete-logs", "-dl", dest="delete_logs", action="store_true")

    args = parser.parse_args()

    if args.config:
        config_path, config_name = os.path.split(args.config)
    if args.delete_logs:
        if os.path.exists("debug.log"):
            os.remove("debug.log")
        if os.path.exists("error.log"):
            os.remove("error.log")

    run()
