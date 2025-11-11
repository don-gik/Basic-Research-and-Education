from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
import xarray as xr


def discover_grib_variables(grib_path: Path) -> list[dict[str, str]]:
    """Return metadata for every variable present inside the GRIB file."""
    import cfgrib  # Imported lazily to avoid hard dependency unless used.

    datasets = cfgrib.open_datasets(
        str(grib_path),
        backend_kwargs={"indexpath": ""},
    )
    records: list[dict[str, str]] = []
    for ds in datasets:
        try:
            for name, data in ds.data_vars.items():
                attrs = data.attrs
                records.append(
                    {
                        "dataset_var": name,
                        "short_name": attrs.get("GRIB_shortName", ""),
                        "long_name": attrs.get("GRIB_name", ""),
                        "units": attrs.get("GRIB_units", ""),
                        "level_type": attrs.get("GRIB_typeOfLevel", ""),
                    }
                )
        finally:
            ds.close()
    return records


def _load_short_name_array(grib_path: Path, short_name: str) -> np.ndarray:
    """Load a single GRIB short name into a numpy array."""
    backend_kwargs = {
        "filter_by_keys": {"shortName": short_name},
        "indexpath": "",
    }
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs=backend_kwargs,
    )
    try:
        data_var = next(iter(ds.data_vars))
        data = ds[data_var].astype(np.float32).to_numpy()
    finally:
        ds.close()
    return data


def normalize_arrays_to_tensor(
    channel_arrays: Mapping[str, np.ndarray],
    channel_short_names: Mapping[str, str],
    tensor_path: Path,
    stats_path: Path,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize provided arrays, persist them as a tensor, and store stats."""
    if not channel_arrays:
        raise ValueError("No channel arrays were provided.")

    normalized_channels: list[np.ndarray] = []
    stats: dict[str, dict[str, float | str]] = {}
    reference_shape = None

    for alias, array in channel_arrays.items():
        arr = np.asarray(array, dtype=np.float32)
        if reference_shape is None:
            reference_shape = arr.shape
        elif arr.shape != reference_shape:
            raise ValueError(f"Channel '{alias}' shape {arr.shape} does not match {reference_shape}")

        mean = float(arr.mean())
        std = float(arr.std())
        scale = std if std > eps else eps
        normalized_channels.append((arr - mean) / scale)
        stats[alias] = {
            "short_name": channel_short_names.get(alias, alias),
            "mean": mean,
            "std": scale,
        }

    stacked = np.stack(normalized_channels, axis=0)
    tensor = torch.from_numpy(stacked)

    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, tensor_path)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return tensor


def save_normalized_tensor(
    grib_path: Path,
    channel_short_names: Mapping[str, str],
    tensor_path: Path,
    stats_path: Path,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Load GRIB variables, normalize, and persist as tensor + stats JSON."""
    arrays = {alias: _load_short_name_array(grib_path, short_name) for alias, short_name in channel_short_names.items()}
    return normalize_arrays_to_tensor(arrays, channel_short_names, tensor_path, stats_path, eps=eps)


if __name__ == "__main__":
    GRIB_PATH = Path("../data/raw/monthly.grib")
    TENSOR_PATH = Path("../data/processed/monthly_tensor.pt")
    STATS_PATH = Path("../data/processed/monthly_tensor_stats.json")
    CHANNEL_SHORT_NAMES = {
        "t2m": "2t",
        "u10": "10u",
        "v10": "10v",
        "u100": "100u",
        "v100": "100v",
        "msl": "msl",
        "sst": "sst",
    }

    metadata = discover_grib_variables(GRIB_PATH)
    print(
        pd.DataFrame(metadata).drop_duplicates(subset=["short_name"]).sort_values("short_name").reset_index(drop=True)
    )

    tensor = save_normalized_tensor(
        GRIB_PATH,
        CHANNEL_SHORT_NAMES,
        TENSOR_PATH,
        STATS_PATH,
    )
    print(tensor.shape)
