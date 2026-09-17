"""CLI and helpers for generating VAR datasets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

from .autoreg import VectAutoReg
from .config import Config


def _generate(
    autoreg: VectAutoReg,
    trajectory_length: int,
    sample_count: int,
    idx_len: int,
    dest_folder: str,
    *,
    stationary_init: bool,
) -> None:
    for idx in range(sample_count):
        series = autoreg(trajectory_length, stationary_init=stationary_init)
        data_columns = [f"x_{i}" for i in range(1, series.shape[0] + 1)]
        df = pd.DataFrame(series.T, columns=data_columns)
        df.insert(0, "time", np.arange(len(df)))
        seq_id = str(idx).zfill(idx_len)
        file_path = os.path.join(dest_folder, f"seq_{seq_id}.parquet")
        try:
            df.to_parquet(file_path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Writing Parquet requires pyarrow. Install the project with its "
                "declared dependencies (for example: pip install -e .)."
            ) from exc


def generate(config: Config) -> Tuple[str, Config, bool]:
    coeffs = config.ordered_coeffs()
    init_values = np.asarray(config.INIT, dtype=float) if config.INIT is not None else None
    bias = np.asarray(config.BIAS, dtype=float) if config.BIAS is not None else None
    noise_cov = (
        np.asarray(config.NOISE_COV, dtype=float) if config.NOISE_COV is not None else None
    )

    if config.stationary_init and init_values is not None:
        raise ValueError("stationary_init=true cannot be combined with INIT")

    autoreg = VectAutoReg(
        coeffs,
        init=init_values,
        bias=bias,
        noise_cov=noise_cov,
        rng=config.seed,
    )

    if config.trajectory_length < autoreg.order:
        raise ValueError(
            "trajectory_length is the total number of returned time points and "
            f"must be at least the VAR order ({autoreg.order})"
        )

    if config.stationary_init and not autoreg.stable:
        raise ValueError("stationary_init=true requires a stable VAR")

    sample_count = config.sample_count
    idx_len = max(1, len(str(sample_count - 1)))
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    dest_folder = os.path.join(config.dest_folder, "VAR_" + timestamp)
    os.makedirs(dest_folder, exist_ok=True)

    _generate(
        autoreg,
        config.trajectory_length,
        sample_count,
        idx_len,
        dest_folder,
        stationary_init=config.stationary_init,
    )

    return dest_folder, config, autoreg.stable


def _model_dump(config: Config) -> dict:
    # Pydantic v2 / v1 compatibility.
    if hasattr(config, "model_dump"):
        return config.model_dump()
    return config.dict()


def _log(dest_folder: str, config: Config, is_stable: bool) -> None:
    log_folder = os.path.join(dest_folder, "log")
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, "log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"is_stable: {is_stable}\n")

    config_file_path = os.path.join(log_folder, "config.yaml")
    with open(config_file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_model_dump(config), f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gaussian VAR trajectories")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    config = Config(**raw_config)
    dest_folder, config, is_stable = generate(config)
    _log(dest_folder, config, is_stable)
    print(dest_folder)


if __name__ == "__main__":
    main()
