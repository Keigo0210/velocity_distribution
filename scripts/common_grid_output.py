"""Common valid-area metrics and configurable common-grid serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VALID_FORMATS = {"none", "csv", "csv.gz", "npz"}


def resolve_grid_output_format(output: dict[str, Any]) -> str:
    if "grid_output_format" in output:
        value = str(output["grid_output_format"]).lower()
        if "save_grid_csv" in output:
            print(
                "Warning: both output.grid_output_format and output.save_grid_csv are set; "
                "grid_output_format takes priority."
            )
    elif "save_grid_csv" in output:
        value = "csv" if bool(output["save_grid_csv"]) else "none"
    else:
        value = "none"
    if value not in VALID_FORMATS:
        raise ValueError(
            f"grid_output_format must be one of {sorted(VALID_FORMATS)}, got {value!r}"
        )
    return value


def valid_grid_metrics(
    valid_mask: np.ndarray, width: float, height: float, grid_resolution: tuple[int, int]
) -> dict[str, float | int]:
    nx, ny = (int(grid_resolution[0]), int(grid_resolution[1]))
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.size != nx * ny:
        raise ValueError(
            f"valid mask size {valid.size} does not match grid resolution {nx} x {ny}"
        )
    grid = valid.reshape((ny, nx))
    valid_cells = grid[:-1, :-1] & grid[1:, :-1] & grid[:-1, 1:] & grid[1:, 1:]
    valid_point_count = int(np.sum(valid))
    valid_cell_count = int(np.sum(valid_cells))
    total_cells = (nx - 1) * (ny - 1)
    point_fraction = valid_point_count / valid.size
    cell_fraction = valid_cell_count / total_cells
    ds = float(width) / (nx - 1)
    dt = float(height) / (ny - 1)
    return {
        "valid_point_count": valid_point_count,
        "valid_point_fraction": point_fraction,
        "valid_cell_count": valid_cell_count,
        "valid_cell_fraction": cell_fraction,
        "common_valid_area": valid_cell_count * ds * dt,
        "legacy_valid_area_estimate": point_fraction * float(width) * float(height),
    }


def save_grid_output(
    base_path: Path,
    output_format: str,
    dataframe: pd.DataFrame | None,
    arrays: dict[str, Any],
    metadata: dict[str, Any],
) -> Path | None:
    if output_format == "none":
        return None
    base_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        if dataframe is None:
            raise ValueError("CSV output requires a dataframe")
        path = base_path.with_suffix(".csv")
        dataframe.to_csv(path, index=False)
        return path
    if output_format == "csv.gz":
        if dataframe is None:
            raise ValueError("CSV.GZ output requires a dataframe")
        path = base_path.with_suffix(".csv.gz")
        dataframe.to_csv(path, index=False, compression="gzip")
        return path
    if output_format == "npz":
        path = base_path.with_suffix(".npz")
        payload = {key: np.asarray(value) for key, value in arrays.items()}
        payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez_compressed(path, **payload)
        return path
    raise ValueError(output_format)
