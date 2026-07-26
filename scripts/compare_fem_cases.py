#!/usr/bin/env python3
"""Compare velocity fields from configurable data sources on common sections."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import pyvista as pv

from section_config import SectionConfigError, resolve_sections_from_config
from common_grid_output import resolve_grid_output_format, save_grid_output, valid_grid_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "compare_fem_cases.json"
SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class ConfigError(ValueError):
    """Raised when the comparison configuration is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare configured velocity data sources on common section grids."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON configuration file")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def require_mapping(parent: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    if key not in parent:
        raise ConfigError(f"{context}: required key {key!r} is missing")
    value = parent[key]
    if not isinstance(value, dict):
        raise ConfigError(f"{context}.{key} must be an object")
    return value


def require_list(parent: dict[str, Any], key: str, context: str) -> list[Any]:
    if key not in parent:
        raise ConfigError(f"{context}: required key {key!r} is missing")
    value = parent[key]
    if not isinstance(value, list):
        raise ConfigError(f"{context}.{key} must be an array")
    return value


def require_keys(mapping: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"{context}: missing required keys: {', '.join(missing)}")


def finite_float(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context} must be a finite number") from exc
    if not math.isfinite(result):
        raise ConfigError(f"{context} must be a finite number")
    return result


def positive_float(value: Any, context: str) -> float:
    result = finite_float(value, context)
    if result <= 0.0:
        raise ConfigError(f"{context} must be greater than zero")
    return result


def nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a non-empty string")
    return value


def boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be true or false")
    return value


def validate_data_source(name: str, source: dict[str, Any]) -> dict[str, Any]:
    context = f"data_sources.{name}"
    require_keys(
        source,
        (
            "label",
            "path",
            "dt",
            "time",
            "velocity_array",
            "data_association",
            "length_unit",
            "velocity_unit",
            "length_scale_to_mm",
            "velocity_scale_to_mm_s",
        ),
        context,
    )
    path_text = nonempty_string(source["path"], f"{context}.path")
    path = resolve_path(path_text)
    if not path.is_file():
        raise ConfigError(f"{context}.path does not exist or is not a file: {path}")
    association = nonempty_string(
        source["data_association"], f"{context}.data_association"
    ).lower()
    if association not in {"point", "cell"}:
        raise ConfigError(f"{context}.data_association must be 'point' or 'cell'")
    time = finite_float(source["time"], f"{context}.time")
    if time < 0.0:
        raise ConfigError(f"{context}.time must be non-negative")
    return {
        **source,
        "label": nonempty_string(source["label"], f"{context}.label"),
        "path": path,
        "dt": positive_float(source["dt"], f"{context}.dt"),
        "time": time,
        "velocity_array": nonempty_string(
            source["velocity_array"], f"{context}.velocity_array"
        ),
        "data_association": association,
        "length_unit": nonempty_string(source["length_unit"], f"{context}.length_unit"),
        "velocity_unit": nonempty_string(
            source["velocity_unit"], f"{context}.velocity_unit"
        ),
        "length_scale_to_mm": positive_float(
            source["length_scale_to_mm"], f"{context}.length_scale_to_mm"
        ),
        "velocity_scale_to_mm_s": positive_float(
            source["velocity_scale_to_mm_s"], f"{context}.velocity_scale_to_mm_s"
        ),
    }


def _selector_warning(context: str, keys: list[str], chosen: str) -> None:
    if len(keys) > 1:
        print(
            f"Warning: {context} contains multiple section selectors {keys}; "
            f"using {chosen!r} according to the documented priority."
        )


def _resolve_top_sections(
    config: dict[str, Any], config_path: Path
) -> list[dict[str, Any]]:
    priority = ["section_set", "section_names", "sections", "section"]
    present = [key for key in priority if key in config]
    if not present:
        if "section_library" in config:
            return []
        raise ConfigError("configuration does not define any sections")
    _selector_warning("configuration", present, present[0])
    try:
        return resolve_sections_from_config(config, config_path=config_path)
    except SectionConfigError as exc:
        raise ConfigError(str(exc)) from exc


def _resolve_comparison_sections(
    comparison: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    top_sections: dict[str, dict[str, Any]],
    context: str,
) -> list[dict[str, Any]]:
    priority = ["section_set", "sections", "section"]
    present = [key for key in priority if key in comparison]
    if present:
        _selector_warning(context, present, present[0])
        chosen = present[0]
        if chosen == "section_set":
            selector = {
                "section_library": config.get("section_library"),
                "section_set": comparison[chosen],
            }
        elif chosen == "sections":
            raw_sections = comparison[chosen]
            if not isinstance(raw_sections, list) or not raw_sections:
                raise ConfigError(f"{context}.sections must be a non-empty array")
            if all(isinstance(value, str) for value in raw_sections):
                if "section_library" in config:
                    selector = {
                        "section_library": config["section_library"],
                        "section_names": raw_sections,
                    }
                else:
                    unknown = [name for name in raw_sections if name not in top_sections]
                    if unknown:
                        raise ConfigError(
                            f"{context}.sections references unknown sections: "
                            + ", ".join(unknown)
                        )
                    return [top_sections[name] for name in raw_sections]
            else:
                selector = {"sections": raw_sections}
        else:
            section_reference = comparison[chosen]
            if isinstance(section_reference, str) and section_reference in top_sections:
                return [top_sections[section_reference]]
            if isinstance(section_reference, str) and "section_library" in config:
                selector = {
                    "section_library": config["section_library"],
                    "section_names": [section_reference],
                }
            else:
                selector = {"section": section_reference}
        try:
            return resolve_sections_from_config(selector, config_path=config_path)
        except SectionConfigError as exc:
            raise ConfigError(f"{context}: {exc}") from exc

    if not top_sections:
        raise ConfigError(
            f"{context} has no section selector and no top-level section selection"
        )
    return list(top_sections.values())


def validate_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ConfigError("configuration root must be an object")
    data_source_config = require_mapping(config, "data_sources", "configuration")
    comparisons = require_list(config, "comparisons", "configuration")
    interpolation = require_mapping(config, "interpolation", "configuration")
    metrics = require_mapping(config, "metrics", "configuration")
    output = require_mapping(config, "output", "configuration")
    if not data_source_config:
        raise ConfigError("data_sources must define at least one source")
    if not comparisons:
        raise ConfigError("comparisons must define at least one comparison")

    sources = {
        name: validate_data_source(name, source)
        for name, source in data_source_config.items()
        if isinstance(name, str) and isinstance(source, dict)
    }
    if len(sources) != len(data_source_config):
        raise ConfigError("every data_sources entry must have a string name and object value")

    top_section_list = _resolve_top_sections(config, config_path)
    sections: dict[str, dict[str, Any]] = {
        section["name"]: section for section in top_section_list
    }
    if len(sections) != len(top_section_list):
        raise ConfigError("resolved top-level section names must be unique")

    validated_comparisons: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for index, comparison in enumerate(comparisons):
        context = f"comparisons[{index}]"
        if not isinstance(comparison, dict):
            raise ConfigError(f"{context} must be an object")
        require_keys(comparison, ("name", "source_1", "source_2", "reference"), context)
        values = {
            key: nonempty_string(comparison[key], f"{context}.{key}")
            for key in ("name", "source_1", "source_2", "reference")
        }
        if not SAFE_OUTPUT_NAME.fullmatch(values["name"]):
            raise ConfigError(
                f"{context}.name may contain only letters, numbers, '_', '-', and '.'"
            )
        if values["name"] in used_names:
            raise ConfigError(f"{context}.name is duplicated: {values['name']}")
        used_names.add(values["name"])
        for key in ("source_1", "source_2", "reference"):
            if values[key] not in sources:
                raise ConfigError(
                    f"{context}.{key} references unknown data source {values[key]!r}"
                )
        if values["source_1"] == values["source_2"]:
            raise ConfigError(f"{context}: source_1 and source_2 must be different")
        if values["reference"] not in {values["source_1"], values["source_2"]}:
            raise ConfigError(f"{context}.reference must equal source_1 or source_2")

        resolved = _resolve_comparison_sections(
            comparison, config, config_path, sections, context
        )
        section_names: list[str] = []
        for section in resolved:
            name = section["name"]
            existing = sections.get(name)
            if existing is not None and not (
                np.allclose(existing["center"], section["center"])
                and np.allclose(existing["normal"], section["normal"])
            ):
                raise ConfigError(f"{context} resolves conflicting definitions for {name!r}")
            sections[name] = section
            section_names.append(name)
        validated_comparisons.append({**values, "sections": section_names})

    require_keys(
        interpolation,
        ("method", "duplicate_point_tolerance", "minimum_common_valid_fraction"),
        "interpolation",
    )
    method = nonempty_string(interpolation["method"], "interpolation.method").lower()
    if method != "linear_tri":
        raise ConfigError("interpolation.method currently supports only 'linear_tri'")
    minimum_valid = finite_float(
        interpolation["minimum_common_valid_fraction"],
        "interpolation.minimum_common_valid_fraction",
    )
    if not 0.0 <= minimum_valid <= 1.0:
        raise ConfigError(
            "interpolation.minimum_common_valid_fraction must be between 0 and 1"
        )
    validated_interpolation = {
        **interpolation,
        "method": method,
        "duplicate_point_tolerance": positive_float(
            interpolation["duplicate_point_tolerance"],
            "interpolation.duplicate_point_tolerance",
        ),
        "minimum_common_valid_fraction": minimum_valid,
    }

    require_keys(metrics, ("epsilon", "u95_percentile", "compute_vector_components"), "metrics")
    percentile = finite_float(metrics["u95_percentile"], "metrics.u95_percentile")
    if not 0.0 < percentile <= 100.0:
        raise ConfigError("metrics.u95_percentile must be greater than 0 and at most 100")
    validated_metrics = {
        **metrics,
        "epsilon": positive_float(metrics["epsilon"], "metrics.epsilon"),
        "u95_percentile": percentile,
        "compute_vector_components": boolean(
            metrics["compute_vector_components"], "metrics.compute_vector_components"
        ),
    }

    require_keys(
        output,
        ("directory", "save_metrics_csv", "save_png", "save_vtp", "dpi"),
        "output",
    )
    dpi = output["dpi"]
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ConfigError("output.dpi must be a positive integer")
    validated_output = {
        **output,
        "directory": resolve_path(nonempty_string(output["directory"], "output.directory")),
        "grid_output_format": resolve_grid_output_format(output),
        "save_grid_csv": (
            boolean(output["save_grid_csv"], "output.save_grid_csv")
            if "save_grid_csv" in output else False
        ),
        "save_metrics_csv": boolean(output["save_metrics_csv"], "output.save_metrics_csv"),
        "save_png": boolean(output["save_png"], "output.save_png"),
        "save_vtp": boolean(output["save_vtp"], "output.save_vtp"),
        "dpi": dpi,
    }
    execution = config.get("execution", {})
    if not isinstance(execution, dict):
        raise ConfigError("execution must be an object")
    fail_fast = boolean(execution.get("fail_fast", True), "execution.fail_fast")
    return {
        "data_sources": sources,
        "sections": sections,
        "comparisons": validated_comparisons,
        "interpolation": validated_interpolation,
        "metrics": validated_metrics,
        "output": validated_output,
        "execution": {"fail_fast": fail_fast},
        "section_library": config.get("section_library"),
    }


def load_and_validate_config(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    try:
        config = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {resolved}: {exc}") from exc
    return validate_config(config, resolved)


def read_velocity_mesh(source_name: str, source: dict[str, Any]) -> pv.DataSet:
    data = pv.read(source["path"])
    if isinstance(data, pv.MultiBlock):
        mesh = data.combine(merge_points=False)
    else:
        mesh = data
    mesh = mesh.copy(deep=True)
    mesh.points = np.asarray(mesh.points, dtype=float) * source["length_scale_to_mm"]
    velocity_name = source["velocity_array"]
    association = source["data_association"]
    if association == "point":
        if velocity_name not in mesh.point_data:
            raise KeyError(
                f"Data source {source_name!r}: point array {velocity_name!r} not found. "
                f"Point arrays: {list(mesh.point_data.keys())}"
            )
        mesh.point_data[velocity_name] = (
            np.asarray(mesh.point_data[velocity_name], dtype=float)
            * source["velocity_scale_to_mm_s"]
        )
    else:
        if velocity_name not in mesh.cell_data:
            raise KeyError(
                f"Data source {source_name!r}: cell array {velocity_name!r} not found. "
                f"Cell arrays: {list(mesh.cell_data.keys())}"
            )
        mesh.cell_data[velocity_name] = (
            np.asarray(mesh.cell_data[velocity_name], dtype=float)
            * source["velocity_scale_to_mm_s"]
        )
        mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
    velocity = np.asarray(mesh.point_data[velocity_name])
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError(
            f"Data source {source_name!r}: {velocity_name!r} must be an N x 3 vector; "
            f"found shape {velocity.shape}"
        )
    if not np.all(np.isfinite(velocity)):
        raise ValueError(
            f"Data source {source_name!r}: {velocity_name!r} contains non-finite values"
        )
    return mesh


def average_duplicate_samples(
    s: np.ndarray,
    t: np.ndarray,
    vectors: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(s) & np.isfinite(t) & np.all(np.isfinite(vectors), axis=1)
    s = s[finite]
    t = t[finite]
    vectors = vectors[finite]
    if s.size == 0:
        return s, t, vectors
    scaled = np.column_stack((s, t)) / tolerance
    if np.max(np.abs(scaled)) > np.iinfo(np.int64).max * 0.5:
        raise ValueError("Section coordinates are too large for duplicate-point tolerance")
    keys = np.rint(scaled).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    averaged_s = np.bincount(inverse, weights=s) / counts
    averaged_t = np.bincount(inverse, weights=t) / counts
    averaged_vectors = np.column_stack(
        [
            np.bincount(inverse, weights=vectors[:, component]) / counts
            for component in range(vectors.shape[1])
        ]
    )
    return averaged_s, averaged_t, averaged_vectors


def extract_section_samples(
    source_name: str,
    mesh: pv.DataSet,
    velocity_name: str,
    section_name: str,
    section: dict[str, Any],
    duplicate_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cut = mesh.slice(origin=section["center"], normal=section["normal"])
    if cut.n_points == 0:
        raise RuntimeError(
            f"Data source {source_name!r}, section {section_name!r}: slice is empty"
        )
    if velocity_name not in cut.point_data:
        raise KeyError(
            f"Data source {source_name!r}, section {section_name!r}: "
            f"{velocity_name!r} not found on slice. Arrays: {list(cut.point_data.keys())}"
        )
    relative = np.asarray(cut.points) - section["center"]
    s = relative @ section["s_axis"]
    t = relative @ section["t_axis"]
    keep = (np.abs(s) <= 0.5 * section["width"]) & (
        np.abs(t) <= 0.5 * section["height"]
    )
    if not np.any(keep):
        raise RuntimeError(
            f"Data source {source_name!r}, section {section_name!r}: "
            "no slice points remain inside width/height"
        )
    vectors = np.asarray(cut.point_data[velocity_name], dtype=float)
    return average_duplicate_samples(
        s[keep], t[keep], vectors[keep], duplicate_tolerance
    )


def make_common_grid(
    section: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = section["grid_resolution"]
    s_values = np.linspace(-0.5 * section["width"], 0.5 * section["width"], nx)
    t_values = np.linspace(-0.5 * section["height"], 0.5 * section["height"], ny)
    ss, tt = np.meshgrid(s_values, t_values, indexing="xy")
    points = (
        section["center"]
        + ss[..., None] * section["s_axis"]
        + tt[..., None] * section["t_axis"]
    )
    return s_values, t_values, ss, tt, points.reshape((-1, 3))


def interpolate_vectors(
    s: np.ndarray,
    t: np.ndarray,
    vectors: np.ndarray,
    ss: np.ndarray,
    tt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if s.size < 3:
        raise RuntimeError("At least three section samples are required for interpolation")
    triangulation = mtri.Triangulation(s, t)
    interpolated = np.full((ss.size, 3), np.nan, dtype=float)
    valid = np.ones(ss.size, dtype=bool)
    for component in range(3):
        interpolator = mtri.LinearTriInterpolator(triangulation, vectors[:, component])
        values = np.ma.asarray(interpolator(ss, tt)).reshape(-1)
        mask = np.ma.getmaskarray(values)
        data = values.filled(np.nan)
        interpolated[:, component] = data
        valid &= ~mask & np.isfinite(data)
    return interpolated, valid


def compute_comparison_data(
    velocity_1: np.ndarray,
    velocity_2: np.ndarray,
    valid_1: np.ndarray,
    valid_2: np.ndarray,
    reference_index: int,
    metrics_config: dict[str, Any],
) -> dict[str, Any]:
    valid = valid_1 & valid_2
    if not np.any(valid):
        raise RuntimeError("The two data sources have no common valid grid points")
    speed_1 = np.linalg.norm(velocity_1, axis=1)
    speed_2 = np.linalg.norm(velocity_2, axis=1)
    difference = velocity_1 - velocity_2
    vector_error = np.linalg.norm(difference, axis=1)
    signed_speed_error = speed_1 - speed_2
    absolute_speed_error = np.abs(signed_speed_error)
    reference_velocity = velocity_1 if reference_index == 1 else velocity_2
    reference_speed = speed_1 if reference_index == 1 else speed_2
    epsilon = metrics_config["epsilon"]
    relative_vector_error = np.full(len(valid), np.nan)
    relative_speed_error = np.full(len(valid), np.nan)
    denominator = np.maximum(reference_speed, epsilon)
    relative_vector_error[valid] = vector_error[valid] / denominator[valid] * 100.0
    relative_speed_error[valid] = (
        absolute_speed_error[valid] / denominator[valid] * 100.0
    )
    reference_u95 = float(
        np.percentile(reference_speed[valid], metrics_config["u95_percentile"])
    )
    u95_denominator = max(reference_u95, epsilon)
    u95_vector_error = np.full(len(valid), np.nan)
    u95_speed_error = np.full(len(valid), np.nan)
    u95_vector_error[valid] = vector_error[valid] / u95_denominator * 100.0
    u95_speed_error[valid] = absolute_speed_error[valid] / u95_denominator * 100.0
    return {
        "valid": valid,
        "velocity_1": velocity_1,
        "velocity_2": velocity_2,
        "speed_1": speed_1,
        "speed_2": speed_2,
        "velocity_difference": difference,
        "vector_error_magnitude": vector_error,
        "signed_speed_error": signed_speed_error,
        "absolute_speed_error": absolute_speed_error,
        "reference_velocity": reference_velocity,
        "reference_speed": reference_speed,
        "relative_vector_error_percent": relative_vector_error,
        "relative_speed_error_percent": relative_speed_error,
        "reference_u95_mm_per_s": reference_u95,
        "u95_normalized_vector_error_percent": u95_vector_error,
        "u95_normalized_speed_error_percent": u95_speed_error,
    }


def valid_stat(values: np.ndarray, valid: np.ndarray, statistic: str) -> float:
    selected = np.asarray(values)[valid & np.isfinite(values)]
    if statistic == "mean":
        return float(np.mean(selected))
    if statistic == "median":
        return float(np.median(selected))
    if statistic == "max":
        return float(np.max(selected))
    if statistic == "p95":
        return float(np.percentile(selected, 95.0))
    raise ValueError(statistic)


def make_metrics_row(
    comparison: dict[str, str],
    source_1: dict[str, Any],
    source_2: dict[str, Any],
    section: dict[str, Any],
    data: dict[str, Any],
    common_valid_fraction: float,
    compute_components: bool,
) -> dict[str, Any]:
    valid = data["valid"]
    reference_norm = float(np.linalg.norm(data["reference_velocity"][valid].ravel()))
    difference_norm = float(np.linalg.norm(data["velocity_difference"][valid].ravel()))
    row: dict[str, Any] = {
        "comparison": comparison["name"],
        "source_1": comparison["source_1"],
        "source_1_label": source_1["label"],
        "source_1_path": str(source_1["path"]),
        "source_1_dt_s": source_1["dt"],
        "source_1_time_s": source_1["time"],
        "source_2": comparison["source_2"],
        "source_2_label": source_2["label"],
        "source_2_path": str(source_2["path"]),
        "source_2_dt_s": source_2["dt"],
        "source_2_time_s": source_2["time"],
        "reference": comparison["reference"],
        "section": comparison["section"],
        "section_label": section["label"],
        "common_valid_fraction": common_valid_fraction,
        "common_valid_points": int(np.sum(valid)),
        "total_grid_points": len(valid),
        "reference_u95_mm_per_s": data["reference_u95_mm_per_s"],
        "relative_vector_L2_percent": (
            difference_norm / reference_norm * 100.0
            if reference_norm > 0.0
            else float("nan")
        ),
    }
    for prefix, values in (
        ("source_1_speed_mm_per_s", data["speed_1"]),
        ("source_2_speed_mm_per_s", data["speed_2"]),
        ("vector_error_mm_per_s", data["vector_error_magnitude"]),
        ("absolute_speed_error_mm_per_s", data["absolute_speed_error"]),
        ("relative_vector_error_percent", data["relative_vector_error_percent"]),
        ("relative_speed_error_percent", data["relative_speed_error_percent"]),
        (
            "u95_normalized_vector_error_percent",
            data["u95_normalized_vector_error_percent"],
        ),
        (
            "u95_normalized_speed_error_percent",
            data["u95_normalized_speed_error_percent"],
        ),
    ):
        for statistic in ("mean", "median", "p95", "max"):
            row[f"{prefix}_{statistic}"] = valid_stat(values, valid, statistic)
    center = section["center"]
    normal = section["normal"]
    row.update(
        {
            "comparison_name": comparison["name"],
            "section_name": comparison["section"],
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "center_z": float(center[2]),
            "normal_x": float(normal[0]),
            "normal_y": float(normal[1]),
            "normal_z": float(normal[2]),
            "dt_source_1": source_1["dt"],
            "dt_source_2": source_2["dt"],
            "time_source_1": source_1["time"],
            "time_source_2": source_2["time"],
            "relative_l2_error": row["relative_vector_L2_percent"],
            "mae": row["absolute_speed_error_mm_per_s_mean"],
            "max_absolute_error": row["absolute_speed_error_mm_per_s_max"],
            "normalized_mae": row["u95_normalized_speed_error_percent_mean"],
            **valid_grid_metrics(
                valid, section["width"], section["height"], section["grid_resolution"]
            ),
            "valid_fraction": common_valid_fraction,
            "valid_area": common_valid_fraction * section["width"] * section["height"],
        }
    )
    if compute_components:
        for component, component_name in enumerate(("u", "v", "w")):
            reference_component_norm = float(
                np.linalg.norm(data["reference_velocity"][valid, component])
            )
            difference_component_norm = float(
                np.linalg.norm(data["velocity_difference"][valid, component])
            )
            row[f"relative_{component_name}_L2_percent"] = (
                difference_component_norm / reference_component_norm * 100.0
                if reference_component_norm > 0.0
                else float("nan")
            )
    return row


def make_grid_dataframe(
    points: np.ndarray,
    ss: np.ndarray,
    tt: np.ndarray,
    data: dict[str, Any],
    compute_components: bool,
) -> pd.DataFrame:
    columns: dict[str, Any] = {
        "x_mm": points[:, 0],
        "y_mm": points[:, 1],
        "z_mm": points[:, 2],
        "s_mm": ss.ravel(),
        "t_mm": tt.ravel(),
        "valid": data["valid"],
        "source_1_speed_mm_per_s": data["speed_1"],
        "source_2_speed_mm_per_s": data["speed_2"],
        "reference_speed_mm_per_s": data["reference_speed"],
        "signed_speed_error_mm_per_s": data["signed_speed_error"],
        "absolute_speed_error_mm_per_s": data["absolute_speed_error"],
        "vector_error_magnitude_mm_per_s": data["vector_error_magnitude"],
        "relative_speed_error_percent": data["relative_speed_error_percent"],
        "relative_vector_error_percent": data["relative_vector_error_percent"],
        "u95_normalized_speed_error_percent": data[
            "u95_normalized_speed_error_percent"
        ],
        "u95_normalized_vector_error_percent": data[
            "u95_normalized_vector_error_percent"
        ],
    }
    if compute_components:
        for component, name in enumerate(("u", "v", "w")):
            columns[f"source_1_{name}_mm_per_s"] = data["velocity_1"][:, component]
            columns[f"source_2_{name}_mm_per_s"] = data["velocity_2"][:, component]
            columns[f"difference_{name}_mm_per_s"] = data[
                "velocity_difference"
            ][:, component]
    return pd.DataFrame(columns)


def finite_max(*arrays: np.ndarray) -> float:
    values = np.concatenate(
        [np.asarray(array)[np.isfinite(array)] for array in arrays]
    )
    return max(float(np.max(values)), np.finfo(float).tiny)


def plot_comparison(
    path: Path,
    comparison: dict[str, str],
    source_1: dict[str, Any],
    source_2: dict[str, Any],
    section: dict[str, Any],
    ss: np.ndarray,
    tt: np.ndarray,
    data: dict[str, Any],
    dpi: int,
) -> None:
    valid = data["valid"]

    def field(values: np.ndarray) -> np.ma.MaskedArray:
        reshaped = np.asarray(values).reshape(ss.shape)
        mask = (~valid).reshape(ss.shape) | ~np.isfinite(reshaped)
        return np.ma.array(reshaped, mask=mask)

    speed_max = finite_max(data["speed_1"][valid], data["speed_2"][valid])
    panels = (
        (data["speed_1"], f"{source_1['label']} speed", "viridis", 0.0, speed_max, "mm/s"),
        (data["speed_2"], f"{source_2['label']} speed", "viridis", 0.0, speed_max, "mm/s"),
        (
            data["absolute_speed_error"],
            "Absolute speed error",
            "magma",
            0.0,
            None,
            "mm/s",
        ),
        (
            data["vector_error_magnitude"],
            "Vector error magnitude",
            "magma",
            0.0,
            None,
            "mm/s",
        ),
        (
            data["relative_speed_error_percent"],
            "Local relative speed error",
            "coolwarm",
            0.0,
            None,
            "%",
        ),
        (
            data["u95_normalized_vector_error_percent"],
            "U95-normalized vector error",
            "coolwarm",
            0.0,
            None,
            "%",
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (values, title, cmap, vmin, vmax, unit) in zip(axes.ravel(), panels):
        mesh = ax.pcolormesh(
            ss,
            tt,
            field(values),
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(mesh, ax=ax, label=unit)
        ax.set_title(title)
        ax.set_xlabel("s [mm]")
        ax.set_ylabel("t [mm]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{comparison['name']} — {section['label']}\n"
        f"{source_1['label']} ({source_1['time']:g} s) vs "
        f"{source_2['label']} ({source_2['time']:g} s); "
        f"reference={comparison['reference']}"
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)



def plot_scalar_field(
    path: Path,
    ss: np.ndarray,
    tt: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str,
    dpi: int,
    vmax: float | None = None,
) -> None:
    field = np.asarray(values).reshape(ss.shape)
    mask = (~valid).reshape(ss.shape) | ~np.isfinite(field)
    display = np.ma.array(field, mask=mask)
    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    image = ax.pcolormesh(
        ss, tt, display, shading="auto", cmap=cmap, vmin=0.0, vmax=vmax
    )
    fig.colorbar(image, ax=ax, label=colorbar_label)
    ax.set_xlabel("s [mm]")
    ax.set_ylabel("t [mm]")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_comparison_pngs(
    directory: Path,
    comparison: dict[str, Any],
    source_1: dict[str, Any],
    source_2: dict[str, Any],
    section: dict[str, Any],
    ss: np.ndarray,
    tt: np.ndarray,
    data: dict[str, Any],
    dpi: int,
) -> None:
    valid = data["valid"]
    speed_max = finite_max(data["speed_1"][valid], data["speed_2"][valid])
    plot_scalar_field(
        directory / "speed_case1.png", ss, tt, data["speed_1"], valid,
        f"{source_1['label']} speed — {section['label']}", "Speed [mm/s]",
        "viridis", dpi, speed_max,
    )
    plot_scalar_field(
        directory / "speed_case2.png", ss, tt, data["speed_2"], valid,
        f"{source_2['label']} speed — {section['label']}", "Speed [mm/s]",
        "viridis", dpi, speed_max,
    )
    plot_scalar_field(
        directory / "absolute_difference.png", ss, tt,
        data["absolute_speed_error"], valid,
        f"Absolute speed difference — {section['label']}", "|Speed difference| [mm/s]",
        "magma", dpi,
    )
    plot_scalar_field(
        directory / "normalized_difference.png", ss, tt,
        data["u95_normalized_speed_error_percent"], valid,
        f"U95-normalized speed difference — {section['label']}", "Difference [%]",
        "coolwarm", dpi,
    )
    side_by_side = directory / "side_by_side.png"
    plot_comparison(
        side_by_side, comparison, source_1, source_2, section, ss, tt, data, dpi
    )
    shutil.copyfile(side_by_side, directory / "comparison.png")

def save_vtp(
    path: Path,
    points: np.ndarray,
    data: dict[str, Any],
    compute_components: bool,
) -> None:
    polydata = pv.PolyData(points)
    polydata.point_data["valid"] = data["valid"].astype(np.uint8)
    for name in (
        "speed_1",
        "speed_2",
        "reference_speed",
        "signed_speed_error",
        "absolute_speed_error",
        "vector_error_magnitude",
        "relative_speed_error_percent",
        "relative_vector_error_percent",
        "u95_normalized_speed_error_percent",
        "u95_normalized_vector_error_percent",
    ):
        polydata.point_data[name] = data[name]
    if compute_components:
        polydata.point_data["velocity_1"] = data["velocity_1"]
        polydata.point_data["velocity_2"] = data["velocity_2"]
        polydata.point_data["velocity_difference"] = data["velocity_difference"]
    polydata.save(path)


def print_comparison_settings(
    comparison: dict[str, str],
    source_1: dict[str, Any],
    source_2: dict[str, Any],
    section: dict[str, Any],
    output_dir: Path,
) -> None:
    print()
    print(f"Comparison: {comparison['name']}")
    for role, key, source in (
        ("source_1", comparison["source_1"], source_1),
        ("source_2", comparison["source_2"], source_2),
    ):
        print(f"  {role}: {key} ({source['label']})")
        print(f"    path: {source['path']}")
        print(f"    dt: {source['dt']:g} s")
        print(f"    comparison time: {source['time']:g} s")
        print(
            f"    velocity: {source['velocity_array']} "
            f"({source['data_association']} data)"
        )
        print(
            f"    input units: length={source['length_unit']}, "
            f"velocity={source['velocity_unit']}"
        )
        print(
            f"    scales to mm/mm/s: {source['length_scale_to_mm']:g}, "
            f"{source['velocity_scale_to_mm_s']:g}"
        )
    print(f"  section: {comparison['section']} ({section['label']})")
    print(f"    center [mm]: {section['center'].tolist()}")
    print(f"    normalized normal: {section['normal'].tolist()}")
    print(f"    s axis: {section['s_axis'].tolist()}")
    print(f"    t axis: {section['t_axis'].tolist()}")
    print(f"    width x height [mm]: {section['width']:g} x {section['height']:g}")
    print(f"    grid resolution: {list(section['grid_resolution'])}")
    print(f"  reference source: {comparison['reference']}")
    print(f"  output: {output_dir / comparison['name'] / comparison['section']}")


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    config = load_and_validate_config(config_path)
    output_config = config["output"]
    output_dir: Path = output_config["directory"]
    section_order = [
        f"{comparison['name']}:{section_name}"
        for comparison in config["comparisons"]
        for section_name in comparison["sections"]
    ]
    print(f"Configuration: {config_path}")
    print(f"Section library: {config.get('section_library') or '(inline/legacy)'}")
    print(f"Resolved section tasks: {len(section_order)}")
    print(f"Section processing order: {section_order}")
    print(f"Output root: {output_dir}")

    for comparison in config["comparisons"]:
        source_1 = config["data_sources"][comparison["source_1"]]
        source_2 = config["data_sources"][comparison["source_2"]]
        for section_name in comparison["sections"]:
            scoped = {**comparison, "section": section_name}
            print_comparison_settings(
                scoped, source_1, source_2, config["sections"][section_name], output_dir
            )

    meshes = {
        name: read_velocity_mesh(name, source)
        for name, source in config["data_sources"].items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    section_sample_cache: dict[
        tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    metrics_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    duplicate_tolerance = config["interpolation"]["duplicate_point_tolerance"]
    fail_fast = config["execution"]["fail_fast"]

    for comparison in config["comparisons"]:
        source_1_name = comparison["source_1"]
        source_2_name = comparison["source_2"]
        source_1 = config["data_sources"][source_1_name]
        source_2 = config["data_sources"][source_2_name]
        for section_name in comparison["sections"]:
            scoped = {**comparison, "section": section_name}
            section = config["sections"][section_name]
            section_dir = output_dir / comparison["name"] / section_name
            try:
                print(f"Processing {comparison['name']} / {section_name} ({section['label']})")
                for source_name, source in (
                    (source_1_name, source_1),
                    (source_2_name, source_2),
                ):
                    cache_key = (source_name, section_name)
                    if cache_key not in section_sample_cache:
                        section_sample_cache[cache_key] = extract_section_samples(
                            source_name,
                            meshes[source_name],
                            source["velocity_array"],
                            section_name,
                            section,
                            duplicate_tolerance,
                        )
                _, _, ss, tt, points = make_common_grid(section)
                s_1, t_1, samples_1 = section_sample_cache[(source_1_name, section_name)]
                s_2, t_2, samples_2 = section_sample_cache[(source_2_name, section_name)]
                velocity_1, valid_1 = interpolate_vectors(s_1, t_1, samples_1, ss, tt)
                velocity_2, valid_2 = interpolate_vectors(s_2, t_2, samples_2, ss, tt)
                reference_index = 1 if comparison["reference"] == source_1_name else 2
                data = compute_comparison_data(
                    velocity_1, velocity_2, valid_1, valid_2,
                    reference_index, config["metrics"],
                )
                common_valid_fraction = float(np.mean(data["valid"]))
                minimum_valid = config["interpolation"]["minimum_common_valid_fraction"]
                if common_valid_fraction < minimum_valid:
                    raise RuntimeError(
                        f"common valid fraction {common_valid_fraction:.6f} is below "
                        f"configured minimum {minimum_valid:.6f}"
                    )
                section_dir.mkdir(parents=True, exist_ok=True)
                compute_components = config["metrics"]["compute_vector_components"]
                metrics_row = make_metrics_row(
                    scoped, source_1, source_2, section, data,
                    common_valid_fraction, compute_components,
                )
                metrics_rows.append(metrics_row)
                grid_format = output_config["grid_output_format"]
                if grid_format != "none":
                    frame = (
                        make_grid_dataframe(points, ss, tt, data, compute_components)
                        if grid_format in {"csv", "csv.gz"} else None
                    )
                    saved_grid = save_grid_output(
                        section_dir / "comparison_grid",
                        grid_format,
                        frame,
                        {
                            "s_grid": ss,
                            "t_grid": tt,
                            "source_1_velocity": data["velocity_1"],
                            "source_2_velocity": data["velocity_2"],
                            "source_1_speed": data["speed_1"],
                            "source_2_speed": data["speed_2"],
                            "absolute_difference": data["absolute_speed_error"],
                            "normalized_difference": data["u95_normalized_speed_error_percent"],
                            "common_valid_mask": data["valid"],
                        },
                        {
                            "comparison_name": comparison["name"],
                            "section_name": section_name,
                            "section_label": section["label"],
                            "center": section["center"].tolist(),
                            "normal": section["normal"].tolist(),
                            "source_1": source_1_name,
                            "source_2": source_2_name,
                            "reference": comparison["reference"],
                            "source_1_path": str(source_1["path"]),
                            "source_2_path": str(source_2["path"]),
                        },
                    )
                    print(f"  grid output: {saved_grid}")
                if output_config["save_metrics_csv"]:
                    pd.DataFrame([metrics_row]).to_csv(section_dir / "metrics.csv", index=False)
                if output_config["save_png"]:
                    save_comparison_pngs(
                        section_dir, scoped, source_1, source_2, section,
                        ss, tt, data, output_config["dpi"],
                    )
                if output_config["save_vtp"]:
                    save_vtp(
                        section_dir / "comparison_grid.vtp", points, data, compute_components
                    )
                print(
                    f"Completed {comparison['name']} / {section_name}: "
                    f"valid_fraction={common_valid_fraction:.6f}, "
                    f"relative_L2={metrics_row['relative_vector_L2_percent']:.6g}%"
                )
            except Exception as exc:
                failures.append(
                    {"comparison_name": comparison["name"], "section_name": section_name,
                     "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"FAILED {comparison['name']} / {section_name}: {type(exc).__name__}: {exc}")
                if fail_fast:
                    raise

    summary_path = output_dir / "comparison_metrics.csv"
    if output_config["save_metrics_csv"]:
        pd.DataFrame(metrics_rows).to_csv(summary_path, index=False)
        if failures:
            pd.DataFrame(failures).to_csv(output_dir / "comparison_failures.csv", index=False)
    print(f"Successful sections: {len(metrics_rows)}")
    print(f"Failed sections: {len(failures)}")
    print(f"Output directory: {output_dir}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
