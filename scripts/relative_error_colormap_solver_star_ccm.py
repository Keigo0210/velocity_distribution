from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import pyvista as pv

from section_config import SectionConfigError, resolve_sections_from_config
from common_grid_output import resolve_grid_output_format, save_grid_output, valid_grid_metrics
from streamlines_common import (
    ROOT,
    load_config,
    physical_time_value,
    ensure_point_vectors,
    resolve_output_dir,
    resolve_path,
    scale_mesh,
    time_points_from_config,
)


DEFAULT_CONFIG = ROOT / "config" / "relative_error_colormap_solver_star_ccm.json"


def section_name(section: dict, index: int) -> str:
    if section.get("name"):
        return str(section["name"])
    center = "_".join(f"{float(value):.6g}".replace("-", "m").replace(".", "p") for value in section["center"])
    return f"section_{index + 1}_{center}"


def format_coord_for_filename(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def section_center_tag(section: dict) -> str:
    x, y, z = (format_coord_for_filename(value) for value in section["center"])
    return f"cx{x}_cy{y}_cz{z}"


def make_section_points(section: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(section["center"], dtype=float)
    width = float(section.get("width", 10.0))
    height = float(section.get("height", 10.0))
    nx, ny = section["grid_resolution"]
    e1 = np.asarray(section["s_axis"], dtype=float)
    e2 = np.asarray(section["t_axis"], dtype=float)
    s_values = np.linspace(-0.5 * width, 0.5 * width, nx)
    t_values = np.linspace(-0.5 * height, 0.5 * height, ny)
    ss, tt = np.meshgrid(s_values, t_values, indexing="xy")
    points = center + ss[..., None] * e1 + tt[..., None] * e2
    return points.reshape((-1, 3)), ss, tt, points


def prepare_mesh_for_sampling(mesh: pv.DataSet, velocity_name: str, association: str) -> pv.DataSet:
    association = association.lower()
    if association == "point":
        return ensure_point_vectors(mesh, velocity_name)
    if association == "cell":
        if velocity_name not in mesh.cell_data:
            raise KeyError(
                f"{velocity_name!r} was requested as cell data, but was not found in cell_data. "
                f"Cell data: {list(mesh.cell_data.keys())}, Point data: {list(mesh.point_data.keys())}"
            )
        return mesh
    raise ValueError("data_association must be 'point' or 'cell'")


def read_solver_mesh_for_sampling(path: Path, config: dict) -> pv.DataSet:
    velocity_name = config.get("velocity_name", "solution_velocity")
    mesh = pv.read(path)
    mesh = scale_mesh(
        mesh,
        coordinate_scale=float(config.get("coordinate_scale", 1.0)),
        velocity_name=velocity_name,
        velocity_scale=float(config.get("velocity_scale", 1.0)),
    )
    return prepare_mesh_for_sampling(mesh, velocity_name, config.get("data_association", "point"))


def read_star_mesh_for_sampling(reader, time_point: int, config: dict) -> pv.DataSet:
    velocity_name = config.get("velocity_name", "Velocity")
    reader.set_active_time_point(int(time_point))
    data = reader.read()
    merge_points = bool(config.get("merge_points", False))
    mesh = data.combine(merge_points=merge_points) if isinstance(data, pv.MultiBlock) else data
    mesh = scale_mesh(
        mesh,
        coordinate_scale=float(config.get("coordinate_scale", 1000.0)),
        velocity_name=velocity_name,
        velocity_scale=float(config.get("velocity_scale", 1000.0)),
    )
    return prepare_mesh_for_sampling(mesh, velocity_name, config.get("data_association", "cell"))


def sample_vectors(mesh: pv.DataSet, points: np.ndarray, velocity_name: str, association: str) -> tuple[np.ndarray, np.ndarray]:
    association = association.lower()
    cloud = pv.PolyData(points)
    sampled = cloud.sample(
        mesh,
        locator="static_cell",
        mark_blank=True,
        pass_cell_data=(association == "cell"),
        pass_point_data=(association == "point"),
    )
    if velocity_name not in sampled.point_data:
        raise KeyError(f"{velocity_name!r} was not sampled. Available arrays: {list(sampled.point_data.keys())}")
    vectors = np.asarray(sampled.point_data[velocity_name], dtype=float)
    if vectors.ndim == 1:
        vectors = vectors.reshape((-1, 1))
    valid = np.asarray(sampled.point_data.get("vtkValidPointMask", np.ones(points.shape[0], dtype=np.uint8))).astype(bool)
    valid &= np.all(np.isfinite(vectors), axis=1)
    return vectors, valid


def compute_reference_speed(
    solver_speed: np.ndarray,
    star_speed: np.ndarray,
    valid: np.ndarray,
    comparison_config: dict,
) -> float:
    source = str(comparison_config.get("reference_speed_source", "star")).lower()
    if source == "solver":
        values = solver_speed
    elif source == "mean":
        values = 0.5 * (solver_speed + star_speed)
    elif source == "max":
        values = np.maximum(solver_speed, star_speed)
    else:
        values = star_speed

    finite = valid & np.isfinite(values)
    if not np.any(finite):
        return float("nan")

    if "reference_speed" in comparison_config:
        return float(comparison_config["reference_speed"])

    statistic = str(comparison_config.get("reference_speed_statistic", "percentile")).lower()
    if statistic == "mean":
        reference_speed = float(np.nanmean(values[finite]))
    elif statistic == "median":
        reference_speed = float(np.nanmedian(values[finite]))
    elif statistic == "max":
        reference_speed = float(np.nanmax(values[finite]))
    else:
        percentile = float(comparison_config.get("reference_speed_percentile", 95.0))
        reference_speed = float(np.nanpercentile(values[finite], percentile))

    minimum = float(comparison_config.get("minimum_reference_speed", 0.0))
    return max(reference_speed, minimum)


def compute_error_denominator(
    solver_speed: np.ndarray,
    star_speed: np.ndarray,
    valid: np.ndarray,
    zero_speed_tolerance: float,
    comparison_config: dict,
) -> tuple[np.ndarray, float]:
    mode = str(comparison_config.get("denominator", "local_star_speed_with_floor")).lower()
    reference_speed = compute_reference_speed(solver_speed, star_speed, valid, comparison_config)

    if mode in {"reference_speed", "global_reference_speed", "global_star_speed"}:
        denominator = np.full_like(star_speed, reference_speed, dtype=float)
    elif mode in {"symmetric_speed", "mean_speed"}:
        denominator = 0.5 * (solver_speed + star_speed)
    elif mode in {"max_speed", "larger_speed"}:
        denominator = np.maximum(solver_speed, star_speed)
    else:
        denominator = star_speed.copy()

    floor_absolute = comparison_config.get("denominator_floor")
    floor_fraction = comparison_config.get("reference_speed_floor_fraction")
    floor = zero_speed_tolerance
    if floor_absolute is not None:
        floor = max(floor, float(floor_absolute))
    if floor_fraction is not None and np.isfinite(reference_speed):
        floor = max(floor, float(floor_fraction) * reference_speed)
    if "floor" in mode and np.isfinite(reference_speed):
        floor = max(floor, float(comparison_config.get("reference_speed_floor_fraction", 0.05)) * reference_speed)

    denominator = np.maximum(denominator, floor)
    return denominator, reference_speed


def compute_relative_error(
    solver_mesh: pv.DataSet,
    solver_velocity_name: str,
    solver_association: str,
    star_mesh: pv.DataSet,
    star_velocity_name: str,
    star_association: str,
    points: np.ndarray,
    zero_speed_tolerance: float,
    comparison_config: dict | None = None,
) -> dict[str, np.ndarray]:
    comparison_config = comparison_config or {}
    solver_velocity, solver_valid = sample_vectors(solver_mesh, points, solver_velocity_name, solver_association)
    star_velocity, star_valid = sample_vectors(star_mesh, points, star_velocity_name, star_association)
    velocity_error = solver_velocity - star_velocity
    velocity_error_magnitude = np.linalg.norm(velocity_error, axis=1)
    solver_speed = np.linalg.norm(solver_velocity, axis=1)
    star_speed = np.linalg.norm(star_velocity, axis=1)

    base_valid = solver_valid & star_valid
    denominator, reference_speed = compute_error_denominator(
        solver_speed,
        star_speed,
        base_valid,
        zero_speed_tolerance,
        comparison_config,
    )
    valid = base_valid & np.isfinite(denominator) & (denominator > zero_speed_tolerance)
    relative_error = np.full(points.shape[0], np.nan, dtype=float)
    relative_error[valid] = velocity_error_magnitude[valid] / denominator[valid]
    return {
        "solver_velocity": solver_velocity,
        "star_velocity": star_velocity,
        "velocity_error": velocity_error,
        "velocity_error_magnitude": velocity_error_magnitude,
        "solver_speed": solver_speed,
        "star_speed": star_speed,
        "relative_velocity_error_denominator": denominator,
        "relative_velocity_error_reference_speed": np.full(points.shape[0], reference_speed, dtype=float),
        "relative_velocity_error": relative_error,
        "relative_velocity_error_percent": relative_error * 100.0,
        "representative_normalized_velocity_error_percent": (
            velocity_error_magnitude / reference_speed * 100.0
            if np.isfinite(reference_speed) and reference_speed > 0.0
            else np.full_like(velocity_error_magnitude, np.nan)
        ),
        "valid_mask": valid,
        "solver_valid_mask": solver_valid,
        "star_valid_mask": star_valid,
    }


def ensure_point_vectors_for_slice(mesh: pv.DataSet, velocity_name: str) -> pv.DataSet:
    if velocity_name in mesh.point_data:
        return mesh
    if velocity_name in mesh.cell_data:
        print(f"Converting cell_data '{velocity_name}' to point_data for section interpolation.")
        return mesh.cell_data_to_point_data(pass_cell_data=True)
    raise KeyError(
        f"{velocity_name!r} was not found. "
        f"Point data: {list(mesh.point_data.keys())}, Cell data: {list(mesh.cell_data.keys())}"
    )


def average_duplicate_section_samples(
    s: np.ndarray,
    t: np.ndarray,
    values: np.ndarray,
    decimals: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(s) & np.isfinite(t) & np.all(np.isfinite(values), axis=1)
    s = s[finite]
    t = t[finite]
    values = values[finite]
    if s.size == 0:
        return s, t, values

    coords = np.round(np.column_stack([s, t]), decimals=decimals)
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    averaged = np.empty((unique_coords.shape[0], values.shape[1]), dtype=float)
    for component in range(values.shape[1]):
        averaged[:, component] = np.bincount(inverse, weights=values[:, component]) / counts
    return unique_coords[:, 0], unique_coords[:, 1], averaged


def extract_section_velocity_samples(
    mesh: pv.DataSet,
    velocity_name: str,
    section: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = ensure_point_vectors_for_slice(mesh, velocity_name)
    center = np.asarray(section["center"], dtype=float)
    normal = np.asarray(section["normalized_normal"], dtype=float)
    e1 = np.asarray(section["s_axis"], dtype=float)
    e2 = np.asarray(section["t_axis"], dtype=float)
    cut = mesh.slice(origin=center, normal=normal)
    if cut.n_points == 0:
        raise RuntimeError(f"No section points were created for section {section.get('name', section['center'])}")
    if velocity_name not in cut.point_data:
        raise KeyError(f"{velocity_name!r} was not found on the sliced section. Available: {list(cut.point_data.keys())}")

    rel = cut.points - center
    s = rel @ e1
    t = rel @ e2
    velocity = np.asarray(cut.point_data[velocity_name], dtype=float)
    if velocity.ndim == 1:
        velocity = velocity.reshape((-1, 1))

    keep = np.ones(cut.n_points, dtype=bool)
    if "width" in section:
        keep &= np.abs(s) <= 0.5 * float(section["width"])
    if "height" in section:
        keep &= np.abs(t) <= 0.5 * float(section["height"])
    if not np.any(keep):
        raise RuntimeError(f"No section points remained inside width/height for section {section.get('name', section['center'])}")

    return average_duplicate_section_samples(s[keep], t[keep], velocity[keep])


def interpolate_section_vectors(
    s: np.ndarray,
    t: np.ndarray,
    vectors: np.ndarray,
    ss: np.ndarray,
    tt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if s.size < 3:
        raise RuntimeError("At least three section points are required for triangular interpolation")
    triangulation = mtri.Triangulation(s, t)
    interpolated = np.full((ss.size, vectors.shape[1]), np.nan, dtype=float)
    valid = np.ones(ss.size, dtype=bool)

    for component in range(vectors.shape[1]):
        interpolator = mtri.LinearTriInterpolator(triangulation, vectors[:, component])
        component_values = np.ma.asarray(interpolator(ss, tt)).reshape(-1)
        component_mask = np.ma.getmaskarray(component_values)
        component_data = component_values.filled(np.nan)
        interpolated[:, component] = component_data
        valid &= ~component_mask & np.isfinite(component_data)

    return interpolated, valid


def compute_slice_interpolated_relative_error(
    solver_mesh: pv.DataSet,
    solver_velocity_name: str,
    star_mesh: pv.DataSet,
    star_velocity_name: str,
    section: dict,
    ss: np.ndarray,
    tt: np.ndarray,
    zero_speed_tolerance: float,
    comparison_config: dict | None = None,
) -> dict[str, np.ndarray]:
    comparison_config = comparison_config or {}
    solver_s, solver_t, solver_section_velocity = extract_section_velocity_samples(solver_mesh, solver_velocity_name, section)
    star_s, star_t, star_section_velocity = extract_section_velocity_samples(star_mesh, star_velocity_name, section)

    solver_velocity, solver_valid = interpolate_section_vectors(solver_s, solver_t, solver_section_velocity, ss, tt)
    star_velocity, star_valid = interpolate_section_vectors(star_s, star_t, star_section_velocity, ss, tt)

    velocity_error = solver_velocity - star_velocity
    vector_velocity_error_magnitude = np.linalg.norm(velocity_error, axis=1)
    solver_speed = np.linalg.norm(solver_velocity, axis=1)
    star_speed = np.linalg.norm(star_velocity, axis=1)
    speed_error = solver_speed - star_speed

    quantity = str(comparison_config.get("quantity", "speed")).lower()
    if quantity in {"speed", "speed_magnitude", "magnitude", "velocity_magnitude"}:
        velocity_error_magnitude = np.abs(speed_error)
    else:
        velocity_error_magnitude = vector_velocity_error_magnitude

    base_valid = solver_valid & star_valid & np.isfinite(velocity_error_magnitude)
    denominator, reference_speed = compute_error_denominator(
        solver_speed,
        star_speed,
        base_valid,
        zero_speed_tolerance,
        comparison_config,
    )
    valid = base_valid & np.isfinite(denominator) & (denominator > zero_speed_tolerance)
    relative_error = np.full(ss.size, np.nan, dtype=float)
    relative_error[valid] = velocity_error_magnitude[valid] / denominator[valid]

    return {
        "solver_velocity": solver_velocity,
        "star_velocity": star_velocity,
        "velocity_error": velocity_error,
        "velocity_error_magnitude": velocity_error_magnitude,
        "vector_velocity_error_magnitude": vector_velocity_error_magnitude,
        "solver_speed": solver_speed,
        "star_speed": star_speed,
        "speed_error": speed_error,
        "relative_velocity_error_denominator": denominator,
        "relative_velocity_error_reference_speed": np.full(ss.size, reference_speed, dtype=float),
        "relative_velocity_error": relative_error,
        "relative_velocity_error_percent": relative_error * 100.0,
        "representative_normalized_velocity_error_percent": (
            velocity_error_magnitude / reference_speed * 100.0
            if np.isfinite(reference_speed) and reference_speed > 0.0
            else np.full_like(velocity_error_magnitude, np.nan)
        ),
        "valid_mask": valid,
        "solver_valid_mask": solver_valid,
        "star_valid_mask": star_valid,
    }


def save_vtp(path: Path, points: np.ndarray, data: dict[str, np.ndarray], time_index: int, time_value: float, section: dict) -> None:
    poly = pv.PolyData(points)
    for name, values in data.items():
        if values.shape[0] == points.shape[0]:
            poly.point_data[name] = values
    poly.field_data["time_index"] = np.asarray([time_index], dtype=np.int32)
    poly.field_data["time_value"] = np.asarray([time_value], dtype=float)
    poly.field_data["section_center"] = np.asarray(section["center"], dtype=float)
    poly.field_data["section_normal"] = np.asarray(section["normal"], dtype=float)
    poly.save(path)


def make_grid_dataframe(points: np.ndarray, ss: np.ndarray, tt: np.ndarray, data: dict[str, np.ndarray]) -> pd.DataFrame:
    solver_velocity = data["solver_velocity"]
    star_velocity = data["star_velocity"]
    velocity_error = data["velocity_error"]
    df = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "s": ss.ravel(),
            "t": tt.ravel(),
            "solver_u": solver_velocity[:, 0],
            "solver_v": solver_velocity[:, 1],
            "solver_w": solver_velocity[:, 2],
            "star_u": star_velocity[:, 0],
            "star_v": star_velocity[:, 1],
            "star_w": star_velocity[:, 2],
            "error_u": velocity_error[:, 0],
            "error_v": velocity_error[:, 1],
            "error_w": velocity_error[:, 2],
            "solver_speed": data["solver_speed"],
            "star_speed": data["star_speed"],
            "velocity_error_magnitude": data["velocity_error_magnitude"],
            "vector_velocity_error_magnitude": data.get("vector_velocity_error_magnitude", data["velocity_error_magnitude"]),
            "speed_error": data.get("speed_error", data["solver_speed"] - data["star_speed"]),
            "relative_velocity_error_denominator": data["relative_velocity_error_denominator"],
            "relative_velocity_error_reference_speed": data["relative_velocity_error_reference_speed"],
            "relative_velocity_error_percent": data["relative_velocity_error_percent"],
            "representative_normalized_velocity_error_percent": data[
                "representative_normalized_velocity_error_percent"
            ],
            "valid": data["valid_mask"].astype(np.uint8),
            "solver_valid": data["solver_valid_mask"].astype(np.uint8),
            "star_valid": data["star_valid_mask"].astype(np.uint8),
        }
    )
    return df


def save_colormap_png(
    path: Path,
    ss: np.ndarray,
    tt: np.ndarray,
    data: dict[str, np.ndarray],
    section: dict,
    time_index: int,
    time_value: float,
    plot_config: dict,
    error_visualization: dict,
) -> None:
    local_min = float(error_visualization.get("display_min_percent", 0.0))
    local_max = float(error_visualization.get("display_max_percent", 40.0))
    representative_min = float(error_visualization.get("representative_display_min_percent", 0.0))
    representative_max = float(error_visualization.get("representative_display_max_percent", 10.0))
    cmap_name = plot_config.get("cmap", "viridis")
    invalid_color = error_visualization.get("invalid_color", "white")

    valid_2d = data["valid_mask"].reshape(ss.shape)
    local_error_2d = np.asarray(data["relative_velocity_error_percent"], dtype=float).reshape(ss.shape)
    representative_error_2d = np.asarray(
        data["representative_normalized_velocity_error_percent"], dtype=float
    ).reshape(ss.shape)
    star_speed_2d = np.asarray(data["star_speed"], dtype=float).reshape(ss.shape)
    denominator_2d = np.asarray(
        data["relative_velocity_error_denominator"], dtype=float
    ).reshape(ss.shape)
    reference_speed = float(data["relative_velocity_error_reference_speed"][0])

    local_display = np.ma.masked_where(~valid_2d, local_error_2d)
    representative_display = np.ma.masked_where(~valid_2d, representative_error_2d)
    floor_active = valid_2d & np.isfinite(star_speed_2d) & np.isfinite(denominator_2d)
    floor_active &= denominator_2d > star_speed_2d + max(1.0e-12, abs(reference_speed) * 1.0e-12)

    extent = [
        float(np.nanmin(ss)),
        float(np.nanmax(ss)),
        float(np.nanmin(tt)),
        float(np.nanmax(tt)),
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=plot_config.get("figsize", [13.0, 6.0]),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    local_cmap = plt.get_cmap(cmap_name).copy()
    local_cmap.set_bad(invalid_color)
    local_cmap.set_over(error_visualization.get("local_over_limit_color", "black"))
    local_image = axes[0].imshow(
        local_display,
        origin="lower",
        extent=extent,
        cmap=local_cmap,
        vmin=local_min,
        vmax=local_max,
        aspect="equal",
    )
    local_colorbar = fig.colorbar(
        local_image,
        ax=axes[0],
        extend="max",
    )
    local_colorbar.set_label("Floor-limited local relative speed error [%]")

    floor_fraction = None
    if np.any(floor_active) and np.any(~floor_active & valid_2d):
        axes[0].contour(
            ss,
            tt,
            floor_active.astype(float),
            levels=[0.5],
            colors=[error_visualization.get("floor_contour_color", "black")],
            linewidths=float(error_visualization.get("floor_contour_width", 1.2)),
            linestyles="--",
        )
        floor_fraction = float(np.nanmedian(denominator_2d[floor_active]) / reference_speed)
    representative_cmap = plt.get_cmap(cmap_name).copy()
    representative_cmap.set_bad(invalid_color)
    representative_cmap.set_over(representative_cmap(1.0))
    representative_image = axes[1].imshow(
        representative_display,
        origin="lower",
        extent=extent,
        cmap=representative_cmap,
        vmin=representative_min,
        vmax=representative_max,
        aspect="equal",
    )
    representative_colorbar = fig.colorbar(
        representative_image,
        ax=axes[1],
        extend="max",
    )
    representative_colorbar.set_label("|FEM speed - Star speed| / Star U95 [%]")

    local_title = "Local relative error"
    if floor_fraction is not None:
        local_title += f"\nDashed: denominator floor boundary ({floor_fraction:g} x U95)"
    axes[0].set_title(local_title)
    if np.isfinite(reference_speed):
        axes[1].set_title(
            "Representative-speed normalized error\n"
            f"U95={reference_speed:.4g} mm/s; 1%={0.01 * reference_speed:.4g} mm/s"
        )
    else:
        axes[1].set_title("Representative-speed normalized error")

    for ax in axes:
        ax.set_xlabel(plot_config.get("s_label", "s"))
        if plot_config.get("grid", True):
            ax.grid(color="white", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel(plot_config.get("t_label", "t"))

    section_label = section.get("name", "section")
    title = plot_config.get("title", "Velocity-distribution error comparison")
    title_lines = [f"{title}: {section_label} / index {time_index} / time {time_value:g} s"]
    if bool(plot_config.get("show_section_info", True)):
        center_text = ", ".join(f"{float(value):g}" for value in section["center"])
        normal_text = ", ".join(f"{float(value):g}" for value in section["normal"])
        title_lines.append(
            f"center=({center_text}), normal=({normal_text}), "
            f"width={float(section.get('width', 10.0)):g}, height={float(section.get('height', 10.0)):g}"
        )
    fig.suptitle("\n".join(title_lines))
    fig.savefig(path, dpi=int(plot_config.get("dpi", 180)))
    plt.close(fig)


def save_scalar_png(
    path: Path,
    ss: np.ndarray,
    tt: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    title: str,
    label: str,
    cmap: str,
    plot_config: dict,
    vmax: float | None = None,
) -> None:
    field = np.asarray(values, dtype=float).reshape(ss.shape)
    mask = (~np.asarray(valid, dtype=bool)).reshape(ss.shape) | ~np.isfinite(field)
    display = np.ma.array(field, mask=mask)
    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    image = ax.imshow(
        display,
        origin="lower",
        extent=[float(np.min(ss)), float(np.max(ss)), float(np.min(tt)), float(np.max(tt))],
        aspect="equal",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )
    fig.colorbar(image, ax=ax, label=label)
    ax.set_xlabel(plot_config.get("s_label", "s [mm]"))
    ax.set_ylabel(plot_config.get("t_label", "t [mm]"))
    ax.set_title(title)
    fig.savefig(path, dpi=int(plot_config.get("dpi", 180)))
    plt.close(fig)


def save_section_pngs(
    directory: Path,
    suffix: str,
    ss: np.ndarray,
    tt: np.ndarray,
    data: dict[str, np.ndarray],
    section: dict,
    time_index: int,
    time_value: float,
    plot_config: dict,
    error_visualization: dict,
) -> None:
    valid = data["valid_mask"]
    speed_max = max(
        float(np.nanmax(data["solver_speed"][valid])),
        float(np.nanmax(data["star_speed"][valid])),
    )
    save_scalar_png(
        directory / f"fem_speed{suffix}.png", ss, tt, data["solver_speed"], valid,
        f"FEM speed — {section['label']}", "Speed [mm/s]", "viridis", plot_config, speed_max,
    )
    save_scalar_png(
        directory / f"star_speed{suffix}.png", ss, tt, data["star_speed"], valid,
        f"STAR-CCM+ speed — {section['label']}", "Speed [mm/s]", "viridis", plot_config, speed_max,
    )
    save_scalar_png(
        directory / f"absolute_difference{suffix}.png", ss, tt,
        np.abs(data["solver_speed"] - data["star_speed"]), valid,
        f"Absolute speed difference — {section['label']}", "|Speed difference| [mm/s]",
        "magma", plot_config,
    )
    save_scalar_png(
        directory / f"normalized_difference{suffix}.png", ss, tt,
        data["representative_normalized_velocity_error_percent"], valid,
        f"Representative normalized difference — {section['label']}", "Difference [%]",
        "coolwarm", plot_config,
    )
    save_colormap_png(
        directory / f"side_by_side{suffix}.png", ss, tt, data, section,
        time_index, time_value, plot_config, error_visualization,
    )

def build_time_pairs(config: dict, reader) -> list[dict]:
    explicit = config.get("time_pairs")
    if explicit is not None:
        return explicit

    star_config = config.get("star_ccm", {})
    solver_config = config.get("solver", {})
    display_offset = int(star_config.get("display_time_index_offset", 1))
    star_points = time_points_from_config(star_config, reader.number_time_points)
    solver_template = solver_config.get("vtu_template")
    if not solver_template:
        raise ValueError("Set solver.vtu_template when time_pairs is not provided")
    return [
        {
            "label": int(star_time_point) + display_offset,
            "solver_vtu": solver_template.format(time_index=int(star_time_point) + display_offset),
            "star_time_point": int(star_time_point),
        }
        for star_time_point in star_points
    ]


def section_metadata(section: dict) -> dict:
    center = np.asarray(section["center"], dtype=float)
    normal = np.asarray(section["normalized_normal"], dtype=float)
    return {
        "section_name": section["name"],
        "section_label": section["label"],
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "normal_z": float(normal[2]),
        "section_width": float(section["width"]),
        "section_height": float(section["height"]),
    }


def summarize(
    comparison_name: str,
    section: dict,
    time_index: int,
    time_value: float,
    data: dict[str, np.ndarray],
    display_max: float,
) -> dict:
    valid = np.asarray(data["valid_mask"], dtype=bool)
    metadata = section_metadata(section)
    base = {
        "comparison_name": comparison_name,
        **metadata,
        "source_1": "fem",
        "source_2": "star_ccm",
        "reference": "star_ccm",
        "time_index": time_index,
        "time_value": time_value,
        **valid_grid_metrics(
            valid, section["width"], section["height"], section["grid_resolution"]
        ),
        "valid_fraction": float(np.mean(valid)),
        "valid_area": float(np.mean(valid)) * section["width"] * section["height"],
    }
    if not np.any(valid):
        return {
            **base,
            "relative_l2_error": np.nan,
            "mae": np.nan,
            "max_absolute_error": np.nan,
            "normalized_mae": np.nan,
        }

    solver_velocity = np.asarray(data["solver_velocity"], dtype=float)
    star_velocity = np.asarray(data["star_velocity"], dtype=float)
    difference = solver_velocity - star_velocity
    reference_norm = float(np.linalg.norm(star_velocity[valid].ravel()))
    difference_norm = float(np.linalg.norm(difference[valid].ravel()))
    absolute_speed_error = np.abs(
        np.asarray(data["solver_speed"]) - np.asarray(data["star_speed"])
    )
    representative = np.asarray(
        data["representative_normalized_velocity_error_percent"], dtype=float
    )
    relative = np.asarray(data["relative_velocity_error_percent"], dtype=float)
    valid_relative = valid & np.isfinite(relative)
    valid_representative = valid & np.isfinite(representative)
    return {
        **base,
        "relative_l2_error": (
            difference_norm / reference_norm * 100.0 if reference_norm > 0.0 else np.nan
        ),
        "mae": float(np.mean(absolute_speed_error[valid])),
        "max_absolute_error": float(np.max(absolute_speed_error[valid])),
        "normalized_mae": float(np.mean(representative[valid_representative])),
        "mean_relative_velocity_error_percent": float(np.mean(relative[valid_relative])),
        "max_relative_velocity_error_percent": float(np.max(relative[valid_relative])),
        "p95_relative_velocity_error_percent": float(np.percentile(relative[valid_relative], 95)),
        "over_limit_point_ratio_percent": float(np.mean(relative[valid_relative] > display_max) * 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create relative velocity error colormaps between FEM and STAR-CCM+ data."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()
    config = load_config(config_path)
    try:
        sections = resolve_sections_from_config(config, config_path=config_path)
    except SectionConfigError as exc:
        raise ValueError(f"Invalid section configuration: {exc}") from exc

    output_dir = resolve_output_dir(config, "output/fem_vs_star")
    solver_config = config.get("solver", {})
    star_config = config.get("star_ccm", {})
    solver_velocity_name = solver_config.get("velocity_name", "solution_velocity")
    star_velocity_name = star_config.get("velocity_name", "Velocity")
    zero_speed_tolerance = float(config.get("zero_speed_tolerance", 1.0e-9))
    comparison_config = config.get("comparison", {})
    error_visualization = config.get("error_visualization", {})
    display_max = float(error_visualization.get("display_max_percent", 10.0))
    plot_config = config.get("plot", {})
    execution = config.get("execution", {})
    fail_fast = bool(execution.get("fail_fast", True))
    comparison_name = str(config.get("comparison_name", "fem_vs_star"))

    case_file = resolve_path(star_config.get("case_file"))
    if case_file is None or not case_file.exists():
        raise FileNotFoundError(f"Case file not found: {case_file}")
    reader = pv.get_reader(case_file)
    time_pairs = build_time_pairs(config, reader)
    output_config = config.get("output", {})
    save_vtp_enabled = bool(output_config.get("save_vtp", True))
    grid_output_format = resolve_grid_output_format(output_config)
    minimum_valid = float(comparison_config.get("minimum_common_valid_fraction", 0.0))

    print(f"Configuration: {config_path}")
    print(f"Section library: {config.get('section_library', '(inline/legacy)')}")
    print(f"Resolved sections: {len(sections)}")
    print(f"Section processing order: {[section['name'] for section in sections]}")
    print(f"FEM input: {solver_config}")
    print(f"STAR input: {case_file}")
    print(f"Output root: {output_dir}")

    summaries: list[dict] = []
    failures: list[dict[str, str]] = []
    per_section: dict[str, list[dict]] = {section["name"]: [] for section in sections}
    for pair in time_pairs:
        time_index = int(pair.get("label", pair.get("star_time_point", 0)))
        solver_vtu = resolve_path(pair["solver_vtu"])
        star_time_point = int(pair["star_time_point"])
        if solver_vtu is None or not solver_vtu.exists():
            raise FileNotFoundError(f"Solver VTU not found: {solver_vtu}")

        print(f"Reading time index {time_index}: solver={solver_vtu}, star_time={star_time_point}")
        solver_mesh = read_solver_mesh_for_sampling(solver_vtu, solver_config)
        star_mesh = read_star_mesh_for_sampling(reader, star_time_point, star_config)
        time_value = physical_time_value(time_index, config)
        suffix = "" if len(time_pairs) == 1 else f"_{time_index:06d}"

        for section in sections:
            section_name_value = section["name"]
            section_dir = output_dir / section_name_value
            try:
                section_dir.mkdir(parents=True, exist_ok=True)
                print(f"Processing section: {section_name_value} ({section['label']})")
                print(f"  center: {section['center'].tolist()}")
                print(f"  normalized normal: {section['normalized_normal'].tolist()}")
                print(f"  s_axis: {section['s_axis'].tolist()}")
                print(f"  t_axis: {section['t_axis'].tolist()}")
                print(f"  width/height: {section['width']} / {section['height']}")
                print(f"  grid_resolution: {list(section['grid_resolution'])}")
                print(f"  output: {section_dir}")
                points, ss, tt, _grid_points = make_section_points(section)
                method = str(comparison_config.get("method", "direct_sample")).lower()
                if method in {"slice", "section", "slice_interpolated", "section_interpolated"}:
                    data = compute_slice_interpolated_relative_error(
                        solver_mesh, solver_velocity_name, star_mesh, star_velocity_name,
                        section, ss, tt, zero_speed_tolerance, comparison_config,
                    )
                else:
                    data = compute_relative_error(
                        solver_mesh, solver_velocity_name,
                        solver_config.get("data_association", "point"),
                        star_mesh, star_velocity_name,
                        star_config.get("data_association", "cell"),
                        points, zero_speed_tolerance, comparison_config,
                    )
                valid_fraction = float(np.mean(data["valid_mask"]))
                if valid_fraction < minimum_valid:
                    raise RuntimeError(
                        f"common valid fraction {valid_fraction:.6f} is below "
                        f"configured minimum {minimum_valid:.6f}"
                    )
                if grid_output_format != "none":
                    frame = (
                        make_grid_dataframe(points, ss, tt, data)
                        if grid_output_format in {"csv", "csv.gz"} else None
                    )
                    saved_grid = save_grid_output(
                        section_dir / f"comparison_grid{suffix}",
                        grid_output_format,
                        frame,
                        {
                            "s_grid": ss,
                            "t_grid": tt,
                            "source_1_velocity": data["solver_velocity"],
                            "source_2_velocity": data["star_velocity"],
                            "source_1_speed": data["solver_speed"],
                            "source_2_speed": data["star_speed"],
                            "absolute_difference": np.abs(data["solver_speed"] - data["star_speed"]),
                            "normalized_difference": data["representative_normalized_velocity_error_percent"],
                            "common_valid_mask": data["valid_mask"],
                        },
                        {
                            "comparison_name": comparison_name,
                            "section_name": section_name_value,
                            "section_label": section["label"],
                            "center": section["center"].tolist(),
                            "normal": section["normalized_normal"].tolist(),
                            "source_1": "fem",
                            "source_2": "star_ccm",
                            "reference": "star_ccm",
                            "solver_vtu": str(solver_vtu),
                            "star_case": str(case_file),
                        },
                    )
                    print(f"  grid output: {saved_grid}")
                save_section_pngs(
                    section_dir, suffix, ss, tt, data, section,
                    time_index, time_value, plot_config, error_visualization,
                )
                if save_vtp_enabled:
                    save_vtp(
                        section_dir / f"comparison_grid{suffix}.vtp",
                        points, data, time_index, time_value, section,
                    )
                row = summarize(
                    comparison_name, section, time_index, time_value,
                    data, display_max,
                )
                summaries.append(row)
                per_section[section_name_value].append(row)
                print(f"Completed {section_name_value}: valid_fraction={valid_fraction:.6f}")
            except Exception as exc:
                failures.append(
                    {"section_name": section_name_value, "time_index": str(time_index),
                     "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"FAILED {section_name_value}: {type(exc).__name__}: {exc}")
                if fail_fast:
                    raise

    for section_name_value, rows in per_section.items():
        if rows:
            pd.DataFrame(rows).to_csv(output_dir / section_name_value / "metrics.csv", index=False)
    summary_path = output_dir / "summary_metrics.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)
    print(f"Successful sections: {len(summaries)}")
    print(f"Failed sections: {len(failures)}")
    print(f"Output directory: {output_dir}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
