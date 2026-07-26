#!/usr/bin/env python3
"""Audit FEM/STAR inlet and pre-branch velocity profiles without new CFD runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from audit_boundary_flow import (
    _find_named_block,
    _map_point_vectors_by_coordinate,
    _scaled_copy,
    load_source,
    select_boundary,
)
from profile_metrics import (
    ProfileMetricsError,
    area_weighted_histogram,
    classify_fem_inlet_nodes,
    common_grid_profile_metrics,
    compute_profile_metrics,
)
from relative_error_colormap_solver_star_ccm import (
    average_duplicate_section_samples,
    extract_section_velocity_samples,
    interpolate_section_vectors,
    make_section_points,
)
from section_config import (
    SectionConfigError,
    load_section_library,
    resolve_section,
    validate_section,
)
from surface_flow import (
    SurfaceFlowError,
    integrate_native_volume_cell_section,
    integrate_point_surface,
    polygon_area_centroid_2d,
    triangulate_surface,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "audit_inlet_upstream_profile.json"


class ProfileAuditError(ValueError):
    """Raised for an invalid 4B-2 profile audit configuration or input."""


def resolve_config_path(value: str | Path, config_path: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config_path).resolve().parent / path).resolve()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileAuditError(f"{context} must be a JSON object")
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_sections(config: Mapping[str, Any], config_path: Path) -> dict[str, dict[str, Any]]:
    library_path = resolve_config_path(str(config.get("section_library", "")), config_path)
    library = load_section_library(library_path)
    locations = _mapping(config.get("locations"), "locations")
    resolved: dict[str, dict[str, Any]] = {}
    for name, raw_value in locations.items():
        raw = dict(_mapping(raw_value, f"locations.{name}"))
        location_type = str(raw.get("type", ""))
        if location_type == "boundary":
            section_raw = dict(_mapping(raw.get("section"), f"locations.{name}.section"))
            section = validate_section(section_raw)
            section["name"] = name
        elif location_type == "internal_section":
            reference = str(raw.get("section_name", ""))
            section = resolve_section(reference, library)
        else:
            raise ProfileAuditError(f"locations.{name}.type must be boundary or internal_section")
        flow = np.asarray(raw.get("flow_direction_normal"), dtype=float)
        if flow.shape != (3,) or not np.all(np.isfinite(flow)) or np.linalg.norm(flow) == 0.0:
            raise ProfileAuditError(f"locations.{name}.flow_direction_normal must be non-zero")
        flow /= np.linalg.norm(flow)
        if abs(abs(float(np.dot(flow, section["normalized_normal"]))) - 1.0) > 1.0e-8:
            raise ProfileAuditError(f"locations.{name}: flow direction must be collinear with section normal")
        section = dict(section)
        section["location_label"] = raw.get("label", section["label"])
        section["location_type"] = location_type
        section["boundary_name"] = raw.get("boundary_name")
        section["flow_direction_normal"] = flow
        resolved[name] = section
    required = {"inlet", "upstream_z30"}
    if set(resolved) != required:
        raise ProfileAuditError(f"locations must contain exactly {sorted(required)} for step 4B-2")
    return resolved


def point_surface_quadrature(
    surface: pv.DataSet, velocity_name: str, section: Mapping[str, Any]
) -> dict[str, np.ndarray | int]:
    """Positive degree-4 triangle quadrature, exact through cubic linear-velocity moments."""

    triangles, connectivity = triangulate_surface(surface)
    if velocity_name not in triangles.point_data:
        raise ProfileAuditError(f"point velocity {velocity_name!r} missing from surface")
    points = np.asarray(triangles.points, dtype=float)
    velocity = np.asarray(triangles.point_data[velocity_name], dtype=float)
    vertices = points[connectivity]
    triangle_velocity = velocity[connectivity]
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1
    )
    valid = np.isfinite(triangle_areas) & (triangle_areas > 0.0)
    if not np.all(valid):
        vertices = vertices[valid]
        triangle_velocity = triangle_velocity[valid]
        triangle_areas = triangle_areas[valid]
    barycentric = np.asarray([
        [0.816847572980459, 0.091576213509771, 0.091576213509771],
        [0.091576213509771, 0.816847572980459, 0.091576213509771],
        [0.091576213509771, 0.091576213509771, 0.816847572980459],
        [0.108103018168070, 0.445948490915965, 0.445948490915965],
        [0.445948490915965, 0.108103018168070, 0.445948490915965],
        [0.445948490915965, 0.445948490915965, 0.108103018168070],
    ])
    quadrature_weights = np.asarray([
        0.109951743655322, 0.109951743655322, 0.109951743655322,
        0.223381589678011, 0.223381589678011, 0.223381589678011,
    ])
    sample_points = np.einsum("qv,tvc->tqc", barycentric, vertices).reshape((-1, 3))
    sample_velocity = np.einsum("qv,tvc->tqc", barycentric, triangle_velocity).reshape((-1, 3))
    sample_areas = (triangle_areas[:, None] * quadrature_weights[None, :]).reshape(-1)
    relative = sample_points - np.asarray(section["center"], dtype=float)
    return {
        "areas": sample_areas,
        "velocities": sample_velocity,
        "s": relative @ np.asarray(section["s_axis"], dtype=float),
        "t": relative @ np.asarray(section["t_axis"], dtype=float),
        "element_count": int(len(triangle_areas)),
        "sample_count": int(len(sample_areas)),
    }


def cell_surface_samples(
    surface: pv.DataSet, velocity_name: str, section: Mapping[str, Any]
) -> dict[str, np.ndarray | int]:
    if velocity_name not in surface.cell_data:
        raise ProfileAuditError(f"cell velocity {velocity_name!r} missing from surface")
    velocity = np.asarray(surface.cell_data[velocity_name], dtype=float)
    if velocity.shape != (surface.n_cells, 3):
        raise ProfileAuditError("surface cell velocity must have three components")
    center = np.asarray(section["center"], dtype=float)
    s_axis = np.asarray(section["s_axis"], dtype=float)
    t_axis = np.asarray(section["t_axis"], dtype=float)
    areas: list[float] = []
    centroids_s: list[float] = []
    centroids_t: list[float] = []
    kept_velocity: list[np.ndarray] = []
    for cell_id in range(surface.n_cells):
        relative = np.asarray(surface.get_cell(cell_id).points, dtype=float) - center
        polygon = np.column_stack((relative @ s_axis, relative @ t_axis))
        try:
            area, centroid = polygon_area_centroid_2d(polygon)
        except SurfaceFlowError:
            continue
        areas.append(area)
        centroids_s.append(float(centroid[0]))
        centroids_t.append(float(centroid[1]))
        kept_velocity.append(velocity[cell_id])
    if not areas:
        raise ProfileAuditError("surface has no valid polygon cells")
    return {
        "areas": np.asarray(areas), "velocities": np.asarray(kept_velocity),
        "s": np.asarray(centroids_s), "t": np.asarray(centroids_t),
        "element_count": len(areas), "sample_count": len(areas),
    }


def native_record_samples(records: list[dict[str, Any]], section: Mapping[str, Any]) -> dict[str, np.ndarray | int]:
    valid = [record for record in records if record["valid"]]
    center = np.asarray(section["center"], dtype=float)
    s_axis = np.asarray(section["s_axis"], dtype=float)
    t_axis = np.asarray(section["t_axis"], dtype=float)
    centroids = np.asarray([[r["polygon_centroid_x_mm"], r["polygon_centroid_y_mm"], r["polygon_centroid_z_mm"]] for r in valid])
    relative = centroids - center
    return {
        "areas": np.asarray([r["polygon_area_mm2"] for r in valid]),
        "velocities": np.asarray([[r["velocity_x_mm_s"], r["velocity_y_mm_s"], r["velocity_z_mm_s"]] for r in valid]),
        "s": relative @ s_axis, "t": relative @ t_axis,
        "element_count": len(valid), "sample_count": len(valid),
    }


def fem_internal_section_samples(
    mesh: pv.DataSet, velocity_name: str, section: Mapping[str, Any]
) -> dict[str, np.ndarray | int]:
    """Extract a configured FEM internal section using the established clip rule."""

    cut = mesh.slice(
        origin=np.asarray(section["center"], dtype=float),
        normal=np.asarray(section["normalized_normal"], dtype=float),
    )
    if cut.n_points == 0:
        raise ProfileAuditError(f"{section.get('name', 'section')}: FEM slice is empty")
    relative = np.asarray(cut.points) - np.asarray(section["center"], dtype=float)
    keep = (
        np.abs(relative @ np.asarray(section["s_axis"], dtype=float))
        <= 0.5 * float(section["width"])
    ) & (
        np.abs(relative @ np.asarray(section["t_axis"], dtype=float))
        <= 0.5 * float(section["height"])
    )
    clipped = cut.extract_points(keep, adjacent_cells=False)
    if clipped.n_cells == 0:
        raise ProfileAuditError(f"{section.get('name', 'section')}: FEM clipped slice is empty")
    return point_surface_quadrature(clipped, velocity_name, section)


def star_internal_section_samples(
    mesh: pv.DataSet,
    velocity_name: str,
    section: Mapping[str, Any],
    minimum_polygon_area_mm2: float = 1.0e-12,
) -> tuple[dict[str, np.ndarray | int], dict[str, Any], list[dict[str, Any]], pv.PolyData, dict[str, Any]]:
    """Return formal native STAR samples plus full 4B-1 diagnostics."""

    metrics, records, polydata, diagnostics = integrate_native_volume_cell_section(
        mesh,
        velocity_name,
        section["center"],
        section["normalized_normal"],
        section["s_axis"],
        section["t_axis"],
        float(section["width"]),
        float(section["height"]),
        minimum_polygon_area_mm2=minimum_polygon_area_mm2,
        validate_original_cell_ids=True,
        fail_on_unmapped_polygon=True,
        clip_to_section_window=True,
    )
    return native_record_samples(records, section), metrics, records, polydata, diagnostics


def compute_location_profile(
    solver: str, location: str, samples: Mapping[str, Any], section: Mapping[str, Any], low_fraction: float
) -> dict[str, Any]:
    metrics = compute_profile_metrics(
        samples["areas"], samples["velocities"], samples["s"], samples["t"],
        section["normalized_normal"], section["flow_direction_normal"],
        section["s_axis"], section["t_axis"], low_fraction,
    )
    return {
        "solver": solver, "location": location,
        "section_normal_x": float(section["normalized_normal"][0]),
        "section_normal_y": float(section["normalized_normal"][1]),
        "section_normal_z": float(section["normalized_normal"][2]),
        "flow_direction_normal_x": float(section["flow_direction_normal"][0]),
        "flow_direction_normal_y": float(section["flow_direction_normal"][1]),
        "flow_direction_normal_z": float(section["flow_direction_normal"][2]),
        "element_count": int(samples["element_count"]),
        **metrics, "status": "success", "warning": "",
    }


def _surface_point_interpolation_samples(
    surface: pv.DataSet, velocity_name: str, association: str, section: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if association == "cell":
        surface = surface.cell_data_to_point_data(pass_cell_data=True)
    if velocity_name not in surface.point_data:
        raise ProfileAuditError(f"visualization point velocity {velocity_name!r} is missing")
    poly = surface.extract_surface(algorithm="dataset_surface")
    relative = np.asarray(poly.points) - np.asarray(section["center"])
    s = relative @ np.asarray(section["s_axis"])
    t = relative @ np.asarray(section["t_axis"])
    velocity = np.asarray(poly.point_data[velocity_name], dtype=float)
    keep = (np.abs(s) <= 0.5 * float(section["width"]) + 1e-10) & (np.abs(t) <= 0.5 * float(section["height"]) + 1e-10)
    return average_duplicate_section_samples(s[keep], t[keep], velocity[keep])


def common_grid_for_location(
    location: str,
    section: Mapping[str, Any],
    fem_mesh: pv.DataSet,
    fem_surface: pv.DataSet,
    fem_velocity_name: str,
    star_mesh: pv.DataSet,
    star_surface: pv.DataSet,
    star_velocity_name: str,
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _, ss, tt, _ = make_section_points(dict(section))
    if location == "inlet":
        fem_s, fem_t, fem_values = _surface_point_interpolation_samples(fem_surface, fem_velocity_name, "point", section)
        star_s, star_t, star_values = _surface_point_interpolation_samples(star_surface, star_velocity_name, "cell", section)
    else:
        fem_s, fem_t, fem_values = extract_section_velocity_samples(fem_mesh, fem_velocity_name, dict(section))
        star_s, star_t, star_values = extract_section_velocity_samples(star_mesh, star_velocity_name, dict(section))
    fem_velocity, fem_valid = interpolate_section_vectors(fem_s, fem_t, fem_values, ss, tt)
    star_velocity, star_valid = interpolate_section_vectors(star_s, star_t, star_values, ss, tt)
    n_section = np.asarray(section["normalized_normal"])
    n_flow = np.asarray(section["flow_direction_normal"])
    s_axis = np.asarray(section["s_axis"])
    t_axis = np.asarray(section["t_axis"])
    fem_signed = fem_velocity @ n_section
    star_signed = star_velocity @ n_section
    fem_flow = fem_velocity @ n_flow
    star_flow = star_velocity @ n_flow
    fem_secondary = np.sqrt((fem_velocity @ s_axis) ** 2 + (fem_velocity @ t_axis) ** 2)
    star_secondary = np.sqrt((star_velocity @ s_axis) ** 2 + (star_velocity @ t_axis) ** 2)
    fem_mean = float(profiles[("fem", location)]["mean_flow_velocity_mm_s"])
    star_mean = float(profiles[("star", location)]["mean_flow_velocity_mm_s"])
    metrics, common_valid = common_grid_profile_metrics(
        fem_flow, star_flow, fem_secondary, star_secondary, fem_valid, star_valid,
        fem_mean, star_mean, float(section["width"]), float(section["height"]),
        tuple(section["grid_resolution"]),
    )
    metrics.update({"location": location, "interpolation_method": "linear_tri_point_visualization", "status": "success", "warning": "STAR cell-to-point smoothing is used only for the visualization/common grid."})
    arrays = {
        "s_grid": ss, "t_grid": tt,
        "fem_velocity_mm_s": fem_velocity.reshape((*ss.shape, 3)),
        "star_velocity_mm_s": star_velocity.reshape((*ss.shape, 3)),
        "fem_signed_normal_velocity_mm_s": fem_signed.reshape(ss.shape),
        "star_signed_normal_velocity_mm_s": star_signed.reshape(ss.shape),
        "fem_flow_velocity_mm_s": fem_flow.reshape(ss.shape),
        "star_flow_velocity_mm_s": star_flow.reshape(ss.shape),
        "fem_normalized_flow_velocity": (fem_flow / fem_mean).reshape(ss.shape),
        "star_normalized_flow_velocity": (star_flow / star_mean).reshape(ss.shape),
        "fem_secondary_speed_mm_s": fem_secondary.reshape(ss.shape),
        "star_secondary_speed_mm_s": star_secondary.reshape(ss.shape),
        "fem_valid": fem_valid.reshape(ss.shape), "star_valid": star_valid.reshape(ss.shape),
        "common_valid": common_valid.reshape(ss.shape),
    }
    return metrics, arrays


def _boundary_perimeter_mask(surface: pv.DataSet, tolerance: float) -> np.ndarray:
    tri = surface.extract_surface(algorithm="dataset_surface").triangulate()
    faces = np.asarray(tri.faces).reshape((-1, 4))[:, 1:]
    counts: Counter[tuple[int, int]] = Counter()
    for a, b, c in faces:
        for first, second in ((a, b), (b, c), (c, a)):
            counts[tuple(sorted((int(first), int(second))))] += 1
    boundary_points = {point for edge, count in counts.items() if count == 1 for point in edge}
    key = lambda point: tuple(np.rint(np.asarray(point) / tolerance).astype(np.int64))
    keys = {key(tri.points[index]) for index in boundary_points}
    return np.asarray([key(point) in keys for point in surface.points], dtype=bool)


def _stats(prefix: str, values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {f"{prefix}_count": 0, f"{prefix}_minimum_mm_s": math.nan, f"{prefix}_maximum_mm_s": math.nan, f"{prefix}_mean_mm_s": math.nan, f"{prefix}_std_mm_s": math.nan}
    return {f"{prefix}_count": int(len(values)), f"{prefix}_minimum_mm_s": float(np.min(values)), f"{prefix}_maximum_mm_s": float(np.max(values)), f"{prefix}_mean_mm_s": float(np.mean(values)), f"{prefix}_std_mm_s": float(np.std(values))}


def fem_inlet_node_audit(
    surface: pv.DataSet,
    velocity_name: str,
    section: Mapping[str, Any],
    source: Mapping[str, Any],
    settings: Mapping[str, Any],
    config_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    velocity = np.asarray(surface.point_data[velocity_name], dtype=float)
    flow_velocity = velocity @ np.asarray(section["flow_direction_normal"])
    tolerance = float(settings["coordinate_tolerance_mm"])
    perimeter = _boundary_perimeter_mask(surface, tolerance)
    mesh_path = resolve_config_path(str(source["boundary_mesh_path"]), config_path)
    # audit_boundary_flow uses repository-prefixed paths; preserve that established rule.
    if not mesh_path.is_file() and str(source["boundary_mesh_path"]).startswith("data/"):
        mesh_path = (ROOT / str(source["boundary_mesh_path"])).resolve()
    mesh = pv.read(mesh_path)
    ids = np.asarray(mesh.cell_data[str(source.get("boundary_id_array", "gmsh:physical"))])
    wall_ids = np.flatnonzero(ids == int(settings["wall_boundary_id"]))
    wall = mesh.extract_cells(wall_ids).extract_surface(algorithm="dataset_surface")
    key = lambda point: tuple(np.rint(np.asarray(point) / tolerance).astype(np.int64))
    wall_keys = {key(point) for point in wall.points}
    shared = np.asarray([key(point) in wall_keys for point in surface.points], dtype=bool)
    labels, counts = classify_fem_inlet_nodes(
        flow_velocity, perimeter, shared, float(settings["expected_velocity_mm_s"]),
        float(settings["velocity_tolerance_mm_s"]),
    )
    relative = np.asarray(surface.points) - np.asarray(section["center"])
    s = relative @ np.asarray(section["s_axis"])
    t = relative @ np.asarray(section["t_axis"])
    rows = []
    for index in range(surface.n_points):
        rows.append({
            "point_id": index, "x_mm": surface.points[index, 0], "y_mm": surface.points[index, 1], "z_mm": surface.points[index, 2],
            "s_mm": s[index], "t_mm": t[index],
            "velocity_x_mm_s": velocity[index, 0], "velocity_y_mm_s": velocity[index, 1], "velocity_z_mm_s": velocity[index, 2],
            "speed_mm_s": float(np.linalg.norm(velocity[index])), "signed_normal_velocity_mm_s": float(velocity[index] @ np.asarray(section["normalized_normal"])),
            "flow_direction_velocity_mm_s": flow_velocity[index], "velocity_class": labels[index],
            "is_perimeter": bool(perimeter[index]), "is_interior": bool(not perimeter[index]), "is_inlet_wall_shared": bool(shared[index]),
        })
    actual = integrate_point_surface(surface, velocity_name, section["normalized_normal"])
    expected = float(settings["expected_velocity_mm_s"])
    uniform = surface.copy(deep=True)
    uniform.point_data["audit_uniform_velocity"] = np.tile(expected * np.asarray(section["flow_direction_normal"]), (surface.n_points, 1))
    uniform_flow = integrate_point_surface(uniform, "audit_uniform_velocity", section["normalized_normal"])
    shared_zero = uniform.copy(deep=True)
    reconstructed = np.asarray(shared_zero.point_data["audit_uniform_velocity"]).copy()
    reconstructed[shared] = 0.0
    shared_zero.point_data["audit_shared_zero_velocity"] = reconstructed
    reconstructed_flow = integrate_point_surface(shared_zero, "audit_shared_zero_velocity", section["normalized_normal"])
    summary = {
        **counts,
        "triangle_count": int(surface.extract_surface(algorithm="dataset_surface").triangulate().n_cells),
        "perimeter_equals_wall_shared": bool(np.array_equal(perimeter, shared)),
        "actual_signed_flow_mm3_s": float(actual["signed_flow_rate_mm3_s"]),
        "actual_flow_magnitude_mm3_s": abs(float(actual["signed_flow_rate_mm3_s"])),
        "uniform_theoretical_flow_magnitude_mm3_s": abs(float(uniform_flow["signed_flow_rate_mm3_s"])),
        "shared_zero_reconstructed_flow_magnitude_mm3_s": abs(float(reconstructed_flow["signed_flow_rate_mm3_s"])),
        "actual_vs_shared_zero_difference_mm3_s": abs(float(actual["signed_flow_rate_mm3_s"]) - float(reconstructed_flow["signed_flow_rate_mm3_s"])),
        **_stats("perimeter_flow_velocity", flow_velocity[perimeter]),
        **_stats("interior_flow_velocity", flow_velocity[~perimeter]),
        **_stats("inlet_wall_shared_flow_velocity", flow_velocity[shared]),
    }
    return rows, summary


def star_inlet_uniformity(profile: Mapping[str, Any], samples: Mapping[str, Any], section: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    velocity = np.asarray(samples["velocities"])
    u_flow = velocity @ np.asarray(section["flow_direction_normal"])
    expected = float(settings["expected_velocity_mm_s"])
    tolerance = float(settings["velocity_tolerance_mm_s"])
    difference = u_flow - expected
    within = np.abs(difference) <= tolerance
    return {
        "surface_cell_count": int(len(u_flow)),
        "minimum_flow_velocity_mm_s": float(np.min(u_flow)),
        "maximum_flow_velocity_mm_s": float(np.max(u_flow)),
        "mean_flow_velocity_mm_s": float(np.average(u_flow, weights=samples["areas"])),
        "std_flow_velocity_mm_s": float(np.sqrt(np.average((u_flow - np.average(u_flow, weights=samples["areas"]))**2, weights=samples["areas"]))),
        "minimum_difference_from_expected_mm_s": float(np.min(difference)),
        "maximum_difference_from_expected_mm_s": float(np.max(difference)),
        "uniform_cell_fraction_percent": 100.0 * float(np.mean(within)),
        "all_cells_uniform_within_tolerance": bool(np.all(within)),
        "tolerance_mm_s": tolerance,
        "profile_uniformity_coefficient_of_variation_percent": 100.0 * float(profile["velocity_std_mm_s"]) / float(profile["mean_flow_velocity_mm_s"]),
        "beta": profile["beta"], "alpha": profile["alpha"],
        "reverse_flow_area_mm2": profile["reverse_flow_area_mm2"],
        "rms_secondary_velocity_mm_s": profile["rms_secondary_velocity_mm_s"],
    }


def comparison_row(location: str, fem: Mapping[str, Any], star: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"location": location}
    fields = [
        "area_mm2", "flow_magnitude_mm3_s", "mean_flow_velocity_mm_s", "velocity_std_mm_s",
        "normalized_velocity_std", "beta", "alpha", "reverse_flow_area_fraction_percent",
        "low_velocity_area_fraction_percent", "flux_centroid_s_mm", "flux_centroid_t_mm",
        "flux_centroid_offset_mm", "rms_secondary_velocity_mm_s", "secondary_velocity_ratio_percent",
    ]
    for field in fields:
        first = float(fem[field]); second = float(star[field])
        row[f"fem_{field}"] = first; row[f"star_{field}"] = second
        row[f"signed_difference_{field}"] = first - second
        row[f"absolute_difference_{field}"] = abs(first - second)
        row[f"relative_difference_percent_{field}"] = 100.0 * abs(first - second) / max(abs(second), 1.0e-12)
    return row


def _masked(array: np.ndarray, valid: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.array(array, mask=~valid)


def plot_location(location: str, arrays: Mapping[str, np.ndarray], output_dir: Path, plot: Mapping[str, Any], hist_samples: Mapping[str, Mapping[str, Any]], bins: int) -> None:
    target = output_dir / "figures" / ("inlet" if location == "inlet" else "upstream_z30")
    target.mkdir(parents=True, exist_ok=True)
    ss=np.asarray(arrays["s_grid"]); tt=np.asarray(arrays["t_grid"]); valid=np.asarray(arrays["common_valid"],bool)
    fem_flow=np.asarray(arrays["fem_flow_velocity_mm_s"]); star_flow=np.asarray(arrays["star_flow_velocity_mm_s"])
    fem_norm=np.asarray(arrays["fem_normalized_flow_velocity"]); star_norm=np.asarray(arrays["star_normalized_flow_velocity"])
    fem_sec=np.asarray(arrays["fem_secondary_speed_mm_s"]); star_sec=np.asarray(arrays["star_secondary_speed_mm_s"])
    dpi=int(plot.get("dpi",180)); cmap=str(plot.get("cmap","viridis")); dcmap=str(plot.get("difference_cmap","coolwarm"))
    prefix="inlet" if location=="inlet" else "upstream"

    def one_field(values: np.ndarray, filename: str, title: str, vmin: float, vmax: float, color_map: str=cmap) -> None:
        fig,ax=plt.subplots(figsize=(6,4)); m=ax.pcolormesh(ss,tt,_masked(values,valid),shading="auto",cmap=color_map,vmin=vmin,vmax=vmax); fig.colorbar(m,ax=ax);ax.set(xlabel="s [mm]",ylabel="t [mm]",title=title,aspect="equal");fig.tight_layout();fig.savefig(target/filename,dpi=dpi);plt.close(fig)
    flow_min=float(np.nanmin(np.concatenate([fem_flow[valid],star_flow[valid]]))); flow_max=float(np.nanmax(np.concatenate([fem_flow[valid],star_flow[valid]])))
    one_field(fem_flow, f"fem_{prefix}_flow_velocity.png", f"FEM {location} flow velocity [mm/s]", flow_min, flow_max)
    one_field(star_flow, f"star_{prefix}_flow_velocity.png", f"STAR {location} flow velocity [mm/s]", flow_min, flow_max)
    norm_min=float(np.nanmin(np.concatenate([fem_norm[valid],star_norm[valid]]))); norm_max=float(np.nanmax(np.concatenate([fem_norm[valid],star_norm[valid]])))
    fig,axes=plt.subplots(1,2,figsize=(10,4));
    for ax,data,title in zip(axes,(fem_norm,star_norm),("FEM","STAR")):
        m=ax.pcolormesh(ss,tt,_masked(data,valid),shading="auto",cmap=cmap,vmin=norm_min,vmax=norm_max);ax.set(xlabel="s [mm]",ylabel="t [mm]",title=title,aspect="equal")
    fig.colorbar(m,ax=axes.tolist());fig.suptitle(f"{location} normalized flow velocity");fig.savefig(target/f"{prefix}_normalized_velocity_comparison.png",dpi=dpi,bbox_inches="tight");plt.close(fig)
    difference=fem_norm-star_norm; limit=float(np.nanmax(np.abs(difference[valid])));one_field(difference,f"{prefix}_normalized_velocity_difference.png",f"{location} FEM - STAR normalized velocity",-limit,limit,dcmap)
    sec_max=float(np.nanmax(np.concatenate([fem_sec[valid],star_sec[valid]])));fig,axes=plt.subplots(1,2,figsize=(10,4));
    for ax,data,title in zip(axes,(fem_sec,star_sec),("FEM","STAR")):
        m=ax.pcolormesh(ss,tt,_masked(data,valid),shading="auto",cmap=cmap,vmin=0,vmax=sec_max);ax.set(xlabel="s [mm]",ylabel="t [mm]",title=title,aspect="equal")
    fig.colorbar(m,ax=axes.tolist());fig.suptitle(f"{location} secondary speed [mm/s]");fig.savefig(target/f"{prefix}_secondary_velocity_comparison.png",dpi=dpi,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for solver in ("fem","star"):
        sample=hist_samples[solver];vel=np.asarray(sample["velocities"])@np.asarray(sample["flow_normal"]); area_fraction, fraction, edges=area_weighted_histogram(vel,sample["areas"],bins=bins); centers=.5*(edges[:-1]+edges[1:]);ax.step(centers,100*fraction,where="mid",label=solver.upper())
    ax.set(xlabel="flow velocity [mm/s]",ylabel="area fraction per bin [%]",title=f"{location} area-weighted histogram");ax.legend();ax.grid(True,alpha=.3);fig.tight_layout();fig.savefig(target/f"{prefix}_area_weighted_histogram.png",dpi=dpi);plt.close(fig)


def plot_node_classification(rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    target=output_dir/"figures/inlet"; target.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(6,4));
    for label,marker in (("expected","o"),("zero","s"),("intermediate","x")):
        selected=[r for r in rows if r["velocity_class"]==label];ax.scatter([r["s_mm"] for r in selected],[r["t_mm"] for r in selected],s=18,marker=marker,label=label)
    ax.set(xlabel="s [mm]",ylabel="t [mm]",title="FEM inlet boundary node classification",aspect="equal");ax.legend();fig.tight_layout();fig.savefig(target/"fem_inlet_boundary_node_classification.png",dpi=dpi);plt.close(fig)


def plot_overview(profiles: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    target=output_dir/"figures/overview";target.mkdir(parents=True,exist_ok=True);locations=["inlet","upstream_z30"];x=np.arange(len(locations));width=.36
    by={(r["solver"],r["location"]):r for r in profiles}
    metrics=[("mean_flow_velocity_mm_s","Mean flow velocity"),("normalized_velocity_std","Normalized velocity std"),("secondary_velocity_ratio_percent","Secondary / mean [%]")]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(field,title) in zip(axes,metrics):
        ax.bar(x-width/2,[by[("fem",l)][field] for l in locations],width,label="FEM");ax.bar(x+width/2,[by[("star",l)][field] for l in locations],width,label="STAR");ax.set_xticks(x,locations,rotation=15);ax.set_title(title);ax.grid(True,axis="y",alpha=.3)
    axes[0].legend();fig.tight_layout();fig.savefig(target/"profile_metrics_comparison.png",dpi=dpi);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,field,title in zip(axes,("beta","alpha"),("Momentum correction β","Kinetic-energy correction α")):
        ax.bar(x-width/2,[by[("fem",l)][field] for l in locations],width,label="FEM");ax.bar(x+width/2,[by[("star",l)][field] for l in locations],width,label="STAR");ax.set_xticks(x,locations,rotation=15);ax.set_title(title);ax.grid(True,axis="y",alpha=.3)
    axes[0].legend();fig.tight_layout();fig.savefig(target/"alpha_beta_comparison.png",dpi=dpi);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,location in zip(axes,locations):
        for solver,marker in (("fem","o"),("star","s")):
            row=by[(solver,location)];ax.scatter(row["flux_centroid_s_mm"],row["flux_centroid_t_mm"],marker=marker,label=solver.upper());ax.plot([0,row["flux_centroid_s_mm"]],[0,row["flux_centroid_t_mm"]],alpha=.5)
        ax.axhline(0,color="gray",lw=.7);ax.axvline(0,color="gray",lw=.7);ax.set(xlabel="s [mm]",ylabel="t [mm]",title=location,aspect="equal");ax.legend()
    fig.tight_layout();fig.savefig(target/"flux_centroid_comparison.png",dpi=dpi);plt.close(fig)


def classify_case(common_rows: list[dict[str, Any]], profiles: Mapping[tuple[str,str], Mapping[str,Any]], thresholds: Mapping[str,Any]) -> dict[str,Any]:
    grid={r["location"]:r for r in common_rows}; shape_limit=float(thresholds["normalized_velocity_relative_l2_percent"]); centroid_limit=float(thresholds["flux_centroid_difference_mm"]); sec_limit=float(thresholds["secondary_ratio_difference_percentage_points"]); ab_limit=float(thresholds["alpha_beta_relative_difference_percent"])
    evidence={}
    for location in ("inlet","upstream_z30"):
        f=profiles[("fem",location)];s=profiles[("star",location)]
        centroid=math.hypot(float(f["flux_centroid_s_mm"])-float(s["flux_centroid_s_mm"]),float(f["flux_centroid_t_mm"])-float(s["flux_centroid_t_mm"]))
        sec=abs(float(f["secondary_velocity_ratio_percent"])-float(s["secondary_velocity_ratio_percent"]))
        ab=max(100*abs(float(f["alpha"])-float(s["alpha"]))/max(abs(float(s["alpha"])),1e-12),100*abs(float(f["beta"])-float(s["beta"]))/max(abs(float(s["beta"])),1e-12))
        evidence[location]={"normalized_l2_percent":grid[location]["normalized_velocity_relative_l2_percent"],"flux_centroid_difference_mm":centroid,"secondary_ratio_difference_percentage_points":sec,"maximum_alpha_beta_relative_difference_percent":ab,"shape_different":bool(grid[location]["normalized_velocity_relative_l2_percent"]>shape_limit or ab>ab_limit),"asymmetry_or_secondary_different":bool(centroid>centroid_limit or sec>sec_limit)}
    inlet=evidence["inlet"];up=evidence["upstream_z30"]
    if not inlet["shape_different"] and not up["shape_different"] and (inlet["asymmetry_or_secondary_different"] or up["asymmetry_or_secondary_different"]): case="D";text="総流量・主速度形状より、流量重心または二次流れの差が目立つ観測です。非対称な流入運動量が分岐配分へ影響する可能性はありますが断定しません。"
    elif inlet["shape_different"] and not up["shape_different"]: case="B";text="入口差はある一方、分岐前では設定基準内です。入口差が減衰し、主因が分岐部以降にある可能性があります。"
    elif not inlet["shape_different"] and not up["shape_different"]: case="C";text="入口・分岐前とも主速度形状は設定基準内です。入口profileだけでは分岐配分差を説明しにくい観測です。"
    else: case="A";text="入口または分岐前で主速度形状、alpha/betaに設定基準を超える差があります。入口条件または発達差が分岐配分へ影響する可能性があります。"
    return {"case":case,"interpretation":text,"thresholds":dict(thresholds),"evidence":evidence,"causation_determined":False}


def build_markdown(audit: Mapping[str,Any]) -> str:
    lines=["# 入口・分岐前速度プロファイル監査","",f"作成日: {date.today().isoformat()}","","正式流量・profile指標はFEM point三角形積分、STAR入口native surface-cell、STAR上流native volume-cell intersectionを使用した。共通グリッドは可視化・空間比較専用で、STARのcell-to-point平滑化を正式流量へ混ぜていない。","","## Profile metrics","","| Solver | Location | Area | |Q| | Umean | std/Umean | beta | alpha | centroid offset | secondary ratio |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in audit["profile_metrics"]: lines.append(f"| {r['solver']} | {r['location']} | {r['area_mm2']:.8g} | {r['flow_magnitude_mm3_s']:.8g} | {r['mean_flow_velocity_mm_s']:.8g} | {r['normalized_velocity_std']:.8g} | {r['beta']:.8g} | {r['alpha']:.8g} | {r['flux_centroid_offset_mm']:.8g} | {r['secondary_velocity_ratio_percent']:.8g}% |")
    lines.extend(["","## Common-grid shape metrics","","| Location | normalized L2 | MAE | RMSE | correlation | secondary L2 | common area |","|---|---:|---:|---:|---:|---:|---:|"])
    for r in audit["common_grid_metrics"]: lines.append(f"| {r['location']} | {r['normalized_velocity_relative_l2_percent']:.8g}% | {r['normalized_velocity_mae']:.8g} | {r['normalized_velocity_rmse']:.8g} | {r['normalized_velocity_correlation']:.8g} | {r['secondary_velocity_relative_l2_percent']:.8g}% | {r['common_valid_area_mm2']:.8g} |")
    fem=audit["fem_inlet_node_audit"];star=audit["star_inlet_uniformity"];decision=audit["decision"]
    lines.extend(["","## FEM inlet nodes","",f"入口{fem['node_count']}点のうちwall共有/外周は{fem['inlet_wall_shared_node_count']}点で、速度0 mm/s。内部{fem['interior_node_count']}点は10 mm/sだった。外周集合とwall共有集合の一致: `{fem['perimeter_equals_wall_shared']}`。",f"実流量={fem['actual_flow_magnitude_mm3_s']:.9g}、全点一様10 mm/s理論={fem['uniform_theoretical_flow_magnitude_mm3_s']:.9g}、共有点だけ0の再現={fem['shared_zero_reconstructed_flow_magnitude_mm3_s']:.9g} mm³/s。","","## STAR inlet native cells","",f"{star['surface_cell_count']} cells、10 mm/s許容差内割合={star['uniform_cell_fraction_percent']:.8g}%、全cell一様判定=`{star['all_cells_uniform_within_tolerance']}`。","","## 判定","",f"ケース **{decision['case']}**: {decision['interpretation']}","","補間は画像・共通gridだけに線形三角形補間を使用するため、STAR cell-to-pointによる平滑化の可能性がある。原因は自動的に断定しない。","","## 実行","","```bash","cd /workspace","python scripts/audit_inlet_upstream_profile.py --config config/audit_inlet_upstream_profile.json","```",""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG);args=parser.parse_args(argv)
    config_path=args.config if args.config.is_absolute() else ROOT/args.config;config_path=config_path.resolve();config=dict(_mapping(json.loads(config_path.read_text(encoding="utf-8")),"configuration"));sections=_resolve_sections(config,config_path)
    output=resolve_config_path(str(_mapping(config["output"],"output")["directory"]),config_path);execution=_mapping(config.get("execution",{}),"execution")
    if execution.get("refuse_nonempty_output_directory",True) and output.exists() and any(output.iterdir()): raise ProfileAuditError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True,exist_ok=True)
    audit_config_path=resolve_config_path(str(config["boundary_audit_config"]),config_path);audit_config=dict(_mapping(json.loads(audit_config_path.read_text(encoding="utf-8")),"boundary audit configuration"));solver_config=_mapping(config["solvers"],"solvers")
    source_names={solver:str(_mapping(raw,f"solvers.{solver}")["source_name"]) for solver,raw in solver_config.items()};loaded={};sources={}
    for solver,source_name in source_names.items():
        source=dict(_mapping(audit_config["data_sources"][source_name],f"data source {source_name}"));sources[solver]=source;loaded[solver]=load_source(source_name,source,audit_config_path)
    inlet=sections["inlet"];upstream=sections["upstream_z30"]
    fem_name=source_names["fem"];star_name=source_names["star"]
    fem_surface,_=select_boundary(fem_name,sources["fem"],loaded["fem"],str(inlet["boundary_name"]),audit_config["boundaries"][fem_name][inlet["boundary_name"]],audit_config_path)
    fem_velocity_name=str(sources["fem"]["velocity_array"]);mapped,mapping_diag=_map_point_vectors_by_coordinate(loaded["fem"]["primary"],fem_surface,fem_velocity_name,float(sources["fem"].get("point_mapping_tolerance_mm",1e-9)));fem_surface=fem_surface.copy(deep=True);fem_surface.point_data[fem_velocity_name]=mapped
    star_surface,_=select_boundary(star_name,sources["star"],loaded["star"],str(inlet["boundary_name"]),audit_config["boundaries"][star_name][inlet["boundary_name"]],audit_config_path);star_velocity_name=str(sources["star"]["velocity_array"])
    part_name=str(_mapping(solver_config["star"],"solvers.star")["volume_part_name"]);star_mesh=_scaled_copy(_find_named_block(loaded["star"]["raw"],part_name),sources["star"]);fem_mesh=loaded["fem"]["primary"]
    low=float(_mapping(config["profile"],"profile").get("low_velocity_fraction_of_mean",.1))
    sample_sets={}
    sample_sets[("fem","inlet")]=point_surface_quadrature(fem_surface,fem_velocity_name,inlet)
    sample_sets[("star","inlet")]=cell_surface_samples(star_surface,star_velocity_name,inlet)
    fem_cut=fem_mesh.slice(origin=upstream["center"],normal=upstream["normalized_normal"]);rel=np.asarray(fem_cut.points)-np.asarray(upstream["center"]);keep=(np.abs(rel@np.asarray(upstream["s_axis"]))<=.5*float(upstream["width"]))&(np.abs(rel@np.asarray(upstream["t_axis"]))<=.5*float(upstream["height"]));fem_clipped=fem_cut.extract_points(keep,adjacent_cells=False);sample_sets[("fem","upstream_z30")]=point_surface_quadrature(fem_clipped,fem_velocity_name,upstream)
    native_metrics,native_records,native_vtp,native_diag=integrate_native_volume_cell_section(star_mesh,star_velocity_name,upstream["center"],upstream["normalized_normal"],upstream["s_axis"],upstream["t_axis"],float(upstream["width"]),float(upstream["height"]),minimum_polygon_area_mm2=1e-12,validate_original_cell_ids=True,fail_on_unmapped_polygon=True,clip_to_section_window=True);sample_sets[("star","upstream_z30")]=native_record_samples(native_records,upstream)
    profiles=[];by_profile={}
    for solver in ("fem","star"):
        for location,section in (("inlet",inlet),("upstream_z30",upstream)):
            row=compute_location_profile(solver,location,sample_sets[(solver,location)],section,low);profiles.append(row);by_profile[(solver,location)]=row
    node_rows,node_summary=fem_inlet_node_audit(fem_surface,fem_velocity_name,inlet,sources["fem"],_mapping(config["fem_inlet_audit"],"fem_inlet_audit"),audit_config_path)
    star_uniformity=star_inlet_uniformity(by_profile[("star","inlet")],sample_sets[("star","inlet")],inlet,_mapping(config["star_inlet_audit"],"star_inlet_audit"))
    common_rows=[];grid_arrays={}
    for location,section in (("inlet",inlet),("upstream_z30",upstream)):
        row,arrays=common_grid_for_location(location,section,fem_mesh,fem_surface,fem_velocity_name,star_mesh,star_surface,star_velocity_name,by_profile);common_rows.append(row);grid_arrays[location]=arrays
        metadata={"location":location,"formal_integration_uses_common_grid":False,"interpolation_method":"linear_tri_point_visualization","star_cell_to_point_for_visualization":True,"grid_resolution":list(section["grid_resolution"]),"width_mm":section["width"],"height_mm":section["height"]}
        path=output/"grids"/("inlet_common_grid.npz" if location=="inlet" else "upstream_z30_common_grid.npz");path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays,metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
    comparison_rows=[comparison_row(location,by_profile[("fem",location)],by_profile[("star",location)]) for location in ("inlet","upstream_z30")]
    _write_csv(output/"profile_metrics_summary.csv",profiles);_write_csv(output/"fem_inlet_node_audit.csv",node_rows);_write_csv(output/"inlet_profile_comparison.csv",[comparison_rows[0]]);_write_csv(output/"upstream_profile_comparison.csv",[comparison_rows[1]]);_write_csv(output/"common_grid_metrics.csv",common_rows)
    plot_cfg=_mapping(config.get("plot",{}),"plot");bins=int(_mapping(config["profile"],"profile").get("histogram_bins",30))
    for location in ("inlet","upstream_z30"):
        hist={solver:{**sample_sets[(solver,location)],"flow_normal":sections[location]["flow_direction_normal"]} for solver in ("fem","star")};plot_location(location,grid_arrays[location],output,plot_cfg,hist,bins)
    plot_node_classification(node_rows,output,int(plot_cfg.get("dpi",180)));plot_overview(profiles,output,int(plot_cfg.get("dpi",180)))
    decision=classify_case(common_rows,by_profile,_mapping(config["decision_thresholds"],"decision_thresholds"))
    audit={"configuration":str(config_path),"data":{"fem":{"path":str(loaded["fem"]["path"]),"time_s":sources["fem"].get("time"),"velocity_array":fem_velocity_name,"association":"point"},"star":{"path":str(loaded["star"]["path"]),"time_index":loaded["star"]["reader_time_point"],"time_s":sources["star"].get("time"),"velocity_array":star_velocity_name,"boundary_part":audit_config["boundaries"][star_name]["inlet"].get("part_name"),"volume_part":part_name,"association":"native cell"}},"sections":sections,"methods":{"fem_profile":"positive degree-4 triangle quadrature of linearly interpolated point velocity","star_inlet_profile":"native surface-cell polygon area","star_upstream_profile":"native volume-cell intersection polygon","common_grid":"linear triangular point interpolation; STAR cell-to-point only for visualization","formal_flow_uses_common_grid":False},"profile_metrics":profiles,"comparisons":comparison_rows,"common_grid_metrics":common_rows,"fem_inlet_node_audit":node_summary,"star_inlet_uniformity":star_uniformity,"diagnostics":{"fem_point_mapping":mapping_diag,"star_upstream_native":native_diag,"star_upstream_native_summary":native_metrics},"decision":decision,"limitations":["Common-grid STAR fields are smoothed by cell-to-point conversion and are not used for formal flow/moment metrics.","Observed differences do not prove causation for branch partitioning.","No additional section, new CFD run, boundary-condition change, mesh change, or downstream detailed profile comparison was performed."]}
    (output/"inlet_upstream_profile_audit.json").write_text(json.dumps(_json_ready(audit),ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(output/"inlet_upstream_profile_audit.md").write_text(build_markdown(audit),encoding="utf-8")
    print(f"Configuration: {config_path}");print(f"Output: {output}")
    for row in profiles: print(f"{row['solver']}/{row['location']}: area={row['area_mm2']:.9g}, |Q|={row['flow_magnitude_mm3_s']:.9g}, U={row['mean_flow_velocity_mm_s']:.9g}, beta={row['beta']:.9g}, alpha={row['alpha']:.9g}, secondary={row['secondary_velocity_ratio_percent']:.6g}%")
    for row in common_rows: print(f"grid/{row['location']}: normalized L2={row['normalized_velocity_relative_l2_percent']:.6g}%, common area={row['common_valid_area_mm2']:.9g} mm2")
    print(f"FEM inlet nodes: wall-shared={node_summary['inlet_wall_shared_node_count']}, interior={node_summary['interior_node_count']}, reconstruction difference={node_summary['actual_vs_shared_zero_difference_mm3_s']:.3g}")
    print(f"STAR inlet uniform: {star_uniformity['all_cells_uniform_within_tolerance']} ({star_uniformity['uniform_cell_fraction_percent']:.6g}%)")
    print(f"Decision: case {decision['case']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
