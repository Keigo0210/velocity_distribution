#!/usr/bin/env python3
"""Shared surface-flow integration utilities.

The functions in this module do not select boundaries. They integrate a caller-
provided surface using either linearly interpolated point vectors or native cell
vectors and a caller-provided normal direction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyvista as pv


ORIGINAL_VOLUME_CELL_ID = "__original_volume_cell_id"


class SurfaceFlowError(ValueError):
    """Raised when a surface or its velocity data cannot be integrated."""


def normalize_normal(normal: Any) -> np.ndarray:
    vector = np.asarray(normal, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise SurfaceFlowError("normal must be a finite three-component vector")
    magnitude = float(np.linalg.norm(vector))
    if magnitude == 0.0:
        raise SurfaceFlowError("normal must be non-zero")
    return vector / magnitude


def flow_unit_values(signed_flow_mm3_s: float) -> dict[str, float]:
    """Convert mm^3/s to the requested millilitre time units."""

    ml_s = float(signed_flow_mm3_s) / 1000.0
    return {
        "signed_flow_rate_ml_s": ml_s,
        "signed_flow_rate_ml_min": ml_s * 60.0,
        "signed_flow_rate_ml_h": ml_s * 3600.0,
        "absolute_flow_rate_ml_s": abs(ml_s),
        "absolute_flow_rate_ml_min": abs(ml_s) * 60.0,
        "absolute_flow_rate_ml_h": abs(ml_s) * 3600.0,
    }


def integrate_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    velocity: np.ndarray,
    normal: np.ndarray,
) -> dict[str, float | int]:
    """Integrate linearly interpolated point velocity over triangles."""

    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    velocity = np.asarray(velocity, dtype=float)
    unit_normal = normalize_normal(normal)
    if points.ndim != 2 or points.shape[1] != 3:
        raise SurfaceFlowError("points must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise SurfaceFlowError("triangles must have shape (N, 3)")
    if len(triangles) == 0:
        raise SurfaceFlowError("surface contains no triangles")
    if np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise SurfaceFlowError("triangle connectivity references an invalid point")
    if velocity.shape != (len(points), 3):
        raise SurfaceFlowError("velocity must have shape (number_of_points, 3)")

    vertices = points[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
        axis=1,
    )
    normal_velocity = velocity @ unit_normal
    speed = np.linalg.norm(velocity, axis=1)
    triangle_normal_velocity = np.mean(normal_velocity[triangles], axis=1)
    triangle_speed = np.mean(speed[triangles], axis=1)
    valid = (
        np.isfinite(areas)
        & (areas > 0.0)
        & np.all(np.isfinite(velocity[triangles]), axis=(1, 2))
    )
    valid_area = float(np.sum(areas[valid]))
    signed_flow = float(np.sum(areas[valid] * triangle_normal_velocity[valid]))
    mean_speed = (
        float(np.sum(areas[valid] * triangle_speed[valid]) / valid_area)
        if valid_area > 0.0
        else math.nan
    )
    max_speed = (
        float(np.max(speed[np.unique(triangles[valid])])) if np.any(valid) else math.nan
    )
    result: dict[str, float | int] = {
        "section_area_mm2": valid_area,
        "signed_flow_rate_mm3_s": signed_flow,
        "absolute_flow_rate_mm3_s": abs(signed_flow),
        "area_mean_normal_velocity_mm_s": (
            signed_flow / valid_area if valid_area > 0.0 else math.nan
        ),
        "mean_speed_mm_s": mean_speed,
        "max_speed_mm_s": max_speed,
        "triangle_count": int(len(triangles)),
        "valid_triangle_count": int(np.sum(valid)),
        "point_count": int(len(points)),
        "cell_count": int(len(triangles)),
        "valid_cell_count": int(np.sum(valid)),
    }
    result.update(flow_unit_values(signed_flow))
    return result


def triangulate_surface(surface: pv.DataSet) -> tuple[pv.PolyData, np.ndarray]:
    """Convert a 2-D surface dataset to PolyData triangles."""

    if surface.n_cells == 0:
        raise SurfaceFlowError("surface contains no cells")
    poly = surface.extract_surface(algorithm="dataset_surface")
    triangles = poly.triangulate()
    faces = np.asarray(triangles.faces, dtype=np.int64)
    if faces.size == 0:
        raise SurfaceFlowError("surface triangulation is empty")
    packed = faces.reshape((-1, 4))
    if np.any(packed[:, 0] != 3):
        raise SurfaceFlowError("surface triangulation produced non-triangle cells")
    return triangles, packed[:, 1:]


def integrate_point_surface(
    surface: pv.DataSet, velocity_name: str, normal: Any
) -> dict[str, float | int]:
    """Triangulate and integrate a point-associated velocity field."""

    triangles, connectivity = triangulate_surface(surface)
    if velocity_name not in triangles.point_data:
        raise SurfaceFlowError(
            f"point velocity {velocity_name!r} is missing; available: "
            f"{list(triangles.point_data.keys())}"
        )
    return integrate_triangles(
        np.asarray(triangles.points),
        connectivity,
        np.asarray(triangles.point_data[velocity_name], dtype=float),
        np.asarray(normal, dtype=float),
    )


def integrate_cell_surface(
    surface: pv.DataSet, velocity_name: str, normal: Any
) -> dict[str, float | int]:
    """Integrate native cell vectors using each original surface-cell area."""

    if surface.n_cells == 0:
        raise SurfaceFlowError("surface contains no cells")
    if velocity_name not in surface.cell_data:
        raise SurfaceFlowError(
            f"cell velocity {velocity_name!r} is missing; available: "
            f"{list(surface.cell_data.keys())}"
        )
    velocity = np.asarray(surface.cell_data[velocity_name], dtype=float)
    if velocity.shape != (surface.n_cells, 3):
        raise SurfaceFlowError("cell velocity must have shape (number_of_cells, 3)")
    unit_normal = normalize_normal(normal)
    sized = surface.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.asarray(sized.cell_data["Area"], dtype=float)
    speed = np.linalg.norm(velocity, axis=1)
    normal_velocity = velocity @ unit_normal
    valid = (
        np.isfinite(areas)
        & (areas > 0.0)
        & np.all(np.isfinite(velocity), axis=1)
    )
    valid_area = float(np.sum(areas[valid]))
    signed_flow = float(np.sum(areas[valid] * normal_velocity[valid]))
    result: dict[str, float | int] = {
        "section_area_mm2": valid_area,
        "signed_flow_rate_mm3_s": signed_flow,
        "absolute_flow_rate_mm3_s": abs(signed_flow),
        "area_mean_normal_velocity_mm_s": (
            signed_flow / valid_area if valid_area > 0.0 else math.nan
        ),
        "mean_speed_mm_s": (
            float(np.sum(areas[valid] * speed[valid]) / valid_area)
            if valid_area > 0.0
            else math.nan
        ),
        "max_speed_mm_s": float(np.max(speed[valid])) if np.any(valid) else math.nan,
        "triangle_count": 0,
        "valid_triangle_count": 0,
        "point_count": int(surface.n_points),
        "cell_count": int(surface.n_cells),
        "valid_cell_count": int(np.sum(valid)),
    }
    result.update(flow_unit_values(signed_flow))
    return result


def partition_value(
    row: dict[str, Any], mode: str, orientation_factors: dict[str, float]
) -> float:
    signed = float(row["signed_flow_rate_mm3_s"])
    if mode == "absolute":
        return abs(signed)
    if mode == "signed":
        return signed
    if mode == "orientation_adjusted":
        name = str(row.get("section_name", row.get("boundary_name", "")))
        return signed * float(orientation_factors.get(name, 1.0))
    raise SurfaceFlowError(
        "partition_mode must be absolute, signed, or orientation_adjusted"
    )


def flow_balance(
    rows: list[dict[str, Any]], balance_config: dict[str, Any], epsilon: float = 1.0e-12
) -> dict[str, Any]:
    """Keep the section-flow balance calculation backward compatible."""

    by_name = {str(row["section_name"]): row for row in rows}
    inlets = list(balance_config.get("inlets", []))
    outlets = list(balance_config.get("outlets", []))
    unknown = [name for name in inlets + outlets if name not in by_name]
    if unknown:
        raise SurfaceFlowError(f"flow_balance references unknown sections: {unknown}")
    mode = str(balance_config.get("partition_mode", "absolute"))
    factors = balance_config.get("orientation_factors", {})
    inlet_total = float(
        sum(partition_value(by_name[name], mode, factors) for name in inlets)
    )
    outlet_values = {
        name: partition_value(by_name[name], mode, factors) for name in outlets
    }
    outlet_total = float(sum(outlet_values.values()))
    error = outlet_total - inlet_total
    result: dict[str, Any] = {
        "partition_mode": mode,
        "inlet_flow_total_mm3_s": inlet_total,
        "outlet_flow_total_mm3_s": outlet_total,
        "flow_balance_error_mm3_s": error,
        "flow_balance_error_percent": 100.0 * abs(error) / max(abs(inlet_total), epsilon),
        "signed_inlet_flow_total_mm3_s": float(
            sum(float(by_name[name]["signed_flow_rate_mm3_s"]) for name in inlets)
        ),
        "signed_outlet_flow_total_mm3_s": float(
            sum(float(by_name[name]["signed_flow_rate_mm3_s"]) for name in outlets)
        ),
    }
    for name, value in outlet_values.items():
        result[f"{name}_fraction"] = (
            value / outlet_total if outlet_total != 0.0 else math.nan
        )
    return result



def _deduplicate_adjacent_vertices(vertices: np.ndarray, tolerance: float) -> np.ndarray:
    """Remove adjacent/closing duplicate 2-D polygon vertices."""

    result: list[np.ndarray] = []
    for vertex in np.asarray(vertices, dtype=float):
        if not result or float(np.linalg.norm(vertex - result[-1])) > tolerance:
            result.append(vertex)
    if len(result) > 1 and float(np.linalg.norm(result[0] - result[-1])) <= tolerance:
        result.pop()
    return np.asarray(result, dtype=float).reshape((-1, 2))


def polygon_area_centroid_2d(vertices: Any) -> tuple[float, np.ndarray]:
    """Return unsigned area and centroid of an ordered planar polygon."""

    polygon = np.asarray(vertices, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise SurfaceFlowError("polygon vertices must have shape (N, 2)")
    if len(polygon) < 3:
        raise SurfaceFlowError("polygon must contain at least three vertices")
    if not np.all(np.isfinite(polygon)):
        raise SurfaceFlowError("polygon vertices must be finite")
    x = polygon[:, 0]
    y = polygon[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_twice_area = float(np.sum(cross))
    if abs(signed_twice_area) <= np.finfo(float).eps:
        raise SurfaceFlowError("polygon area is zero")
    centroid = np.array(
        [np.sum((x + x_next) * cross), np.sum((y + y_next) * cross)],
        dtype=float,
    ) / (3.0 * signed_twice_area)
    return abs(0.5 * signed_twice_area), centroid


def clip_polygon_to_rectangle_2d(
    vertices: Any,
    width: float,
    height: float,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Clip an ordered polygon to a centered width-by-height rectangle."""

    polygon = np.asarray(vertices, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise SurfaceFlowError("polygon vertices must have shape (N, 2)")
    if not np.all(np.isfinite(polygon)):
        raise SurfaceFlowError("polygon vertices must be finite")
    if not math.isfinite(width) or width <= 0.0:
        raise SurfaceFlowError("section width must be positive")
    if not math.isfinite(height) or height <= 0.0:
        raise SurfaceFlowError("section height must be positive")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise SurfaceFlowError("clipping tolerance must be finite and non-negative")

    polygon = _deduplicate_adjacent_vertices(polygon, tolerance)
    boundaries = (
        (0, -0.5 * width, True),
        (0, 0.5 * width, False),
        (1, -0.5 * height, True),
        (1, 0.5 * height, False),
    )
    for axis, bound, keep_greater in boundaries:
        if len(polygon) == 0:
            break
        clipped: list[np.ndarray] = []
        previous = polygon[-1]
        previous_inside = (
            previous[axis] >= bound - tolerance
            if keep_greater
            else previous[axis] <= bound + tolerance
        )
        for current in polygon:
            current_inside = (
                current[axis] >= bound - tolerance
                if keep_greater
                else current[axis] <= bound + tolerance
            )
            if current_inside != previous_inside:
                denominator = current[axis] - previous[axis]
                if abs(float(denominator)) > np.finfo(float).eps:
                    fraction = (bound - previous[axis]) / denominator
                    intersection = previous + fraction * (current - previous)
                    intersection[axis] = bound
                    clipped.append(intersection)
            if current_inside:
                clipped.append(current.copy())
            previous = current
            previous_inside = current_inside
        polygon = _deduplicate_adjacent_vertices(
            np.asarray(clipped, dtype=float).reshape((-1, 2)), tolerance
        )
    return polygon


def tag_original_volume_cell_ids(
    mesh: pv.DataSet, array_name: str = ORIGINAL_VOLUME_CELL_ID
) -> pv.DataSet:
    """Copy a volume mesh and attach an exact int64 source-cell identifier."""

    if mesh.n_cells == 0:
        raise SurfaceFlowError("volume mesh contains no cells")
    if array_name in mesh.cell_data or array_name in mesh.point_data:
        raise SurfaceFlowError(f"reserved original-cell array already exists: {array_name}")
    tagged = mesh.copy(deep=True)
    tagged.cell_data[array_name] = np.arange(mesh.n_cells, dtype=np.int64)
    return tagged


def _validated_cut_original_ids(
    cut: pv.DataSet,
    original_cell_count: int,
    array_name: str,
    validate: bool,
) -> tuple[np.ndarray, int]:
    if array_name not in cut.cell_data:
        if validate:
            raise SurfaceFlowError(
                "plane cutter did not preserve the original volume-cell ID array"
            )
        return np.full(cut.n_cells, -1, dtype=np.int64), int(cut.n_cells)
    raw = np.asarray(cut.cell_data[array_name])
    if raw.shape != (cut.n_cells,):
        raise SurfaceFlowError("cut original-cell ID array has an invalid shape")
    if not np.issubdtype(raw.dtype, np.integer):
        raise SurfaceFlowError(
            "cut original-cell IDs are not integer-valued; refusing interpolation/rounding"
        )
    ids = raw.astype(np.int64, copy=False)
    invalid = (ids < 0) | (ids >= original_cell_count)
    count = int(np.sum(invalid))
    if validate and count:
        raise SurfaceFlowError(
            f"plane cutter produced {count} out-of-range original volume-cell IDs"
        )
    return ids, count


def integrate_native_volume_cell_section(
    mesh: pv.DataSet,
    velocity_name: str,
    center: Any,
    normal: Any,
    s_axis: Any,
    t_axis: Any,
    width: float,
    height: float,
    minimum_polygon_area_mm2: float = 1.0e-12,
    validate_original_cell_ids: bool = True,
    fail_on_unmapped_polygon: bool = True,
    clip_to_section_window: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], pv.PolyData, dict[str, Any]]:
    """Integrate native volume-cell vectors over plane-intersection polygons.

    The cutter receives an explicit int64 original-cell ID. Velocity is then
    looked up from the original volume mesh using that ID; cutter-generated
    values are never used for the quantitative integration.
    """

    if velocity_name not in mesh.cell_data:
        raise SurfaceFlowError(
            f"cell velocity {velocity_name!r} is missing; available: "
            f"{list(mesh.cell_data.keys())}"
        )
    velocity = np.asarray(mesh.cell_data[velocity_name], dtype=float)
    if velocity.shape != (mesh.n_cells, 3):
        raise SurfaceFlowError("native cell velocity must have shape (number_of_cells, 3)")
    if not np.all(np.isfinite(velocity)):
        raise SurfaceFlowError("native cell velocity contains non-finite values")
    center_array = np.asarray(center, dtype=float)
    s_array = np.asarray(s_axis, dtype=float)
    t_array = np.asarray(t_axis, dtype=float)
    unit_normal = normalize_normal(normal)
    for value, name in ((center_array, "center"), (s_array, "s_axis"), (t_array, "t_axis")):
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise SurfaceFlowError(f"{name} must be a finite three-component vector")
    s_array = normalize_normal(s_array)
    t_array = normalize_normal(t_array)
    orthogonality = max(
        abs(float(np.dot(unit_normal, s_array))),
        abs(float(np.dot(unit_normal, t_array))),
        abs(float(np.dot(s_array, t_array))),
    )
    if orthogonality > 1.0e-8:
        raise SurfaceFlowError("normal, s_axis, and t_axis must be mutually perpendicular")
    if not math.isfinite(minimum_polygon_area_mm2) or minimum_polygon_area_mm2 < 0.0:
        raise SurfaceFlowError("minimum polygon area must be finite and non-negative")

    tagged = tag_original_volume_cell_ids(mesh)
    cut = tagged.slice(origin=center_array, normal=unit_normal)
    if cut.n_cells == 0:
        raise SurfaceFlowError("section plane does not intersect the volume mesh")
    original_ids, initial_unmapped = _validated_cut_original_ids(
        cut, mesh.n_cells, ORIGINAL_VOLUME_CELL_ID, validate_original_cell_ids
    )

    records: list[dict[str, Any]] = []
    valid_geometry: list[tuple[np.ndarray, dict[str, Any]]] = []
    invalid_count = 0
    unmapped_count = initial_unmapped
    seen_signatures: set[tuple[int, tuple[tuple[float, float], ...]]] = set()
    mapped_ids: list[int] = []
    exact_duplicate_count = 0

    for polygon_id in range(cut.n_cells):
        original_id = int(original_ids[polygon_id])
        record: dict[str, Any] = {
            "original_volume_cell_id": original_id,
            "polygon_id": polygon_id,
            "polygon_area_mm2": math.nan,
            "velocity_x_mm_s": math.nan,
            "velocity_y_mm_s": math.nan,
            "velocity_z_mm_s": math.nan,
            "normal_velocity_mm_s": math.nan,
            "signed_flow_contribution_mm3_s": math.nan,
            "polygon_centroid_x_mm": math.nan,
            "polygon_centroid_y_mm": math.nan,
            "polygon_centroid_z_mm": math.nan,
            "valid": False,
        }
        if original_id < 0 or original_id >= mesh.n_cells:
            unmapped_count += int(initial_unmapped == 0)
            invalid_count += 1
            records.append(record)
            continue
        points = np.asarray(cut.get_cell(polygon_id).points, dtype=float)
        if len(points) < 3 or not np.all(np.isfinite(points)):
            invalid_count += 1
            records.append(record)
            continue
        relative = points - center_array
        polygon_st = np.column_stack((relative @ s_array, relative @ t_array))
        if clip_to_section_window:
            polygon_st = clip_polygon_to_rectangle_2d(polygon_st, width, height)
        else:
            polygon_st = _deduplicate_adjacent_vertices(polygon_st, 1.0e-12)
        if len(polygon_st) < 3:
            invalid_count += 1
            records.append(record)
            continue
        try:
            area, centroid_st = polygon_area_centroid_2d(polygon_st)
        except SurfaceFlowError:
            invalid_count += 1
            records.append(record)
            continue
        if area <= minimum_polygon_area_mm2:
            invalid_count += 1
            records.append(record)
            continue
        canonical = tuple(sorted(tuple(value) for value in np.round(polygon_st, 11)))
        signature = (original_id, canonical)
        if signature in seen_signatures:
            exact_duplicate_count += 1
            invalid_count += 1
            records.append(record)
            continue
        seen_signatures.add(signature)
        mapped_ids.append(original_id)
        cell_velocity = velocity[original_id]
        normal_velocity = float(np.dot(cell_velocity, unit_normal))
        contribution = area * normal_velocity
        centroid_xyz = center_array + centroid_st[0] * s_array + centroid_st[1] * t_array
        record.update(
            {
                "polygon_area_mm2": area,
                "velocity_x_mm_s": float(cell_velocity[0]),
                "velocity_y_mm_s": float(cell_velocity[1]),
                "velocity_z_mm_s": float(cell_velocity[2]),
                "normal_velocity_mm_s": normal_velocity,
                "signed_flow_contribution_mm3_s": contribution,
                "polygon_centroid_x_mm": float(centroid_xyz[0]),
                "polygon_centroid_y_mm": float(centroid_xyz[1]),
                "polygon_centroid_z_mm": float(centroid_xyz[2]),
                "valid": True,
            }
        )
        records.append(record)
        polygon_xyz = (
            center_array
            + polygon_st[:, 0, None] * s_array
            + polygon_st[:, 1, None] * t_array
        )
        valid_geometry.append((polygon_xyz, record))

    if fail_on_unmapped_polygon and unmapped_count:
        raise SurfaceFlowError(
            f"{unmapped_count} intersection polygons could not be mapped to original cells"
        )
    valid_records = [record for record in records if record["valid"]]
    if not valid_records:
        raise SurfaceFlowError("no valid intersection polygons remain after clipping")
    areas = np.asarray([record["polygon_area_mm2"] for record in valid_records], float)
    flows = np.asarray(
        [record["signed_flow_contribution_mm3_s"] for record in valid_records], float
    )
    speeds = np.asarray(
        [
            np.linalg.norm(
                [record["velocity_x_mm_s"], record["velocity_y_mm_s"], record["velocity_z_mm_s"]]
            )
            for record in valid_records
        ],
        float,
    )
    total_area = float(np.sum(areas))
    signed_flow = float(np.sum(flows))
    id_counts = np.unique(np.asarray(mapped_ids, dtype=np.int64), return_counts=True)[1]
    repeated_id_excess = int(np.sum(np.maximum(id_counts - 1, 0)))
    summary: dict[str, Any] = {
        "native_intersection_area_mm2": total_area,
        "signed_flow_mm3_s": signed_flow,
        "absolute_flow_mm3_s": abs(signed_flow),
        "signed_flow_ml_s": signed_flow / 1000.0,
        "signed_flow_ml_min": signed_flow * 0.06,
        "signed_flow_ml_h": signed_flow * 3.6,
        "area_mean_normal_velocity_mm_s": signed_flow / total_area,
        "area_weighted_mean_speed_mm_s": float(np.sum(areas * speeds) / total_area),
        "maximum_cell_speed_mm_s": float(np.max(speeds)),
        "intersected_volume_cell_count": int(len(set(mapped_ids))),
        "generated_polygon_count": int(cut.n_cells),
        "valid_polygon_count": int(len(valid_records)),
        "invalid_polygon_count": int(invalid_count),
        "duplicate_original_cell_count": exact_duplicate_count,
        "multiple_polygon_original_cell_count": repeated_id_excess,
        "unmapped_polygon_count": int(unmapped_count),
        "minimum_polygon_area_mm2": float(np.min(areas)),
        "maximum_polygon_area_mm2": float(np.max(areas)),
        "status": "success",
        "warning": "; ".join(
            part for part in (
                f"{exact_duplicate_count} exact duplicate polygon(s) excluded"
                if exact_duplicate_count else "",
                f"{repeated_id_excess} additional distinct polygon component(s) share original cell IDs"
                if repeated_id_excess else "",
            ) if part
        ),
    }

    output_points: list[np.ndarray] = []
    faces: list[int] = []
    for polygon_xyz, _ in valid_geometry:
        offset = len(output_points)
        output_points.extend(polygon_xyz)
        faces.extend([len(polygon_xyz), *range(offset, offset + len(polygon_xyz))])
    polydata = pv.PolyData(
        np.asarray(output_points, dtype=float), np.asarray(faces, dtype=np.int64)
    )
    for key in (
        "original_volume_cell_id",
        "polygon_id",
        "polygon_area_mm2",
        "velocity_x_mm_s",
        "velocity_y_mm_s",
        "velocity_z_mm_s",
        "normal_velocity_mm_s",
        "signed_flow_contribution_mm3_s",
    ):
        values = [record[key] for _, record in valid_geometry]
        polydata.cell_data[key] = np.asarray(values)
    diagnostic = {
        "original_cell_id_array": ORIGINAL_VOLUME_CELL_ID,
        "original_cell_id_input_dtype": "int64",
        "original_cell_id_cut_dtype": str(np.asarray(cut.cell_data[ORIGINAL_VOLUME_CELL_ID]).dtype),
        "cut_cell_count": int(cut.n_cells),
        "exact_duplicate_polygon_count": exact_duplicate_count,
        "maximum_basis_dot_product": orthogonality,
        "velocity_lookup": "original volume cell array indexed by preserved int64 ID",
    }
    return summary, records, polydata, diagnostic
