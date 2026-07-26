#!/usr/bin/env python3
"""Audit native inlet/outlet surface flow rates for configured datasets."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import date
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyvista as pv

from surface_flow import (
    SurfaceFlowError,
    integrate_cell_surface,
    integrate_point_surface,
    normalize_normal,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "audit_boundary_flow.json"


class BoundaryAuditError(ValueError):
    """Raised when an audit input or boundary selection is invalid."""


def resolve_config_path(value: str | Path, config_path: str | Path) -> Path:
    """Resolve repository paths while supporting config-relative external files."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    config_path = Path(config_path).resolve()
    prefixes = {"data", "output", "config", "manual", "scripts", "tests"}
    if path.parts and path.parts[0] in prefixes:
        return (config_path.parent.parent / path).resolve()
    return (config_path.parent / path).resolve()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BoundaryAuditError(f"{context} must be a JSON object")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BoundaryAuditError(f"configuration file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BoundaryAuditError(f"invalid JSON in {path}: {exc}") from exc
    return dict(_mapping(value, "configuration"))


def _cell_type_counts(dataset: pv.DataSet) -> dict[str, int]:
    if not hasattr(dataset, "celltypes"):
        return {}
    values, counts = np.unique(np.asarray(dataset.celltypes), return_counts=True)
    result: dict[str, int] = {}
    for value, count in zip(values, counts):
        try:
            name = pv.CellType(int(value)).name
        except (ValueError, AttributeError):
            name = f"VTK_{int(value)}"
        result[f"{int(value)}:{name}"] = int(count)
    return result


def dataset_inventory(dataset: pv.DataSet, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "dataset_type": type(dataset).__name__,
        "bounds": [float(value) for value in dataset.bounds],
        "point_count": int(dataset.n_points),
        "cell_count": int(dataset.n_cells),
        "cell_types": _cell_type_counts(dataset),
        "point_data_arrays": list(dataset.point_data.keys()),
        "cell_data_arrays": list(dataset.cell_data.keys()),
        "field_data_arrays": list(dataset.field_data.keys()),
        "surface_mesh_available": bool(
            dataset.n_cells
            and dataset.extract_surface(algorithm="dataset_surface").n_cells
        ),
    }


def multiblock_inventory(data: pv.DataObject, prefix: str = "root") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(data, pv.MultiBlock):
        for index in range(data.n_blocks):
            block = data[index]
            name = str(data.get_block_name(index) or index).strip()
            path = f"{prefix}/{index}:{name}"
            if block is None:
                records.append({"name": path, "dataset_type": "None"})
            else:
                records.extend(multiblock_inventory(block, path))
    else:
        records.append(dataset_inventory(data, prefix))
    return records


def _scaled_copy(dataset: pv.DataSet, source: Mapping[str, Any]) -> pv.DataSet:
    result = dataset.copy(deep=True)
    length_scale = float(source.get("length_scale_to_mm", 1.0))
    velocity_scale = float(source.get("velocity_scale_to_mm_s", 1.0))
    if length_scale != 1.0:
        result.points = np.asarray(result.points, dtype=float) * length_scale
    velocity_name = str(source["velocity_array"])
    if velocity_name in result.point_data:
        result.point_data[velocity_name] = (
            np.asarray(result.point_data[velocity_name], dtype=float) * velocity_scale
        )
    if velocity_name in result.cell_data:
        result.cell_data[velocity_name] = (
            np.asarray(result.cell_data[velocity_name], dtype=float) * velocity_scale
        )
    return result


def load_source(
    source_name: str, source: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    path = resolve_config_path(str(source["path"]), config_path)
    if not path.is_file():
        raise BoundaryAuditError(f"{source_name}: input file does not exist: {path}")
    source_type = str(source.get("type", "fem_vtu")).lower()
    if source_type in {"star_ensight", "ensight", "case", "star_ccm"}:
        reader = pv.get_reader(path)
        point = int(source.get("time_point", reader.number_time_points - 1))
        if not 0 <= point < reader.number_time_points:
            raise BoundaryAuditError(
                f"{source_name}: time_point {point} outside 0..{reader.number_time_points - 1}"
            )
        reader.set_active_time_point(point)
        raw = reader.read()
        inventory = multiblock_inventory(raw)
        return {
            "raw": raw,
            "primary": None,
            "path": path,
            "reader_time_point": point,
            "reader_time_count": int(reader.number_time_points),
            "inventory": inventory,
        }
    raw = pv.read(path)
    if isinstance(raw, pv.MultiBlock):
        primary = raw.combine(merge_points=False)
    else:
        primary = raw
    return {
        "raw": raw,
        "primary": _scaled_copy(primary, source),
        "path": path,
        "inventory": multiblock_inventory(raw),
    }


def _find_named_block(data: pv.DataObject, target: str) -> pv.DataSet:
    matches: list[pv.DataSet] = []
    target = target.strip()

    def walk(obj: pv.DataObject) -> None:
        if not isinstance(obj, pv.MultiBlock):
            return
        for index in range(obj.n_blocks):
            block = obj[index]
            name = str(obj.get_block_name(index) or "").strip()
            if block is not None and name == target and not isinstance(block, pv.MultiBlock):
                matches.append(block)
            if isinstance(block, pv.MultiBlock):
                walk(block)

    walk(data)
    if not matches:
        raise BoundaryAuditError(f"named surface part/block was not found: {target!r}")
    if len(matches) > 1:
        raise BoundaryAuditError(f"named surface part/block is ambiguous: {target!r}")
    return matches[0]


def _map_point_vectors_by_coordinate(
    volume: pv.DataSet,
    surface: pv.DataSet,
    velocity_name: str,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if velocity_name not in volume.point_data:
        raise BoundaryAuditError(
            f"point velocity {velocity_name!r} missing from source mesh"
        )
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise BoundaryAuditError("point_mapping_tolerance_mm must be positive")
    source_points = np.asarray(volume.points, dtype=float)
    source_velocity = np.asarray(volume.point_data[velocity_name], dtype=float)
    if source_velocity.shape != (volume.n_points, 3):
        raise BoundaryAuditError("source point velocity must have three components")
    buckets: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for point, velocity in zip(source_points, source_velocity):
        key = tuple(np.rint(point / tolerance).astype(np.int64))
        buckets.setdefault(key, []).append(velocity)
    mapped: list[np.ndarray] = []
    multiplicities: list[int] = []
    maximum_spread = 0.0
    for point in np.asarray(surface.points, dtype=float):
        key = tuple(np.rint(point / tolerance).astype(np.int64))
        candidates = buckets.get(key)
        if not candidates:
            raise BoundaryAuditError(
                "boundary surface point has no matching result-mesh point within "
                f"the configured coordinate quantization: {point.tolist()}"
            )
        values = np.asarray(candidates, dtype=float)
        mapped.append(np.mean(values, axis=0))
        multiplicities.append(len(values))
        if len(values) > 1:
            maximum_spread = max(
                maximum_spread,
                float(np.max(np.linalg.norm(values - np.mean(values, axis=0), axis=1))),
            )
    return np.asarray(mapped), {
        "mapping_method": "coordinate_key_average",
        "mapping_tolerance_mm": tolerance,
        "minimum_multiplicity": int(min(multiplicities)),
        "maximum_multiplicity": int(max(multiplicities)),
        "maximum_velocity_spread_mm_s": maximum_spread,
        "all_surface_points_matched": True,
    }


def _select_geometric_plane(
    primary: pv.DataSet, boundary: Mapping[str, Any]
) -> pv.DataSet:
    center = np.asarray(boundary.get("center"), dtype=float)
    normal = normalize_normal(boundary.get("normal"))
    tolerance = float(boundary.get("tolerance"))
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise BoundaryAuditError("geometric_plane center must have three finite values")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise BoundaryAuditError("geometric_plane tolerance must be positive")
    surface = primary.extract_surface(algorithm="dataset_surface")
    keep: list[int] = []
    for cell_id in range(surface.n_cells):
        cell = surface.get_cell(cell_id)
        distances = np.abs((np.asarray(cell.points) - center) @ normal)
        if np.all(distances <= tolerance):
            keep.append(cell_id)
    if not keep:
        raise BoundaryAuditError("geometric_plane selected no boundary cells")
    return surface.extract_cells(keep)


def select_boundary(
    source_name: str,
    source: Mapping[str, Any],
    loaded: Mapping[str, Any],
    boundary_name: str,
    boundary: Mapping[str, Any],
    config_path: Path,
) -> tuple[pv.DataSet, dict[str, Any]]:
    method = str(boundary.get("selection_method", ""))
    identifier: str
    diagnostic: dict[str, Any] = {}
    if method in {"part_name", "block_name"}:
        key = "part_name" if method == "part_name" else "block_name"
        identifier = str(boundary.get(key, ""))
        if not identifier:
            raise BoundaryAuditError(f"{source_name}/{boundary_name}: {key} is required")
        selected = _find_named_block(loaded["raw"], identifier)
        surface = _scaled_copy(selected, source)
    elif method == "boundary_id":
        if "boundary_id" not in boundary:
            raise BoundaryAuditError(
                f"{source_name}/{boundary_name}: boundary_id is required"
            )
        boundary_id = int(boundary["boundary_id"])
        identifier = str(boundary_id)
        mesh_value = source.get("boundary_mesh_path")
        if not isinstance(mesh_value, str) or not mesh_value:
            raise BoundaryAuditError(
                f"{source_name}: boundary_mesh_path is required for boundary_id selection"
            )
        mesh_path = resolve_config_path(mesh_value, config_path)
        if not mesh_path.is_file():
            raise BoundaryAuditError(f"boundary mesh does not exist: {mesh_path}")
        boundary_mesh = pv.read(mesh_path)
        if isinstance(boundary_mesh, pv.MultiBlock):
            boundary_mesh = boundary_mesh.combine(merge_points=False)
        array_name = str(source.get("boundary_id_array", "gmsh:physical"))
        if array_name not in boundary_mesh.cell_data:
            raise BoundaryAuditError(
                f"boundary ID array {array_name!r} missing from {mesh_path}"
            )
        ids = np.flatnonzero(
            np.asarray(boundary_mesh.cell_data[array_name]) == boundary_id
        )
        if not len(ids):
            raise BoundaryAuditError(
                f"boundary ID {boundary_id} does not exist in {mesh_path}"
            )
        selected = boundary_mesh.extract_cells(ids)
        surface = selected.extract_surface(algorithm="dataset_surface")
        surface = _scaled_copy(surface, {**source, "velocity_array": source["velocity_array"]})
        diagnostic.update(
            {
                "boundary_mesh_path": str(mesh_path),
                "boundary_id_array": array_name,
                "selected_source_cell_count": int(len(ids)),
            }
        )
    elif method == "explicit_surface_file":
        value = boundary.get("path")
        if not isinstance(value, str) or not value:
            raise BoundaryAuditError("explicit_surface_file requires path")
        surface_path = resolve_config_path(value, config_path)
        if not surface_path.is_file():
            raise BoundaryAuditError(f"surface file does not exist: {surface_path}")
        raw = pv.read(surface_path)
        surface = raw.combine(merge_points=False) if isinstance(raw, pv.MultiBlock) else raw
        surface = _scaled_copy(surface, source)
        identifier = str(surface_path)
    elif method == "geometric_plane":
        if loaded.get("primary") is None:
            raw = loaded["raw"]
            primary = raw.combine(merge_points=True) if isinstance(raw, pv.MultiBlock) else raw
            primary = _scaled_copy(primary, source)
        else:
            primary = loaded["primary"]
        surface = _select_geometric_plane(primary, boundary)
        identifier = json.dumps(
            {
                "center": boundary.get("center"),
                "normal": boundary.get("normal"),
                "tolerance": boundary.get("tolerance"),
            },
            sort_keys=True,
        )
        diagnostic.update(
            {
                "plane_center": boundary.get("center"),
                "plane_normal": boundary.get("normal"),
                "plane_tolerance": boundary.get("tolerance"),
                "geometric_selection_reason": boundary.get("selection_reason", ""),
            }
        )
    else:
        raise BoundaryAuditError(
            f"{source_name}/{boundary_name}: unsupported selection_method {method!r}"
        )
    if surface.n_cells == 0:
        raise BoundaryAuditError(
            f"{source_name}/{boundary_name}: selected boundary surface is empty"
        )
    diagnostic.update(
        {
            "selection_method": method,
            "boundary_identifier": identifier,
            "selected_point_count": int(surface.n_points),
            "selected_cell_count": int(surface.n_cells),
            "selection_evidence": boundary.get("selection_evidence", ""),
        }
    )
    return surface, diagnostic


def boundary_normal(boundary: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    mode = str(boundary.get("normal_mode", "explicit"))
    if mode in {"explicit", "configured_outward"}:
        normal = normalize_normal(boundary.get("normal"))
        source = (
            "configured explicit outward normal"
            if mode == "configured_outward"
            else "configured explicit normal"
        )
        return normal, source
    raise BoundaryAuditError(
        "normal_mode must be explicit or configured_outward; automatic normals are "
        "not used because native face orientation may be inconsistent"
    )


def audit_one_boundary(
    source_name: str,
    source: Mapping[str, Any],
    loaded: Mapping[str, Any],
    boundary_name: str,
    boundary: Mapping[str, Any],
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    surface, selection = select_boundary(
        source_name, source, loaded, boundary_name, boundary, config_path
    )
    normal, normal_source = boundary_normal(boundary)
    association = str(source.get("data_association", "point")).lower()
    velocity_name = str(source["velocity_array"])
    warning_parts: list[str] = []
    conversion: dict[str, Any] | None = None
    diagnostics = dict(selection)
    if association == "point":
        if velocity_name not in surface.point_data:
            primary = loaded.get("primary")
            if primary is None:
                raise BoundaryAuditError(
                    f"{source_name}/{boundary_name}: point velocity is not present on the "
                    "selected surface and no primary point mesh is available"
                )
            mapped, mapping = _map_point_vectors_by_coordinate(
                primary,
                surface,
                velocity_name,
                float(source.get("point_mapping_tolerance_mm", 1.0e-8)),
            )
            surface = surface.copy(deep=True)
            surface.point_data[velocity_name] = mapped
            diagnostics["point_mapping"] = mapping
            if mapping["maximum_velocity_spread_mm_s"] > 1.0e-10:
                warning_parts.append(
                    "duplicate coordinate nodes had different velocities and were averaged"
                )
        metrics = integrate_point_surface(surface, velocity_name, normal)
        integration_method = "triangulated_point_linear"
    elif association == "cell":
        if velocity_name not in surface.cell_data:
            raise BoundaryAuditError(
                f"{source_name}/{boundary_name}: native cell velocity {velocity_name!r} "
                "is not available on the selected boundary part"
            )
        metrics = integrate_cell_surface(surface, velocity_name, normal)
        integration_method = "native_surface_cell_area"
        converted = surface.cell_data_to_point_data(pass_cell_data=True)
        converted_metrics = integrate_point_surface(converted, velocity_name, normal)
        native = float(metrics["signed_flow_rate_mm3_s"])
        point_value = float(converted_metrics["signed_flow_rate_mm3_s"])
        difference = point_value - native
        conversion = {
            "source_name": source_name,
            "boundary_name": boundary_name,
            "native_cell_flow_mm3_s": native,
            "converted_point_flow_mm3_s": point_value,
            "signed_difference_mm3_s": difference,
            "absolute_difference_mm3_s": abs(difference),
            "relative_difference_percent": (
                100.0 * abs(difference) / max(abs(native), 1.0e-12)
            ),
            "native_cell_flow": native,
            "converted_point_flow": point_value,
            "absolute_difference": abs(difference),
            "native_cell_area_mm2": metrics["section_area_mm2"],
            "converted_point_area_mm2": converted_metrics["section_area_mm2"],
            "status": "success",
            "warning": "",
        }
    else:
        raise BoundaryAuditError(
            f"{source_name}: data_association must be point or cell"
        )
    row: dict[str, Any] = {
        "source_name": source_name,
        "source_label": source.get("label", source_name),
        "boundary_name": boundary_name,
        "boundary_role": boundary.get("role", boundary_name),
        "selection_method": selection["selection_method"],
        "boundary_identifier": selection["boundary_identifier"],
        "data_association": association,
        "integration_method": integration_method,
        "area_mm2": metrics["section_area_mm2"],
        "signed_flow_mm3_s": metrics["signed_flow_rate_mm3_s"],
        "absolute_flow_mm3_s": metrics["absolute_flow_rate_mm3_s"],
        "signed_flow_ml_s": metrics["signed_flow_rate_ml_s"],
        "signed_flow_ml_min": metrics["signed_flow_rate_ml_min"],
        "signed_flow_ml_h": metrics["signed_flow_rate_ml_h"],
        "area_mean_normal_velocity_mm_s": metrics["area_mean_normal_velocity_mm_s"],
        "mean_speed_mm_s": metrics["mean_speed_mm_s"],
        "max_speed_mm_s": metrics["max_speed_mm_s"],
        "point_count": metrics["point_count"],
        "cell_count": metrics["cell_count"],
        "valid_cell_count": metrics["valid_cell_count"],
        "selected_point_count": selection["selected_point_count"],
        "selected_cell_count": selection["selected_cell_count"],
        "integration_cell_count": metrics["cell_count"],
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "normal_z": float(normal[2]),
        "normal_mode": boundary.get("normal_mode", "explicit"),
        "normal_source": normal_source,
        "status": "success",
        "warning": "; ".join(warning_parts),
    }
    return row, conversion, diagnostics


def boundary_balance(
    rows: list[dict[str, Any]], epsilon: float = 1.0e-12
) -> dict[str, Any]:
    inlets = [row for row in rows if str(row["boundary_role"]) == "inlet"]
    outlets = [row for row in rows if str(row["boundary_role"]) == "outlet"]
    if not inlets or not outlets:
        raise BoundaryAuditError("boundary balance requires inlet and outlet roles")
    inlet_total = float(sum(abs(float(row["signed_flow_mm3_s"])) for row in inlets))
    outlet_total = float(sum(abs(float(row["signed_flow_mm3_s"])) for row in outlets))
    signed_sum = float(sum(float(row["signed_flow_mm3_s"]) for row in rows))
    magnitude_error = outlet_total - inlet_total
    return {
        "inlet_flow_total": inlet_total,
        "outlet_flow_total": outlet_total,
        "signed_flow_sum": signed_sum,
        "magnitude_balance_error": magnitude_error,
        "signed_balance_error": signed_sum,
        "inlet_flow_total_mm3_s": inlet_total,
        "outlet_flow_total_mm3_s": outlet_total,
        "signed_flow_sum_mm3_s": signed_sum,
        "magnitude_balance_error_mm3_s": magnitude_error,
        "magnitude_balance_error_percent": (
            100.0 * abs(magnitude_error) / max(abs(inlet_total), epsilon)
        ),
        "signed_balance_error_mm3_s": signed_sum,
        "signed_balance_error_percent": (
            100.0 * abs(signed_sum) / max(abs(inlet_total), epsilon)
        ),
    }


def compare_boundary_internal(
    boundary_flow: float, internal_flow: float, epsilon: float = 1.0e-12
) -> dict[str, float]:
    signed_difference = float(boundary_flow - internal_flow)
    magnitude_difference = float(abs(boundary_flow) - abs(internal_flow))
    return {
        "signed_difference_mm3_s": signed_difference,
        "absolute_difference_mm3_s": abs(magnitude_difference),
        "magnitude_difference_mm3_s": magnitude_difference,
        "relative_difference_percent": (
            100.0 * abs(magnitude_difference) / max(abs(boundary_flow), epsilon)
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str] | None = None) -> None:
    if columns is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
    else:
        keys = list(columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "未評価"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for column in columns:
            value = row.get(column)
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                values.append("未評価")
            elif isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _solver_comparisons(
    rows: list[dict[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    comparison = _mapping(config.get("solver_comparison", {}), "solver_comparison")
    source_1 = comparison.get("source_1")
    source_2 = comparison.get("source_2")
    if not isinstance(source_1, str) or not isinstance(source_2, str):
        return []
    by_key = {(row["source_name"], row["boundary_name"]): row for row in rows}
    names_1 = {
        row["boundary_name"] for row in rows if row["source_name"] == source_1
    }
    names_2 = {
        row["boundary_name"] for row in rows if row["source_name"] == source_2
    }
    result: list[dict[str, Any]] = []
    for name in sorted(names_1 & names_2):
        first = by_key[(source_1, name)]
        second = by_key[(source_2, name)]
        q1 = float(first["signed_flow_mm3_s"])
        q2 = float(second["signed_flow_mm3_s"])
        a1 = float(first["area_mm2"])
        a2 = float(second["area_mm2"])
        result.append(
            {
                "boundary_name": name,
                "boundary_role": first["boundary_role"],
                "source_1": source_1,
                "source_2": source_2,
                "source_1_area_mm2": a1,
                "source_2_area_mm2": a2,
                "area_absolute_difference_mm2": abs(a1 - a2),
                "area_relative_difference_percent": 100.0 * abs(a1 - a2) / max(abs(a1), 1.0e-12),
                "source_1_signed_flow_mm3_s": q1,
                "source_2_signed_flow_mm3_s": q2,
                "flow_absolute_difference_mm3_s": abs(abs(q1) - abs(q2)),
                "flow_relative_difference_percent": 100.0 * abs(abs(q1) - abs(q2)) / max(abs(q1), 1.0e-12),
            }
        )
    return result


def _hypothesis_text(
    solver_rows: list[dict[str, Any]],
    internal_rows: list[dict[str, Any]],
    conversion_rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> str:
    limits = _mapping(config.get("decision_thresholds", {}), "decision_thresholds")
    match = float(limits.get("flow_match_percent", 1.0))
    conversion_limit = float(limits.get("cell_to_point_attention_percent", 1.0))
    inlet = next((row for row in solver_rows if row["boundary_role"] == "inlet"), None)
    inlet_internal = [
        row for row in internal_rows if str(row.get("boundary_role")) == "inlet"
    ]
    statements: list[str] = []
    if inlet is None:
        statements.append("A/B判定: ソルバー間の入口境界比較を構成できず未評価です。")
    elif float(inlet["flow_relative_difference_percent"]) > match:
        statements.append(
            "仮説Aを支持する観測: 入口境界流量差が設定した一致目安を超えています。"
            "入口面積、単位換算、入口速度設定・実装を次に照合してください。"
        )
    else:
        statements.append(
            "仮説Aは強く支持されません: 入口境界流量は設定した一致目安内です。"
        )
        if len(inlet_internal) >= 2:
            values = [abs(float(row["internal_section_flow_mm3_s"])) for row in inlet_internal]
            internal_difference = 100.0 * abs(values[0] - values[1]) / max(values[0], 1.0e-12)
            if internal_difference > match:
                statements.append(
                    "仮説Bに対応する観測があります: 入口境界は近い一方、対応する上流内部断面では"
                    "ソルバー間差が大きくなっています。ただし、cell-to-point変換や内部断面積分法の"
                    "影響も含むため、物理的な流量損失とは断定できません。"
                )
            else:
                statements.append(
                    "上流内部断面も一致目安内で、差が分岐後に増えるなら仮説Cの検討対象です。"
                )
    conversion_max = max(
        (float(row["relative_difference_percent"]) for row in conversion_rows),
        default=math.nan,
    )
    if math.isfinite(conversion_max) and conversion_max > conversion_limit:
        statements.append(
            f"仮説Dに注意が必要です: surface cell-to-point変換による最大流量差は"
            f"{conversion_max:.6g}%です。定量積分にはnative cell値を優先してください。"
        )
    elif math.isfinite(conversion_max):
        statements.append(
            f"cell-to-point変換差の最大値は{conversion_max:.6g}%で、設定した注意目安内です。"
        )
    else:
        statements.append("cell-to-point変換影響はnative cell boundary dataがなく未評価です。")
    return "\n\n".join(statements)


def build_report(
    config: Mapping[str, Any],
    flow_rows: list[dict[str, Any]],
    balance_rows: list[dict[str, Any]],
    internal_rows: list[dict[str, Any]],
    conversion_rows: list[dict[str, Any]],
    solver_rows: list[dict[str, Any]],
    bc_rows: list[dict[str, Any]],
    diagnostics: Mapping[str, Any],
) -> str:
    hypotheses = _hypothesis_text(solver_rows, internal_rows, conversion_rows, config)
    geometric = [row for row in flow_rows if row["selection_method"] == "geometric_plane"]
    undecided = [row for row in bc_rows if str(row.get("matched")) == "確認不能"]
    return f"""# 境界流量監査

作成日: {date.today().isoformat()}

## 1. 使用データ

{_markdown_table(list(diagnostics.get('sources', [])), ['source_name', 'label', 'path', 'time_s', 'velocity_array', 'data_association', 'length_unit', 'velocity_unit'])}

## 2. 境界面の特定方法

{_markdown_table(flow_rows, ['source_name', 'boundary_name', 'boundary_role', 'selection_method', 'boundary_identifier', 'normal_source', 'selected_point_count', 'selected_cell_count', 'integration_cell_count'])}

`boundary_id`は元メッシュの明示タグ、`part_name`はEnsightの明示surface partです。
`geometric_plane`はnative境界ではなく最後の手段として区別します。今回のgeometric plane使用数は{len(geometric)}です。

## 3. 境界面積・入口出口流量

{_markdown_table(flow_rows, ['source_name', 'boundary_name', 'area_mm2', 'signed_flow_mm3_s', 'absolute_flow_mm3_s', 'signed_flow_ml_s', 'signed_flow_ml_min', 'signed_flow_ml_h', 'area_mean_normal_velocity_mm_s'])}

signed flowは設定した外向き法線に対する符号です。入口は通常負、出口は通常正です。

## 4. 各ソルバーの境界流量保存

{_markdown_table(balance_rows, ['source_name', 'inlet_flow_total_mm3_s', 'outlet_flow_total_mm3_s', 'signed_flow_sum_mm3_s', 'magnitude_balance_error_mm3_s', 'magnitude_balance_error_percent'])}

## 5. 境界と内部断面

{_markdown_table(internal_rows, ['source_name', 'boundary_name', 'internal_section_name', 'boundary_flow_mm3_s', 'internal_section_flow_mm3_s', 'signed_difference_mm3_s', 'absolute_difference_mm3_s', 'relative_difference_percent'])}

符号比較と大きさ比較を分離しています。内部断面法線と境界外向き法線が逆の場合、signed値が逆でも大きさは一致し得ます。

## 6. FEMとSTARの境界差

{_markdown_table(solver_rows, ['boundary_name', 'source_1_area_mm2', 'source_2_area_mm2', 'area_relative_difference_percent', 'source_1_signed_flow_mm3_s', 'source_2_signed_flow_mm3_s', 'flow_relative_difference_percent'])}

## 7. STAR surface cell-to-point変換

{_markdown_table(conversion_rows, ['source_name', 'boundary_name', 'native_cell_flow_mm3_s', 'converted_point_flow_mm3_s', 'absolute_difference_mm3_s', 'relative_difference_percent'])}

native surface cell速度を直接面積積分した値を主値とし、変換後point積分は影響確認用です。体積cell値を根拠なく境界へ投影していません。

## 8. 仮説の整理

{hypotheses}

この監査だけで原因を断定しません。観測が各仮説を支持するかどうかだけを整理しています。

## 9. 境界条件監査

{_markdown_table(bc_rows, ['item', 'FEM', 'STAR', 'matched', 'evidence_file', 'evidence_location', 'note'])}

## 10. 判定できなかった項目

{_markdown_table(undecided, ['item', 'FEM', 'STAR', 'note'])}

## 11. 次に推奨される解析

- STAR体積cell速度を内部断面でcell-wiseに積分し、既存cell-to-point断面値との差を分離する。
- 入口境界からupstream断面までについて、まず少数の確認断面だけでnative fluxを照合する。
- 境界条件入力ファイルを入手し、密度、粘度、入口profile、出口圧力、壁面条件を直接比較する。
- これらを確認するまでは新規計算、メッシュ変更、流量正規化比較へ進まない。

## 12. ユーザーへ確認する情報

- FEM実行時の完全なコマンドラインと入力設定
- STAR-CCM+の`.sim`または境界条件レポート
- 両ソルバーの密度・粘度
- 入口速度profileの定義方法
- 2出口の圧力・流出・逆流条件
- 壁面条件と圧力基準点
"""


def run_audit(config_path: str | Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    config = _load_json(config_path)
    sources = _mapping(config.get("data_sources"), "data_sources")
    boundaries = _mapping(config.get("boundaries"), "boundaries")
    if not sources:
        raise BoundaryAuditError("data_sources must not be empty")
    output_config = _mapping(config.get("output", {}), "output")
    output_dir = resolve_config_path(
        str(output_config.get("directory", "output/boundary_flow_audit")),
        config_path,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise BoundaryAuditError(
            f"output directory already contains files; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    flow_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    conversion_rows: list[dict[str, Any]] = []
    selection_diagnostics: dict[str, Any] = {}
    source_diagnostics: list[dict[str, Any]] = []
    inventories: dict[str, Any] = {}
    loaded_sources: dict[str, Any] = {}

    for source_name, raw_source in sources.items():
        source = _mapping(raw_source, f"data_sources.{source_name}")
        loaded = load_source(source_name, source, config_path)
        loaded_sources[source_name] = loaded
        inventories[source_name] = loaded["inventory"]
        source_diagnostics.append(
            {
                "source_name": source_name,
                "label": source.get("label", source_name),
                "path": str(loaded["path"]),
                "time_s": source.get("time", ""),
                "velocity_array": source.get("velocity_array"),
                "data_association": source.get("data_association"),
                "length_unit": source.get("length_unit", ""),
                "velocity_unit": source.get("velocity_unit", ""),
            }
        )
        source_boundaries = _mapping(
            boundaries.get(source_name), f"boundaries.{source_name}"
        )
        rows_for_source: list[dict[str, Any]] = []
        for boundary_name, raw_boundary in source_boundaries.items():
            boundary = _mapping(
                raw_boundary, f"boundaries.{source_name}.{boundary_name}"
            )
            row, conversion, diagnostic = audit_one_boundary(
                source_name,
                source,
                loaded,
                boundary_name,
                boundary,
                config_path,
            )
            flow_rows.append(row)
            rows_for_source.append(row)
            selection_diagnostics[f"{source_name}/{boundary_name}"] = diagnostic
            if conversion is not None:
                conversion_rows.append(conversion)
        balance_rows.append(
            {"source_name": source_name, **boundary_balance(rows_for_source)}
        )

    internal_rows: list[dict[str, Any]] = []
    internal_config = _mapping(config.get("internal_sections", {}), "internal_sections")
    summary_value = internal_config.get("summary_csv")
    if isinstance(summary_value, str) and summary_value:
        summary_path = resolve_config_path(summary_value, config_path)
        if not summary_path.is_file():
            raise BoundaryAuditError(f"internal section summary does not exist: {summary_path}")
        summary = pd.read_csv(summary_path)
        source_mapping = _mapping(
            internal_config.get("source_mapping", {}), "internal_sections.source_mapping"
        )
        boundary_mapping = _mapping(
            internal_config.get("boundary_mapping", {}), "internal_sections.boundary_mapping"
        )
        by_boundary = {
            (row["source_name"], row["boundary_name"]): row for row in flow_rows
        }
        for audit_source, section_source in source_mapping.items():
            for boundary_name, section_name in boundary_mapping.items():
                key = (audit_source, boundary_name)
                if key not in by_boundary:
                    raise BoundaryAuditError(f"internal mapping references missing boundary: {key}")
                matches = summary[
                    (summary["source_name"].astype(str) == str(section_source))
                    & (summary["section_name"].astype(str) == str(section_name))
                ]
                if len(matches) != 1:
                    raise BoundaryAuditError(
                        f"expected one internal summary row for {section_source}/{section_name}, got {len(matches)}"
                    )
                boundary_row = by_boundary[key]
                internal_flow = float(matches.iloc[0]["signed_flow_rate_mm3_s"])
                comparison = compare_boundary_internal(
                    float(boundary_row["signed_flow_mm3_s"]), internal_flow
                )
                internal_rows.append(
                    {
                        "source_name": audit_source,
                        "boundary_name": boundary_name,
                        "boundary_role": boundary_row["boundary_role"],
                        "internal_source_name": section_source,
                        "internal_section_name": section_name,
                        "boundary_flow": boundary_row["signed_flow_mm3_s"],
                        "internal_section_flow": internal_flow,
                        "signed_difference": comparison["signed_difference_mm3_s"],
                        "absolute_difference": comparison["absolute_difference_mm3_s"],
                        "boundary_flow_mm3_s": boundary_row["signed_flow_mm3_s"],
                        "internal_section_flow_mm3_s": internal_flow,
                        **comparison,
                    }
                )

    solver_rows = _solver_comparisons(flow_rows, config)
    bc_rows = [dict(_mapping(row, "boundary_condition_audit row")) for row in config.get("boundary_condition_audit", [])]
    diagnostics = {
        "configuration_path": str(config_path),
        "sources": source_diagnostics,
        "inventories": inventories,
        "boundary_selections": selection_diagnostics,
    }

    flow_path = output_dir / "boundary_flow_summary.csv"
    balance_path = output_dir / "boundary_balance_summary.csv"
    internal_path = output_dir / "boundary_vs_internal_sections.csv"
    conversion_path = output_dir / "star_cell_to_point_comparison.csv"
    solver_path = output_dir / "solver_boundary_comparison.csv"
    bc_path = output_dir / "boundary_condition_audit.csv"
    _write_csv(flow_path, flow_rows)
    _write_csv(balance_path, balance_rows)
    _write_csv(internal_path, internal_rows)
    _write_csv(conversion_path, conversion_rows)
    _write_csv(solver_path, solver_rows)
    _write_csv(bc_path, bc_rows)

    audit_payload = {
        "configuration": config,
        "diagnostics": diagnostics,
        "boundary_flow": flow_rows,
        "boundary_balance": balance_rows,
        "boundary_vs_internal": internal_rows,
        "cell_to_point_comparison": conversion_rows,
        "solver_boundary_comparison": solver_rows,
        "boundary_condition_audit": bc_rows,
    }
    json_path = output_dir / "boundary_audit.json"
    json_path.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = build_report(
        config,
        flow_rows,
        balance_rows,
        internal_rows,
        conversion_rows,
        solver_rows,
        bc_rows,
        diagnostics,
    )
    report_path = output_dir / "boundary_audit.md"
    report_path.write_text(report, encoding="utf-8")
    return {
        "output_dir": output_dir,
        "flow": flow_path,
        "balance": balance_path,
        "internal": internal_path,
        "conversion": conversion_path,
        "solver": solver_path,
        "boundary_conditions": bc_path,
        "json": json_path,
        "report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    try:
        outputs = run_audit(config_path)
    except (BoundaryAuditError, SurfaceFlowError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Boundary audit: {outputs['output_dir']}")
    for name, path in outputs.items():
        if name != "output_dir":
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
