#!/usr/bin/env python3
"""Compare existing Re=10/Re=100 FEM and STAR profile/branch-flow audits."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import date
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from audit_boundary_flow import _find_named_block, _scaled_copy, load_source
from audit_inlet_development_profiles import main as development_main
from audit_inlet_upstream_profile import (
    compute_location_profile,
    fem_internal_section_samples,
    main as profile_main,
    star_internal_section_samples,
)
from check_velocity_steady_state import mesh_hash, relative_l2
from section_config import load_section_library, resolve_section
from section_series import generate_section_series


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "audit_reynolds_profile_comparison.json"


class ReynoldsAuditError(ValueError):
    """Raised for invalid or inconsistent 4B-4 inputs."""


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base.parent / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReynoldsAuditError(f"{path}: top level must be an object")
    return value


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float:
    if value in (None, "", "unknown"):
        return math.nan
    return float(value)


def relative_percent(first: float, reference: float) -> float:
    return 100.0 * abs(first - reference) / abs(reference) if reference else math.nan


def calculate_reynolds(
    density_kg_m3: Any, velocity_mm_s: Any, diameter_mm: Any, viscosity_pa_s: Any
) -> float:
    rho, velocity, diameter, mu = map(
        number, (density_kg_m3, velocity_mm_s, diameter_mm, viscosity_pa_s)
    )
    if not all(math.isfinite(value) for value in (rho, velocity, diameter, mu)) or mu <= 0:
        return math.nan
    return rho * (velocity / 1000.0) * (diameter / 1000.0) / mu


def branch_partition(upstream: float, straight: float, side: float) -> dict[str, float]:
    upstream, straight, side = abs(upstream), abs(straight), abs(side)
    outlet = straight + side
    if upstream <= 0 or outlet <= 0:
        raise ReynoldsAuditError("branch flows must be non-zero")
    return {
        "upstream_flow_magnitude_mm3_s": upstream,
        "straight_flow_magnitude_mm3_s": straight,
        "side_flow_magnitude_mm3_s": side,
        "straight_branch_fraction_percent": 100.0 * straight / outlet,
        "side_branch_fraction_percent": 100.0 * side / outlet,
        "conservation_residual_percent": 100.0 * (outlet - upstream) / upstream,
    }


def safe_amplification(re100: float, re10: float, epsilon: float) -> tuple[float, str]:
    if not math.isfinite(re10) or abs(re10) <= epsilon:
        return math.nan, "not robust: Re=10 denominator is zero, non-finite, or extremely small"
    return re100 / re10, "ok"


def classify_reynolds_case(
    re10: Mapping[str, float],
    re100: Mapping[str, float],
    thresholds: Mapping[str, float],
    data_adequate: bool,
) -> dict[str, Any]:
    if not data_adequate:
        conditional = classify_reynolds_case(re10, re100, thresholds, True)["case"]
        return {
            "case": "R5",
            "conditional_observed_case": conditional,
            "reason": (
                "rho and dynamic viscosity are absent, so nominal Reynolds numbers cannot be "
                "independently recalculated. Numerical comparisons remain valid conditionally "
                "on the supplied Re labels."
            ),
        }
    profile10 = re10["profile_l2"] <= thresholds["normalized_profile_l2_percent"]
    profile100 = re100["profile_l2"] <= thresholds["normalized_profile_l2_percent"]
    split10 = re10["branch_fraction_pp"] <= thresholds[
        "branch_fraction_difference_percentage_points"
    ]
    split100 = re100["branch_fraction_pp"] <= thresholds[
        "branch_fraction_difference_percentage_points"
    ]
    flow10 = re10["upstream_flow"] <= thresholds["upstream_flow_difference_percent"]
    moments10 = (
        re10["beta"] <= thresholds["beta_relative_difference_percent"]
        and re10["alpha"] <= thresholds["alpha_relative_difference_percent"]
    )
    if flow10 and profile10 and moments10 and split10 and (not profile100 or not split100):
        case = "R1"
    elif not profile10 and split10:
        case = "R3"
    elif not split10:
        case = "R4"
    else:
        case = "R2"
    return {"case": case, "conditional_observed_case": case, "reason": "threshold-based observation"}


def validate_case(case_name: str, case: Mapping[str, Any], config_path: Path) -> None:
    for solver in ("fem", "star"):
        if solver not in case or not isinstance(case[solver], Mapping):
            raise ReynoldsAuditError(f"cases.{case_name}.{solver} is required")
        path = resolve_path(str(case[solver].get("path", "")), config_path)
        if not path.is_file():
            raise ReynoldsAuditError(f"{case_name}/{solver}: data file not found: {path}")
        if not str(case[solver].get("velocity_array", "")).strip():
            raise ReynoldsAuditError(f"{case_name}/{solver}: velocity_array is required")
    if float(case.get("nominal_reynolds_number", 0)) <= 0:
        raise ReynoldsAuditError(f"{case_name}: nominal_reynolds_number must be positive")


def resolved_source(
    case_name: str, solver: str, raw: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    result = dict(raw)
    result["path"] = str(resolve_path(str(raw["path"]), config_path))
    if "data_directory" in raw:
        result["data_directory"] = str(resolve_path(str(raw["data_directory"]), config_path))
    if "result_log" in raw:
        result["result_log"] = str(resolve_path(str(raw["result_log"]), config_path))
    if "boundary_mesh_path" in raw:
        result["boundary_mesh_path"] = str(
            resolve_path(str(raw["boundary_mesh_path"]), config_path)
        )
    result["case_name"] = case_name
    result["solver"] = solver
    return result


def inspect_inputs(
    cases: Mapping[str, Any], config_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    resolved: dict[str, Any] = {}
    conditions: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for case_name, raw_case in cases.items():
        validate_case(case_name, raw_case, config_path)
        case_result: dict[str, Any] = {}
        condition = raw_case["conditions"]
        recalculated = calculate_reynolds(
            condition.get("density_kg_m3"),
            condition.get("nominal_inlet_velocity_mm_s"),
            condition.get("characteristic_length_mm"),
            condition.get("dynamic_viscosity_pa_s"),
        )
        for solver in ("fem", "star"):
            source = resolved_source(case_name, solver, raw_case[solver], config_path)
            path = Path(source["path"])
            if solver == "fem":
                mesh = pv.read(path)
                velocity = np.asarray(mesh.point_data[source["velocity_array"]])
                source.update({
                    "point_count": int(mesh.n_points),
                    "cell_count": int(mesh.n_cells),
                    "bounds_in_configured_length_unit": list(map(float, mesh.bounds)),
                    "point_data_arrays": list(mesh.point_data),
                    "cell_data_arrays": list(mesh.cell_data),
                    "mesh_hash": mesh_hash(mesh),
                    "velocity_shape": list(velocity.shape),
                    "actual_data_association": "point",
                })
                directory = Path(source["data_directory"])
                files = sorted(directory.glob("solution_*.vtu"))
                source["available_timestep_count"] = len(files)
                source["first_timestep_file"] = str(files[0])
                source["last_timestep_file"] = str(files[-1])
            else:
                reader = pv.get_reader(path)
                times = list(map(float, reader.time_values))
                index = int(source["time_index"])
                if not 0 <= index < len(times):
                    raise ReynoldsAuditError(f"{case_name}/star: invalid time_index {index}")
                reader.set_active_time_point(index)
                raw = reader.read()
                volume = _find_named_block(raw, str(source["volume_part_name"]))
                if source["velocity_array"] not in volume.cell_data:
                    raise ReynoldsAuditError(
                        f"{case_name}/star: cell array {source['velocity_array']!r} not found"
                    )
                source.update({
                    "available_time_count": len(times),
                    "first_time_s": times[0],
                    "last_time_s": times[-1],
                    "actual_time_s": times[index],
                    "volume_point_count": int(volume.n_points),
                    "volume_cell_count": int(volume.n_cells),
                    "volume_bounds_m": list(map(float, volume.bounds)),
                    "volume_cell_data_arrays": list(volume.cell_data),
                    "actual_data_association": "cell",
                    "boundary_parts": [
                        "branch_duct_test.inlet1", "branch_duct_test.outlet1",
                        "branch_duct_test.outlet2", "branch_duct_test.wall",
                    ],
                })
            case_result[solver] = source
            conditions.append({
                "case": case_name,
                "solver": solver,
                "nominal_reynolds_number": raw_case["nominal_reynolds_number"],
                "density_kg_m3": condition.get("density_kg_m3", "unknown"),
                "dynamic_viscosity_pa_s": condition.get("dynamic_viscosity_pa_s", "unknown"),
                "nominal_inlet_velocity_mm_s": condition["nominal_inlet_velocity_mm_s"],
                "characteristic_length_mm": condition["characteristic_length_mm"],
                "reynolds_number_calculated": recalculated,
                "coordinate_unit": source["length_unit"],
                "velocity_unit": source["velocity_unit"],
                "physical_time_s": source["physical_time"],
                "timestep_index": source["timestep_index"],
                "dt_s": source["dt"],
                "mesh_identifier": source.get("mesh_hash", "STAR Ensight geometry"),
                "inlet_boundary_identifier": (
                    "Physical Surface 10 inlet" if solver == "fem"
                    else "branch_duct_test.inlet1"
                ),
                "straight_outlet_identifier": (
                    "Physical Surface 20 outlet_main" if solver == "fem"
                    else "branch_duct_test.outlet1"
                ),
                "side_outlet_identifier": (
                    "Physical Surface 30 outlet_branch" if solver == "fem"
                    else "branch_duct_test.outlet2"
                ),
                "wall_identifier": (
                    "Physical Surface 40 wall" if solver == "fem"
                    else "branch_duct_test.wall"
                ),
            })
        resolved[case_name] = case_result
        if not math.isfinite(recalculated):
            issues.append({
                "case": case_name, "severity": "warning",
                "item": "reynolds_number_calculated",
                "message": "density and dynamic viscosity input files are absent; calculated Re is unknown",
            })
    return resolved, conditions, issues


def runtime_configs(
    case_name: str,
    case: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    output: Path,
) -> tuple[Path, Path]:
    base = config["base_configs"]
    boundary = load_json(resolve_path(base["boundary_audit"], config_path))
    fem_name, star_name = f"fem_{case_name}", f"star_{case_name}"
    boundary["data_sources"] = {
        fem_name: {
            "label": f"FEM {case_name} at {case['fem']['physical_time']} s",
            "type": "fem_vtu", "path": str(resolve_path(case["fem"]["path"], config_path)),
            "time": case["fem"]["physical_time"],
            "velocity_array": case["fem"]["velocity_array"], "data_association": "point",
            "length_unit": case["fem"]["length_unit"], "velocity_unit": case["fem"]["velocity_unit"],
            "length_scale_to_mm": case["fem"]["length_scale_to_mm"],
            "velocity_scale_to_mm_s": case["fem"]["velocity_scale_to_mm_s"],
            "boundary_mesh_path": str(resolve_path(case["fem"]["boundary_mesh_path"], config_path)),
            "boundary_id_array": "gmsh:physical", "point_mapping_tolerance_mm": 1.0e-9,
        },
        star_name: {
            "label": f"STAR {case_name} at {case['star']['physical_time']} s",
            "type": "star_ensight", "path": str(resolve_path(case["star"]["path"], config_path)),
            "time_point": case["star"]["time_index"], "time": case["star"]["physical_time"],
            "velocity_array": case["star"]["velocity_array"], "data_association": "cell",
            "length_unit": case["star"]["length_unit"], "velocity_unit": case["star"]["velocity_unit"],
            "length_scale_to_mm": case["star"]["length_scale_to_mm"],
            "velocity_scale_to_mm_s": case["star"]["velocity_scale_to_mm_s"],
        },
    }
    boundary["boundaries"] = {
        fem_name: deepcopy(boundary["boundaries"]["fem_re100"]),
        star_name: deepcopy(boundary["boundaries"]["star_re100"]),
    }
    boundary_path = output / "_runtime_configs" / f"boundary_{case_name}.json"
    write_json(boundary_path, boundary)

    profile = load_json(resolve_path(base["profile_audit"], config_path))
    profile["boundary_audit_config"] = str(boundary_path)
    profile["section_library"] = str(resolve_path(config["section_library"], config_path))
    profile["solvers"] = {
        "fem": {"source_name": fem_name},
        "star": {"source_name": star_name, "volume_part_name": case["star"]["volume_part_name"]},
    }
    expected = case["conditions"]["nominal_inlet_velocity_mm_s"]
    profile["fem_inlet_audit"]["expected_velocity_mm_s"] = expected
    profile["star_inlet_audit"]["expected_velocity_mm_s"] = expected
    profile_output = output / "_case_runs" / case_name / "profile"
    profile["output"]["directory"] = str(profile_output)
    profile_path = output / "_runtime_configs" / f"profile_{case_name}.json"
    write_json(profile_path, profile)

    development = load_json(resolve_path(base["development_audit"], config_path))
    development["base_profile_audit_config"] = str(profile_path)
    development["section_library"] = str(resolve_path(config["section_library"], config_path))
    development["section_series"] = deepcopy(config["section_series"])
    development["position_sensitivity"] = deepcopy(config["position_sensitivity"])
    development["existing_boundary_audit"] = {
        "profile_metrics_summary": str(profile_output / "profile_metrics_summary.csv"),
        "common_grid_metrics": str(profile_output / "common_grid_metrics.csv"),
        "audit_json": str(profile_output / "inlet_upstream_profile_audit.json"),
    }
    development["output"]["directory"] = str(output / "_case_runs" / case_name / "development")
    development_path = output / "_runtime_configs" / f"development_{case_name}.json"
    write_json(development_path, development)
    return profile_path, development_path


def run_existing_audits(
    cases: Mapping[str, Any], config: Mapping[str, Any], config_path: Path, output: Path
) -> None:
    for case_name, case in cases.items():
        profile_path, development_path = runtime_configs(
            case_name, case, config, config_path, output
        )
        print(f"\n[{case_name}] existing 4B-2 profile audit")
        profile_main(["--config", str(profile_path)])
        print(f"\n[{case_name}] existing 4B-3 development audit")
        development_main(["--config", str(development_path)])


def centerline_metrics(
    grid_path: Path, profile_by_solver: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    with np.load(grid_path) as data:
        radius2 = np.asarray(data["s_grid"]) ** 2 + np.asarray(data["t_grid"]) ** 2
        index = np.unravel_index(int(np.argmin(radius2)), radius2.shape)
        result: dict[str, float] = {}
        for solver in ("fem", "star"):
            center = float(data[f"{solver}_flow_velocity_mm_s"][index])
            mean = float(profile_by_solver[solver]["mean_flow_velocity_mm_s"])
            result[f"{solver}_centerline_velocity_mm_s"] = center
            result[f"{solver}_centerline_to_mean_velocity_ratio"] = center / mean
        return result


def collect_development(
    case_name: str, output: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source = output / "_case_runs" / case_name / "development"
    profiles = [dict(row, case=case_name) for row in read_csv(source / "profile_development_summary.csv")]
    comparisons = [dict(row, case=case_name) for row in read_csv(source / "fem_star_profile_comparison.csv")]
    sensitivity = [dict(row, case=case_name) for row in read_csv(source / "position_sensitivity_summary.csv")]
    by_key = {(row["section_name"], row["solver"]): row for row in profiles}
    for comparison in comparisons:
        section = comparison["section_name"]
        center = centerline_metrics(
            source / "grids" / f"{section}_common_grid.npz",
            {solver: by_key[(section, solver)] for solver in ("fem", "star")},
        )
        comparison.update(center)
        comparison["centerline_to_mean_ratio_difference"] = abs(
            center["fem_centerline_to_mean_velocity_ratio"]
            - center["star_centerline_to_mean_velocity_ratio"]
        )
        for solver in ("fem", "star"):
            row = by_key[(section, solver)]
            row["centerline_velocity_mm_s"] = center[f"{solver}_centerline_velocity_mm_s"]
            row["centerline_to_mean_velocity_ratio"] = center[
                f"{solver}_centerline_to_mean_velocity_ratio"
            ]
            row["centerline_to_mean_difference_from_fully_developed"] = (
                number(row["centerline_to_mean_velocity_ratio"]) - 2.0
            )
    destination = output / "grids" / case_name
    destination.mkdir(parents=True, exist_ok=True)
    for path in (source / "grids").glob("*.npz"):
        shutil.copy2(path, destination / path.name)
    return profiles, comparisons, sensitivity


def load_case_meshes(
    case_name: str, case: Mapping[str, Any], config_path: Path
) -> tuple[pv.DataSet, pv.DataSet]:
    fem_source = {
        "type": "fem_vtu", "path": str(resolve_path(case["fem"]["path"], config_path)),
        "velocity_array": case["fem"]["velocity_array"],
        "length_scale_to_mm": case["fem"]["length_scale_to_mm"],
        "velocity_scale_to_mm_s": case["fem"]["velocity_scale_to_mm_s"],
    }
    star_source = {
        "type": "star_ensight", "path": str(resolve_path(case["star"]["path"], config_path)),
        "time_point": case["star"]["time_index"], "velocity_array": case["star"]["velocity_array"],
        "length_scale_to_mm": case["star"]["length_scale_to_mm"],
        "velocity_scale_to_mm_s": case["star"]["velocity_scale_to_mm_s"],
    }
    fem = load_source(f"{case_name}_fem", fem_source, config_path)["primary"]
    loaded_star = load_source(f"{case_name}_star", star_source, config_path)
    star = _scaled_copy(
        _find_named_block(loaded_star["raw"], case["star"]["volume_part_name"]), star_source
    )
    return fem, star


def branch_rows(
    cases: Mapping[str, Any], config: Mapping[str, Any], config_path: Path
) -> list[dict[str, Any]]:
    library = load_section_library(resolve_path(config["section_library"], config_path))
    rows: list[dict[str, Any]] = []
    directions = {
        "upstream_z30": [0.0, 0.0, -1.0],
        "straight_z10": [0.0, 0.0, -1.0],
        "side_branch": [1.0, 0.0, -1.0],
    }
    for case_name, case in cases.items():
        fem, star = load_case_meshes(case_name, case, config_path)
        by_solver: dict[str, dict[str, float]] = {"fem": {}, "star": {}}
        for section_name in directions:
            section = resolve_section(section_name, library)
            section["flow_direction_normal"] = directions[section_name]
            fem_samples = fem_internal_section_samples(
                fem, case["fem"]["velocity_array"], section
            )
            star_samples, star_extraction, _, _, _ = star_internal_section_samples(
                star, case["star"]["velocity_array"], section, 1.0e-12
            )
            fem_extraction = {
                "intersected_cell_count": fem_samples["element_count"],
                "generated_polygon_count": fem_samples["element_count"],
                "valid_polygon_count": fem_samples["element_count"],
                "unmapped_polygon_count": 0,
                "duplicate_original_cell_count": 0,
            }
            for solver, samples, extraction in (
                ("fem", fem_samples, fem_extraction),
                ("star", star_samples, star_extraction),
            ):
                profile = compute_location_profile(
                    solver, section_name, samples, section, 0.1
                )
                by_solver[solver][section_name] = float(profile["signed_flow_mm3_s"])
                rows.append({
                    "case": case_name, "solver": solver, "section_name": section_name,
                    **profile,
                    "intersected_cell_count": extraction.get(
                        "intersected_cell_count", extraction.get("intersected_volume_cell_count")
                    ),
                    "generated_polygon_count": extraction["generated_polygon_count"],
                    "valid_polygon_count": extraction["valid_polygon_count"],
                    "unmapped_polygon_count": extraction["unmapped_polygon_count"],
                    "duplicate_original_cell_count": extraction.get(
                        "duplicate_original_cell_count", 0
                    ),
                })
        for solver in ("fem", "star"):
            flow = by_solver[solver]
            partition = branch_partition(
                flow["upstream_z30"], flow["straight_z10"], flow["side_branch"]
            )
            for row in rows:
                if row["case"] == case_name and row["solver"] == solver:
                    row.update(partition)
    return rows


def inlet_rows(case_names: list[str], output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_name in case_names:
        source = output / "_case_runs" / case_name / "profile"
        profiles = read_csv(source / "profile_metrics_summary.csv")
        audit = load_json(source / "inlet_upstream_profile_audit.json")
        for row in profiles:
            if row["location"] == "inlet":
                out = dict(row, case=case_name)
                if row["solver"] == "fem":
                    out.update({
                        f"fem_node_{key}": value
                        for key, value in audit["fem_inlet_node_audit"].items()
                    })
                else:
                    out.update({
                        f"star_uniformity_{key}": value
                        for key, value in audit["star_inlet_uniformity"].items()
                    })
                rows.append(out)
    return rows


def steady_state_rows(
    cases: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
) -> list[dict[str, Any]]:
    steps = list(map(int, config["steady_state"]["last_timestep_indices"]))
    library = load_section_library(resolve_path(config["section_library"], config_path))
    sections = {
        name: resolve_section(name, library)
        for name in ("upstream_z30", "straight_z10", "side_branch")
    }
    directions = {
        "upstream_z30": [0, 0, -1], "straight_z10": [0, 0, -1],
        "side_branch": [1, 0, -1],
    }
    rows: list[dict[str, Any]] = []
    for case_name, case in cases.items():
        previous: dict[str, np.ndarray] = {}
        previous_section: dict[str, np.ndarray] = {}
        star_reader = pv.get_reader(resolve_path(case["star"]["path"], config_path))
        for step in steps:
            fem_path = resolve_path(case["fem"]["data_directory"], config_path) / f"solution_{step:06d}.vtu"
            fem_mesh = pv.read(fem_path)
            star_reader.set_active_time_point(step - 1)
            star_source = case["star"]
            star_mesh = _scaled_copy(
                _find_named_block(star_reader.read(), star_source["volume_part_name"]),
                star_source,
            )
            for solver, mesh, velocity_name in (
                ("fem", fem_mesh, case["fem"]["velocity_array"]),
                ("star", star_mesh, case["star"]["velocity_array"]),
            ):
                velocity = np.asarray(
                    mesh.point_data[velocity_name] if solver == "fem"
                    else mesh.cell_data[velocity_name],
                    dtype=float,
                )
                key = f"{case_name}/{solver}"
                row: dict[str, Any] = {
                    "case": case_name, "solver": solver, "timestep_index": step,
                    "physical_time_s": step * float(case[solver]["dt"]),
                    "whole_field_relative_l2_from_previous_percent": (
                        relative_l2(velocity, previous[key]) if key in previous else math.nan
                    ),
                }
                previous[key] = velocity.copy()
                for name, section in sections.items():
                    section["flow_direction_normal"] = directions[name]
                    if solver == "fem":
                        samples = fem_internal_section_samples(mesh, velocity_name, section)
                    else:
                        samples, _, _, _, _ = star_internal_section_samples(
                            mesh, velocity_name, section, 1.0e-12
                        )
                    profile = compute_location_profile(solver, name, samples, section, 0.1)
                    section_key = f"{case_name}/{solver}/{name}"
                    section_velocity = np.asarray(samples["velocities"], dtype=float)
                    row[f"{name}_velocity_relative_l2_from_previous_percent"] = (
                        relative_l2(section_velocity, previous_section[section_key])
                        if section_key in previous_section else math.nan
                    )
                    previous_section[section_key] = section_velocity.copy()
                    row[f"{name}_signed_flow_mm3_s"] = profile["signed_flow_mm3_s"]
                rows.append(row)
    return rows


def make_comparison_rows(
    comparisons: list[dict[str, Any]], branches: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_case_section = {(r["case"], r["section_name"]): r for r in comparisons}
    summary: list[dict[str, Any]] = []
    for case in sorted({r["case"] for r in comparisons}):
        row = by_case_section[(case, "inlet_development_z30p0")]
        summary.append({
            "case": case, "comparison": "z30",
            "upstream_flow_difference_percent": number(
                row["relative_difference_percent_flow_magnitude_mm3_s"]
            ),
            "normalized_velocity_l2_percent": number(
                row["normalized_velocity_relative_l2_percent"]
            ),
            "dimensional_velocity_l2_percent": number(
                row["dimensional_velocity_relative_l2_percent"]
            ),
            "beta_fem": number(row["fem_beta"]), "beta_star": number(row["star_beta"]),
            "beta_difference_percent": number(row["relative_difference_percent_beta"]),
            "alpha_fem": number(row["fem_alpha"]), "alpha_star": number(row["star_alpha"]),
            "alpha_difference_percent": number(row["relative_difference_percent_alpha"]),
            "normalized_velocity_std_fem": number(row["fem_normalized_velocity_std"]),
            "normalized_velocity_std_star": number(row["star_normalized_velocity_std"]),
            "centerline_to_mean_fem": number(row["fem_centerline_to_mean_velocity_ratio"]),
            "centerline_to_mean_star": number(row["star_centerline_to_mean_velocity_ratio"]),
            "flux_centroid_distance_mm": number(row["flux_centroid_distance_mm"]),
            "secondary_velocity_ratio_difference_percentage_points": number(
                row["secondary_velocity_ratio_difference_percentage_points"]
            ),
        })
    branch_summary: list[dict[str, Any]] = []
    for case in sorted({r["case"] for r in branches}):
        selected = {
            (r["solver"], r["section_name"]): r for r in branches if r["case"] == case
        }
        fem = selected[("fem", "upstream_z30")]
        star = selected[("star", "upstream_z30")]
        branch_summary.append({
            "case": case,
            "upstream_flow_difference_percent": relative_percent(
                number(fem["upstream_flow_magnitude_mm3_s"]),
                number(star["upstream_flow_magnitude_mm3_s"]),
            ),
            "straight_branch_flow_difference_percent": relative_percent(
                number(fem["straight_flow_magnitude_mm3_s"]),
                number(star["straight_flow_magnitude_mm3_s"]),
            ),
            "side_branch_flow_difference_percent": relative_percent(
                number(fem["side_flow_magnitude_mm3_s"]),
                number(star["side_flow_magnitude_mm3_s"]),
            ),
            "straight_branch_fraction_fem_percent": number(
                fem["straight_branch_fraction_percent"]
            ),
            "straight_branch_fraction_star_percent": number(
                star["straight_branch_fraction_percent"]
            ),
            "straight_branch_fraction_difference_percentage_points": abs(
                number(fem["straight_branch_fraction_percent"])
                - number(star["straight_branch_fraction_percent"])
            ),
            "side_branch_fraction_fem_percent": number(fem["side_branch_fraction_percent"]),
            "side_branch_fraction_star_percent": number(star["side_branch_fraction_percent"]),
            "side_branch_fraction_difference_percentage_points": abs(
                number(fem["side_branch_fraction_percent"])
                - number(star["side_branch_fraction_percent"])
            ),
            "conservation_residual_fem_percent": number(fem["conservation_residual_percent"]),
            "conservation_residual_star_percent": number(star["conservation_residual_percent"]),
        })
    return summary, branch_summary


def effect_rows(
    comparisons: list[dict[str, Any]],
    z30: list[dict[str, Any]],
    branch: list[dict[str, Any]],
    epsilon: float,
) -> list[dict[str, Any]]:
    by = {(r["case"], r["section_name"]): r for r in comparisons}
    rows: list[dict[str, Any]] = []
    metrics = [
        ("normalized_velocity_l2", "normalized_velocity_relative_l2_percent"),
        ("dimensional_velocity_l2", "dimensional_velocity_relative_l2_percent"),
        ("beta_relative_difference", "relative_difference_percent_beta"),
        ("alpha_relative_difference", "relative_difference_percent_alpha"),
        ("upstream_flow_difference", "relative_difference_percent_flow_magnitude_mm3_s"),
    ]
    for name, field in metrics:
        low = number(by[("re10", "inlet_development_z30p0")][field])
        high = number(by[("re100", "inlet_development_z30p0")][field])
        factor, status = safe_amplification(high, low, epsilon)
        rows.append({
            "metric": name, "location": "z30", "difference_re10": low,
            "difference_re100": high, "amplification_factor": factor,
            "amplification_status": status,
        })
    b = {row["case"]: row for row in branch}
    for name, field in (
        ("straight_branch_fraction_difference", "straight_branch_fraction_difference_percentage_points"),
        ("side_branch_fraction_difference", "side_branch_fraction_difference_percentage_points"),
        ("straight_branch_flow_difference", "straight_branch_flow_difference_percent"),
        ("side_branch_flow_difference", "side_branch_flow_difference_percent"),
    ):
        low, high = number(b["re10"][field]), number(b["re100"][field])
        factor, status = safe_amplification(high, low, epsilon)
        rows.append({
            "metric": name, "location": "branch_partition", "difference_re10": low,
            "difference_re100": high, "amplification_factor": factor,
            "amplification_status": status,
        })
    return rows


def plot_line(
    path: Path, rows: list[Mapping[str, Any]], x: str, ys: list[tuple[str, str]],
    ylabel: str, title: str, dpi: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ordered = sorted(rows, key=lambda row: number(row[x]))
    for field, label in ys:
        ax.plot([number(r[x]) for r in ordered], [number(r[field]) for r in ordered],
                marker="o", label=label)
    ax.set(xlabel="Distance from inlet / D", ylabel=ylabel, title=title)
    ax.grid(True, alpha=.3)
    if len(ys) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_bar(path: Path, labels: list[str], series: list[tuple[str, list[float]]],
             ylabel: str, title: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels)); width = .75 / len(series)
    for index, (name, values) in enumerate(series):
        ax.bar(x + (index - (len(series)-1)/2)*width, values, width, label=name)
    ax.set_xticks(x, labels)
    ax.set(ylabel=ylabel, title=title)
    ax.grid(True, axis="y", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def create_plots(
    profiles: list[dict[str, Any]], comparisons: list[dict[str, Any]],
    branch: list[dict[str, Any]], output: Path, dpi: int
) -> None:
    for case in ("re10", "re100"):
        comp = [r for r in comparisons if r["case"] == case]
        prof = [r for r in profiles if r["case"] == case]
        target = output / "figures" / case
        plot_line(target / f"normalized_velocity_l2_vs_distance_{case}.png", comp,
                  "distance_from_inlet_over_diameter",
                  [("normalized_velocity_relative_l2_percent", "FEM–STAR common-grid")],
                  "Normalized profile L2 [%]", f"{case}: common-grid profile difference", dpi)
        profile_by_key = {(r["section_name"], r["solver"]): r for r in prof}
        profile_plot_rows = []
        for section_name in sorted({r["section_name"] for r in prof}):
            fem_row = profile_by_key[(section_name, "fem")]
            star_row = profile_by_key[(section_name, "star")]
            profile_plot_rows.append({
                "distance_from_inlet_over_diameter": fem_row["distance_from_inlet_over_diameter"],
                **{f"fem_{key}": value for key, value in fem_row.items()},
                **{f"star_{key}": value for key, value in star_row.items()},
            })
        for field, filename, ylabel in (
            ("momentum_correction_factor_beta", f"beta_vs_distance_{case}.png", "β [native/formal]"),
            ("kinetic_energy_correction_factor_alpha", f"alpha_vs_distance_{case}.png", "α [native/formal]"),
            ("flow_magnitude_mm3_s", f"flow_vs_distance_{case}.png", "|Q| [mm³/s, native/formal]"),
        ):
            plot_line(target / filename, profile_plot_rows, "distance_from_inlet_over_diameter",
                      [(f"fem_{field}", "FEM"), (f"star_{field}", "STAR")],
                      ylabel, f"{case}: {ylabel}", dpi)
        b = {(r["solver"], r["section_name"]): r for r in branch if r["case"] == case}
        plot_bar(target / f"branch_fraction_{case}.png", ["Straight", "Side"], [
            ("FEM", [number(b[("fem","upstream_z30")]["straight_branch_fraction_percent"]),
                     number(b[("fem","upstream_z30")]["side_branch_fraction_percent"])]),
            ("STAR", [number(b[("star","upstream_z30")]["straight_branch_fraction_percent"]),
                      number(b[("star","upstream_z30")]["side_branch_fraction_percent"])])
        ], "Fraction [% of outlet magnitudes]", f"{case}: native/formal branch partition", dpi)

    target = output / "figures" / "reynolds_comparison"
    by_case = {case: [r for r in comparisons if r["case"] == case] for case in ("re10","re100")}
    for field, filename, ylabel in (
        ("normalized_velocity_relative_l2_percent", "normalized_velocity_l2_re10_vs_re100.png", "Normalized L2 [%]"),
        ("dimensional_velocity_relative_l2_percent", "dimensional_velocity_l2_re10_vs_re100.png", "Dimensional L2 [%]"),
        ("relative_difference_percent_beta", "beta_difference_re10_vs_re100.png", "β relative difference [%]"),
        ("relative_difference_percent_alpha", "alpha_difference_re10_vs_re100.png", "α relative difference [%]"),
        ("relative_difference_percent_flow_magnitude_mm3_s", "upstream_flow_difference_re10_vs_re100.png", "Flow difference [%]"),
    ):
        rows = []
        for case, selected in by_case.items():
            for row in selected:
                rows.append({**row, f"{field}_{case}": row[field]})
        merged = []
        keys = sorted({r["section_name"] for r in comparisons})
        for key in keys:
            merged.append({
                "distance_from_inlet_over_diameter": next(
                    number(r["distance_from_inlet_over_diameter"]) for r in comparisons
                    if r["section_name"] == key
                ),
                f"{field}_re10": next(r[field] for r in by_case["re10"] if r["section_name"]==key),
                f"{field}_re100": next(r[field] for r in by_case["re100"] if r["section_name"]==key),
            })
        plot_line(target / filename, merged, "distance_from_inlet_over_diameter",
                  [(f"{field}_re10","Re=10"),(f"{field}_re100","Re=100")],
                  ylabel, f"Reynolds comparison: {ylabel}", dpi)
    plot_line(target / "profile_development_re10_vs_re100.png", [
        {
            "distance_from_inlet_over_diameter": row["distance_from_inlet_over_diameter"],
            "re10": next(r["normalized_velocity_relative_l2_percent"] for r in by_case["re10"]
                         if r["section_name"]==row["section_name"]),
            "re100": next(r["normalized_velocity_relative_l2_percent"] for r in by_case["re100"]
                          if r["section_name"]==row["section_name"]),
        } for row in by_case["re10"]
    ], "distance_from_inlet_over_diameter", [("re10","Re=10"),("re100","Re=100")],
       "Normalized L2 [%]", "Profile development (common-grid)", dpi)
    branch_by = {(r["case"],r["solver"],r["section_name"]):r for r in branch}
    labels = ["Re=10","Re=100"]
    straight_diff=[]; side_diff=[]; straight_fem=[]; straight_star=[]; side_fem=[]; side_star=[]
    for case in ("re10","re100"):
        f=branch_by[(case,"fem","upstream_z30")];s=branch_by[(case,"star","upstream_z30")]
        straight_diff.append(abs(number(f["straight_branch_fraction_percent"])-number(s["straight_branch_fraction_percent"])))
        side_diff.append(abs(number(f["side_branch_fraction_percent"])-number(s["side_branch_fraction_percent"])))
        straight_fem.append(number(f["straight_branch_fraction_percent"]));straight_star.append(number(s["straight_branch_fraction_percent"]))
        side_fem.append(number(f["side_branch_fraction_percent"]));side_star.append(number(s["side_branch_fraction_percent"]))
    plot_bar(target/"branch_fraction_difference_re10_vs_re100.png",labels,[("Straight Δ",straight_diff),("Side Δ",side_diff)],"Difference [percentage points]","Branch-fraction FEM–STAR difference",dpi)
    plot_bar(target/"straight_branch_fraction_re10_vs_re100.png",labels,[("FEM",straight_fem),("STAR",straight_star)],"Straight fraction [%]","Straight branch fraction",dpi)
    plot_bar(target/"side_branch_fraction_re10_vs_re100.png",labels,[("FEM",side_fem),("STAR",side_star)],"Side fraction [%]","Side branch fraction",dpi)

    for token in ("z49p5", "z48p0", "z45p0", "z30p0"):
        section = f"inlet_development_{token}"
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        for row_index, case in enumerate(("re10","re100")):
            with np.load(output/"grids"/case/f"{section}_common_grid.npz") as data:
                valid=np.asarray(data["common_valid"],bool)
                for col, solver in enumerate(("fem","star")):
                    values=np.where(valid,data[f"{solver}_normalized_flow_velocity"],np.nan)
                    image=axes[row_index,col].pcolormesh(data["s_grid"],data["t_grid"],values,
                                                         shading="auto",vmin=0,vmax=2,cmap="viridis")
                    axes[row_index,col].set(title=f"{case} {solver.upper()} common-grid",aspect="equal",
                                            xlabel="s [mm]",ylabel="t [mm]")
        fig.colorbar(image,ax=axes.ravel().tolist(),label="u_n / formal mean [-]")
        fig.suptitle(f"{section}: normalized common-grid profiles")
        fig.savefig(target/f"{token}_normalized_profile_re10_vs_re100.png",dpi=dpi,bbox_inches="tight")
        plt.close(fig)


def regression_check(output: Path) -> dict[str, Any]:
    new_dir = output / "_case_runs" / "re100" / "development"
    old_dir = ROOT / "output" / "inlet_development_profile_audit" / "re100"
    results: dict[str, Any] = {}
    for filename in ("profile_development_summary.csv", "fem_star_profile_comparison.csv",
                     "position_sensitivity_summary.csv"):
        new, old = read_csv(new_dir/filename), read_csv(old_dir/filename)
        maximum = 0.0
        for new_row, old_row in zip(new, old):
            for key in set(new_row) & set(old_row):
                try:
                    a, b = float(new_row[key]), float(old_row[key])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(a) and math.isfinite(b):
                    maximum = max(maximum, abs(a-b))
        results[filename] = {"row_count_new":len(new),"row_count_existing":len(old),
                             "maximum_absolute_numeric_difference":maximum}
    return results


def build_markdown(audit: Mapping[str, Any]) -> str:
    decision = audit["decision"]
    lines = [
        "# Step 4B-4 Reynolds profile comparison", "",
        f"Generated: {date.today().isoformat()}", "",
        "Re=10 and Re=100 use the same seven internal sections. FEM formal values use "
        "point-linear triangle quadrature; STAR formal values use native volume-cell "
        "intersection polygons. Common grids are used only for profile comparison and figures.", "",
        "## Decision", "",
        f"Primary case: **{decision['case']}**. Conditional numerical pattern: "
        f"**{decision['conditional_observed_case']}**.", "",
        decision["reason"], "",
        "This audit does not establish causation. A mechanical issue (Re-dependent inertia and "
        "momentum partition sensitivity) is distinct from a numerical issue (mesh resolution, "
        "discretization, numerical diffusion, time-step/CFL, or field representation).", "",
        "## Important limitation", "",
        "Original density and dynamic-viscosity inputs are absent. Nominal inlet velocities and "
        "geometry are observable, but Re=rho*U*D/mu cannot be independently recalculated. STAR "
        "pressure and original solver/STAR setup files are also absent.", "",
        "## Reproduction", "", "```bash", "cd /workspace",
        "python scripts/audit_reynolds_profile_comparison.py --config config/audit_reynolds_profile_comparison.json",
        "python -m unittest discover -s tests -p 'test_*.py' -v", "```", "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config_path = args.config.resolve() if args.config.is_absolute() else (ROOT/args.config).resolve()
    config = load_json(config_path)
    cases = config.get("cases")
    if not isinstance(cases, dict) or set(cases) != {"re10", "re100"}:
        raise ReynoldsAuditError("cases must contain exactly re10 and re100")
    output = resolve_path(config["output"]["directory"], config_path)
    if config["output"].get("refuse_nonempty_output_directory", True) and output.exists() and any(output.iterdir()):
        raise ReynoldsAuditError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Configuration: {config_path}\nOutput: {output}")

    resolved, conditions, issues = inspect_inputs(cases, config_path)
    write_json(output/"resolved_input_data.json", resolved)
    write_csv(output/"simulation_conditions_summary.csv", conditions)
    run_existing_audits(cases, config, config_path, output)

    profiles: list[dict[str, Any]]=[]; comparisons: list[dict[str, Any]]=[]; sensitivity: list[dict[str, Any]]=[]
    for case_name in cases:
        p,c,s=collect_development(case_name,output);profiles+=p;comparisons+=c;sensitivity+=s
    branches=branch_rows(cases,config,config_path)
    inlets=inlet_rows(list(cases),output)
    steady=steady_state_rows(cases,config,config_path)
    z30,branch_summary=make_comparison_rows(comparisons,branches)
    effects=effect_rows(comparisons,z30,branch_summary,
                        float(config["decision_thresholds"]["amplification_denominator_epsilon"]))

    write_csv(output/"inlet_boundary_comparison.csv",inlets)
    write_csv(output/"profile_development_by_reynolds.csv",profiles)
    write_csv(output/"fem_star_comparison_by_reynolds.csv",comparisons)
    write_csv(output/"branch_flow_comparison_by_reynolds.csv",branches)
    write_csv(output/"reynolds_effect_summary.csv",effects)
    write_csv(output/"position_sensitivity_by_reynolds.csv",sensitivity)
    write_csv(output/"issues.csv",issues)

    z={row["case"]:row for row in z30}; b={row["case"]:row for row in branch_summary}
    t=config["decision_thresholds"]
    evidence={}
    for case in ("re10","re100"):
        evidence[case]={
            "profile_l2":z[case]["normalized_velocity_l2_percent"],
            "upstream_flow":z[case]["upstream_flow_difference_percent"],
            "beta":z[case]["beta_difference_percent"],"alpha":z[case]["alpha_difference_percent"],
            "branch_fraction_pp":max(
                b[case]["straight_branch_fraction_difference_percentage_points"],
                b[case]["side_branch_fraction_difference_percentage_points"]),
        }
    decision=classify_reynolds_case(evidence["re10"],evidence["re100"],t,False)
    regressions=regression_check(output)
    audit={
        "configuration":str(config_path),"output_directory":str(output),
        "resolved_input_data":resolved,"simulation_conditions":conditions,
        "steady_state_last_three_steps":steady,"inlet_boundary_comparison":inlets,
        "profile_development":profiles,"fem_star_comparisons":comparisons,
        "branch_flows":branches,"z30_comparison":z30,"branch_comparison":branch_summary,
        "reynolds_effects":effects,"position_sensitivity":sensitivity,
        "decision":decision,"decision_evidence":evidence,"thresholds":t,
        "regression_checks":regressions,"issues":issues,
        "methods":{
            "fem_formal":"existing point-linear triangle quadrature",
            "star_formal":"existing native volume-cell intersection polygon integration",
            "common_grid":"existing linear triangular interpolation; profile/visualization only",
            "branch_fraction":"absolute straight and side formal flow divided by their sum",
            "sign_convention":"section normal defines signed flow; magnitudes define partition",
        },
        "limitations":[
            "rho and dynamic viscosity are unavailable, so nominal Re cannot be independently recalculated.",
            "STAR pressure and original setup files are unavailable.",
            "Observed Re dependence does not identify a mechanical or numerical cause.",
        ],
    }
    write_json(output/"reynolds_profile_comparison.json",audit)
    (output/"reynolds_profile_comparison.md").write_text(build_markdown(audit),encoding="utf-8")
    create_plots(profiles,comparisons,branches,output,int(config["output"].get("dpi",180)))
    print(f"Decision: {decision['case']} (conditional {decision['conditional_observed_case']})")
    print(f"Re=100 regression maximum differences: {regressions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
