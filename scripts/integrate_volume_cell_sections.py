#!/usr/bin/env python3
"""Integrate STAR internal-section flow using native volume-cell velocities."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyvista as pv

from section_config import SectionConfigError, resolve_sections_from_config
from surface_flow import SurfaceFlowError, integrate_native_volume_cell_section


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "integrate_volume_cell_sections.json"

SUMMARY_COLUMNS = [
    "source_name", "source_label", "section_name", "section_label",
    "center_x", "center_y", "center_z", "normal_x", "normal_y", "normal_z",
    "width_mm", "height_mm", "native_intersection_area_mm2",
    "signed_flow_mm3_s", "absolute_flow_mm3_s", "signed_flow_ml_s",
    "signed_flow_ml_min", "signed_flow_ml_h", "area_mean_normal_velocity_mm_s",
    "area_weighted_mean_speed_mm_s", "maximum_cell_speed_mm_s",
    "intersected_volume_cell_count", "generated_polygon_count",
    "valid_polygon_count", "invalid_polygon_count",
    "duplicate_original_cell_count", "unmapped_polygon_count",
    "minimum_polygon_area_mm2", "maximum_polygon_area_mm2", "status", "warning",
]
POLYGON_COLUMNS = [
    "original_volume_cell_id", "polygon_id", "polygon_area_mm2",
    "velocity_x_mm_s", "velocity_y_mm_s", "velocity_z_mm_s",
    "normal_velocity_mm_s", "signed_flow_contribution_mm3_s",
    "polygon_centroid_x_mm", "polygon_centroid_y_mm", "polygon_centroid_z_mm",
    "valid",
]
COMPARISON_COLUMNS = [
    "section_name", "boundary_name", "native_volume_cell_flow_mm3_s",
    "current_point_converted_flow_mm3_s", "native_boundary_flow_mm3_s",
    "native_vs_point_signed_difference", "native_vs_point_magnitude_difference",
    "native_vs_point_absolute_difference",
    "native_vs_point_relative_difference_percent",
    "native_vs_boundary_signed_difference", "native_vs_boundary_magnitude_difference",
    "native_vs_boundary_absolute_difference",
    "native_vs_boundary_relative_difference_percent",
    "point_vs_boundary_signed_difference", "point_vs_boundary_magnitude_difference",
    "point_vs_boundary_absolute_difference",
    "point_vs_boundary_relative_difference_percent",
    "native_intersection_area_mm2", "boundary_area_mm2", "area_difference_percent",
    "native_point_sign_relation", "native_boundary_sign_relation",
    "point_boundary_sign_relation",
]


class NativeSectionError(ValueError):
    """Raised for invalid native volume-cell integration configuration/data."""


def resolve_config_path(value: str | Path, config_path: str | Path) -> Path:
    """Resolve every relative path against the configuration JSON directory."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config_path).resolve().parent / path).resolve()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeSectionError(f"{context} must be a JSON object")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise NativeSectionError(f"configuration file does not exist: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NativeSectionError(f"invalid JSON in {path}: {exc}") from exc
    return dict(_mapping(config, "configuration"))


def _find_named_dataset(data: pv.DataObject, target: str) -> pv.DataSet:
    matches: list[pv.DataSet] = []

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
        raise NativeSectionError(f"volume part/block was not found: {target!r}")
    if len(matches) > 1:
        raise NativeSectionError(f"volume part/block is ambiguous: {target!r}")
    return matches[0]


def _cell_type_counts(mesh: pv.DataSet) -> dict[str, int]:
    if not hasattr(mesh, "celltypes"):
        return {}
    values, counts = np.unique(np.asarray(mesh.celltypes), return_counts=True)
    result: dict[str, int] = {}
    for value, count in zip(values, counts):
        try:
            name = pv.CellType(int(value)).name
        except (ValueError, AttributeError):
            name = f"VTK_{int(value)}"
        result[f"{int(value)}:{name}"] = int(count)
    return result


def scale_native_volume_mesh(
    mesh: pv.DataSet,
    velocity_name: str,
    length_scale_to_mm: float,
    velocity_scale_to_mm_s: float,
) -> pv.DataSet:
    """Return a scaled copy without changing cell association."""

    if not math.isfinite(length_scale_to_mm) or length_scale_to_mm <= 0.0:
        raise NativeSectionError("length_scale_to_mm must be positive")
    if not math.isfinite(velocity_scale_to_mm_s) or velocity_scale_to_mm_s <= 0.0:
        raise NativeSectionError("velocity_scale_to_mm_s must be positive")
    if velocity_name not in mesh.cell_data:
        raise NativeSectionError(f"cell velocity {velocity_name!r} is missing")
    velocity = np.asarray(mesh.cell_data[velocity_name], dtype=float)
    if velocity.shape != (mesh.n_cells, 3):
        raise NativeSectionError("cell velocity must have three components")
    result = mesh.copy(deep=True)
    result.points = np.asarray(result.points, dtype=float) * length_scale_to_mm
    result.cell_data[velocity_name] = velocity * velocity_scale_to_mm_s
    return result


def load_native_volume_source(
    source_name: str, source: Mapping[str, Any], config_path: Path
) -> tuple[pv.DataSet, dict[str, Any]]:
    path = resolve_config_path(str(source.get("path", "")), config_path)
    if not path.is_file():
        raise NativeSectionError(f"{source_name}: input file does not exist: {path}")
    source_type = str(source.get("type", "")).lower()
    if source_type not in {"star_ensight", "ensight", "case", "star_ccm"}:
        raise NativeSectionError(
            f"{source_name}: 4B-1 requires a STAR/Ensight source, got {source_type!r}"
        )
    if str(source.get("data_association", "")).lower() != "cell":
        raise NativeSectionError(f"{source_name}: data_association must be 'cell'")
    part_name = str(source.get("volume_part_name", "")).strip()
    if not part_name:
        raise NativeSectionError(f"{source_name}: volume_part_name is required")
    velocity_name = str(source.get("velocity_array", "")).strip()
    if not velocity_name:
        raise NativeSectionError(f"{source_name}: velocity_array is required")

    reader = pv.get_reader(path)
    time_index = int(source.get("time_index", source.get("time_point", -1)))
    count = int(reader.number_time_points)
    if not 0 <= time_index < count:
        raise NativeSectionError(
            f"{source_name}: time_index {time_index} outside 0..{count - 1}"
        )
    time_values = [float(value) for value in reader.time_values]
    reader.set_active_time_point(time_index)
    raw = reader.read()
    mesh = _find_named_dataset(raw, part_name).copy(deep=True)
    if velocity_name not in mesh.cell_data:
        raise NativeSectionError(
            f"{source_name}: cell array {velocity_name!r} missing from part {part_name!r}; "
            f"available: {list(mesh.cell_data.keys())}"
        )
    velocity = np.asarray(mesh.cell_data[velocity_name], dtype=float)
    if velocity.shape != (mesh.n_cells, 3):
        raise NativeSectionError(
            f"{source_name}: {velocity_name!r} must have shape ({mesh.n_cells}, 3)"
        )
    length_scale = float(source.get("length_scale_to_mm", 1.0))
    velocity_scale = float(source.get("velocity_scale_to_mm_s", 1.0))
    if not math.isfinite(length_scale) or length_scale <= 0.0:
        raise NativeSectionError(f"{source_name}: length_scale_to_mm must be positive")
    if not math.isfinite(velocity_scale) or velocity_scale <= 0.0:
        raise NativeSectionError(f"{source_name}: velocity_scale_to_mm_s must be positive")
    mesh = scale_native_volume_mesh(mesh, velocity_name, length_scale, velocity_scale)
    metadata = {
        "path": str(path),
        "time_index": time_index,
        "time_point_count": count,
        "actual_time_s": time_values[time_index],
        "configured_time_s": source.get("time"),
        "volume_part_name": part_name,
        "velocity_array": velocity_name,
        "data_association": "cell",
        "length_scale_to_mm": length_scale,
        "velocity_scale_to_mm_s": velocity_scale,
        "point_count": int(mesh.n_points),
        "cell_count": int(mesh.n_cells),
        "cell_types": _cell_type_counts(mesh),
        "bounds_mm": [float(value) for value in mesh.bounds],
        "point_data_arrays": list(mesh.point_data.keys()),
        "cell_data_arrays": list(mesh.cell_data.keys()),
    }
    return mesh, metadata


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=POLYGON_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_index(
    path: Path, key_name: str, value_name: str, filters: Mapping[str, str]
) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise NativeSectionError(f"comparison CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or key_name not in reader.fieldnames:
            raise NativeSectionError(f"{path}: required column missing: {key_name}")
        if value_name not in reader.fieldnames:
            raise NativeSectionError(f"{path}: required column missing: {value_name}")
        result: dict[str, dict[str, str]] = {}
        for row in reader:
            if all(row.get(name) == value for name, value in filters.items()):
                key = str(row[key_name])
                if key in result:
                    raise NativeSectionError(f"{path}: duplicate comparison row for {key!r}")
                result[key] = row
    if not result:
        raise NativeSectionError(f"{path}: no rows matched comparison filters {dict(filters)}")
    return result


def _relative_magnitude_difference(a: float, b: float) -> float:
    denominator = abs(b)
    return 100.0 * abs(abs(a) - abs(b)) / denominator if denominator else math.nan


def _sign_relation(a: float, b: float) -> str:
    if a == 0.0 or b == 0.0:
        return "contains_zero"
    return "same" if math.copysign(1.0, a) == math.copysign(1.0, b) else "opposite"


def build_comparison_rows(
    native_rows: list[dict[str, Any]], comparison: Mapping[str, Any], config_path: Path
) -> list[dict[str, Any]]:
    internal_path = resolve_config_path(
        str(comparison.get("existing_internal_flow_summary", "")), config_path
    )
    boundary_path = resolve_config_path(
        str(comparison.get("boundary_flow_summary", "")), config_path
    )
    internal_source = str(comparison.get("existing_internal_source_name", ""))
    boundary_source = str(comparison.get("boundary_source_name", ""))
    section_to_boundary = _mapping(
        comparison.get("section_to_boundary"), "comparison.section_to_boundary"
    )
    point_rows = _read_csv_index(
        internal_path, "section_name", "signed_flow_rate_mm3_s",
        {"source_name": internal_source},
    )
    boundary_rows = _read_csv_index(
        boundary_path, "boundary_name", "signed_flow_mm3_s",
        {"source_name": boundary_source},
    )
    result: list[dict[str, Any]] = []
    for native in native_rows:
        section_name = str(native["section_name"])
        if section_name not in section_to_boundary:
            raise NativeSectionError(
                f"comparison.section_to_boundary lacks {section_name!r}"
            )
        boundary_name = str(section_to_boundary[section_name])
        if section_name not in point_rows:
            raise NativeSectionError(f"point-converted summary lacks {section_name!r}")
        if boundary_name not in boundary_rows:
            raise NativeSectionError(f"boundary summary lacks {boundary_name!r}")
        q_native = float(native["signed_flow_mm3_s"])
        q_point = float(point_rows[section_name]["signed_flow_rate_mm3_s"])
        q_boundary = float(boundary_rows[boundary_name]["signed_flow_mm3_s"])
        area_native = float(native["native_intersection_area_mm2"])
        area_boundary = float(boundary_rows[boundary_name]["area_mm2"])

        def comparison_fields(prefix: str, a: float, b: float) -> dict[str, float]:
            return {
                f"{prefix}_signed_difference": a - b,
                f"{prefix}_magnitude_difference": abs(a) - abs(b),
                f"{prefix}_absolute_difference": abs(abs(a) - abs(b)),
                f"{prefix}_relative_difference_percent": _relative_magnitude_difference(a, b),
            }

        row: dict[str, Any] = {
            "section_name": section_name,
            "boundary_name": boundary_name,
            "native_volume_cell_flow_mm3_s": q_native,
            "current_point_converted_flow_mm3_s": q_point,
            "native_boundary_flow_mm3_s": q_boundary,
            **comparison_fields("native_vs_point", q_native, q_point),
            **comparison_fields("native_vs_boundary", q_native, q_boundary),
            **comparison_fields("point_vs_boundary", q_point, q_boundary),
            "native_intersection_area_mm2": area_native,
            "boundary_area_mm2": area_boundary,
            "area_difference_percent": 100.0 * (area_native - area_boundary) / area_boundary,
            "native_point_sign_relation": _sign_relation(q_native, q_point),
            "native_boundary_sign_relation": _sign_relation(q_native, q_boundary),
            "point_boundary_sign_relation": _sign_relation(q_point, q_boundary),
        }
        result.append(row)
    return result


def classify_observed_pattern(
    comparisons: list[dict[str, Any]], close_percent: float
) -> dict[str, Any]:
    differences = [float(row["native_vs_boundary_relative_difference_percent"]) for row in comparisons]
    point_differences = [float(row["point_vs_boundary_relative_difference_percent"]) for row in comparisons]
    close = [value <= close_percent for value in differences]
    point_worse = [point > native for point, native in zip(point_differences, differences)]
    if all(close) and all(point_worse):
        pattern = "A"
        statement = (
            "All native volume-cell section magnitudes are within the configured boundary "
            "threshold and point-converted values are farther from the boundary. This is "
            "consistent with a cell-to-point post-processing contribution, but does not by "
            "itself prove sole causation."
        )
    elif all(not value for value in close):
        pattern = "B"
        statement = (
            "All native volume-cell section magnitudes remain outside the configured boundary "
            "threshold. Cutter/clipping, finite-volume cell-value meaning, output conservation, "
            "time/part correspondence, and true internal-boundary differences remain candidates."
        )
    else:
        pattern = "C"
        statement = (
            "The native-to-boundary tendency differs by section. Branch cell shapes, intersection "
            "polygon quality, near-wall cells, and branch-specific post-processing remain candidates."
        )
    return {
        "pattern": pattern,
        "configured_native_boundary_close_percent": close_percent,
        "native_vs_boundary_relative_difference_percent": differences,
        "point_vs_boundary_relative_difference_percent": point_differences,
        "interpretation": statement,
        "causation_determined": False,
    }


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


def write_audit_markdown(path: Path, audit: Mapping[str, Any]) -> None:
    lines = [
        "# Native volume-cell internal-section audit", "",
        "STAR volume-cell `Velocity` is mapped to each cutter polygon through an explicit ",
        "`int64` original-cell ID. No cell-to-point conversion is used.", "",
        "## Results", "",
        "| Section | Area (mm²) | Native Q (mm³/s) | Point-converted Q | Boundary Q | Native–boundary magnitude diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    comparison_by_name = {row["section_name"]: row for row in audit["comparisons"]}
    for row in audit["summary"]:
        comparison = comparison_by_name[row["section_name"]]
        lines.append(
            f"| {row['section_name']} | {row['native_intersection_area_mm2']:.9g} | "
            f"{row['signed_flow_mm3_s']:.9g} | "
            f"{comparison['current_point_converted_flow_mm3_s']:.9g} | "
            f"{comparison['native_boundary_flow_mm3_s']:.9g} | "
            f"{comparison['native_vs_boundary_relative_difference_percent']:.6g}% |"
        )
    decision = audit["observed_pattern"]
    lines.extend([
        "", "## Observed pattern", "",
        f"Pattern: **{decision['pattern']}**", "", decision["interpretation"], "",
        "This numerical pattern is evidence for diagnosis, not an automatic causal conclusion.", "",
        "## Sign convention", "",
        "Signed flow uses each section library's normalized normal. Boundary outward normals may "
        "be opposite to an internal-section normal, so magnitude and signed differences are reported separately.", "",
        "## Reproduction", "", "```bash", "cd /workspace",
        "python scripts/integrate_volume_cell_sections.py --config config/integrate_volume_cell_sections.json",
        "```", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_configuration(config: Mapping[str, Any]) -> None:
    sources = _mapping(config.get("data_sources"), "data_sources")
    if not sources:
        raise NativeSectionError("data_sources must not be empty")
    integration = _mapping(config.get("integration"), "integration")
    if integration.get("method") != "native_volume_cell_intersection":
        raise NativeSectionError(
            "integration.method must be 'native_volume_cell_intersection'"
        )
    minimum = float(integration.get("minimum_polygon_area_mm2", -1.0))
    if not math.isfinite(minimum) or minimum < 0.0:
        raise NativeSectionError("minimum_polygon_area_mm2 must be non-negative")
    output = _mapping(config.get("output"), "output")
    if not str(output.get("directory", "")).strip():
        raise NativeSectionError("output.directory is required")
    if output.get("polygon_details_format", "csv.gz") != "csv.gz":
        raise NativeSectionError("only polygon_details_format='csv.gz' is supported")
    _mapping(config.get("comparison"), "comparison")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    config = load_config(config_path)
    validate_configuration(config)
    try:
        sections = resolve_sections_from_config(config, config_path=config_path)
    except SectionConfigError as exc:
        raise NativeSectionError(f"invalid section configuration: {exc}") from exc
    integration = dict(_mapping(config["integration"], "integration"))
    output_config = dict(_mapping(config["output"], "output"))
    execution = dict(_mapping(config.get("execution", {}), "execution"))
    output_dir = resolve_config_path(output_config["directory"], config_path)
    if (
        execution.get("refuse_nonempty_output_directory", True)
        and output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise NativeSectionError(
            f"refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration: {config_path}")
    print(f"Output: {output_dir}")
    all_summary: list[dict[str, Any]] = []
    all_diagnostics: dict[str, Any] = {}
    source_metadata: dict[str, Any] = {}
    sources = _mapping(config["data_sources"], "data_sources")
    for source_name, source_value in sources.items():
        source = dict(_mapping(source_value, f"data_sources.{source_name}"))
        mesh, metadata = load_native_volume_source(source_name, source, config_path)
        source_metadata[source_name] = metadata
        velocity_name = str(source["velocity_array"])
        print(f"Source: {source_name} -> {metadata['path']}")
        print(
            f"  time_index={metadata['time_index']}, actual_time={metadata['actual_time_s']:.12g} s, "
            f"part={metadata['volume_part_name']!r}, velocity={velocity_name!r} (cell)"
        )
        for section in sections:
            center = np.asarray(section["center"], dtype=float)
            normal = np.asarray(section["normalized_normal"], dtype=float)
            s_axis = np.asarray(section["s_axis"], dtype=float)
            t_axis = np.asarray(section["t_axis"], dtype=float)
            print(f"Section: {section['name']}")
            print(f"  center={center.tolist()}, normal={normal.tolist()}")
            print(f"  s_axis={s_axis.tolist()}, t_axis={t_axis.tolist()}")
            print(f"  width={section['width']} mm, height={section['height']} mm")
            try:
                metrics, records, polydata, diagnostic = integrate_native_volume_cell_section(
                    mesh, velocity_name, center, normal, s_axis, t_axis,
                    float(section["width"]), float(section["height"]),
                    minimum_polygon_area_mm2=float(
                        integration.get("minimum_polygon_area_mm2", 1.0e-12)
                    ),
                    validate_original_cell_ids=bool(
                        integration.get("validate_original_cell_ids", True)
                    ),
                    fail_on_unmapped_polygon=bool(
                        integration.get("fail_on_unmapped_polygon", True)
                    ),
                    clip_to_section_window=bool(
                        integration.get("clip_to_section_window", True)
                    ),
                )
            except SurfaceFlowError as exc:
                raise NativeSectionError(f"{source_name}/{section['name']}: {exc}") from exc
            row = {
                "source_name": source_name,
                "source_label": source.get("label", source_name),
                "section_name": section["name"],
                "section_label": section["label"],
                "center_x": float(center[0]), "center_y": float(center[1]),
                "center_z": float(center[2]), "normal_x": float(normal[0]),
                "normal_y": float(normal[1]), "normal_z": float(normal[2]),
                "width_mm": float(section["width"]),
                "height_mm": float(section["height"]),
                **metrics,
            }
            all_summary.append(row)
            all_diagnostics[f"{source_name}/{section['name']}"] = diagnostic
            if output_config.get("save_polygon_details", True):
                _write_csv_gz(output_dir / "polygons" / f"{section['name']}.csv.gz", records)
            if output_config.get("save_vtp", True):
                path = output_dir / "vtp" / f"{section['name']}.vtp"
                path.parent.mkdir(parents=True, exist_ok=True)
                polydata.save(path)
            print(
                f"  area={metrics['native_intersection_area_mm2']:.9g} mm2, "
                f"signed_Q={metrics['signed_flow_mm3_s']:.9g} mm3/s, "
                f"cells={metrics['intersected_volume_cell_count']}, "
                f"polygons={metrics['valid_polygon_count']}"
            )

    if output_config.get("save_summary_csv", True):
        _write_csv(
            output_dir / "native_cell_section_flow_summary.csv",
            all_summary, SUMMARY_COLUMNS,
        )
    comparison_config = dict(_mapping(config["comparison"], "comparison"))
    comparison_rows = build_comparison_rows(all_summary, comparison_config, config_path)
    if output_config.get("save_comparison_csv", True):
        _write_csv(
            output_dir / "native_vs_point_vs_boundary.csv",
            comparison_rows, COMPARISON_COLUMNS,
        )
    close_percent = float(comparison_config.get("native_boundary_close_percent", 1.0))
    observed = classify_observed_pattern(comparison_rows, close_percent)
    audit = {
        "configuration": str(config_path),
        "output_directory": str(output_dir),
        "method": {
            "name": "native_volume_cell_intersection",
            "cell_to_point_conversion": False,
            "velocity_assignment": "original STAR volume-cell Velocity by preserved int64 ID",
            "section_window_clipping": "Sutherland-Hodgman in orthonormal section (s,t) coordinates",
            "polygon_area": "2-D shoelace formula in orthonormal (s,t) coordinates",
            "signed_flow": "sum(area * dot(original_cell_velocity, normalized_section_normal))",
            "absolute_flow": "absolute value of net signed flow",
        },
        "source_metadata": source_metadata,
        "sections": [
            {
                "name": section["name"], "label": section["label"],
                "center": section["center"], "normal": section["normalized_normal"],
                "s_axis": section["s_axis"], "t_axis": section["t_axis"],
                "width_mm": section["width"], "height_mm": section["height"],
            }
            for section in sections
        ],
        "summary": all_summary,
        "diagnostics": all_diagnostics,
        "comparisons": comparison_rows,
        "observed_pattern": observed,
        "limitations": [
            "A volume-cell output value's exact finite-volume meaning cannot be inferred from Ensight files alone.",
            "This audit does not prove causation even when native and boundary flow magnitudes agree.",
            "No new CFD run, boundary-condition change, mesh change, or multi-section sweep was performed.",
        ],
    }
    audit_path = output_dir / "native_cell_section_audit.json"
    audit_path.write_text(
        json.dumps(_json_ready(audit), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_audit_markdown(output_dir / "native_cell_section_audit.md", audit)
    print(f"Observed pattern: {observed['pattern']}")
    print(f"Summary: {output_dir / 'native_cell_section_flow_summary.csv'}")
    print(f"Comparison: {output_dir / 'native_vs_point_vs_boundary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
