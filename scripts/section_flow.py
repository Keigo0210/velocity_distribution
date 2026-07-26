#!/usr/bin/env python3
"""Integrate volumetric flow rates on reusable configured sections."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from section_config import SectionConfigError, resolve_sections_from_config
from surface_flow import flow_balance, integrate_triangles


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "section_flow.json"


def resolve_config_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    repository_prefixes = {"data", "output", "config", "manual", "scripts", "tests"}
    if path.parts and path.parts[0] in repository_prefixes:
        return (config_path.parent.parent / path).resolve()
    candidates = [config_path.parent / path, config_path.parent.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def prepare_mesh(source_name: str, source: dict[str, Any], config_path: Path) -> tuple[pv.DataSet, str]:
    path = resolve_config_path(source["path"], config_path)
    if not path.is_file():
        raise FileNotFoundError(f"{source_name}: input file not found: {path}")
    source_type = str(source.get("type", "vtu")).lower()
    if source_type in {"ensight", "case", "star_ccm"}:
        reader = pv.get_reader(path)
        time_point = int(source.get("time_point", reader.number_time_points - 1))
        if not 0 <= time_point < reader.number_time_points:
            raise ValueError(f"{source_name}: time_point {time_point} is out of range")
        reader.set_active_time_point(time_point)
        data = reader.read()
    else:
        data = pv.read(path)
    mesh = data.combine(merge_points=True) if isinstance(data, pv.MultiBlock) else data
    velocity_name = str(source["velocity_array"])
    association = str(source.get("data_association", "point")).lower()
    if association == "cell":
        if velocity_name not in mesh.cell_data:
            raise KeyError(f"{source_name}: {velocity_name!r} missing from cell data")
        mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
    elif association != "point":
        raise ValueError(f"{source_name}: data_association must be point or cell")
    if velocity_name not in mesh.point_data:
        if velocity_name in mesh.cell_data:
            mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
        else:
            raise KeyError(f"{source_name}: {velocity_name!r} missing from point/cell data")
    length_scale = float(source.get("length_scale_to_mm", 1.0))
    velocity_scale = float(source.get("velocity_scale_to_mm_s", 1.0))
    mesh = mesh.copy(deep=True)
    if length_scale != 1.0:
        mesh.points *= length_scale
    mesh.point_data[velocity_name] = (
        np.asarray(mesh.point_data[velocity_name], dtype=float) * velocity_scale
    )
    return mesh, velocity_name


def section_flow(
    mesh: pv.DataSet, velocity_name: str, section: dict[str, Any]
) -> dict[str, Any]:
    center = np.asarray(section["center"], dtype=float)
    normal = np.asarray(section["normalized_normal"], dtype=float)
    s_axis = np.asarray(section["s_axis"], dtype=float)
    t_axis = np.asarray(section["t_axis"], dtype=float)
    cut = mesh.slice(origin=center, normal=normal)
    if cut.n_points == 0:
        raise RuntimeError(f"{section['name']}: section extraction is empty")
    if velocity_name not in cut.point_data:
        raise KeyError(f"{section['name']}: {velocity_name!r} missing from section")
    relative = np.asarray(cut.points) - center
    keep = (np.abs(relative @ s_axis) <= 0.5 * float(section["width"])) & (
        np.abs(relative @ t_axis) <= 0.5 * float(section["height"])
    )
    clipped = cut.extract_points(keep, adjacent_cells=False)
    if clipped.n_cells == 0:
        raise RuntimeError(f"{section['name']}: no section cells remain inside width/height")
    surface = clipped.extract_surface(algorithm="dataset_surface").triangulate()
    faces = np.asarray(surface.faces, dtype=np.int64)
    if len(faces) == 0:
        raise RuntimeError(f"{section['name']}: triangulation is empty")
    triangles = faces.reshape((-1, 4))[:, 1:]
    result = integrate_triangles(
        np.asarray(surface.points), triangles,
        np.asarray(surface.point_data[velocity_name], dtype=float), normal,
    )
    result.update(
        {
            "section_name": section["name"],
            "section_label": section["label"],
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "center_z": float(center[2]),
            "normal_x": float(normal[0]),
            "normal_y": float(normal[1]),
            "normal_z": float(normal[2]),
        }
    )
    return result


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        sections = resolve_sections_from_config(config, config_path=config_path)
    except SectionConfigError as exc:
        raise ValueError(f"Invalid section configuration: {exc}") from exc
    output_dir = resolve_config_path(config.get("output", {}).get("directory", "output/section_flow"), config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = config.get("data_sources", {})
    if not isinstance(sources, dict) or not sources:
        raise ValueError("data_sources must be a non-empty object")
    balance_config = config.get("flow_balance", {})
    all_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    print(f"Configuration: {config_path}")
    print(f"Sections: {[section['name'] for section in sections]}")
    print(f"Output: {output_dir}")
    for source_name, source in sources.items():
        mesh, velocity_name = prepare_mesh(source_name, source, config_path)
        rows = []
        for section in sections:
            row = section_flow(mesh, velocity_name, section)
            row.update(
                {
                    "source_name": source_name,
                    "source_label": source.get("label", source_name),
                    "input_path": str(resolve_config_path(source["path"], config_path)),
                    "time_s": source.get("time", ""),
                }
            )
            rows.append(row)
            print(
                f"{source_name}/{section['name']}: area={row['section_area_mm2']:.6g} mm2, "
                f"signed_Q={row['signed_flow_rate_mm3_s']:.6g} mm3/s"
            )
        balance = {"source_name": source_name, **flow_balance(rows, balance_config)}
        balance_rows.append(balance)
        for row in rows:
            row.update(balance)
        source_dir = output_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        write_rows(source_dir / "flow_metrics.csv", rows)
        all_rows.extend(rows)
    write_rows(output_dir / "flow_summary.csv", all_rows)
    write_rows(output_dir / "flow_balance_summary.csv", balance_rows)
    print(f"Flow summary: {output_dir / 'flow_summary.csv'}")
    print(f"Balance summary: {output_dir / 'flow_balance_summary.csv'}")


if __name__ == "__main__":
    main()
