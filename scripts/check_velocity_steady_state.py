#!/usr/bin/env python3
"""Quantify temporal convergence of a velocity field stored in VTU files.

The primary comparison is performed over a user-specified physical interval
(1 s by default).  One-output-step changes are also reported.  All whole-field
statistics are unweighted point statistics because the solver velocity in the
examined data is point-associated.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from section_config import SectionConfigError, make_plane_basis as shared_make_plane_basis, resolve_sections_from_config


ROOT = Path(__file__).resolve().parents[1]
STEP_RE = re.compile(r"(\d+)(?=\.vtu$)")
LOG_RE = re.compile(
    r"Time step\s+(\d+)\s*/\s*(\d+),\s*t\s*=\s*([0-9.eE+-]+),\s*dt\s*=\s*([0-9.eE+-]+)"
)
DEFAULT_CHECK_TIMES = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 35.0, 37.5)


@dataclass(frozen=True)
class TimeEntry:
    step: int
    time: float
    path: Path


@dataclass(frozen=True)
class RunLog:
    times: dict[int, float]
    dts: dict[int, float]
    nsteps: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration for multi-source/region analysis")
    parser.add_argument("--section-library", type=Path, help="Override section_library in --config mode")
    parser.add_argument("--section-set", help="Override section_set in --config mode")
    parser.add_argument("--section-name", action="append", default=[], help="Override with named section; repeatable")
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=ROOT / "data" / "260629_solver",
        help="Directory containing the VTU time series",
    )
    parser.add_argument("--pattern", default="solution_*.vtu", help="VTU glob pattern")
    parser.add_argument("--log-file", type=Path, help="Solver log containing step/time/dt records")
    parser.add_argument("--dt", type=float, help="Fallback time step when no complete log is available")
    parser.add_argument(
        "--velocity-name",
        help="Point-associated velocity vector. If omitted, exactly one 3-component point array is required.",
    )
    parser.add_argument("--lag-seconds", type=float, default=1.0, help="Primary physical comparison interval")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "steady_state" / "solver_Re100_260629",
    )
    parser.add_argument("--section-center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--section-normal", nargs=3, type=float, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--section-width", type=float, default=12.0)
    parser.add_argument("--section-height", type=float, default=10.0)
    parser.add_argument("--no-section", action="store_true", help="Disable section metrics")
    parser.add_argument(
        "--reference-config",
        type=Path,
        action="append",
        default=[],
        help="Existing configuration file establishing case/section provenance; repeatable",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_step(path: Path) -> int:
    match = STEP_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not extract a step number from {path.name}")
    return int(match.group(1))


def parse_log(path: Path | None) -> RunLog:
    if path is None or not path.exists():
        return RunLog({}, {}, None)
    times: dict[int, float] = {}
    dts: dict[int, float] = {}
    nsteps_values: set[int] = set()
    for line in path.read_text(errors="replace").splitlines():
        match = LOG_RE.search(line)
        if match:
            step, nsteps, time, dt = match.groups()
            times[int(step)] = float(time)
            dts[int(step)] = float(dt)
            nsteps_values.add(int(nsteps))
    if len(nsteps_values) > 1:
        raise ValueError(f"Inconsistent nsteps values in {path}: {sorted(nsteps_values)}")
    return RunLog(times, dts, next(iter(nsteps_values), None))


def build_timeline(files: list[Path], run_log: RunLog, fallback_dt: float | None) -> list[TimeEntry]:
    steps = [parse_step(path) for path in files]
    if len(set(steps)) != len(steps):
        raise ValueError("Duplicate step numbers were found")
    if run_log.times and all(step in run_log.times for step in steps):
        times = [run_log.times[step] for step in steps]
    elif fallback_dt is not None:
        times = [step * fallback_dt for step in steps]
    else:
        missing = [step for step in steps if step not in run_log.times]
        raise ValueError(f"Physical times are unavailable for steps {missing[:5]}; provide --dt")
    entries = sorted(
        (TimeEntry(step, time, path) for step, time, path in zip(steps, times, files)),
        key=lambda entry: entry.time,
    )
    if any(right.time <= left.time for left, right in zip(entries, entries[1:])):
        raise ValueError("Physical times are not strictly increasing")
    return entries


def choose_velocity_name(mesh: pv.DataSet, requested: str | None) -> str:
    if requested:
        if requested not in mesh.point_data:
            raise KeyError(
                f"{requested!r} is not point data. Available point arrays: {list(mesh.point_data.keys())}"
            )
        array = np.asarray(mesh.point_data[requested])
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"{requested!r} is not an N x 3 point vector: shape={array.shape}")
        return requested
    candidates = [
        name
        for name in mesh.point_data.keys()
        if np.asarray(mesh.point_data[name]).ndim == 2
        and np.asarray(mesh.point_data[name]).shape[1] == 3
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Velocity name was not specified and a unique 3-component point array "
            f"could not be identified. Candidates: {candidates}"
        )
    return candidates[0]


def update_hash(digest: hashlib._Hash, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())


def mesh_hash(mesh: pv.DataSet) -> str:
    digest = hashlib.sha256()
    update_hash(digest, np.asarray(mesh.points))
    for attribute in ("offset", "cell_connectivity", "celltypes"):
        if not hasattr(mesh, attribute):
            raise TypeError(f"Mesh type {type(mesh).__name__} lacks {attribute}")
        update_hash(digest, np.asarray(getattr(mesh, attribute)))
    return digest.hexdigest()


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return shared_make_plane_basis(normal)


def extract_section(
    mesh: pv.DataSet,
    velocity_name: str,
    center: np.ndarray,
    normal: np.ndarray,
    width: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    cut = mesh.slice(origin=center, normal=normal)
    if velocity_name not in cut.point_data:
        raise KeyError(f"{velocity_name!r} was not interpolated onto the section")
    _, s_axis, t_axis = plane_basis(normal)
    relative = np.asarray(cut.points) - center
    s_coord = relative @ s_axis
    t_coord = relative @ t_axis
    keep = (np.abs(s_coord) <= 0.5 * width) & (np.abs(t_coord) <= 0.5 * height)
    points = np.asarray(cut.points)[keep]
    vectors = np.asarray(cut.point_data[velocity_name], dtype=float)[keep]
    if len(points) == 0:
        raise RuntimeError("No section points remain inside the requested width/height")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return points[order], vectors[order]


def relative_l2(current: np.ndarray, previous: np.ndarray) -> float:
    denominator = float(np.linalg.norm(current.ravel()))
    numerator = float(np.linalg.norm((current - previous).ravel()))
    return 100.0 * numerator / denominator if denominator > 0 else math.nan


def max_vector_difference(current: np.ndarray, previous: np.ndarray) -> float:
    return float(np.linalg.norm(current - previous, axis=1).max())


def sustained_time(rows: list[dict[str, object]], column: str, threshold: float) -> tuple[float | None, float | None]:
    valid = [(index, float(row[column])) for index, row in enumerate(rows) if not math.isnan(float(row[column]))]
    first = next((float(rows[index]["time_s"]) for index, value in valid if value < threshold), None)
    sustained = None
    for index, value in valid:
        tail = [float(row[column]) for row in rows[index:] if not math.isnan(float(row[column]))]
        if value < threshold and tail and all(item < threshold for item in tail):
            sustained = float(rows[index]["time_s"])
            break
    return first, sustained


def nearest_rows(rows: list[dict[str, object]], target_times: Iterable[float]) -> list[dict[str, object]]:
    return [min(rows, key=lambda row: abs(float(row["time_s"]) - target)) for target in target_times]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_xy(rows: list[dict[str, object]], column: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([float(row["time_s"]) for row in rows])
    y = np.asarray([float(row[column]) for row in rows])
    keep = np.isfinite(y)
    return x[keep], y[keep]


def plot_changes(path: Path, rows: list[dict[str, object]], lag_seconds: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for column, label, style in (
        ("relative_L2_lag_percent", f"{lag_seconds:g} s interval", "-"),
        ("relative_L2_one_step_percent", "one output step", "--"),
    ):
        x, y = finite_xy(rows, column)
        ax.semilogy(x, np.maximum(y, np.finfo(float).tiny), style, label=label, linewidth=1.4)
    ax.axhline(0.5, color="#CC6677", linestyle=":", label="0.5% criterion")
    ax.axhline(0.1, color="#4477AA", linestyle=":", label="0.1% criterion")
    ax.set(xlabel="Time [s]", ylabel="Relative L2 velocity change [%]", title="Velocity-field temporal change")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_speed_stats(path: Path, rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for column, label in (
        ("mean_speed_mm_per_s", "mean"),
        ("max_speed_mm_per_s", "maximum"),
        ("p95_speed_mm_per_s", "95th percentile"),
    ):
        x, y = finite_xy(rows, column)
        ax.plot(x, y, label=label, linewidth=1.4)
    ax.set(xlabel="Time [s]", ylabel="Speed [mm/s]", title="Whole-field point speed statistics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_section(path: Path, rows: list[dict[str, object]], lag_seconds: float) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for column, label in (
        ("section_mean_speed_mm_per_s", "mean"),
        ("section_max_speed_mm_per_s", "maximum"),
    ):
        x, y = finite_xy(rows, column)
        axes[0].plot(x, y, label=label, linewidth=1.4)
    x, y = finite_xy(rows, "section_relative_L2_lag_percent")
    axes[1].semilogy(x, np.maximum(y, np.finfo(float).tiny), linewidth=1.4)
    axes[1].axhline(0.5, color="#CC6677", linestyle=":")
    axes[1].axhline(0.1, color="#4477AA", linestyle=":")
    axes[0].set(ylabel="Speed [mm/s]", title="Section point-speed statistics")
    axes[1].set(
        xlabel="Time [s]",
        ylabel="Relative L2 change [%]",
        title=f"Section velocity-distribution change ({lag_seconds:g} s interval)",
    )
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_time(value: float | None) -> str:
    return "not reached" if value is None else f"{value:g} s"


def write_report(
    path: Path,
    entries: list[TimeEntry],
    rows: list[dict[str, object]],
    run_log: RunLog,
    log_path: Path | None,
    velocity_name: str,
    lag_steps: int,
    lag_seconds_actual: float,
    reference_hash: str,
    section_enabled: bool,
    section_center: np.ndarray | None,
    section_normal: np.ndarray | None,
    section_width: float,
    section_height: float,
    reference_configs: list[Path],
    inlet_summary: str,
) -> None:
    first05, sustained05 = sustained_time(rows, "relative_L2_lag_percent", 0.5)
    first01, sustained01 = sustained_time(rows, "relative_L2_lag_percent", 0.1)
    selected = nearest_rows(rows, DEFAULT_CHECK_TIMES)
    dt_values = sorted(set(run_log.dts.values()))
    dt_text = ", ".join(f"{value:g}" for value in dt_values) if dt_values else "not available from log"
    config_lines = "\n".join(f"- `{item}`" for item in reference_configs) or "- none supplied"
    selected_lines = [
        "| time [s] | step | relative L2 / 1 s [%] | max difference [mm/s] | mean [mm/s] | max [mm/s] | p95 [mm/s] |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        selected_lines.append(
            f"| {float(row['time_s']):.3f} | {int(row['step'])} | "
            f"{float(row['relative_L2_lag_percent']):.6g} | "
            f"{float(row['max_difference_lag_mm_per_s']):.6g} | "
            f"{float(row['mean_speed_mm_per_s']):.6g} | "
            f"{float(row['max_speed_mm_per_s']):.6g} | "
            f"{float(row['p95_speed_mm_per_s']):.6g} |"
        )
    section_text = "disabled"
    if section_enabled:
        section_first05, section_sustained05 = sustained_time(rows, "section_relative_L2_lag_percent", 0.5)
        section_first01, section_sustained01 = sustained_time(rows, "section_relative_L2_lag_percent", 0.1)
        section_text = (
            f"center={section_center.tolist()}, normal={section_normal.tolist()}, "
            f"width={section_width:g} mm, height={section_height:g} mm. "
            f"Section 0.5%: first {format_time(section_first05)}, sustained {format_time(section_sustained05)}; "
            f"section 0.1%: first {format_time(section_first01)}, sustained {format_time(section_sustained01)}."
        )
    text = f"""# Re=100 velocity steady-state analysis

Generated: {datetime.now(timezone.utc).isoformat()}

## Inputs and provenance

- Result directory: `{entries[0].path.parent}`
- Solver log: `{log_path if log_path else "not supplied"}`
- Reference configurations:
{config_lines}
- VTU files: {len(entries)} (`{entries[0].path.name}` through `{entries[-1].path.name}`)
- Step/time range: {entries[0].step} / {entries[0].time:g} s through {entries[-1].step} / {entries[-1].time:g} s
- Output interval: {lag_seconds_actual / lag_steps:g} s (one VTU per solver step)
- Log dt values: {dt_text} s; log nsteps: {run_log.nsteps}
- Velocity array: point data `{velocity_name}`
- Inlet-boundary data check: {inlet_summary}
- Mesh SHA-256 (points, offsets, connectivity, cell types): `{reference_hash}`
- Mesh identity: exact hash equality was confirmed for every VTU file.
- Original solver input/configuration: not present in this repository. The run log verifies dt/nsteps,
  while existing post-processing configurations establish the Re=100 case mapping and output interval.

## Method

At every velocity point, speed is `||u||`. Whole-field mean, maximum, and 95th-percentile speeds are
unweighted point statistics. The primary change is
`100 ||u(t)-u(t-{lag_seconds_actual:g} s)||_2 / ||u(t)||_2`; the maximum pointwise vector difference is
also reported. The actual lag is {lag_steps} output steps ({lag_seconds_actual:g} s). One-step changes
are reported separately. Thresholds use strict `<` comparisons. “Sustained” means that every later
available primary comparison also remains below the threshold.

Section analysis: {section_text}

## Whole-field threshold results

- 0.5%: first below at **{format_time(first05)}**; sustained from **{format_time(sustained05)}**.
- 0.1%: first below at **{format_time(first01)}**; sustained from **{format_time(sustained01)}**.

## Requested check times

{chr(10).join(selected_lines)}

## Output files

- `velocity_steady_state_metrics.csv`
- `relative_l2_change.png`
- `speed_statistics.png`
{("- `section_metrics.png`" if section_enabled else "")}
"""
    path.write_text(text)


def legacy_main(args: argparse.Namespace) -> None:
    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    log_path = resolve(args.log_file) if args.log_file else input_dir / "result.txt"
    reference_configs = [resolve(path) for path in args.reference_config]
    files = sorted(input_dir.glob(args.pattern), key=parse_step)
    if not files:
        raise FileNotFoundError(f"No VTU files matched {input_dir / args.pattern}")
    run_log = parse_log(log_path)
    entries = build_timeline(files, run_log, args.dt)
    time_deltas = np.diff([entry.time for entry in entries])
    output_interval = float(np.median(time_deltas))
    if not np.allclose(time_deltas, output_interval, rtol=0, atol=1e-12):
        raise ValueError("Output times are not uniformly spaced; this script currently requires a uniform interval")
    lag_steps = int(round(args.lag_seconds / output_interval))
    if lag_steps < 1:
        raise ValueError("--lag-seconds is shorter than half an output interval")
    lag_seconds_actual = lag_steps * output_interval

    first_mesh = pv.read(entries[0].path)
    velocity_name = choose_velocity_name(first_mesh, args.velocity_name)
    reference_hash = mesh_hash(first_mesh)
    reference_points = np.asarray(first_mesh.points).copy()
    n_points = first_mesh.n_points
    n_cells = first_mesh.n_cells

    section_enabled = not args.no_section
    section_center = np.asarray(args.section_center, dtype=float) if args.section_center else None
    section_normal = np.asarray(args.section_normal, dtype=float) if args.section_normal else None
    if section_enabled and (section_center is None or section_normal is None):
        raise ValueError("Supply both --section-center and --section-normal, or use --no-section")
    if section_enabled and np.linalg.norm(section_normal) == 0:
        raise ValueError("Section normal must be nonzero")

    output_dir.mkdir(parents=True, exist_ok=True)
    vector_history: deque[np.ndarray] = deque(maxlen=lag_steps)
    section_history: deque[np.ndarray] = deque(maxlen=lag_steps)
    reference_section_points: np.ndarray | None = None
    rows: list[dict[str, object]] = []

    inlet_summary = "not evaluated (mesh does not expose the expected z=50 mm boundary)"
    for index, entry in enumerate(entries):
        mesh = first_mesh if index == 0 else pv.read(entry.path)
        current_hash = reference_hash if index == 0 else mesh_hash(mesh)
        if current_hash != reference_hash:
            raise RuntimeError(f"Mesh coordinates/connectivity differ at {entry.path}")
        if mesh.n_points != n_points or mesh.n_cells != n_cells:
            raise RuntimeError(f"Mesh size differs at {entry.path}")
        if velocity_name not in mesh.point_data:
            raise KeyError(f"{velocity_name!r} missing from {entry.path}")
        velocity = np.asarray(mesh.point_data[velocity_name], dtype=float).copy()
        if velocity.shape != (n_points, 3):
            raise ValueError(f"Unexpected velocity shape in {entry.path}: {velocity.shape}")
        speed = np.linalg.norm(velocity, axis=1)
        one_previous = vector_history[-1] if vector_history else None
        lag_previous = vector_history[0] if len(vector_history) == lag_steps else None
        row: dict[str, object] = {
            "step": entry.step,
            "time_s": entry.time,
            "file": entry.path.name,
            "mean_speed_mm_per_s": float(np.mean(speed)),
            "max_speed_mm_per_s": float(np.max(speed)),
            "p95_speed_mm_per_s": float(np.percentile(speed, 95)),
            "relative_L2_one_step_percent": (
                relative_l2(velocity, one_previous) if one_previous is not None else math.nan
            ),
            "max_difference_one_step_mm_per_s": (
                max_vector_difference(velocity, one_previous) if one_previous is not None else math.nan
            ),
            "relative_L2_lag_percent": (
                relative_l2(velocity, lag_previous) if lag_previous is not None else math.nan
            ),
            "max_difference_lag_mm_per_s": (
                max_vector_difference(velocity, lag_previous) if lag_previous is not None else math.nan
            ),
        }
        vector_history.append(velocity)

        if section_enabled:
            section_points, section_velocity = extract_section(
                mesh,
                velocity_name,
                section_center,
                section_normal,
                args.section_width,
                args.section_height,
            )
            if reference_section_points is None:
                reference_section_points = section_points.copy()
            elif not np.array_equal(section_points, reference_section_points):
                raise RuntimeError(f"Section sampling points differ at {entry.path}")
            section_speed = np.linalg.norm(section_velocity, axis=1)
            one_section_previous = section_history[-1] if section_history else None
            lag_section_previous = section_history[0] if len(section_history) == lag_steps else None
            row.update(
                {
                    "section_point_count": len(section_points),
                    "section_mean_speed_mm_per_s": float(np.mean(section_speed)),
                    "section_max_speed_mm_per_s": float(np.max(section_speed)),
                    "section_relative_L2_one_step_percent": (
                        relative_l2(section_velocity, one_section_previous)
                        if one_section_previous is not None
                        else math.nan
                    ),
                    "section_relative_L2_lag_percent": (
                        relative_l2(section_velocity, lag_section_previous)
                        if lag_section_previous is not None
                        else math.nan
                    ),
                    "section_max_difference_lag_mm_per_s": (
                        max_vector_difference(section_velocity, lag_section_previous)
                        if lag_section_previous is not None
                        else math.nan
                    ),
                }
            )
            section_history.append(section_velocity.copy())
        rows.append(row)

        if index == len(entries) - 1:
            inlet_mask = np.isclose(reference_points[:, 2], 50.0, rtol=0, atol=1e-12)
            if np.any(inlet_mask):
                inlet_speeds = speed[inlet_mask]
                unique = np.unique(np.round(inlet_speeds, decimals=12))
                inlet_summary = (
                    f"{inlet_mask.sum()} points at z=50 mm; unique speed values "
                    f"{unique.tolist()} mm/s; maximum {inlet_speeds.max():g} mm/s"
                )

    write_csv(output_dir / "velocity_steady_state_metrics.csv", rows)
    plot_changes(output_dir / "relative_l2_change.png", rows, lag_seconds_actual)
    plot_speed_stats(output_dir / "speed_statistics.png", rows)
    if section_enabled:
        plot_section(output_dir / "section_metrics.png", rows, lag_seconds_actual)
    write_report(
        output_dir / "analysis_report.md",
        entries,
        rows,
        run_log,
        log_path if log_path.exists() else None,
        velocity_name,
        lag_steps,
        lag_seconds_actual,
        reference_hash,
        section_enabled,
        section_center,
        section_normal,
        args.section_width,
        args.section_height,
        reference_configs,
        inlet_summary,
    )
    print(f"Analyzed {len(entries)} files; results written to {output_dir}")


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


def normalized_threshold_percent(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("relative L2 thresholds must be positive finite values")
    return value * 100.0 if value <= 0.01 else value


def extract_config_section(
    mesh: pv.DataSet, velocity_name: str, section: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(section["center"], dtype=float)
    normal = np.asarray(section["normalized_normal"], dtype=float)
    s_axis = np.asarray(section["s_axis"], dtype=float)
    t_axis = np.asarray(section["t_axis"], dtype=float)
    cut = mesh.slice(origin=center, normal=normal)
    if cut.n_points == 0:
        raise RuntimeError(f"{section['name']}: section extraction is empty")
    if velocity_name not in cut.point_data:
        raise KeyError(
            f"{velocity_name!r} was not interpolated onto section {section['name']!r}"
        )
    relative = np.asarray(cut.points) - center
    s_coord = relative @ s_axis
    t_coord = relative @ t_axis
    keep = (np.abs(s_coord) <= 0.5 * float(section["width"])) & (
        np.abs(t_coord) <= 0.5 * float(section["height"])
    )
    points = np.asarray(cut.points)[keep]
    vectors = np.asarray(cut.point_data[velocity_name], dtype=float)[keep]
    if len(points) == 0:
        raise RuntimeError(f"{section['name']}: no points remain inside width/height")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return points[order], vectors[order]


def prepare_velocity_mesh(
    mesh: pv.DataSet, source_name: str, source: dict[str, object]
) -> tuple[pv.DataSet, np.ndarray]:
    velocity_name = str(source["velocity_array"])
    association = str(source.get("data_association", "point")).lower()
    if association == "cell":
        if velocity_name not in mesh.cell_data:
            raise KeyError(
                f"{source_name}: {velocity_name!r} missing from cell data; "
                f"available={list(mesh.cell_data.keys())}"
            )
        mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
    elif association != "point":
        raise ValueError(f"{source_name}: data_association must be point or cell")
    if velocity_name not in mesh.point_data:
        raise KeyError(
            f"{source_name}: {velocity_name!r} missing from point data; "
            f"available={list(mesh.point_data.keys())}"
        )
    length_scale = float(source.get("length_scale_to_mm", 1.0))
    velocity_scale = float(source.get("velocity_scale_to_mm_s", 1.0))
    if length_scale != 1.0:
        mesh = mesh.copy(deep=True)
        mesh.points *= length_scale
    velocity = np.asarray(mesh.point_data[velocity_name], dtype=float) * velocity_scale
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError(f"{source_name}: velocity shape must be N x 3, got {velocity.shape}")
    return mesh, velocity


def make_region_row(
    entry: TimeEntry,
    previous_entry: TimeEntry | None,
    velocity: np.ndarray,
    previous_velocity: np.ndarray | None,
) -> dict[str, object]:
    speed = np.linalg.norm(velocity, axis=1)
    return {
        "step": entry.step,
        "time_s": entry.time,
        "comparison_step": previous_entry.step if previous_entry else "",
        "comparison_time_s": previous_entry.time if previous_entry else math.nan,
        "elapsed_time_s": entry.time - previous_entry.time if previous_entry else math.nan,
        "relative_L2_percent": (
            relative_l2(velocity, previous_velocity)
            if previous_velocity is not None
            else math.nan
        ),
        "max_difference_mm_per_s": (
            max_vector_difference(velocity, previous_velocity)
            if previous_velocity is not None
            else math.nan
        ),
        "mean_speed_mm_per_s": float(np.mean(speed)),
        "max_speed_mm_per_s": float(np.max(speed)),
        "p95_speed_mm_per_s": float(np.percentile(speed, 95)),
        "valid_point_count": int(len(velocity)),
        "file": entry.path.name,
    }


def write_region_plots(
    directory: Path, rows: list[dict[str, object]], thresholds: list[float], dpi: int
) -> None:
    times = np.asarray([float(row["time_s"]) for row in rows])
    relative = np.asarray([float(row["relative_L2_percent"]) for row in rows])
    valid = np.isfinite(relative)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.semilogy(times[valid], np.maximum(relative[valid], np.finfo(float).tiny))
    for threshold in thresholds:
        ax.axhline(threshold, linestyle=":", label=f"{threshold:g}%")
    ax.set(xlabel="Time [s]", ylabel="Relative L2 change [%]", title="Velocity temporal change")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "relative_l2.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(times, [float(row["mean_speed_mm_per_s"]) for row in rows], label="mean")
    ax.plot(times, [float(row["max_speed_mm_per_s"]) for row in rows], label="max")
    ax.set(xlabel="Time [s]", ylabel="Speed [mm/s]", title="Speed statistics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "mean_speed.png", dpi=dpi)
    plt.close(fig)


def analyze_config_source(
    source_name: str,
    source: dict[str, object],
    sections: list[dict[str, object]],
    config_path: Path,
    output_root: Path,
    interval_seconds: float,
    thresholds: list[float],
    evaluate_whole: bool,
    evaluate_sections: bool,
    save_csv: bool,
    save_png: bool,
    dpi: int,
) -> list[dict[str, object]]:
    source_dir = resolve_config_path(str(source["directory"]), config_path)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"{source_name}: invalid data directory: {source_dir}")
    pattern = str(source.get("file_pattern", "solution_*.vtu"))
    files = sorted(source_dir.glob(pattern), key=parse_step)
    if not files:
        raise FileNotFoundError(f"{source_name}: no files matched {source_dir / pattern}")
    dt = float(source["dt"])
    if dt <= 0.0:
        raise ValueError(f"{source_name}.dt must be positive")
    entries = build_timeline(files, RunLog({}, {}, None), dt)
    output_intervals = np.diff([entry.time for entry in entries])
    output_interval = float(np.median(output_intervals))
    if not np.allclose(output_intervals, output_interval, rtol=0, atol=1.0e-10):
        raise ValueError(f"{source_name}: output times are not uniformly spaced")
    lag_steps = int(round(interval_seconds / output_interval))
    if lag_steps < 1 or not math.isclose(
        lag_steps * output_interval, interval_seconds, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError(
            f"{source_name}: comparison interval {interval_seconds:g} s is not an "
            f"integer multiple of output interval {output_interval:g} s"
        )

    regions: list[tuple[str, str, dict[str, object] | None]] = []
    if evaluate_whole:
        regions.append(("whole_domain", "Whole domain", None))
    if evaluate_sections:
        regions.extend((str(s["name"]), str(s["label"]), s) for s in sections)
    histories = {name: deque(maxlen=lag_steps) for name, _, _ in regions}
    entry_histories = {name: deque(maxlen=lag_steps) for name, _, _ in regions}
    rows_by_region: dict[str, list[dict[str, object]]] = {name: [] for name, _, _ in regions}
    reference_section_points: dict[str, np.ndarray] = {}
    reference_mesh_hash: str | None = None
    reference_n_points: int | None = None

    print(
        f"Source {source_name}: files={len(entries)}, dt={dt:g} s, "
        f"output_interval={output_interval:g} s, lag_steps={lag_steps}"
    )
    for index, entry in enumerate(entries):
        raw_mesh = pv.read(entry.path)
        mesh, whole_velocity = prepare_velocity_mesh(raw_mesh, source_name, source)
        current_hash = mesh_hash(mesh)
        if reference_mesh_hash is None:
            reference_mesh_hash = current_hash
            reference_n_points = mesh.n_points
        elif current_hash != reference_mesh_hash or mesh.n_points != reference_n_points:
            raise RuntimeError(f"{source_name}: mesh differs at {entry.path}")

        for region_name, _region_label, section in regions:
            if section is None:
                velocity = whole_velocity
            else:
                points, velocity = extract_config_section(
                    mesh, str(source["velocity_array"]), section
                )
                velocity = velocity * float(source.get("velocity_scale_to_mm_s", 1.0))
                if region_name not in reference_section_points:
                    reference_section_points[region_name] = points.copy()
                elif not np.array_equal(points, reference_section_points[region_name]):
                    raise RuntimeError(
                        f"{source_name}/{region_name}: section points differ at {entry.path}"
                    )
            previous = histories[region_name][0] if len(histories[region_name]) == lag_steps else None
            previous_entry = (
                entry_histories[region_name][0]
                if len(entry_histories[region_name]) == lag_steps
                else None
            )
            rows_by_region[region_name].append(
                make_region_row(entry, previous_entry, velocity, previous)
            )
            histories[region_name].append(velocity.copy())
            entry_histories[region_name].append(entry)

    summaries: list[dict[str, object]] = []
    for region_name, region_label, section in regions:
        rows = rows_by_region[region_name]
        region_dir = output_root / source_name / region_name
        region_dir.mkdir(parents=True, exist_ok=True)
        if save_csv:
            write_csv(region_dir / "steady_state.csv", rows)
        if save_png:
            write_region_plots(region_dir, rows, thresholds, dpi)
        for threshold in thresholds:
            first, continuous = sustained_time(rows, "relative_L2_percent", threshold)
            summaries.append(
                {
                    "source_name": source_name,
                    "source_label": source.get("label", source_name),
                    "region_type": "whole_domain" if section is None else "section",
                    "section_name": region_name,
                    "section_label": region_label,
                    "threshold": threshold,
                    "first_below_threshold_time": first if first is not None else "",
                    "continuous_below_threshold_time": continuous if continuous is not None else "",
                    "final_relative_l2": rows[-1]["relative_L2_percent"],
                    "final_mean_speed": rows[-1]["mean_speed_mm_per_s"],
                    "final_max_speed": rows[-1]["max_speed_mm_per_s"],
                    "analysis_start_time": entries[0].time,
                    "analysis_end_time": entries[-1].time,
                    "comparison_interval_seconds": interval_seconds,
                    "actual_comparison_interval_seconds": lag_steps * output_interval,
                    "number_of_files": len(entries),
                    "status": "success",
                    "error_message": "",
                }
            )
    return summaries


def config_main(args: argparse.Namespace) -> None:
    config_path = resolve(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    section_selector = dict(config)
    if args.section_library:
        section_selector["section_library"] = str(args.section_library)
    if args.section_set:
        section_selector["section_set"] = args.section_set
        section_selector.pop("section_names", None)
    if args.section_name:
        section_selector["section_names"] = args.section_name
        section_selector.pop("section_set", None)
    try:
        sections = resolve_sections_from_config(section_selector, config_path=config_path)
    except SectionConfigError as exc:
        raise ValueError(f"Invalid section configuration: {exc}") from exc

    steady = config.get("steady_state", {})
    interval = float(steady.get("comparison_interval_seconds", 1.0))
    thresholds = [
        normalized_threshold_percent(value)
        for value in steady.get("relative_l2_thresholds", [0.005, 0.001])
    ]
    output = config.get("output", {})
    output_root = resolve_config_path(output.get("directory", "output/steady_state/configured"), config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    evaluation = config.get("evaluation_regions", {})
    execution = config.get("execution", {})
    fail_fast = bool(execution.get("fail_fast", True))
    summaries: list[dict[str, object]] = []
    failures = 0
    print(f"Configuration: {config_path}")
    print(f"Section library: {section_selector.get('section_library', '(inline/legacy)')}")
    print(f"Sections: {[section['name'] for section in sections]}")
    print(f"Comparison interval: {interval:g} s; thresholds [%]: {thresholds}")
    print(f"Output root: {output_root}")
    sources = config.get("data_sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("data_sources must be a non-empty object")
    for source_name, source in sources.items():
        try:
            summaries.extend(
                analyze_config_source(
                    source_name, source, sections, config_path, output_root,
                    interval, thresholds,
                    bool(evaluation.get("whole_domain", True)),
                    bool(evaluation.get("sections", True)),
                    bool(output.get("save_csv", True)),
                    bool(output.get("save_png", True)),
                    int(output.get("dpi", 200)),
                )
            )
        except Exception as exc:
            failures += 1
            print(f"FAILED {source_name}: {type(exc).__name__}: {exc}")
            for threshold in thresholds:
                summaries.append(
                    {
                        "source_name": source_name,
                        "source_label": source.get("label", source_name),
                        "region_type": "source",
                        "section_name": "",
                        "section_label": "",
                        "threshold": threshold,
                        "status": "failed",
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                )
            if fail_fast:
                raise
    summary_path = output_root / "steady_state_summary.csv"
    if summaries:
        all_keys = []
        for row in summaries:
            for key in row:
                if key not in all_keys:
                    all_keys.append(key)
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(summaries)
    print(f"Successful sources: {len(sources) - failures}")
    print(f"Failed sources: {failures}")
    print(f"Summary CSV: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.config is not None:
        config_main(args)
    else:
        legacy_main(args)


if __name__ == "__main__":
    main()
