from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyvista as pv
import vtk


ROOT = Path(__file__).resolve().parents[1]

# Suppress repetitive VTK/X11 warnings during off-screen rendering inside containers.
vtk.vtkObject.GlobalWarningDisplayOff()
if hasattr(vtk, 'vtkLogger'):
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
if hasattr(vtk, 'vtkStringOutputWindow') and hasattr(vtk, 'vtkOutputWindow'):
    vtk.vtkOutputWindow.SetInstance(vtk.vtkStringOutputWindow())


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def resolve_output_dir(config: dict, default: str) -> Path:
    output_dir = resolve_path(config.get("output_dir", default))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_vtp_dir(output_dir: Path, config: dict) -> Path:
    vtp_dir_name = config.get("vtp_dir_name", "vtp")
    vtp_dir = output_dir / vtp_dir_name
    vtp_dir.mkdir(parents=True, exist_ok=True)
    return vtp_dir


def resolve_png_dir(output_dir: Path, config: dict) -> Path:
    png_dir_name = config.get("png_dir_name", "png")
    png_dir = output_dir / png_dir_name
    png_dir.mkdir(parents=True, exist_ok=True)
    return png_dir


def make_plane_basis(normal: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal_array = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(normal_array)
    if norm == 0.0:
        raise ValueError("seed_source.normal must be a non-zero vector")
    normal_array /= norm

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(tmp, normal_array))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(normal_array, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal_array, e1)
    e2 /= np.linalg.norm(e2)
    return normal_array, e1, e2


def make_seed_points(seed_config: dict) -> np.ndarray:
    center = np.asarray(seed_config.get("center", [0.0, 0.0, 0.0]), dtype=float)
    _, e1, e2 = make_plane_basis(seed_config.get("normal", [0.0, 0.0, 1.0]))
    radius = float(seed_config.get("radius", 1.0))
    n_rings = int(seed_config.get("rings", 2))
    points_per_ring = int(seed_config.get("points_per_ring", 8))
    include_center = bool(seed_config.get("include_center", True))
    explicit_points = seed_config.get("points")

    if explicit_points:
        return np.asarray(explicit_points, dtype=float)

    points: list[np.ndarray] = []
    if include_center:
        points.append(center)

    for ring in range(1, n_rings + 1):
        ring_radius = radius * ring / max(n_rings, 1)
        for index in range(points_per_ring):
            theta = 2.0 * math.pi * index / points_per_ring
            points.append(center + ring_radius * (math.cos(theta) * e1 + math.sin(theta) * e2))

    if not points:
        raise ValueError("seed_source must create at least one seed point")
    return np.vstack(points)


def make_seed_source(seed_config: dict) -> pv.PolyData:
    points = make_seed_points(seed_config)
    source = pv.PolyData(points)
    source.point_data["seed_id"] = np.arange(points.shape[0], dtype=np.int32)
    return source


def sample_point_vectors(mesh: pv.DataSet, points: np.ndarray, velocity_name: str) -> tuple[np.ndarray, np.ndarray]:
    cloud = pv.PolyData(np.asarray(points, dtype=float))
    sampled = cloud.sample(mesh, locator="static_cell", mark_blank=True)
    vectors = np.asarray(sampled.point_data[velocity_name], dtype=float)
    valid = np.asarray(sampled.point_data.get("vtkValidPointMask", np.ones(len(points), dtype=np.uint8))).astype(bool)
    if vectors.ndim == 1:
        vectors = vectors.reshape((-1, 1))
    valid &= np.all(np.isfinite(vectors), axis=1)
    return vectors, valid


def make_head_points(pathlines: pv.PolyData) -> pv.PolyData:
    if pathlines.n_points == 0 or "head_mask" not in pathlines.point_data:
        return pv.PolyData()
    mask = np.asarray(pathlines.point_data["head_mask"], dtype=bool)
    if not mask.any():
        return pv.PolyData()
    points = np.asarray(pathlines.points)[mask]
    heads = pv.PolyData(points)
    for name, values in pathlines.point_data.items():
        heads.point_data[name] = np.asarray(values)[mask]
    return heads


def build_pathline_polydata(
    point_histories: list[list[np.ndarray]],
    speed_histories: list[list[float]],
    time_index_histories: list[list[int]],
    time_value_histories: list[list[float]],
) -> pv.PolyData:
    points = []
    lines = []
    seed_ids = []
    speeds = []
    time_indices = []
    time_values = []
    head_mask = []
    next_point_id = 0

    for seed_id, history in enumerate(point_histories):
        if len(history) < 2:
            continue
        n_points = len(history)
        lines.extend([n_points, *range(next_point_id, next_point_id + n_points)])
        next_point_id += n_points
        points.extend(history)
        seed_ids.extend([seed_id] * n_points)
        speeds.extend(speed_histories[seed_id])
        time_indices.extend(time_index_histories[seed_id])
        time_values.extend(time_value_histories[seed_id])
        head_mask.extend([0] * (n_points - 1) + [1])

    if not points:
        return pv.PolyData()

    polydata = pv.PolyData(np.asarray(points, dtype=float))
    polydata.verts = np.empty(0, dtype=np.int64)
    polydata.lines = np.asarray(lines, dtype=np.int64)
    polydata.point_data["seed_id"] = np.asarray(seed_ids, dtype=np.int32)
    polydata.point_data["speed"] = np.asarray(speeds, dtype=float)
    polydata.point_data["time_index"] = np.asarray(time_indices, dtype=np.int32)
    polydata.point_data["time_value"] = np.asarray(time_values, dtype=float)
    polydata.point_data["head_mask"] = np.asarray(head_mask, dtype=np.uint8)
    return polydata


class PathlineTracker:
    def __init__(self, seed_source: pv.PolyData, initial_time_index: int, initial_time_value: float):
        seed_points = np.asarray(seed_source.points, dtype=float)
        self.current_points = seed_points.copy()
        self.active = np.ones(seed_points.shape[0], dtype=bool)
        self.point_histories = [[seed_points[i].copy()] for i in range(seed_points.shape[0])]
        self.speed_histories = [[0.0] for _ in range(seed_points.shape[0])]
        self.time_index_histories = [[int(initial_time_index)] for _ in range(seed_points.shape[0])]
        self.time_value_histories = [[float(initial_time_value)] for _ in range(seed_points.shape[0])]
        self.last_time_value = float(initial_time_value)

    def advance(self, mesh: pv.DataSet, velocity_name: str, time_index: int, time_value: float) -> pv.PolyData:
        dt = max(float(time_value) - self.last_time_value, 0.0)
        active_ids = np.flatnonzero(self.active)
        if active_ids.size == 0:
            return build_pathline_polydata(
                self.point_histories,
                self.speed_histories,
                self.time_index_histories,
                self.time_value_histories,
            )

        velocities, valid = sample_point_vectors(mesh, self.current_points[active_ids], velocity_name)
        speeds = np.linalg.norm(velocities, axis=1)
        next_points = self.current_points[active_ids] + velocities * dt

        if np.any(valid):
            _, still_inside = sample_point_vectors(mesh, next_points[valid], velocity_name)
            updated_valid = valid.copy()
            updated_valid[valid] &= still_inside
        else:
            updated_valid = valid

        for local_index, seed_id in enumerate(active_ids):
            if not updated_valid[local_index]:
                self.active[seed_id] = False
                continue
            point = next_points[local_index].copy()
            self.current_points[seed_id] = point
            self.point_histories[seed_id].append(point)
            self.speed_histories[seed_id].append(float(speeds[local_index]))
            self.time_index_histories[seed_id].append(int(time_index))
            self.time_value_histories[seed_id].append(float(time_value))

        self.last_time_value = float(time_value)
        pathlines = build_pathline_polydata(
            self.point_histories,
            self.speed_histories,
            self.time_index_histories,
            self.time_value_histories,
        )
        pathlines.field_data["time_index"] = np.asarray([int(time_index)], dtype=np.int32)
        pathlines.field_data["time_value"] = np.asarray([float(time_value)], dtype=float)
        return pathlines


def ensure_point_vectors(mesh: pv.DataSet, velocity_name: str) -> pv.DataSet:
    if velocity_name in mesh.point_data:
        return mesh
    if velocity_name in mesh.cell_data:
        print(f"Converting cell_data '{velocity_name}' to point_data.")
        return mesh.cell_data_to_point_data(pass_cell_data=True)
    raise KeyError(
        f"{velocity_name!r} was not found. "
        f"Point data: {list(mesh.point_data.keys())}, Cell data: {list(mesh.cell_data.keys())}"
    )


def scale_mesh(mesh: pv.DataSet, coordinate_scale: float = 1.0, velocity_name: str | None = None, velocity_scale: float = 1.0) -> pv.DataSet:
    if coordinate_scale != 1.0 or velocity_scale != 1.0:
        mesh = mesh.copy(deep=True)
    if coordinate_scale != 1.0:
        mesh.points *= float(coordinate_scale)
    if velocity_name and velocity_scale != 1.0:
        if velocity_name in mesh.point_data:
            mesh.point_data[velocity_name] = np.asarray(mesh.point_data[velocity_name]) * float(velocity_scale)
        if velocity_name in mesh.cell_data:
            mesh.cell_data[velocity_name] = np.asarray(mesh.cell_data[velocity_name]) * float(velocity_scale)
    return mesh


def add_speed_array(mesh: pv.DataSet, velocity_name: str, speed_name: str = "speed") -> pv.DataSet:
    vectors = np.asarray(mesh.point_data[velocity_name])
    mesh.point_data[speed_name] = np.linalg.norm(vectors, axis=1)
    return mesh


def compute_streamlines(mesh: pv.DataSet, source: pv.PolyData, velocity_name: str, options: dict) -> pv.PolyData:
    mesh = ensure_point_vectors(mesh, velocity_name)
    mesh.set_active_vectors(velocity_name)
    add_speed_array(mesh, velocity_name)
    streamlines = mesh.streamlines_from_source(
        source,
        vectors=velocity_name,
        integrator_type=int(options.get("integrator_type", 45)),
        integration_direction=options.get("integration_direction", "forward"),
        initial_step_length=float(options.get("initial_step_length", 0.5)),
        step_unit=options.get("step_unit", "cl"),
        min_step_length=float(options.get("min_step_length", 0.01)),
        max_step_length=float(options.get("max_step_length", 1.0)),
        max_steps=int(options.get("max_steps", 2000)),
        terminal_speed=float(options.get("terminal_speed", 1.0e-12)),
        max_error=float(options.get("max_error", 1.0e-6)),
        max_time=options.get("max_time"),
        max_length=options.get("max_length"),
        interpolator_type=options.get("interpolator_type", "point"),
        compute_vorticity=bool(options.get("compute_vorticity", False)),
        progress_bar=bool(options.get("progress_bar", False)),
    )
    return streamlines


def physical_time_value(time_index: int, config: dict) -> float:
    return float(time_index) * float(config.get("time_step", config.get("time_step_seconds", 0.125)))


def solver_file_id_from_path(path: Path) -> int | None:
    stem = path.stem
    digits = stem.rsplit("_", 1)[-1]
    if digits.isdigit():
        return int(digits)
    return None


def iter_solver_vtu_files(config: dict) -> list[tuple[int, Path]]:
    files = config.get("files")
    if files:
        result = []
        for value in files:
            path = resolve_path(value)
            file_id = solver_file_id_from_path(path) if path is not None else None
            result.append((file_id if file_id is not None else len(result) + 1, path))
        return result

    input_dir = resolve_path(config.get("input_dir"))
    if input_dir is None:
        single_file = resolve_path(config.get("vtu_file"))
        if single_file is None:
            raise ValueError("Set either input_dir, files, or vtu_file in the config")
        return [(int(config.get("time_point", 1)), single_file)]

    pattern = config.get("file_pattern", "solution_*.vtu")
    all_files = sorted(input_dir.glob(pattern))
    if not all_files:
        raise FileNotFoundError(f"No files matched {input_dir / pattern}")

    available = {solver_file_id_from_path(path): path for path in all_files}
    available.pop(None, None)
    if not available:
        raise ValueError(f"Could not infer numeric time indices from files matched by {input_dir / pattern}")

    requested = config.get("time_points")
    if requested is not None:
        if requested == "all":
            points = sorted(available)
        else:
            points = [int(point) for point in requested]
    else:
        start = int(config.get("start", min(available)))
        stop = int(config.get("stop", max(available)))
        stride = int(config.get("stride", 1))
        points = list(range(start, stop + 1, stride))

    result = []
    missing = []
    for point in points:
        path = available.get(point)
        if path is None:
            missing.append(point)
        else:
            result.append((point, path))
    if missing:
        raise FileNotFoundError(f"Requested solver time indices were not found: {missing[:10]}")
    return result


def read_solver_mesh(path: Path, config: dict) -> pv.DataSet:
    velocity_name = config.get("velocity_name", "solution_velocity")
    mesh = pv.read(path)
    mesh = scale_mesh(
        mesh,
        coordinate_scale=float(config.get("coordinate_scale", 1.0)),
        velocity_name=velocity_name,
        velocity_scale=float(config.get("velocity_scale", 1.0)),
    )
    return ensure_point_vectors(mesh, velocity_name)


def read_star_mesh(reader, time_point: int, config: dict) -> tuple[pv.DataSet, float | None]:
    velocity_name = config.get("velocity_name", "Velocity")
    reader.set_active_time_point(int(time_point))
    active_time = getattr(reader, "active_time_value", None)
    data = reader.read()
    mesh = data.combine(merge_points=True) if isinstance(data, pv.MultiBlock) else data
    mesh = scale_mesh(
        mesh,
        coordinate_scale=float(config.get("coordinate_scale", 1000.0)),
        velocity_name=velocity_name,
        velocity_scale=float(config.get("velocity_scale", 1000.0)),
    )
    return ensure_point_vectors(mesh, velocity_name), active_time


def time_points_from_config(config: dict, n_time_points: int) -> list[int]:
    requested = config.get("time_points")
    if requested is not None:
        if requested == "all":
            points = list(range(n_time_points))
        else:
            points = [int(value) for value in requested]
    else:
        start = int(config.get("start", 0))
        stop = int(config.get("stop", n_time_points - 1))
        stride = int(config.get("stride", 1))
        points = list(range(start, stop + 1, stride))

    bad = [value for value in points if value < 0 or value >= n_time_points]
    if bad:
        raise ValueError(f"Time points out of range 0..{n_time_points - 1}: {bad[:10]}")
    return points


def add_time_arrays(streamlines: pv.PolyData, time_index: int, time_value: float | None) -> pv.PolyData:
    streamlines = streamlines.copy(deep=True)
    streamlines.field_data["time_index"] = np.asarray([time_index], dtype=np.int32)
    if time_value is not None:
        streamlines.field_data["time_value"] = np.asarray([float(time_value)], dtype=float)
    if streamlines.n_points:
        streamlines.point_data["time_index"] = np.full(streamlines.n_points, time_index, dtype=np.int32)
        if time_value is not None:
            streamlines.point_data["time_value"] = np.full(streamlines.n_points, float(time_value), dtype=float)
    return streamlines


def save_streamline_outputs(
    streamlines_by_time: list[tuple[int, float | None, pv.PolyData]],
    output_dir: Path,
    prefix: str,
    screenshot_config: dict,
    config: dict,
    geometry: pv.DataSet | None = None,
) -> pv.MultiBlock:
    multiblock = pv.MultiBlock()
    vtp_dir = resolve_vtp_dir(output_dir, config)
    png_dir = resolve_png_dir(output_dir, config)
    for time_index, time_value, streamlines in streamlines_by_time:
        name = f"{prefix}_{time_index:06d}"
        multiblock[name] = streamlines
        streamlines.save(vtp_dir / f"{name}.vtp")

    multiblock.save(output_dir / f"{prefix}_all_times.vtm")
    write_summary_csv(streamlines_by_time, output_dir / f"{prefix}_summary.csv")

    if screenshot_config.get("enabled", True):
        save_overlay_screenshot(streamlines_by_time, png_dir / f"{prefix}_overlay.png", screenshot_config, geometry)

    animation_config = screenshot_config.get("animation", {})
    if animation_config.get("enabled", False):
        save_time_animation(
            streamlines_by_time,
            output_dir / animation_config.get("filename", f"{prefix}_time_animation.gif"),
            screenshot_config,
            geometry,
        )
    return multiblock


def write_summary_csv(streamlines_by_time: list[tuple[int, float | None, pv.PolyData]], csv_path: Path) -> None:
    rows = []
    for time_index, time_value, streamlines in streamlines_by_time:
        length = streamlines.compute_cell_sizes(length=True).cell_data.get("Length") if streamlines.n_cells else []
        speed = np.asarray(streamlines.point_data.get("speed", []), dtype=float)
        rows.append(
            {
                "time_index": time_index,
                "time_value": time_value,
                "n_points": streamlines.n_points,
                "n_lines": streamlines.n_cells,
                "total_length": float(np.sum(length)) if len(length) else 0.0,
                "mean_speed": float(np.nanmean(speed)) if speed.size else np.nan,
                "max_speed": float(np.nanmax(speed)) if speed.size else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def save_overlay_screenshot(
    streamlines_by_time: list[tuple[int, float | None, pv.PolyData]],
    path: Path,
    config: dict,
    geometry: pv.DataSet | None = None,
) -> None:
    plotter = pv.Plotter(off_screen=bool(config.get("off_screen", True)), window_size=config.get("window_size", [1400, 950]))
    plotter.set_background(config.get("background", "white"))
    if geometry is not None and bool(config.get("show_geometry", True)):
        plotter.add_mesh(geometry.extract_surface(algorithm="dataset_surface"), color="lightgray", opacity=float(config.get("geometry_opacity", 0.12)))

    cmap = config.get("cmap", "viridis")
    for time_index, _, streamlines in streamlines_by_time:
        if streamlines.n_points == 0:
            continue
        plotter.add_mesh(
            streamlines.tube(radius=float(config.get("tube_radius", 0.08))),
            scalars="time_index",
            cmap=cmap,
            clim=[streamlines_by_time[0][0], streamlines_by_time[-1][0]],
            show_scalar_bar=False,
        )
    plotter.add_scalar_bar(title=config.get("scalar_bar_title", "time index"))
    camera_position = config.get("camera_position")
    if camera_position:
        plotter.camera_position = camera_position
    else:
        plotter.view_isometric()
    plotter.screenshot(path)
    plotter.close()


def truncate_streamlines(streamlines: pv.PolyData, fraction: float) -> pv.PolyData:
    if streamlines.n_points == 0 or streamlines.n_cells == 0:
        return streamlines
    fraction = max(0.0, min(float(fraction), 1.0))
    if fraction >= 0.999999:
        return streamlines

    lines = np.asarray(streamlines.lines, dtype=np.int64)
    if lines.size == 0:
        return streamlines

    new_points = []
    new_lines = []
    new_point_data = {name: [] for name in streamlines.point_data.keys()}
    cursor = 0
    next_point_id = 0
    while cursor < lines.size:
        n_points = int(lines[cursor])
        ids = lines[cursor + 1 : cursor + 1 + n_points]
        cursor += n_points + 1
        keep_count = max(2, int(np.ceil(n_points * fraction)))
        keep_count = min(keep_count, n_points)
        kept_ids = ids[:keep_count]
        new_lines.extend([keep_count, *range(next_point_id, next_point_id + keep_count)])
        new_points.extend(np.asarray(streamlines.points)[kept_ids])
        for name, values in streamlines.point_data.items():
            new_point_data[name].extend(np.asarray(values)[kept_ids])
        next_point_id += keep_count

    if not new_points:
        return pv.PolyData()

    truncated = pv.PolyData(np.asarray(new_points))
    truncated.lines = np.asarray(new_lines, dtype=np.int64)
    for name, values in new_point_data.items():
        truncated.point_data[name] = np.asarray(values)
    return truncated


def save_time_animation(
    streamlines_by_time: list[tuple[int, float | None, pv.PolyData]],
    path: Path,
    config: dict,
    geometry: pv.DataSet | None = None,
) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save GIF animations") from exc

    animation_config = config.get("animation", {})
    fps = float(animation_config.get("fps", 2.0))
    duration_ms = max(int(1000.0 / max(fps, 1.0e-6)), 1)
    scalar = animation_config.get("scalars", "speed")
    clim = animation_config.get("clim")
    if clim is None:
        values = []
        for _, _, pathlines in streamlines_by_time:
            if scalar in pathlines.point_data:
                arr = np.asarray(pathlines.point_data[scalar], dtype=float)
                if arr.size:
                    values.append(arr[np.isfinite(arr)])
        values = [arr for arr in values if arr.size]
        if values:
            merged = np.concatenate(values)
            clim = [float(np.nanmin(merged)), float(np.nanmax(merged))]

    frames = []
    for time_index, time_value, pathlines in streamlines_by_time:
        plotter = pv.Plotter(
            off_screen=bool(config.get("off_screen", True)),
            window_size=config.get("window_size", [1400, 950]),
        )
        plotter.set_background(config.get("background", "white"))
        if geometry is not None and bool(config.get("show_geometry", True)):
            plotter.add_mesh(
                geometry.extract_surface(algorithm="dataset_surface"),
                color="lightgray",
                opacity=float(config.get("geometry_opacity", 0.12)),
            )

        if pathlines.n_points:
            mesh_to_draw = pathlines.tube(radius=float(config.get("tube_radius", 0.08)))
            kwargs = {
                "cmap": animation_config.get("cmap", config.get("cmap", "viridis")),
                "show_scalar_bar": bool(animation_config.get("show_scalar_bar", True)),
            }
            if scalar in pathlines.point_data:
                kwargs["scalars"] = scalar
            if clim is not None and "scalars" in kwargs:
                kwargs["clim"] = clim
            plotter.add_mesh(mesh_to_draw, **kwargs)

            if bool(animation_config.get("show_heads", True)):
                heads = make_head_points(pathlines)
                if heads.n_points:
                    plotter.add_mesh(
                        heads,
                        color=animation_config.get("head_color", "black"),
                        point_size=float(animation_config.get("head_size", 12.0)),
                        render_points_as_spheres=True,
                    )

        title_value = f"time index: {time_index}"
        if time_value is not None:
            title_value += f" / time: {time_value:g}"
        plotter.add_text(title_value, position="upper_left", font_size=12, color="black")
        camera_position = config.get("camera_position")
        if camera_position:
            plotter.camera_position = camera_position
        else:
            plotter.view_isometric()
        image = plotter.screenshot(return_img=True)
        plotter.close()
        frames.append(Image.fromarray(image).convert("P", palette=Image.Palette.ADAPTIVE))

    if not frames:
        print(f"No animation frames were generated for {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=int(animation_config.get("loop", 0)),
        optimize=bool(animation_config.get("optimize", True)),
    )
    print(f"Saved animation: {path}")


def load_geometry_from_config(config: dict, fallback_mesh: pv.DataSet | None = None) -> pv.DataSet | None:
    geometry_file = resolve_path(config.get("geometry_file"))
    if geometry_file is not None:
        if not geometry_file.exists():
            raise FileNotFoundError(f"Geometry file not found: {geometry_file}")
        try:
            return pv.read(geometry_file)
        except Exception as exc:
            print(f"Could not read geometry file {geometry_file}: {exc}")
    if fallback_mesh is not None and bool(config.get("use_mesh_surface_as_geometry", True)):
        return fallback_mesh.extract_surface(algorithm="dataset_surface")
    return None
