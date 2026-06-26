from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

from streamlines_common import (
    ROOT,
    load_config,
    load_geometry_from_config,
    make_seed_source,
    PathlineTracker,
    physical_time_value,
    read_solver_mesh,
    read_star_mesh,
    resolve_output_dir,
    resolve_path,
    time_points_from_config,
    truncate_streamlines,
    resolve_vtp_dir,
    resolve_png_dir,
)


DEFAULT_CONFIG = ROOT / "config" / "compare_streamlines_solver_star_ccm.json"


def add_relative_velocity_error_on_solver_pathlines(
    solver_pathlines: pv.PolyData,
    solver_velocity_name: str,
    solver_mesh: pv.DataSet,
    star_mesh: pv.DataSet,
    star_velocity_name: str,
    zero_speed_tolerance: float,
) -> pv.PolyData:
    sampled = solver_pathlines.sample(star_mesh)
    solver_sampled = solver_pathlines.sample(solver_mesh)
    if solver_velocity_name not in sampled.point_data and solver_velocity_name in solver_sampled.point_data:
        sampled.point_data[solver_velocity_name] = solver_sampled.point_data[solver_velocity_name]
    if star_velocity_name not in sampled.point_data:
        raise KeyError(f"Could not sample {star_velocity_name!r} from Star-CCM mesh")

    solver_velocity = np.asarray(sampled.point_data[solver_velocity_name], dtype=float)
    star_velocity = np.asarray(sampled.point_data[star_velocity_name], dtype=float)
    error_vector = solver_velocity - star_velocity
    error_magnitude = np.linalg.norm(error_vector, axis=1)
    solver_speed = np.linalg.norm(solver_velocity, axis=1)
    star_speed = np.linalg.norm(star_velocity, axis=1)
    relative_error = np.full(error_magnitude.shape, np.nan, dtype=float)
    valid = star_speed > float(zero_speed_tolerance)
    relative_error[valid] = error_magnitude[valid] / star_speed[valid]

    sampled.point_data["velocity_error"] = error_vector
    sampled.point_data["velocity_error_magnitude"] = error_magnitude
    sampled.point_data["solver_speed"] = solver_speed
    sampled.point_data["star_speed"] = star_speed
    sampled.point_data["speed_error"] = solver_speed - star_speed
    sampled.point_data["relative_velocity_error"] = relative_error
    sampled.point_data["relative_velocity_error_percent"] = relative_error * 100.0
    return sampled


def save_error_screenshot(streamlines: pv.PolyData, geometry: pv.DataSet | None, path: Path, config: dict) -> None:
    plotter = pv.Plotter(off_screen=bool(config.get("off_screen", True)), window_size=config.get("window_size", [1400, 950]))
    plotter.set_background(config.get("background", "white"))
    if geometry is not None and bool(config.get("show_geometry", True)):
        plotter.add_mesh(geometry.extract_surface(algorithm="dataset_surface"), color="lightgray", opacity=float(config.get("geometry_opacity", 0.12)))

    plotter.add_mesh(
        streamlines.tube(radius=float(config.get("tube_radius", 0.08))),
        scalars="relative_velocity_error_percent",
        cmap=config.get("cmap", "coolwarm"),
        scalar_bar_args={"title": config.get("scalar_bar_title", "Relative velocity error [%]")},
    )
    if config.get("camera_position"):
        plotter.camera_position = config["camera_position"]
    else:
        plotter.view_isometric()
    plotter.screenshot(path)
    plotter.close()


def summarize_error(time_index: int, time_value: float | None, streamlines: pv.PolyData) -> dict:
    error = np.asarray(streamlines.point_data.get("relative_velocity_error_percent", []), dtype=float)
    speed_error = np.asarray(streamlines.point_data.get("speed_error", []), dtype=float)
    valid = np.isfinite(error)
    if not valid.any():
        return {
            "time_index": time_index,
            "time_value": time_value,
            "n_points": streamlines.n_points,
            "mean_relative_velocity_error_percent": np.nan,
            "max_relative_velocity_error_percent": np.nan,
            "p95_relative_velocity_error_percent": np.nan,
            "mean_speed_error": np.nan,
        }
    return {
        "time_index": time_index,
        "time_value": time_value,
        "n_points": streamlines.n_points,
        "mean_relative_velocity_error_percent": float(np.nanmean(error)),
        "max_relative_velocity_error_percent": float(np.nanmax(error)),
        "p95_relative_velocity_error_percent": float(np.nanpercentile(error, 95)),
        "mean_speed_error": float(np.nanmean(speed_error)) if speed_error.size else np.nan,
    }


def save_error_animation(
    streamlines_by_time: list[tuple[int, float | None, pv.PolyData]],
    geometry: pv.DataSet | None,
    path: Path,
    config: dict,
) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save GIF animations") from exc

    animation_config = config.get("animation", {})
    fps = float(animation_config.get("fps", 2.0))
    duration_ms = max(int(1000.0 / max(fps, 1.0e-6)), 1)
    grow = bool(animation_config.get("grow", True))
    min_fraction = float(animation_config.get("min_fraction", 0.02))
    values = []
    for _, _, streamlines in streamlines_by_time:
        arr = np.asarray(streamlines.point_data.get("relative_velocity_error_percent", []), dtype=float)
        if arr.size:
            values.append(arr[np.isfinite(arr)])
    clim = animation_config.get("clim")
    if clim is None and values:
        merged = np.concatenate([arr for arr in values if arr.size])
        clim = [float(np.nanmin(merged)), float(np.nanmax(merged))]

    frames = []
    for frame_index, (time_index, time_value, streamlines) in enumerate(streamlines_by_time):
        growth_fraction = max(min_fraction, (frame_index + 1) / max(len(streamlines_by_time), 1))
        visible_streamlines = truncate_streamlines(streamlines, growth_fraction) if grow else streamlines
        plotter = pv.Plotter(off_screen=bool(config.get("off_screen", True)), window_size=config.get("window_size", [1400, 950]))
        plotter.set_background(config.get("background", "white"))
        if geometry is not None and bool(config.get("show_geometry", True)):
            plotter.add_mesh(geometry.extract_surface(algorithm="dataset_surface"), color="lightgray", opacity=float(config.get("geometry_opacity", 0.12)))
        plotter.add_mesh(
            visible_streamlines.tube(radius=float(config.get("tube_radius", 0.08))),
            scalars="relative_velocity_error_percent",
            cmap=animation_config.get("cmap", config.get("cmap", "coolwarm")),
            clim=clim,
            scalar_bar_args={"title": config.get("scalar_bar_title", "Relative velocity error [%]")},
        )
        title_value = f"time index: {time_index}"
        if time_value is not None:
            title_value += f" / time: {time_value:g}"
        plotter.add_text(title_value, position="upper_left", font_size=12, color="black")
        apply_camera_motion(plotter, config, frame_index, len(streamlines_by_time))
        image = plotter.screenshot(return_img=True)
        plotter.close()
        frames.append(Image.fromarray(image).convert("P", palette=Image.Palette.ADAPTIVE))

    if not frames:
        print(f"No animation frames were generated for {path}")
        return
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=int(animation_config.get("loop", 0)),
        optimize=bool(animation_config.get("optimize", True)),
    )
    print(f"Saved animation: {path}")


def apply_camera_motion(plotter: pv.Plotter, config: dict, frame_index: int, frame_count: int) -> None:
    motion = config.get("camera_motion", {})
    if not motion.get("enabled", False):
        if config.get("camera_position"):
            plotter.camera_position = config["camera_position"]
        else:
            plotter.view_isometric()
        return

    if config.get("camera_position"):
        plotter.camera_position = config["camera_position"]
    else:
        plotter.view_isometric()

    mode = motion.get("mode", "orbit")
    if mode == "orbit":
        azimuth_total = float(motion.get("azimuth_total_deg", 120.0))
        elevation_amplitude = float(motion.get("elevation_amplitude_deg", 12.0))
        zoom = float(motion.get("zoom", 1.0))
        if frame_count > 1:
            progress = frame_index / (frame_count - 1)
        else:
            progress = 0.0
        plotter.camera.Azimuth(azimuth_total * progress)
        plotter.camera.Elevation(elevation_amplitude * np.sin(np.pi * progress))
        if zoom != 1.0:
            plotter.camera.Zoom(zoom)
    else:
        raise ValueError(f"Unsupported camera motion mode: {mode}")


def collect_surface_edges(polydata: pv.PolyData, max_edges: int = 6000) -> list[tuple[int, int]]:
    faces = np.asarray(polydata.faces, dtype=np.int64)
    edges = set()
    cursor = 0
    while cursor < len(faces):
        n_points = int(faces[cursor])
        ids = [int(value) for value in faces[cursor + 1 : cursor + 1 + n_points]]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            edges.add(tuple(sorted((a, b))))
        cursor += n_points + 1
    edges = sorted(edges)
    if len(edges) > max_edges:
        step = int(np.ceil(len(edges) / max_edges))
        edges = edges[::step]
    return edges


def as_points(points: np.ndarray) -> list[list[float]]:
    return [[float(coord) for coord in point] for point in np.asarray(points, dtype=float)]


def polyline_segments_with_scalar(pathlines: pv.PolyData, scalar_name: str) -> dict:
    points = np.asarray(pathlines.points, dtype=float)
    values = np.asarray(pathlines.point_data.get(scalar_name, np.zeros(pathlines.n_points)), dtype=float)
    lines = np.asarray(pathlines.lines, dtype=np.int64)
    segments = []
    cursor = 0
    while cursor < len(lines):
        n_points = int(lines[cursor])
        ids = [int(value) for value in lines[cursor + 1 : cursor + 1 + n_points]]
        for a, b in zip(ids, ids[1:]):
            scalar_pair = np.asarray([values[a], values[b]], dtype=float)
            finite = scalar_pair[np.isfinite(scalar_pair)]
            scalar_value = float(np.mean(finite)) if finite.size else 0.0
            segments.append([a, b, scalar_value])
        cursor += n_points + 1
    return {"points": as_points(points), "segments": segments}


def export_interactive_error_html(
    streamlines_by_time: list[tuple[int, float | None, pv.PolyData]],
    geometry: pv.DataSet | None,
    output_dir: Path,
    config: dict,
) -> None:
    html_config = config.get("interactive_html", {})
    if not html_config.get("enabled", True):
        return

    geometry_points = []
    geometry_edges = []
    if geometry is not None:
        surface = geometry.extract_surface(algorithm="dataset_surface")
        geometry_points = as_points(surface.points)
        geometry_edges = collect_surface_edges(surface, max_edges=int(html_config.get("max_geometry_edges", 8000)))

    values = []
    frames = []
    for time_index, time_value, pathlines in streamlines_by_time:
        frame = polyline_segments_with_scalar(pathlines, "relative_velocity_error_percent")
        frames.append({
            "timeIndex": int(time_index),
            "timeValue": None if time_value is None else float(time_value),
            "points": frame["points"],
            "segments": frame["segments"],
        })
        if pathlines.n_points and "relative_velocity_error_percent" in pathlines.point_data:
            arr = np.asarray(pathlines.point_data["relative_velocity_error_percent"], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                values.append(arr)

    scalar_min = 0.0
    scalar_max = 1.0
    if values:
        merged = np.concatenate(values)
        scalar_min = float(np.nanmin(merged))
        scalar_max = float(np.nanmax(merged))
        if scalar_max <= scalar_min:
            scalar_max = scalar_min + 1.0e-9

    data = {
        "title": html_config.get("title", "Relative Velocity Error [%] Viewer"),
        "geometryPoints": geometry_points,
        "geometryEdges": geometry_edges,
        "frames": frames,
        "scalarMin": scalar_min,
        "scalarMax": scalar_max,
    }

    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { margin: 0; font-family: Arial, sans-serif; background: #f7f7f7; color: #222; }
  #toolbar { position: fixed; left: 12px; top: 12px; z-index: 2; background: rgba(255,255,255,0.92); padding: 10px 12px; border: 1px solid #d0d0d0; border-radius: 6px; font-size: 13px; width: 320px; }
  #toolbar input[type=range] { width: 100%; }
  canvas { display: block; width: 100vw; height: 100vh; cursor: grab; }
  canvas:active { cursor: grabbing; }
  .legend { margin-top: 8px; height: 12px; background: linear-gradient(90deg, #2166ac 0%, #67a9cf 25%, #f7f7f7 50%, #ef8a62 75%, #b2182b 100%); border: 1px solid rgba(0,0,0,0.15); }
  .legend-labels { display: flex; justify-content: space-between; font-size: 12px; margin-top: 4px; }
</style>
</head>
<body>
<div id="toolbar">
  <strong>__TITLE__</strong><br>
  Drag: rotate / Wheel: zoom<br>
  <div style="margin-top:8px;">Time index: <span id="timeIndexLabel"></span></div>
  <div style="margin-top:2px;">Frame: <span id="frameCountLabel"></span></div>
  <input id="frameSlider" type="range" min="0" max="__MAX_FRAME__" step="1" value="0">
  <label style="display:block;margin-top:6px;font-size:12px;">Jump to time index</label>
  <input id="frameNumber" type="number" min="__MIN_TIME_INDEX__" max="__MAX_TIME_INDEX__" step="1" value="__MIN_TIME_INDEX__" style="width:100%;box-sizing:border-box;padding:4px 6px;border:1px solid #c8c8c8;border-radius:4px;">
  <div class="legend"></div>
  <div class="legend-labels"><span id="minLabel"></span><span id="maxLabel"></span></div>
  <div style="margin-top:4px;font-size:12px;">Relative velocity error [%]</div>
</div>
<canvas id="view"></canvas>
<script>
const data = __DATA__;
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('frameSlider');
const frameNumber = document.getElementById('frameNumber');
const timeIndexLabel = document.getElementById('timeIndexLabel');
const frameCountLabel = document.getElementById('frameCountLabel');
const minLabel = document.getElementById('minLabel');
const maxLabel = document.getElementById('maxLabel');
let width = 0, height = 0;
let rotX = -0.65, rotY = 0.0, rotZ = 0.7, zoom = 1.0;
let dragging = false, lastX = 0, lastY = 0;
const allPoints = [];
for (const p of data.geometryPoints) allPoints.push(p);
for (const frame of data.frames) for (const p of frame.points) allPoints.push(p);
let center = [0, 0, 0];
if (allPoints.length) {
  for (const p of allPoints) {
    center[0] += p[0]; center[1] += p[1]; center[2] += p[2];
  }
  center = center.map(v => v / allPoints.length);
}
let radius = 1.0;
for (const p of allPoints) {
  const r = Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]);
  if (r > radius) radius = r;
}
function resize() {
  width = canvas.width = window.innerWidth * devicePixelRatio;
  height = canvas.height = window.innerHeight * devicePixelRatio;
  canvas.style.width = window.innerWidth + 'px';
  canvas.style.height = window.innerHeight + 'px';
  draw();
}
window.addEventListener('resize', resize);
function rotatePoint(p) {
  let x = p[0] - center[0], y = p[1] - center[1], z = p[2] - center[2];
  const cx = Math.cos(rotX), sx = Math.sin(rotX);
  const cy = Math.cos(rotY), sy = Math.sin(rotY);
  const cz = Math.cos(rotZ), sz = Math.sin(rotZ);
  let x1 = x * cz - y * sz, y1 = x * sz + y * cz; x = x1; y = y1;
  let y2 = y * cx - z * sx, z1 = y * sx + z * cx; y = y2; z = z1;
  let x2 = x * cy + z * sy, z2 = -x * sy + z * cy; x = x2; z = z2;
  return [x, y, z];
}
function project(p) {
  if (!p || !p.every(Number.isFinite)) return null;
  const r = rotatePoint(p);
  if (!r.every(Number.isFinite)) return null;
  const scale = Math.min(width, height) * 0.42 * zoom / radius;
  const projected = [width * 0.5 + r[0] * scale, height * 0.52 - r[1] * scale, r[2]];
  return projected.every(Number.isFinite) ? projected : null;
}
function colorForValue(value) {
  if (!Number.isFinite(value)) return null;
  const min = data.scalarMin, max = data.scalarMax;
  const t = Math.max(0, Math.min(1, (value - min) / Math.max(max - min, 1e-9)));
  const stops = [[33,102,172],[103,169,207],[247,247,247],[239,138,98],[178,24,43]];
  const scaled = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const local = scaled - i;
  const c0 = stops[i], c1 = stops[i + 1];
  const c = c0.map((v, idx) => Math.round(v + (c1[idx] - v) * local));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}
function drawEdges(points, edges, color, widthPx) {
  ctx.strokeStyle = color;
  ctx.lineWidth = widthPx * devicePixelRatio;
  for (const [a,b] of edges) {
    const p = project(points[a]), q = project(points[b]);
    if (!p || !q) continue;
    ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
  }
}
function drawFrame(frame) {
  const projected = frame.points.map(project);
  const segments = [];
  for (const seg of frame.segments) {
    const a = projected[seg[0]], b = projected[seg[1]];
    if (!a || !b || !Number.isFinite(seg[2])) continue;
    segments.push({ a, b, value: seg[2], depth: (a[2] + b[2]) * 0.5 });
  }
  segments.sort((s1, s2) => s1.depth - s2.depth);
  for (const seg of segments) {
    const color = colorForValue(seg.value);
    if (!color) continue;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.6 * devicePixelRatio;
    ctx.beginPath(); ctx.moveTo(seg.a[0], seg.a[1]); ctx.lineTo(seg.b[0], seg.b[1]); ctx.stroke();
  }
}
function draw() {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);
  if (data.geometryEdges.length) drawEdges(data.geometryPoints, data.geometryEdges, 'rgba(120,120,120,0.28)', 0.9);
  const frame = data.frames[Number(slider.value)] || data.frames[0];
  if (frame) drawFrame(frame);
  if (frame) {
    const sliderIndex = Number(slider.value);
    timeIndexLabel.textContent = frame.timeValue == null ? `${frame.timeIndex}` : `${frame.timeIndex} / ${frame.timeValue.toFixed(3)} s`;
    frameCountLabel.textContent = `${sliderIndex + 1} / ${data.frames.length}`;
    frameNumber.value = String(frame.timeIndex);
  }
}
function setFrameBySliderIndex(index) {
  const clamped = Math.max(0, Math.min(data.frames.length - 1, Math.round(Number(index) || 0)));
  slider.value = String(clamped);
  draw();
}
function setFrameByTimeIndex(timeIndex) {
  const requested = Math.round(Number(timeIndex));
  if (!Number.isFinite(requested)) return;
  let bestIndex = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < data.frames.length; i++) {
    const distance = Math.abs(data.frames[i].timeIndex - requested);
    if (distance < bestDistance) { bestDistance = distance; bestIndex = i; }
  }
  setFrameBySliderIndex(bestIndex);
}
slider.addEventListener('input', () => setFrameBySliderIndex(slider.value));
frameNumber.addEventListener('change', () => setFrameByTimeIndex(frameNumber.value));
frameNumber.addEventListener('keydown', event => {
  if (event.key === 'Enter') { event.preventDefault(); setFrameByTimeIndex(frameNumber.value); }
});
canvas.addEventListener('mousedown', event => { dragging = true; lastX = event.clientX; lastY = event.clientY; });
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', event => {
  if (!dragging) return;
  rotZ += (event.clientX - lastX) * 0.01;
  rotX += (event.clientY - lastY) * 0.01;
  lastX = event.clientX; lastY = event.clientY;
  draw();
});
canvas.addEventListener('wheel', event => {
  event.preventDefault();
  zoom *= Math.exp(-event.deltaY * 0.001);
  zoom = Math.max(0.2, Math.min(zoom, 8));
  draw();
}, { passive: false });
minLabel.textContent = data.scalarMin.toFixed(2) + ' %';
maxLabel.textContent = data.scalarMax.toFixed(2) + ' %';
resize();
</script>
</body>
</html>"""
    min_time_index = min((frame["timeIndex"] for frame in frames), default=0)
    max_time_index = max((frame["timeIndex"] for frame in frames), default=0)
    html = (
        template.replace("__TITLE__", data["title"])
        .replace("__DATA__", json.dumps(data, allow_nan=False))
        .replace("__MAX_FRAME__", str(max(len(frames) - 1, 0)))
        .replace("__MIN_TIME_INDEX__", str(min_time_index))
        .replace("__MAX_TIME_INDEX__", str(max_time_index))
    )
    html_dir = output_dir / html_config.get("dir_name", "html")
    html_dir.mkdir(parents=True, exist_ok=True)
    html_path = html_dir / html_config.get("filename", "streamline_error_interactive.html")
    html_path.write_text(html, encoding="utf-8")
    print(f"Saved interactive HTML: {html_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare solver and Star-CCM streamlines using common seeds.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    output_dir = resolve_output_dir(config, "output/streamlines/comparison")

    solver_config = config.get("solver", {})
    star_config = config.get("star_ccm", {})
    solver_velocity_name = solver_config.get("velocity_name", "solution_velocity")
    star_velocity_name = star_config.get("velocity_name", "Velocity")
    source = make_seed_source(config.get("seed_source", {}))
    zero_speed_tolerance = float(config.get("zero_speed_tolerance", 1.0e-9))

    case_file = resolve_path(star_config.get("case_file"))
    if case_file is None or not case_file.exists():
        raise FileNotFoundError(f"Case file not found: {case_file}")
    reader = pv.get_reader(case_file)

    time_pairs = config.get("time_pairs")
    display_offset = int(star_config.get("display_time_index_offset", 1))
    if time_pairs is None:
        star_points = time_points_from_config(star_config, reader.number_time_points)
        solver_template = solver_config.get("vtu_template", "data/260625_solver/solution_{time_index:06d}.vtu")
        time_pairs = [
            {
                "label": int(star_time_point) + display_offset,
                "solver_vtu": solver_template.format(time_index=int(star_time_point) + display_offset),
                "star_time_point": int(star_time_point),
            }
            for star_time_point in star_points
        ]

    initial_label = int(time_pairs[0].get("label", time_pairs[0].get("star_time_point", 0)))
    initial_time_value = max(physical_time_value(initial_label - 1, config), 0.0)
    tracker = PathlineTracker(source, initial_time_index=max(initial_label - 1, 0), initial_time_value=initial_time_value)

    summaries = []
    error_streamlines_by_time = []
    first_mesh = None
    combined = pv.MultiBlock()
    vtp_dir = resolve_vtp_dir(output_dir, config)
    png_dir = resolve_png_dir(output_dir, config)
    for pair in time_pairs:
        label = int(pair.get("label", pair.get("star_time_point", 0)))
        solver_vtu = resolve_path(pair["solver_vtu"])
        star_time_point = int(pair["star_time_point"])
        if solver_vtu is None or not solver_vtu.exists():
            raise FileNotFoundError(f"Solver VTU not found: {solver_vtu}")

        print(f"Comparing label={label}: solver={solver_vtu.relative_to(ROOT)}, star_time={star_time_point}")
        solver_mesh = read_solver_mesh(solver_vtu, solver_config)
        star_mesh, _star_reader_time_value = read_star_mesh(reader, star_time_point, star_config)
        time_value = physical_time_value(label, config)
        if first_mesh is None:
            first_mesh = solver_mesh

        solver_pathlines = tracker.advance(solver_mesh, solver_velocity_name, label, time_value)
        error_streamlines = add_relative_velocity_error_on_solver_pathlines(
            solver_pathlines, solver_velocity_name, solver_mesh, star_mesh, star_velocity_name, zero_speed_tolerance
        )
        error_streamlines.field_data["time_index"] = np.asarray([label], dtype=np.int32)
        error_streamlines.save(vtp_dir / f"streamline_error_{label:06d}.vtp")
        combined[f"error_{label:06d}"] = error_streamlines
        summaries.append(summarize_error(label, time_value, error_streamlines))
        error_streamlines_by_time.append((label, time_value, error_streamlines))

        if config.get("screenshot", {}).get("per_time", True):
            geometry = load_geometry_from_config(config.get("screenshot", {}), first_mesh)
            save_error_screenshot(error_streamlines, geometry, png_dir / f"streamline_error_{label:06d}.png", config.get("screenshot", {}))

    screenshot_config = config.get("screenshot", {})
    if screenshot_config.get("animation", {}).get("enabled", False):
        geometry = load_geometry_from_config(screenshot_config, first_mesh)
        save_error_animation(
            error_streamlines_by_time,
            geometry,
            output_dir / screenshot_config["animation"].get("filename", "streamline_error_animation.gif"),
            screenshot_config,
        )

    geometry = load_geometry_from_config(screenshot_config, first_mesh)
    export_interactive_error_html(error_streamlines_by_time, geometry, output_dir, config)

    combined.save(output_dir / "streamline_error_all_times.vtm")
    pd.DataFrame(summaries).to_csv(output_dir / "streamline_error_summary.csv", index=False)
    print(f"Saved comparison outputs to: {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
