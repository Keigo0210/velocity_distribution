from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

VTU_DIR = DATA_DIR / "2605201911_output"
VELOCITY_NAME = "solution_velocity"
GEOMETRY_FILE = None
GEOMETRY_EXTENSIONS = (".msh", ".opts", ".vtk", ".vtu", ".vtp", ".stl", ".obj", ".ply")
FLIP_S_AXIS = False
FLIP_T_AXIS = False
SECTIONS_FILE = ROOT / "config" / "sections.json"


def make_plane_basis(normal: np.ndarray):
    normal = normal / np.linalg.norm(normal)

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(normal, tmp)
    e1 = e1 / np.linalg.norm(e1)

    e2 = np.cross(normal, e1)
    e2 = e2 / np.linalg.norm(e2)

    if FLIP_S_AXIS:
        e1 = -e1
    if FLIP_T_AXIS:
        e2 = -e2

    return e1, e2


def format_vector(vector):
    return "[" + ", ".join(f"{value:+.3f}" for value in vector) + "]"



def format_coord_for_name(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))

    text = f"{value:.6g}"
    return text.replace("-", "m").replace(".", "p")


def section_name_from_center(center):
    if len(center) != 3:
        raise ValueError(f"center は [x, y, z] で指定してください: {center}")

    coords = "_".join(format_coord_for_name(value) for value in center)
    return f"section_{coords}"


def make_arrow(start, direction, length):
    direction = np.asarray(direction, dtype=float)
    return pv.Arrow(start=start, direction=direction, scale=length)



def load_section_specs():
    if not SECTIONS_FILE.exists():
        raise FileNotFoundError(f"断面設定ファイルが見つかりません: {SECTIONS_FILE}")

    with SECTIONS_FILE.open() as file:
        config = json.load(file)

    if isinstance(config, list):
        sections = config
    else:
        sections = config.get("sections")

    if not sections:
        raise ValueError(f"{SECTIONS_FILE} に sections が定義されていません。")

    required_keys = {"center", "normal"}
    for index, section in enumerate(sections):
        missing_keys = required_keys - set(section)
        if missing_keys:
            raise ValueError(
                f"{SECTIONS_FILE}: sections[{index}] に {sorted(missing_keys)} がありません。"
            )
        for size_key in ("width", "height"):
            if size_key in section and section[size_key] is not None and section[size_key] <= 0:
                raise ValueError(
                    f"{SECTIONS_FILE}: sections[{index}].{size_key} は正の値にしてください。"
                )

    return sections



def collect_surface_edges(polydata, max_edges=9000):
    faces = np.asarray(polydata.faces)
    edges = set()
    i = 0
    while i < len(faces):
        n_points = int(faces[i])
        ids = [int(value) for value in faces[i + 1 : i + 1 + n_points]]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            edges.add(tuple(sorted((a, b))))
        i += n_points + 1

    edges = sorted(edges)
    if len(edges) > max_edges:
        step = int(np.ceil(len(edges) / max_edges))
        edges = edges[::step]
    return edges


def collect_polyline_edges(polydata):
    if not hasattr(polydata, "lines"):
        return []

    lines = np.asarray(polydata.lines)
    if len(lines) == 0:
        return []

    edges = []
    i = 0
    while i < len(lines):
        n_points = int(lines[i])
        ids = [int(value) for value in lines[i + 1 : i + 1 + n_points]]
        for a, b in zip(ids, ids[1:]):
            edges.append((a, b))
        i += n_points + 1
    return edges


def as_points(points):
    return [[float(coord) for coord in point] for point in np.asarray(points)]


def export_interactive_overview(reference_geometry, section, section_name, origin, normal, width=None, height=None):
    html_path = OUTPUT_DIR / f"{section_name}_overview_interactive.html"
    plane = make_plane_actor(origin, normal, section.points, reference_geometry.bounds, width, height)
    s_axis, t_axis = make_plane_basis(normal)
    arrow_length = max(float(plane.length) * 0.18, 1.0)

    data = {
        "title": section_name,
        "geometryPoints": as_points(reference_geometry.points),
        "geometryEdges": collect_surface_edges(reference_geometry),
        "sectionPoints": as_points(section.points),
        "sectionEdges": collect_polyline_edges(section),
        "planePoints": as_points(plane.points),
        "planeFaces": [[0, 1, 3, 2]],
        "origin": as_points([origin])[0],
        "normalEnd": as_points([origin + normal * arrow_length])[0],
        "sEnd": as_points([origin + s_axis * arrow_length])[0],
        "tEnd": as_points([origin + t_axis * arrow_length])[0],
    }

    html = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{section_name} overview</title>
<style>
  body {{ margin: 0; font-family: Arial, sans-serif; background: #f8f8f8; color: #222; }}
  #toolbar {{ position: fixed; left: 12px; top: 12px; z-index: 2; background: rgba(255,255,255,0.9); padding: 8px 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 13px; }}
  button {{ margin-right: 4px; }}
  canvas {{ display: block; width: 100vw; height: 100vh; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}
</style>
</head>
<body>
<div id="toolbar">
  <strong>{section_name}</strong><br>
  orange arrow: +s<br>
  green arrow: +t<br>
  blue arrow: normal<br>
  Left drag horizontal: rotate around Z<br>
  Left drag vertical: tilt / Wheel: zoom<br>
  <button onclick="setView('iso')">iso</button>
  <button onclick="setView('x')">x</button>
  <button onclick="setView('y')">y</button>
  <button onclick="setView('z')">z</button>
</div>
<canvas id="view"></canvas>
<script>
const data = {json.dumps(data)};
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
let width = 0;
let height = 0;
let rotX = -0.6;
let rotY = 0.0;
let rotZ = 0.75;
let zoom = 1.0;
let dragging = false;
let lastX = 0;
let lastY = 0;
let activeView = 'iso';

function resize() {{
  width = canvas.width = window.innerWidth * devicePixelRatio;
  height = canvas.height = window.innerHeight * devicePixelRatio;
  canvas.style.width = window.innerWidth + 'px';
  canvas.style.height = window.innerHeight + 'px';
  draw();
}}
window.addEventListener('resize', resize);

const allPoints = data.geometryPoints.concat(data.sectionPoints, data.planePoints, [data.origin, data.normalEnd, data.sEnd, data.tEnd]);
const center = [0, 1, 2].map(i => allPoints.reduce((sum, p) => sum + p[i], 0) / allPoints.length);
const radius = Math.max(...allPoints.map(p => Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2])));

function rotatePoint(p) {{
  let x = p[0] - center[0];
  let y = p[1] - center[1];
  let z = p[2] - center[2];

  let cx = Math.cos(rotX), sx = Math.sin(rotX);
  let cy = Math.cos(rotY), sy = Math.sin(rotY);
  let cz = Math.cos(rotZ), sz = Math.sin(rotZ);

  // Apply Z first so horizontal dragging rotates around the real/model Z axis.
  let x1 = x * cz - y * sz;
  let y1 = x * sz + y * cz;
  x = x1; y = y1;

  let y2 = y * cx - z * sx;
  let z1 = y * sx + z * cx;
  y = y2; z = z1;

  let x2 = x * cy + z * sy;
  let z2 = -x * sy + z * cy;
  x = x2; z = z2;

  return [x, y, z];
}}

function dot(a, b) {{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}}

function projectXView(p) {{
  const v = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
  const az = rotZ;
  const tilt = rotY;

  const depthBase = [Math.cos(az), Math.sin(az), 0];
  const right = [-Math.sin(az), Math.cos(az), 0];
  const upBase = [0, 0, 1];

  // Horizontal drag rotates around model Z. Vertical drag tilts Z into/out of screen depth.
  const depth = [
    depthBase[0] * Math.cos(tilt) + upBase[0] * Math.sin(tilt),
    depthBase[1] * Math.cos(tilt) + upBase[1] * Math.sin(tilt),
    depthBase[2] * Math.cos(tilt) + upBase[2] * Math.sin(tilt),
  ];
  const up = [
    -depthBase[0] * Math.sin(tilt) + upBase[0] * Math.cos(tilt),
    -depthBase[1] * Math.sin(tilt) + upBase[1] * Math.cos(tilt),
    -depthBase[2] * Math.sin(tilt) + upBase[2] * Math.cos(tilt),
  ];

  const scale = Math.min(width, height) * 0.42 * zoom / radius;
  return [width * 0.5 + dot(v, right) * scale, height * 0.52 - dot(v, up) * scale, dot(v, depth)];
}}

function project(p) {{
  if (activeView === 'x') {{
    return projectXView(p);
  }}
  const r = rotatePoint(p);
  const scale = Math.min(width, height) * 0.42 * zoom / radius;
  return [width * 0.5 + r[0] * scale, height * 0.52 - r[1] * scale, r[2]];
}}

function line(points, a, b, color, lw) {{
  const p = project(points[a]);
  const q = project(points[b]);
  ctx.strokeStyle = color;
  ctx.lineWidth = lw * devicePixelRatio;
  ctx.beginPath();
  ctx.moveTo(p[0], p[1]);
  ctx.lineTo(q[0], q[1]);
  ctx.stroke();
}}

function drawPoints(points, color, radius) {{
  ctx.fillStyle = color;
  for (const point of points) {{
    const p = project(point);
    ctx.beginPath();
    ctx.arc(p[0], p[1], radius * devicePixelRatio, 0, Math.PI * 2);
    ctx.fill();
  }}
}}

function drawArrow(start, end, color, label) {{
  const p = project(start);
  const q = project(end);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 3 * devicePixelRatio;
  ctx.beginPath();
  ctx.moveTo(p[0], p[1]);
  ctx.lineTo(q[0], q[1]);
  ctx.stroke();

  const angle = Math.atan2(q[1] - p[1], q[0] - p[0]);
  const size = 10 * devicePixelRatio;
  ctx.beginPath();
  ctx.moveTo(q[0], q[1]);
  ctx.lineTo(q[0] - size * Math.cos(angle - 0.45), q[1] - size * Math.sin(angle - 0.45));
  ctx.lineTo(q[0] - size * Math.cos(angle + 0.45), q[1] - size * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fill();

  ctx.font = `${{14 * devicePixelRatio}}px Arial`;
  ctx.fillText(label, q[0] + 6 * devicePixelRatio, q[1] - 6 * devicePixelRatio);
}}

function axisDirection(vector) {{
  const p = project(vector);
  const o = project([0, 0, 0]);
  return [p[0] - o[0], p[1] - o[1]];
}}

function drawAxisIndicator() {{
  const origin = [72 * devicePixelRatio, height - 72 * devicePixelRatio];
  const axisLength = 48 * devicePixelRatio;
  const axes = [
    {{ label: 'X', vector: [1, 0, 0], color: '#d62728' }},
    {{ label: 'Y', vector: [0, 1, 0], color: '#2ca02c' }},
    {{ label: 'Z', vector: [0, 0, 1], color: '#1f77b4' }},
  ];

  ctx.save();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.82)';
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
  ctx.lineWidth = devicePixelRatio;
  ctx.fillRect(16 * devicePixelRatio, height - 142 * devicePixelRatio, 128 * devicePixelRatio, 126 * devicePixelRatio);
  ctx.strokeRect(16 * devicePixelRatio, height - 142 * devicePixelRatio, 128 * devicePixelRatio, 126 * devicePixelRatio);

  for (const axis of axes) {{
    const d = axisDirection(axis.vector);
    const norm = Math.hypot(d[0], d[1]) || 1;
    const end = [origin[0] + d[0] / norm * axisLength, origin[1] + d[1] / norm * axisLength];
    const angle = Math.atan2(end[1] - origin[1], end[0] - origin[0]);
    const head = 8 * devicePixelRatio;

    ctx.strokeStyle = axis.color;
    ctx.fillStyle = axis.color;
    ctx.lineWidth = 2.5 * devicePixelRatio;
    ctx.beginPath();
    ctx.moveTo(origin[0], origin[1]);
    ctx.lineTo(end[0], end[1]);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(end[0], end[1]);
    ctx.lineTo(end[0] - head * Math.cos(angle - 0.45), end[1] - head * Math.sin(angle - 0.45));
    ctx.lineTo(end[0] - head * Math.cos(angle + 0.45), end[1] - head * Math.sin(angle + 0.45));
    ctx.closePath();
    ctx.fill();

    ctx.font = `${{13 * devicePixelRatio}}px Arial`;
    ctx.fillText(axis.label, end[0] + 5 * devicePixelRatio, end[1] - 5 * devicePixelRatio);
  }}
  ctx.restore();
}}

function drawPlane() {{
  const pts = data.planePoints.map(project);
  ctx.fillStyle = 'rgba(30, 144, 255, 0.25)';
  ctx.strokeStyle = 'rgba(30, 144, 255, 0.75)';
  ctx.lineWidth = 1.5 * devicePixelRatio;
  for (const face of data.planeFaces) {{
    ctx.beginPath();
    ctx.moveTo(pts[face[0]][0], pts[face[0]][1]);
    for (const id of face.slice(1)) ctx.lineTo(pts[id][0], pts[id][1]);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }}
}}

function draw() {{
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  drawPlane();
  for (const e of data.geometryEdges) line(data.geometryPoints, e[0], e[1], 'rgba(130,130,130,0.35)', 0.8);
  for (const e of data.sectionEdges) line(data.sectionPoints, e[0], e[1], 'crimson', 2.0);
  drawPoints(data.sectionPoints, 'crimson', 2.2);
  drawArrow(data.origin, data.sEnd, 'orange', '+s');
  drawArrow(data.origin, data.tEnd, 'seagreen', '+t');
  drawArrow(data.origin, data.normalEnd, 'royalblue', 'normal');
  drawAxisIndicator();
}}

canvas.addEventListener('mousedown', event => {{ dragging = true; lastX = event.clientX; lastY = event.clientY; }});
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', event => {{
  if (!dragging) return;
  const dx = event.clientX - lastX;
  const dy = event.clientY - lastY;
  rotZ += dx * 0.01;
  if (activeView === 'x') {{
    rotY += dy * 0.01;
  }} else {{
    rotX += dy * 0.01;
  }}
  lastX = event.clientX;
  lastY = event.clientY;
  draw();
}});
canvas.addEventListener('wheel', event => {{
  event.preventDefault();
  zoom *= Math.exp(-event.deltaY * 0.001);
  zoom = Math.max(0.25, Math.min(zoom, 8));
  draw();
}}, {{ passive: false }});

function setView(view) {{
  activeView = view;
  if (view === 'iso') {{ rotX = -0.6; rotY = 0.0; rotZ = 0.75; }}
  if (view === 'x') {{ rotX = 0.0; rotY = 0.0; rotZ = 0.0; }}
  if (view === 'y') {{ rotX = 0.0; rotY = 0.0; rotZ = Math.PI / 2; }}
  if (view === 'z') {{ rotX = 0.0; rotY = 0.0; rotZ = 0.0; }}
  draw();
}}

resize();
</script>
</body>
</html>
"""
    html_path.write_text(html)
    print(f"Saved: {html_path}")


def load_latest_mesh():
    vtu_files = sorted(VTU_DIR.glob("*.vtu"))

    if not vtu_files:
        raise FileNotFoundError(f"{VTU_DIR} に .vtu ファイルが見つかりません。")

    target_file = vtu_files[-1]
    print(f"Reading: {target_file.name}")

    mesh = pv.read(target_file)

    print("Mesh loaded.")
    print("Number of points:", mesh.n_points)
    print("Number of cells:", mesh.n_cells)
    print("Point data:", list(mesh.point_data.keys()))

    if VELOCITY_NAME not in mesh.point_data:
        raise KeyError(
            f"{VELOCITY_NAME} が point_data に見つかりません。"
            f"Available: {list(mesh.point_data.keys())}"
        )

    return mesh, target_file


def find_geometry_file():
    if GEOMETRY_FILE is not None:
        geometry_file = Path(GEOMETRY_FILE)
        if not geometry_file.is_absolute():
            geometry_file = ROOT / geometry_file
        if not geometry_file.exists():
            raise FileNotFoundError(f"形状ファイルが見つかりません: {geometry_file}")
        return geometry_file

    candidates = []
    for ext in GEOMETRY_EXTENSIONS:
        candidates.extend(DATA_DIR.rglob(f"*{ext}"))

    candidates = [
        path
        for path in candidates
        if not path.name.endswith(":Zone.Identifier") and path.parent != VTU_DIR
    ]

    if not candidates:
        return None

    return sorted(candidates)[0]


def load_reference_geometry(solution_mesh):
    geometry_file = find_geometry_file()

    if geometry_file is None:
        print("Geometry file not found. Using latest VTU surface for overview.")
        return solution_mesh.extract_surface(algorithm="dataset_surface"), "latest VTU surface"

    print(f"Reading geometry: {geometry_file}")
    try:
        geometry = pv.read(geometry_file)
    except Exception as exc:
        print(f"Could not read geometry file: {exc}")
        print("Using latest VTU surface for overview.")
        return solution_mesh.extract_surface(algorithm="dataset_surface"), "latest VTU surface"

    return geometry.extract_surface(algorithm="dataset_surface"), geometry_file.name


def make_plane_actor(origin, normal, section_points, fallback_bounds, width=None, height=None):
    if width is not None:
        i_size = float(width)
    else:
        i_size = None

    if height is not None:
        j_size = float(height)
    else:
        j_size = None

    if section_points.size and (i_size is None or j_size is None):
        e1, e2 = make_plane_basis(normal)
        rel = section_points - origin
        s = rel @ e1
        t = rel @ e2
        if i_size is None:
            i_size = max(float(np.ptp(s)) * 1.25, 1.0)
        if j_size is None:
            j_size = max(float(np.ptp(t)) * 1.25, 1.0)
    else:
        bounds = np.asarray(fallback_bounds, dtype=float)
        lengths = np.array(
            [
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4],
            ]
        )
        size = max(float(np.linalg.norm(lengths)), 1.0)
        if i_size is None:
            i_size = size
        if j_size is None:
            j_size = size

    return pv.Plane(
        center=origin,
        direction=normal,
        i_size=i_size,
        j_size=j_size,
        i_resolution=1,
        j_resolution=1,
    )


def export_section_overview(reference_geometry, geometry_label, section, section_name, origin, normal, width=None, height=None):
    overview_path = OUTPUT_DIR / f"{section_name}_overview.png"
    plane = make_plane_actor(origin, normal, section.points, reference_geometry.bounds, width, height)
    s_axis, t_axis = make_plane_basis(normal)
    arrow_length = max(float(plane.length) * 0.18, 1.0)
    s_arrow = make_arrow(origin, s_axis, arrow_length)
    t_arrow = make_arrow(origin, t_axis, arrow_length)

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    plotter.set_background("white")
    plotter.add_mesh(
        reference_geometry,
        color="lightgray",
        opacity=0.25,
        show_edges=False,
    )
    plotter.add_mesh(section, color="crimson", line_width=4, point_size=4)
    plotter.add_mesh(plane, color="dodgerblue", opacity=0.35, show_edges=True)
    plotter.add_mesh(s_arrow, color="orange")
    plotter.add_mesh(t_arrow, color="seagreen")
    plotter.add_text(
        "\n".join(
            [
                f"Geometry: {geometry_label}",
                f"Plane origin: {format_vector(origin)}",
                f"Plane normal: {format_vector(normal)}",
                f"orange arrow: +s {format_vector(s_axis)}",
                f"green arrow: +t {format_vector(t_axis)}",
                f"section width x height: {width if width is not None else 'auto'} x {height if height is not None else 'auto'}",
            ]
        ),
        position="upper_left",
        font_size=10,
        color="black",
    )
    plotter.add_axes()
    plotter.camera_position = "iso"

    export_interactive_overview(reference_geometry, section, section_name, origin, normal, width, height)

    plotter.show(screenshot=overview_path)
    plotter.close()

    print(f"Saved: {overview_path}")


def export_section(mesh, section_name, center, normal, width=None, height=None):
    origin = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)

    if origin.shape != (3,):
        raise ValueError(f"{section_name}: center は [x, y, z] で指定してください: {center}")
    if normal.shape != (3,):
        raise ValueError(f"{section_name}: normal は [nx, ny, nz] で指定してください: {normal}")

    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise ValueError(f"{section_name}: normal にゼロベクトルは指定できません。")
    normal = normal / normal_norm

    section = mesh.slice(origin=origin, normal=normal)

    if section.n_points == 0:
        raise RuntimeError(
            f"{section_name}: 断面上に点がありません。"
            "origin と normal がメッシュ領域を横切っているか確認してください。"
        )

    if VELOCITY_NAME not in section.point_data:
        raise KeyError(
            f"{section_name}: {VELOCITY_NAME} が断面データにありません。"
            f"Available: {list(section.point_data.keys())}"
        )

    e1, e2 = make_plane_basis(normal)
    rel = section.points - origin
    s_all = rel @ e1
    t_all = rel @ e2

    if width is not None or height is not None:
        keep = np.ones(section.n_points, dtype=bool)
        if width is not None:
            keep &= np.abs(s_all) <= 0.5 * float(width)
        if height is not None:
            keep &= np.abs(t_all) <= 0.5 * float(height)
        section = section.extract_points(keep, adjacent_cells=False)

        if section.n_points == 0:
            raise RuntimeError(
                f"{section_name}: 指定した width/height の範囲内に断面点がありません。"
            )

    points = section.points
    velocity = section.point_data[VELOCITY_NAME]

    speed = np.linalg.norm(velocity, axis=1)
    normal_velocity = velocity @ normal

    rel = points - origin
    s = rel @ e1
    t = rel @ e2

    df = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "s": s,
            "t": t,
            "ux": velocity[:, 0],
            "uy": velocity[:, 1],
            "uz": velocity[:, 2],
            "speed": speed,
            "normal_velocity": normal_velocity,
        }
    )

    csv_path = OUTPUT_DIR / f"{section_name}.csv"
    png_path = OUTPUT_DIR / f"{section_name}.png"

    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    contour = ax.tricontourf(s, t, speed, levels=30)
    fig.colorbar(contour, ax=ax, label="Velocity magnitude")
    ax.set_xlabel(f"s: + direction / plot right = {format_vector(e1)}")
    ax.set_ylabel(f"t: + direction / plot up = {format_vector(e2)}")
    ax.axis("equal")
    ax.set_title(section_name)
    arrow_origin = (-0.18, -0.16)
    s_arrow_end = (-0.06, -0.16)
    t_arrow_end = (-0.18, -0.04)
    ax.annotate(
        "",
        xy=s_arrow_end,
        xytext=arrow_origin,
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "orange", "linewidth": 2},
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=t_arrow_end,
        xytext=arrow_origin,
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "seagreen", "linewidth": 2},
        annotation_clip=False,
    )
    ax.text(
        s_arrow_end[0] + 0.008,
        s_arrow_end[1],
        "+s",
        transform=ax.transAxes,
        color="orange",
        weight="bold",
        va="center",
        clip_on=False,
    )
    ax.text(
        t_arrow_end[0],
        t_arrow_end[1] + 0.008,
        "+t",
        transform=ax.transAxes,
        color="seagreen",
        weight="bold",
        ha="center",
        clip_on=False,
    )
    fig.subplots_adjust(left=0.24, bottom=0.24, right=0.92, top=0.92)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")

    return section, origin, normal


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    mesh, target_file = load_latest_mesh()
    reference_geometry, geometry_label = load_reference_geometry(mesh)

    # まずはメッシュ全体の範囲を表示
    bounds = mesh.bounds
    print()
    print("Mesh bounds:")
    print(f"x: {bounds[0]} to {bounds[1]}")
    print(f"y: {bounds[2]} to {bounds[3]}")
    print(f"z: {bounds[4]} to {bounds[5]}")

    section_specs = load_section_specs()
    print(f"Section config: {SECTIONS_FILE}")

    for sec in section_specs:
        section_name = section_name_from_center(sec["center"])
        section, origin, normal = export_section(
            mesh=mesh,
            section_name=section_name,
            center=sec["center"],
            normal=sec["normal"],
            width=sec.get("width"),
            height=sec.get("height"),
        )
        export_section_overview(
            reference_geometry=reference_geometry,
            geometry_label=geometry_label,
            section=section,
            section_name=section_name,
            origin=origin,
            normal=normal,
            width=sec.get("width"),
            height=sec.get("height"),
        )


if __name__ == "__main__":
    main()