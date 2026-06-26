from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH_DIR = ROOT / "data" / "mesh"

# Gmsh hex8 local node order. Each face is written as a quadrangle element.
HEX8_FACES = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


@dataclass(frozen=True)
class BoundaryPatch:
    name: str
    physical_id: int
    cell_ids: tuple[int, ...]


@dataclass
class MeshData:
    points: np.ndarray
    faces: np.ndarray
    original_element_count: int


def resolve_mesh_path(value: str | None) -> Path:
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        return path

    meshes = sorted(DEFAULT_MESH_DIR.glob("*.msh"))
    meshes = [path for path in meshes if not path.name.endswith("_identified.msh")]
    if not meshes:
        raise FileNotFoundError(f"No .msh files found in {DEFAULT_MESH_DIR}")
    if len(meshes) > 1:
        print("Available mesh files:")
        for index, path in enumerate(meshes, start=1):
            print(f"  {index}: {path.relative_to(ROOT)}")
        selected = input("Mesh number: ").strip()
        return meshes[int(selected) - 1]
    return meshes[0]


def parse_nodes_and_boundary_faces(mesh_path: Path) -> MeshData:
    points: np.ndarray | None = None
    boundary_faces: dict[tuple[int, int, int, int], tuple[int, int, int, int] | None] = {}
    original_element_count = 0

    with mesh_path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            marker = line.strip()
            if marker == "$Nodes":
                node_count = int(next(file))
                points = np.empty((node_count + 1, 3), dtype=np.float64)
                for _ in range(node_count):
                    parts = next(file).split()
                    node_id = int(parts[0])
                    points[node_id] = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif marker == "$Elements":
                original_element_count = int(next(file))
                for _ in range(original_element_count):
                    parts = next(file).split()
                    element_type = int(parts[1])
                    tag_count = int(parts[2])
                    if element_type != 5:
                        continue

                    nodes = tuple(int(value) for value in parts[3 + tag_count : 3 + tag_count + 8])
                    for local_face in HEX8_FACES:
                        face = tuple(nodes[index] for index in local_face)
                        key = tuple(sorted(face))
                        if key in boundary_faces:
                            del boundary_faces[key]
                        else:
                            boundary_faces[key] = face

    if points is None:
        raise ValueError(f"{mesh_path} does not contain a $Nodes section")
    if original_element_count <= 0:
        raise ValueError(f"{mesh_path} does not contain a valid $Elements section")

    faces = np.asarray(list(boundary_faces.values()), dtype=np.int64)
    if faces.size == 0:
        raise ValueError("No boundary faces were found. This script expects hex8 volume elements.")

    return MeshData(points=points, faces=faces, original_element_count=original_element_count)


def make_polydata(mesh: MeshData) -> pv.PolyData:
    vtk_faces = np.empty((len(mesh.faces), 5), dtype=np.int64)
    vtk_faces[:, 0] = 4
    vtk_faces[:, 1:] = mesh.faces - 1
    return pv.PolyData(mesh.points[1:], vtk_faces.ravel())




def make_patch_polydata(
    points: np.ndarray,
    faces: np.ndarray,
    cell_ids: tuple[int, ...],
    normals: np.ndarray,
    offset: float,
) -> pv.PolyData:
    patch_faces = faces[np.asarray(cell_ids, dtype=np.int64)]
    unique_nodes, inverse = np.unique(patch_faces.ravel(), return_inverse=True)
    patch_points = points[unique_nodes].copy()

    node_normals = np.zeros_like(patch_points)
    repeated_normals = np.repeat(normals[np.asarray(cell_ids, dtype=np.int64)], 4, axis=0)
    np.add.at(node_normals, inverse, repeated_normals)
    lengths = np.linalg.norm(node_normals, axis=1)
    lengths[lengths == 0.0] = 1.0
    patch_points += offset * node_normals / lengths[:, None]

    vtk_faces = np.empty((len(cell_ids), 5), dtype=np.int64)
    vtk_faces[:, 0] = 4
    vtk_faces[:, 1:] = inverse.reshape((-1, 4))
    return pv.PolyData(patch_points, vtk_faces.ravel())


def compute_face_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p0 = points[faces[:, 0]]
    p1 = points[faces[:, 1]]
    p3 = points[faces[:, 3]]
    normals = np.cross(p1 - p0, p3 - p0)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0.0] = 1.0
    return normals / lengths[:, None]


def build_face_adjacency(faces: np.ndarray) -> list[list[int]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cell_id, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            edge_to_faces[tuple(sorted((int(a), int(b))))].append(cell_id)

    adjacency: list[set[int]] = [set() for _ in range(len(faces))]
    for cell_ids in edge_to_faces.values():
        if len(cell_ids) < 2:
            continue
        for cell_id in cell_ids:
            adjacency[cell_id].update(other for other in cell_ids if other != cell_id)
    return [sorted(values) for values in adjacency]


def grow_patch(seed_cell_id: int, adjacency: list[list[int]], normals: np.ndarray, max_angle_deg: float) -> tuple[int, ...]:
    min_dot = math.cos(math.radians(max_angle_deg))
    seed_normal = normals[seed_cell_id]
    selected: list[int] = []
    seen = {seed_cell_id}
    queue: deque[int] = deque([seed_cell_id])

    while queue:
        cell_id = queue.popleft()
        selected.append(cell_id)
        for next_cell_id in adjacency[cell_id]:
            if next_cell_id in seen:
                continue
            if abs(float(np.dot(seed_normal, normals[next_cell_id]))) < min_dot:
                continue
            seen.add(next_cell_id)
            queue.append(next_cell_id)

    return tuple(sorted(selected))


def next_default_id(assignments: dict[int, tuple[str, int]]) -> int:
    used = {physical_id for _, physical_id in assignments.values()}
    physical_id = 1
    while physical_id in used:
        physical_id += 1
    return physical_id


def collect_physical_names(assignments: dict[int, tuple[str, int]]) -> list[tuple[int, str]]:
    names_by_id: dict[int, str] = {}
    for name, physical_id in assignments.values():
        if physical_id in names_by_id and names_by_id[physical_id] != name:
            raise ValueError(
                f"Physical ID {physical_id} is used by both {names_by_id[physical_id]!r} and {name!r}"
            )
        names_by_id[physical_id] = name
    return sorted(names_by_id.items())


def default_output_path(mesh_path: Path) -> Path:
    return mesh_path.with_name(f"{mesh_path.stem}_identified{mesh_path.suffix}")


def write_identified_mesh(
    source_path: Path,
    output_path: Path,
    boundary_faces: np.ndarray,
    assignments: dict[int, tuple[str, int]],
    original_element_count: int,
) -> None:
    physical_names = collect_physical_names(assignments)
    selected_cell_ids = sorted(assignments)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with source_path.open("r", encoding="utf-8", errors="replace") as src, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        inserted_physical_names = False
        while True:
            line = src.readline()
            if not line:
                break

            marker = line.strip()
            if marker == "$PhysicalNames":
                while src.readline().strip() != "$EndPhysicalNames":
                    pass
                continue

            dst.write(line)

            if marker == "$EndMeshFormat" and not inserted_physical_names:
                dst.write("$PhysicalNames\n")
                dst.write(f"{len(physical_names)}\n")
                for physical_id, name in physical_names:
                    safe_name = name.replace("\\", "\\\\").replace('"', '\\"')
                    dst.write(f'2 {physical_id} "{safe_name}"\n')
                dst.write("$EndPhysicalNames\n")
                inserted_physical_names = True
            elif marker == "$Elements":
                old_count = int(src.readline())
                if old_count != original_element_count:
                    raise ValueError(
                        f"Element count changed while writing: parsed {original_element_count}, now {old_count}"
                    )
                dst.write(f"{original_element_count + len(selected_cell_ids)}\n")
                for _ in range(original_element_count):
                    dst.write(src.readline())

                next_element_id = original_element_count + 1
                for cell_id in selected_cell_ids:
                    _, physical_id = assignments[cell_id]
                    nodes = " ".join(str(int(node_id)) for node_id in boundary_faces[cell_id])
                    dst.write(f"{next_element_id} 3 2 {physical_id} {physical_id} {nodes}\n")
                    next_element_id += 1


class BoundaryIdentifier:
    def __init__(self, mesh_path: Path, mesh: MeshData, feature_angle: float):
        self.mesh_path = mesh_path
        self.mesh = mesh
        self.feature_angle = feature_angle
        self.polydata = make_polydata(mesh)
        self.normals = compute_face_normals(mesh.points, mesh.faces)
        self.adjacency = build_face_adjacency(mesh.faces)
        self.highlight_offset = max(float(self.polydata.length) * 0.001, 1.0e-6)
        self.assignments: dict[int, tuple[str, int]] = {}
        self.history: list[BoundaryPatch] = []
        self.preview_ids: tuple[int, ...] = ()
        self.assigned_actor_names: list[str] = []
        self.base_actor = None
        self.plotter = pv.Plotter()

    def run(self) -> None:
        self.base_actor = self.plotter.add_mesh(
            self.polydata,
            color="#c8d0d8",
            show_edges=True,
            edge_color="#4b5563",
            line_width=0.2,
            opacity=0.42,
            pickable=True,
        )
        self.plotter.add_text(
            "Click: select | a: assign remaining wall | u: undo | s: save | q: quit",
            position="upper_left",
            font_size=10,
            color="black",
        )
        self.plotter.track_click_position(self.on_click, side="left")
        self.plotter.track_click_position(self.on_click, side="right")
        self.plotter.add_key_event("a", self.assign_all_unassigned)
        self.plotter.add_key_event("s", self.save)
        self.plotter.add_key_event("u", self.undo)
        self.plotter.add_key_event("q", self.close)
        self.plotter.show(title=f"Identify boundaries: {self.mesh_path.name}")

    def on_click(self, position) -> None:
        if self.base_actor is None:
            print("Mesh actor is not ready yet.")
            return

        seed_cell_id = self.pick_cell_id(position)
        if seed_cell_id < 0:
            seed_cell_id = self.closest_cell_id_from_click_position()

        if seed_cell_id < 0:
            print("No boundary face was picked. Rotate/zoom so the end face is visible, then click again.")
            return

        self.preview_ids = grow_patch(seed_cell_id, self.adjacency, self.normals, self.feature_angle)
        self.show_preview()
        print(f"\nSelected patch: {len(self.preview_ids)} boundary faces")
        self.prompt_assignment_menu(self.preview_ids)

    def pick_cell_id(self, position) -> int:
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.01)
        picker.PickFromListOn()
        picker.AddPickList(self.base_actor)
        renderer = self.plotter.iren.get_poked_renderer() or self.plotter.renderer
        picker.Pick(float(position[0]), float(position[1]), 0.0, renderer)
        return int(picker.GetCellId())

    def closest_cell_id_from_click_position(self) -> int:
        try:
            point = np.asarray(self.plotter.pick_click_position(), dtype=float)
        except Exception as exc:
            print(f"Picking fallback failed: {exc}")
            return -1

        if not np.all(np.isfinite(point)):
            return -1

        cell_id, closest_point = self.polydata.find_closest_cell(point, return_closest_point=True)
        distance = float(np.linalg.norm(np.asarray(closest_point) - point))
        max_distance = max(float(self.polydata.length) * 0.08, self.highlight_offset * 20.0)
        if distance > max_distance:
            print(
                "No exact face pick; nearest boundary face is too far "
                f"({distance:.4g} > {max_distance:.4g})."
            )
            return -1

        print(f"No exact face pick; using nearest boundary face (distance {distance:.4g}).")
        return int(cell_id)

    def show_preview(self) -> None:
        if not self.preview_ids:
            return
        preview = make_patch_polydata(
            self.mesh.points, self.mesh.faces, self.preview_ids, self.normals, self.highlight_offset
        )
        self.plotter.add_mesh(
            preview,
            color="#f59e0b",
            show_edges=True,
            edge_color="#111827",
            opacity=0.95,
            name="__preview__",
            pickable=False,
        )
        self.plotter.render()

    def prompt_assignment_menu(self, cell_ids: tuple[int, ...]) -> None:
        choice = input("Assign selected patch: [i] inlet, [o] outlet, [w] wall, [n] custom, [c] cancel: ").strip().lower()
        role_by_choice = {"i": "inlet", "o": "outlet", "w": "wall", "n": None}
        if choice in ("", "c"):
            print("Assignment canceled. The yellow selection remains active.", flush=True)
            return
        if choice not in role_by_choice:
            print("Unknown choice. Assignment canceled. Click the patch again or press c by choosing cancel next time.", flush=True)
            return
        self.prompt_assignment(cell_ids, role_by_choice[choice])

    def prompt_assignment(self, cell_ids: tuple[int, ...], role: str | None = None) -> None:
        default_id = next_default_id(self.assignments)
        if role is None:
            name_prompt = "Boundary name (e.g. inlet1, outlet2, wall; blank cancels): "
            name = input(name_prompt).strip()
        else:
            default_name = self.next_default_name(role)
            name = input(f"Boundary name [{default_name}]: ").strip() or default_name

        if not name:
            print("Assignment canceled. The yellow selection remains active.", flush=True)
            return

        physical_id_text = input(f"Physical ID [{default_id}]: ").strip()
        physical_id = int(physical_id_text) if physical_id_text else default_id
        self.apply_assignment(name, physical_id, cell_ids)

    def next_default_name(self, role: str) -> str:
        existing = {name for name, _ in self.assignments.values()}
        if role == "wall" and "wall" not in existing:
            return "wall"

        index = 1
        while f"{role}{index}" in existing:
            index += 1
        return f"{role}{index}"

    def unassigned_cell_ids(self) -> tuple[int, ...]:
        return tuple(cell_id for cell_id in range(len(self.mesh.faces)) if cell_id not in self.assignments)

    def assign_all_unassigned(self) -> None:
        cell_ids = self.unassigned_cell_ids()
        print(f"\nUnassigned boundary faces: {len(cell_ids)}")
        if not cell_ids:
            return
        self.prompt_assignment(cell_ids, "wall")

    def apply_assignment(self, name: str, physical_id: int, cell_ids: tuple[int, ...]) -> None:
        patch = BoundaryPatch(name=name, physical_id=physical_id, cell_ids=cell_ids)
        self.history.append(patch)
        for cell_id in cell_ids:
            self.assignments[cell_id] = (name, physical_id)
        assigned = make_patch_polydata(
            self.mesh.points, self.mesh.faces, cell_ids, self.normals, self.highlight_offset
        )
        actor_name = f"assigned_{len(self.history)}"
        self.plotter.add_mesh(
            assigned,
            color=self.color_for_name(name),
            show_edges=True,
            edge_color="#111827",
            opacity=0.9,
            name=actor_name,
            pickable=False,
        )
        self.assigned_actor_names.append(actor_name)
        self.clear_preview()
        print(f"Assigned {len(cell_ids)} faces as {name!r} with Physical ID {physical_id}")

    def undo(self) -> None:
        if not self.history:
            print("Nothing to undo.")
            return
        patch = self.history.pop()
        for cell_id in patch.cell_ids:
            if self.assignments.get(cell_id) == (patch.name, patch.physical_id):
                del self.assignments[cell_id]
        self.refresh_assigned_layers()
        print(f"Undid {patch.name!r}")

    def refresh_assigned_layers(self) -> None:
        for actor_name in self.assigned_actor_names:
            self.plotter.remove_actor(actor_name, render=False)
        self.assigned_actor_names.clear()

        for index, patch in enumerate(self.history, start=1):
            assigned = make_patch_polydata(
                self.mesh.points, self.mesh.faces, patch.cell_ids, self.normals, self.highlight_offset
            )
            actor_name = f"assigned_{index}"
            self.plotter.add_mesh(
                assigned,
                color=self.color_for_name(patch.name),
                show_edges=True,
                edge_color="#111827",
                opacity=0.9,
                name=actor_name,
                pickable=False,
            )
            self.assigned_actor_names.append(actor_name)
        self.plotter.render()

    def clear_preview(self) -> None:
        self.preview_ids = ()
        self.plotter.remove_actor("__preview__", render=False)
        self.plotter.render()

    def save(self) -> None:
        if not self.assignments:
            print("No assigned boundaries to save.")
            return

        unassigned = self.unassigned_cell_ids()
        if unassigned:
            choice = input(
                f"{len(unassigned)} boundary faces are still unassigned. "
                "Assign them as wall before saving? [Y/n]: "
            ).strip().lower()
            if choice in ("", "y", "yes"):
                self.prompt_assignment(unassigned, "wall")

        output_path = default_output_path(self.mesh_path)
        write_identified_mesh(
            self.mesh_path,
            output_path,
            self.mesh.faces,
            self.assignments,
            self.mesh.original_element_count,
        )
        print(f"Saved: {output_path.relative_to(ROOT)}")
        self.close()

    def close(self) -> None:
        try:
            if self.plotter.iren is not None:
                self.plotter.iren.terminate_app()
        finally:
            self.plotter.close()

    @staticmethod
    def color_for_name(name: str) -> str:
        lower = name.lower()
        if lower.startswith("inlet"):
            return "#2563eb"
        if lower.startswith("outlet"):
            return "#dc2626"
        if lower.startswith("wall"):
            return "#16a34a"
        return "#7c3aed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively identify inlet/outlet/wall boundary patches in a Gmsh .msh file."
    )
    parser.add_argument("mesh", nargs="?", help="Path to a .msh file. Defaults to data/mesh/*.msh.")
    parser.add_argument(
        "--feature-angle",
        type=float,
        default=20.0,
        help="Maximum normal angle in degrees for click-to-patch growing. Default: 20.",
    )
    args = parser.parse_args()

    mesh_path = resolve_mesh_path(args.mesh)
    if mesh_path.suffix.lower() != ".msh":
        raise ValueError(f"Expected a .msh file: {mesh_path}")

    if not os.environ.get("DISPLAY") and os.name != "nt":
        print(
            "Warning: DISPLAY is not set. PyVista needs an X/WSLg display for interactive selection."
        )
        print("Run this from a desktop-enabled shell or pass the host display into the container.")

    print(f"Loading mesh: {mesh_path.relative_to(ROOT)}")
    mesh = parse_nodes_and_boundary_faces(mesh_path)
    print(f"Boundary faces extracted: {len(mesh.faces)}")
    print("Tip: click inlet/outlet caps, then press 'a' to assign all remaining faces as wall.")

    BoundaryIdentifier(mesh_path, mesh, args.feature_angle).run()


if __name__ == "__main__":
    main()
