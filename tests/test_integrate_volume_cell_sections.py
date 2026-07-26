from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from integrate_volume_cell_sections import (  # noqa: E402
    NativeSectionError,
    build_comparison_rows,
    resolve_config_path,
    scale_native_volume_mesh,
)
from surface_flow import (  # noqa: E402
    SurfaceFlowError,
    clip_polygon_to_rectangle_2d,
    flow_unit_values,
    integrate_native_volume_cell_section,
    polygon_area_centroid_2d,
)


def hexa_mesh(x_ranges=((0.0, 1.0),), velocities=((0.0, 0.0, 2.0),)):
    points = []
    cells = []
    for x_range in x_ranges:
        x0, x1 = x_range
        offset = len(points)
        points.extend([
            [x0, 0, 0], [x1, 0, 0], [x1, 1, 0], [x0, 1, 0],
            [x0, 0, 1], [x1, 0, 1], [x1, 1, 1], [x0, 1, 1],
        ])
        cells.extend([8, *range(offset, offset + 8)])
    mesh = pv.UnstructuredGrid(
        np.asarray(cells, dtype=np.int64),
        np.full(len(x_ranges), pv.CellType.HEXAHEDRON, dtype=np.uint8),
        np.asarray(points, dtype=float),
    )
    mesh.cell_data["Velocity"] = np.asarray(velocities, dtype=float)
    return mesh


def integrate(mesh, normal=(0, 0, 1), width=10.0, height=10.0):
    center_x = 0.5 * (mesh.bounds.x_min + mesh.bounds.x_max)
    return integrate_native_volume_cell_section(
        mesh, "Velocity", center=[center_x, 0.5, 0.5], normal=normal,
        s_axis=[0, 1, 0], t_axis=[-1, 0, 0], width=width, height=height,
    )


class PolygonGeometryTests(unittest.TestCase):
    def test_triangle_quad_and_polygon_area(self):
        shapes = [
            (np.array([[0, 0], [2, 0], [0, 1]], float), 1.0),
            (np.array([[0, 0], [2, 0], [2, 1], [0, 1]], float), 2.0),
            (np.array([[0, 0], [2, 0], [2, 1], [1, 2], [0, 1]], float), 3.0),
        ]
        for polygon, expected in shapes:
            with self.subTest(vertices=len(polygon)):
                area, centroid = polygon_area_centroid_2d(polygon)
                self.assertAlmostEqual(area, expected)
                self.assertEqual(centroid.shape, (2,))

    def test_rectangle_clipping(self):
        polygon = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]], float)
        clipped = clip_polygon_to_rectangle_2d(polygon, 1.0, 0.5)
        area, _ = polygon_area_centroid_2d(clipped)
        self.assertAlmostEqual(area, 0.5)


class NativeCellIntegrationTests(unittest.TestCase):
    def test_uniform_velocity_matches_area_dot_product(self):
        metrics, records, _, diagnostic = integrate(hexa_mesh())
        self.assertAlmostEqual(metrics["native_intersection_area_mm2"], 1.0)
        self.assertAlmostEqual(metrics["signed_flow_mm3_s"], 2.0)
        self.assertEqual(metrics["intersected_volume_cell_count"], 1)
        self.assertEqual(records[0]["original_volume_cell_id"], 0)
        self.assertEqual(diagnostic["original_cell_id_cut_dtype"], "int64")

    def test_multiple_cells_match_manual_area_weighting_and_ids(self):
        mesh = hexa_mesh(((0, 1), (1, 2)), ((0, 0, 1), (0, 0, 3)))
        metrics, records, polydata, _ = integrate(mesh)
        self.assertAlmostEqual(metrics["native_intersection_area_mm2"], 2.0)
        self.assertAlmostEqual(metrics["signed_flow_mm3_s"], 4.0)
        self.assertEqual({r["original_volume_cell_id"] for r in records if r["valid"]}, {0, 1})
        self.assertEqual(metrics["duplicate_original_cell_count"], 0)
        np.testing.assert_array_equal(
            np.sort(polydata.cell_data["original_volume_cell_id"]), [0, 1]
        )

    def test_normal_flip_changes_only_signed_flow(self):
        forward = integrate(hexa_mesh(), normal=(0, 0, 1))[0]
        reverse = integrate(hexa_mesh(), normal=(0, 0, -1))[0]
        self.assertAlmostEqual(
            forward["native_intersection_area_mm2"], reverse["native_intersection_area_mm2"]
        )
        self.assertAlmostEqual(forward["absolute_flow_mm3_s"], reverse["absolute_flow_mm3_s"])
        self.assertAlmostEqual(forward["signed_flow_mm3_s"], -reverse["signed_flow_mm3_s"])

    def test_section_window_clipping(self):
        metrics = integrate(hexa_mesh(), width=0.5, height=0.5)[0]
        self.assertAlmostEqual(metrics["native_intersection_area_mm2"], 0.25)
        self.assertAlmostEqual(metrics["signed_flow_mm3_s"], 0.5)

    def test_empty_intersection_is_clear_error(self):
        mesh = hexa_mesh()
        with self.assertRaisesRegex(SurfaceFlowError, "does not intersect"):
            integrate_native_volume_cell_section(
                mesh, "Velocity", center=[0, 0, 2], normal=[0, 0, 1],
                s_axis=[0, 1, 0], t_axis=[-1, 0, 0], width=2, height=2,
            )

    def test_missing_cell_velocity_is_error(self):
        mesh = hexa_mesh()
        del mesh.cell_data["Velocity"]
        with self.assertRaisesRegex(SurfaceFlowError, "missing"):
            integrate(mesh)

    def test_vtp_can_be_reloaded_with_original_ids(self):
        polydata = integrate(hexa_mesh())[2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cut.vtp"
            polydata.save(path)
            loaded = pv.read(path)
        self.assertEqual(loaded.n_cells, 1)
        self.assertEqual(int(loaded.cell_data["original_volume_cell_id"][0]), 0)
        self.assertGreater(float(loaded.cell_data["polygon_area_mm2"][0]), 0.0)


class UnitsAndComparisonTests(unittest.TestCase):
    def test_length_velocity_and_flow_unit_conversions(self):
        mesh = hexa_mesh()
        mesh.points /= 1000.0
        mesh.cell_data["Velocity"] /= 1000.0
        scaled = scale_native_volume_mesh(mesh, "Velocity", 1000.0, 1000.0)
        np.testing.assert_allclose(scaled.points, hexa_mesh().points)
        np.testing.assert_allclose(scaled.cell_data["Velocity"], [[0, 0, 2]])
        units = flow_unit_values(1000.0)
        self.assertAlmostEqual(units["signed_flow_rate_ml_s"], 1.0)
        self.assertAlmostEqual(units["signed_flow_rate_ml_min"], 60.0)
        self.assertAlmostEqual(units["signed_flow_rate_ml_h"], 3600.0)

    def test_config_relative_path(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config" / "example.json"
            config.parent.mkdir()
            self.assertEqual(
                resolve_config_path("../data/input.case", config),
                (Path(directory) / "data" / "input.case").resolve(),
            )

    def test_comparison_reports_signed_and_magnitude_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            internal = root / "internal.csv"
            boundary = root / "boundary.csv"
            with internal.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "source_name", "section_name", "signed_flow_rate_mm3_s"
                ])
                writer.writeheader()
                writer.writerow({"source_name": "point", "section_name": "s", "signed_flow_rate_mm3_s": -8})
            with boundary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "source_name", "boundary_name", "signed_flow_mm3_s", "area_mm2"
                ])
                writer.writeheader()
                writer.writerow({"source_name": "boundary", "boundary_name": "b", "signed_flow_mm3_s": 10, "area_mm2": 2})
            config_path = root / "config.json"
            config_path.write_text("{}")
            rows = build_comparison_rows(
                [{"section_name": "s", "signed_flow_mm3_s": -9,
                  "native_intersection_area_mm2": 2.2}],
                {"existing_internal_flow_summary": "internal.csv",
                 "existing_internal_source_name": "point",
                 "boundary_flow_summary": "boundary.csv",
                 "boundary_source_name": "boundary",
                 "section_to_boundary": {"s": "b"}},
                config_path,
            )
        row = rows[0]
        self.assertEqual(row["native_vs_boundary_signed_difference"], -19.0)
        self.assertEqual(row["native_vs_boundary_magnitude_difference"], -1.0)
        self.assertAlmostEqual(row["native_vs_boundary_relative_difference_percent"], 10.0)
        self.assertEqual(row["native_boundary_sign_relation"], "opposite")

    def test_invalid_scale_is_error(self):
        with self.assertRaises(NativeSectionError):
            scale_native_volume_mesh(hexa_mesh(), "Velocity", 0.0, 1000.0)


if __name__ == "__main__":
    unittest.main()
