from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
import sys
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_boundary_flow import (
    BoundaryAuditError,
    _find_named_block,
    compare_boundary_internal,
    resolve_config_path,
)
from surface_flow import (
    SurfaceFlowError,
    flow_unit_values,
    integrate_cell_surface,
    integrate_point_surface,
    integrate_triangles,
    normalize_normal,
)


class BoundaryFlowAuditTests(unittest.TestCase):
    @staticmethod
    def square_surface() -> pv.PolyData:
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
             [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        return pv.PolyData(points, faces=np.array([4, 0, 1, 2, 3]))

    def test_uniform_velocity_square_matches_theory(self) -> None:
        surface = self.square_surface()
        surface.point_data["u"] = np.tile([0.0, 0.0, 3.0], (4, 1))
        result = integrate_point_surface(surface, "u", [0.0, 0.0, 1.0])
        self.assertAlmostEqual(result["section_area_mm2"], 1.0)
        self.assertAlmostEqual(result["signed_flow_rate_mm3_s"], 3.0)

    def test_reversing_normal_only_reverses_sign(self) -> None:
        surface = self.square_surface()
        surface.point_data["u"] = np.tile([0.0, 0.0, 2.0], (4, 1))
        forward = integrate_point_surface(surface, "u", [0.0, 0.0, 1.0])
        reverse = integrate_point_surface(surface, "u", [0.0, 0.0, -1.0])
        self.assertAlmostEqual(forward["signed_flow_rate_mm3_s"], -reverse["signed_flow_rate_mm3_s"])
        self.assertAlmostEqual(forward["absolute_flow_rate_mm3_s"], reverse["absolute_flow_rate_mm3_s"])

    def test_triangle_point_data_uses_vertex_mean(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        triangles = np.array([[0, 1, 2]])
        velocity = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
        result = integrate_triangles(points, triangles, velocity, [0.0, 0.0, 1.0])
        self.assertAlmostEqual(result["section_area_mm2"], 1.0)
        self.assertAlmostEqual(result["signed_flow_rate_mm3_s"], 2.0)

    def test_native_cell_data_integration(self) -> None:
        surface = self.square_surface()
        surface.cell_data["u"] = np.array([[0.0, 0.0, 4.0]])
        result = integrate_cell_surface(surface, "u", [0.0, 0.0, 1.0])
        self.assertAlmostEqual(result["section_area_mm2"], 1.0)
        self.assertAlmostEqual(result["signed_flow_rate_mm3_s"], 4.0)

    def test_quad_polygon_area_is_supported(self) -> None:
        surface = self.square_surface()
        surface.cell_data["u"] = np.array([[2.0, 0.0, 0.0]])
        result = integrate_cell_surface(surface, "u", [1.0, 0.0, 0.0])
        self.assertEqual(result["cell_count"], 1)
        self.assertAlmostEqual(result["signed_flow_rate_mm3_s"], 2.0)

    def test_flow_unit_conversions(self) -> None:
        values = flow_unit_values(1000.0)
        self.assertAlmostEqual(values["signed_flow_rate_ml_s"], 1.0)
        self.assertAlmostEqual(values["signed_flow_rate_ml_min"], 60.0)
        self.assertAlmostEqual(values["signed_flow_rate_ml_h"], 3600.0)

    def test_missing_boundary_part_has_clear_error(self) -> None:
        blocks = pv.MultiBlock({"present": self.square_surface()})
        with self.assertRaisesRegex(BoundaryAuditError, "was not found"):
            _find_named_block(blocks, "missing")

    def test_invalid_normal_is_rejected(self) -> None:
        with self.assertRaisesRegex(SurfaceFlowError, "non-zero"):
            normalize_normal([0.0, 0.0, 0.0])

    def test_relative_path_resolves_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "nested" / "audit.json"
            expected = config.parent / "input" / "surface.vtp"
            self.assertEqual(resolve_config_path("input/surface.vtp", config), expected.resolve())
        self.assertEqual(
            resolve_config_path("data/value.vtu", ROOT / "config" / "audit.json"),
            (ROOT / "data" / "value.vtu").resolve(),
        )

    def test_boundary_internal_signed_and_magnitude_comparison(self) -> None:
        result = compare_boundary_internal(10.0, -9.0)
        self.assertAlmostEqual(result["signed_difference_mm3_s"], 19.0)
        self.assertAlmostEqual(result["magnitude_difference_mm3_s"], 1.0)
        self.assertAlmostEqual(result["absolute_difference_mm3_s"], 1.0)
        self.assertAlmostEqual(result["relative_difference_percent"], 10.0)


if __name__ == "__main__":
    unittest.main()
