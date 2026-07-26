from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from section_flow import flow_balance, integrate_triangles


class SectionFlowTests(unittest.TestCase):
    def test_constant_normal_velocity_on_unit_square(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        velocity = np.tile([0.0, 0.0, 2.0], (4, 1))
        result = integrate_triangles(points, triangles, velocity, np.array([0, 0, 1]))
        self.assertAlmostEqual(result["section_area_mm2"], 1.0)
        self.assertAlmostEqual(result["signed_flow_rate_mm3_s"], 2.0)
        self.assertAlmostEqual(result["area_mean_normal_velocity_mm_s"], 2.0)
        self.assertEqual(result["valid_triangle_count"], 2)

    def test_linear_velocity_uses_vertex_mean(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        triangles = np.array([[0, 1, 2]])
        velocity = np.array([[0, 0, 0], [0, 0, 3], [0, 0, 6]], float)
        result = integrate_triangles(points, triangles, velocity, np.array([0, 0, 1]))
        self.assertAlmostEqual(result["section_area_mm2"], 0.5)
        self.assertAlmostEqual(result["signed_flow_rate_mm3_s"], 1.5)

    def test_absolute_flow_balance_and_fractions(self):
        rows = [
            {"section_name": "in", "signed_flow_rate_mm3_s": -10.0},
            {"section_name": "a", "signed_flow_rate_mm3_s": -6.0},
            {"section_name": "b", "signed_flow_rate_mm3_s": 3.0},
        ]
        result = flow_balance(
            rows, {"inlets": ["in"], "outlets": ["a", "b"], "partition_mode": "absolute"}
        )
        self.assertAlmostEqual(result["inlet_flow_total_mm3_s"], 10.0)
        self.assertAlmostEqual(result["outlet_flow_total_mm3_s"], 9.0)
        self.assertAlmostEqual(result["a_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(result["flow_balance_error_percent"], 10.0)

    def test_orientation_adjusted_mode(self):
        rows = [
            {"section_name": "in", "signed_flow_rate_mm3_s": -10.0},
            {"section_name": "out", "signed_flow_rate_mm3_s": 9.0},
        ]
        result = flow_balance(
            rows,
            {"inlets": ["in"], "outlets": ["out"], "partition_mode": "orientation_adjusted", "orientation_factors": {"in": -1, "out": 1}},
        )
        self.assertAlmostEqual(result["flow_balance_error_percent"], 10.0)


if __name__ == "__main__":
    unittest.main()
