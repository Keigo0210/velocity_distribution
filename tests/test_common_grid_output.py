from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common_grid_output import resolve_grid_output_format, save_grid_output, valid_grid_metrics


class CommonGridOutputTests(unittest.TestCase):
    def test_cell_area_requires_all_four_vertices(self):
        valid = np.ones((3, 3), dtype=bool)
        valid[0, 0] = False
        result = valid_grid_metrics(valid.ravel(), 2.0, 2.0, (3, 3))
        self.assertEqual(result["valid_point_count"], 8)
        self.assertEqual(result["valid_cell_count"], 3)
        self.assertAlmostEqual(result["common_valid_area"], 3.0)
        self.assertAlmostEqual(result["legacy_valid_area_estimate"], 32.0 / 9.0)

    def test_legacy_save_grid_csv(self):
        self.assertEqual(resolve_grid_output_format({"save_grid_csv": True}), "csv")
        self.assertEqual(resolve_grid_output_format({"save_grid_csv": False}), "none")

    def test_explicit_format_has_priority(self):
        self.assertEqual(
            resolve_grid_output_format({"grid_output_format": "npz", "save_grid_csv": True}),
            "npz",
        )

    def test_npz_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_grid_output(
                Path(directory) / "grid", "npz", None,
                {"s_grid": np.arange(4).reshape(2, 2), "common_valid_mask": [True, False]},
                {"section_name": "test"},
            )
            with np.load(path) as data:
                np.testing.assert_array_equal(data["s_grid"], np.arange(4).reshape(2, 2))
                self.assertIn("section_name", str(data["metadata_json"]))

    def test_csv_gz_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = pd.DataFrame({"a": [1, 2]})
            path = save_grid_output(Path(directory) / "grid", "csv.gz", frame, {}, {})
            self.assertEqual(pd.read_csv(path)["a"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
