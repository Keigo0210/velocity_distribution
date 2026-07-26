from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_module():
    path = ROOT / "scripts" / "check_velocity_steady_state.py"
    spec = importlib.util.spec_from_file_location("steady_state_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SteadyStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_fraction_thresholds_are_converted_to_percent(self):
        self.assertEqual(self.module.normalized_threshold_percent(0.005), 0.5)
        self.assertEqual(self.module.normalized_threshold_percent(0.001), 0.1)
        self.assertEqual(self.module.normalized_threshold_percent(0.5), 0.5)

    def test_step_number_times_support_sparse_output(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for step in (20, 40, 80):
                path = Path(directory) / f"solution_{step:06d}.vtu"
                path.touch()
                paths.append(path)
            entries = self.module.build_timeline(
                paths, self.module.RunLog({}, {}, None), 0.0125
            )
        self.assertEqual([entry.time for entry in entries], [0.25, 0.5, 1.0])

    def test_invalid_data_directory_is_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "invalid data directory"):
                self.module.analyze_config_source(
                    "bad", {"directory": "missing", "dt": 0.1}, [],
                    Path(directory) / "config.json", Path(directory) / "out",
                    1.0, [0.5], True, False, False, False, 100,
                )

    def test_side_branch_comes_from_common_library(self):
        from section_config import resolve_sections_from_config
        config_path = ROOT / "config" / "check_velocity_steady_state.json"
        sections = resolve_sections_from_config(
            {"section_library": "sections.json", "section_names": ["side_branch"]},
            config_path=config_path,
        )
        self.assertEqual(sections[0]["name"], "side_branch")


if __name__ == "__main__":
    unittest.main()
