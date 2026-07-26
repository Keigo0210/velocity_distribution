from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from section_config import (  # noqa: E402
    SectionConfigError,
    resolve_sections_from_config,
)


def load_compare_module():
    path = ROOT / "scripts" / "compare_fem_cases.py"
    spec = importlib.util.spec_from_file_location("compare_fem_cases_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SectionConfigIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = ROOT / "config" / "compare_fem_cases.json"
        cls.base_config = json.loads(cls.config_path.read_text(encoding="utf-8"))
        cls.compare = load_compare_module()

    def test_library_set_resolves_three_sections(self) -> None:
        sections = resolve_sections_from_config(
            {"section_library": "sections.json", "section_set": "default_three_sections"},
            config_path=self.config_path,
        )
        self.assertEqual(
            [section["name"] for section in sections],
            ["straight_z10", "upstream_z30", "side_branch"],
        )

    def test_library_names_resolves_one_section(self) -> None:
        sections = resolve_sections_from_config(
            {"section_library": "sections.json", "section_names": ["side_branch"]},
            config_path=self.config_path,
        )
        self.assertEqual([section["name"] for section in sections], ["side_branch"])

    def test_existing_sections_array(self) -> None:
        sections = resolve_sections_from_config(
            {
                "sections": [
                    {
                        "name": "inline",
                        "center": [0, 0, 10],
                        "normal": [0, 0, 1],
                        "width": 12,
                        "height": 10,
                        "resolution": [11, 9],
                    }
                ]
            }
        )
        self.assertEqual(sections[0]["name"], "inline")
        self.assertEqual(sections[0]["grid_resolution"], (11, 9))

    def test_single_section_form(self) -> None:
        sections = resolve_sections_from_config(
            {
                "section": {
                    "name": "single",
                    "center": [0, 0, 30],
                    "normal": [0, 0, 1],
                    "width": 12,
                    "height": 10,
                    "grid_resolution": [5, 5],
                }
            }
        )
        self.assertEqual([section["name"] for section in sections], ["single"])

    def _validate_comparison_selector(self, key: str, value, expected: list[str]) -> None:
        config = deepcopy(self.base_config)
        comparison = config["comparisons"][0]
        for selector in ("section", "sections", "section_set"):
            comparison.pop(selector, None)
        comparison[key] = value
        config["comparisons"] = [comparison]
        validated = self.compare.validate_config(config, self.config_path)
        self.assertEqual(validated["comparisons"][0]["sections"], expected)

    def test_comparison_section(self) -> None:
        self._validate_comparison_selector("section", "side_branch", ["side_branch"])

    def test_comparison_sections(self) -> None:
        self._validate_comparison_selector(
            "sections", ["straight_z10", "side_branch"],
            ["straight_z10", "side_branch"],
        )

    def test_comparison_section_set(self) -> None:
        self._validate_comparison_selector(
            "section_set", "default_three_sections",
            ["straight_z10", "upstream_z30", "side_branch"],
        )

    def test_unknown_section_set_is_rejected(self) -> None:
        with self.assertRaisesRegex(SectionConfigError, "unknown section set"):
            resolve_sections_from_config(
                {"section_library": "sections.json", "section_set": "missing"},
                config_path=self.config_path,
            )

    def test_repository_relative_library_path_fallback(self) -> None:
        sections = resolve_sections_from_config(
            {"section_library": "config/sections.json", "section_names": ["side_branch"]},
            config_path=self.config_path,
        )
        self.assertEqual(sections[0]["name"], "side_branch")

    def test_side_branch_basis_matches_existing_orientation(self) -> None:
        section = resolve_sections_from_config(
            {"section_library": "sections.json", "section_names": ["side_branch"]},
            config_path=self.config_path,
        )[0]
        np.testing.assert_allclose(
            section["normalized_normal"], [0.70710678, 0, -0.70710678], atol=1e-8
        )
        np.testing.assert_allclose(section["s_axis"], [0, -1, 0], atol=1e-8)
        np.testing.assert_allclose(
            section["t_axis"], [-0.70710678, 0, -0.70710678], atol=1e-8
        )


if __name__ == "__main__":
    unittest.main()
