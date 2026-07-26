from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from section_config import (  # noqa: E402
    SectionConfigError,
    load_section_library,
    make_plane_basis,
    resolve_section,
    resolve_section_set,
    resolve_sections_from_config,
    validate_section,
)


LIBRARY_PATH = ROOT / "config" / "sections.json"


class SectionConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.library = load_section_library(LIBRARY_PATH)

    def test_loads_three_sections(self) -> None:
        self.assertEqual(
            set(self.library["sections"]),
            {"straight_z10", "upstream_z30", "side_branch"},
        )

    def test_resolves_default_three_sections_in_order(self) -> None:
        sections = resolve_section_set("default_three_sections", self.library)
        self.assertEqual(
            [section["name"] for section in sections],
            ["straight_z10", "upstream_z30", "side_branch"],
        )

    def test_side_branch_uses_expected_automatic_basis(self) -> None:
        section = resolve_section("side_branch", self.library)
        np.testing.assert_allclose(
            section["normal"], [0.70710678, 0.0, -0.70710678], atol=1.0e-8
        )
        np.testing.assert_allclose(section["s_axis"], [0.0, -1.0, 0.0], atol=1.0e-8)
        np.testing.assert_allclose(
            section["t_axis"], [-0.70710678, 0.0, -0.70710678], atol=1.0e-8
        )

    def test_zero_normal_is_rejected(self) -> None:
        with self.assertRaisesRegex(SectionConfigError, "non-zero"):
            validate_section(
                {
                    "center": [0, 0, 0],
                    "normal": [0, 0, 0],
                    "width": 12,
                    "height": 10,
                    "grid_resolution": [5, 5],
                }
            )

    def test_unknown_section_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(SectionConfigError, "unknown section"):
            resolve_section("does_not_exist", self.library)

    def test_explicit_axes_are_used(self) -> None:
        normal, s_axis, t_axis = make_plane_basis(
            [0, 0, 1], s_axis=[1, 0, 0], t_axis=[0, 1, 0]
        )
        np.testing.assert_allclose(normal, [0, 0, 1])
        np.testing.assert_allclose(s_axis, [1, 0, 0])
        np.testing.assert_allclose(t_axis, [0, 1, 0])

    def test_one_explicit_axis_completes_basis(self) -> None:
        _, s_axis, t_axis = make_plane_basis([0, 0, 1], s_axis=[1, 0, 0])
        np.testing.assert_allclose(s_axis, [1, 0, 0])
        np.testing.assert_allclose(t_axis, [0, 1, 0])

    def test_config_resolution_prioritizes_library_set(self) -> None:
        config = {
            "section_library": "sections.json",
            "section_set": "default_three_sections",
            "section_names": ["side_branch"],
            "sections": [
                {
                    "center": [0, 0, 0],
                    "normal": [0, 0, 1],
                    "width": 1,
                    "height": 1,
                    "grid_resolution": [2, 2],
                }
            ],
        }
        sections = resolve_sections_from_config(
            config, config_path=ROOT / "config" / "consumer.json"
        )
        self.assertEqual(len(sections), 3)

    def test_config_resolution_supports_library_names(self) -> None:
        config = {
            "section_library": "sections.json",
            "section_names": ["upstream_z30", "side_branch"],
        }
        sections = resolve_sections_from_config(
            config, config_path=ROOT / "config" / "consumer.json"
        )
        self.assertEqual(
            [section["name"] for section in sections],
            ["upstream_z30", "side_branch"],
        )

    def test_legacy_sections_array_is_supported(self) -> None:
        sections = resolve_sections_from_config(
            {
                "flip_s_axis": True,
                "sections": [
                    {
                        "name": "legacy",
                        "center": [0, 0, 10],
                        "normal": [0, 0, 1],
                        "width": 12,
                        "height": 10,
                        "resolution": [11, 9],
                    }
                ],
            }
        )
        self.assertEqual(sections[0]["grid_resolution"], (11, 9))
        np.testing.assert_allclose(sections[0]["s_axis"], [0, -1, 0])


if __name__ == "__main__":
    unittest.main()
