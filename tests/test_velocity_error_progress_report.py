from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_velocity_error_progress_report.py"
SPEC = importlib.util.spec_from_file_location("progress_report", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class ProgressReportTests(unittest.TestCase):
    def test_config_relative_path_is_exact(self) -> None:
        config_path = ROOT / "config" / "nested" / "report.json"
        self.assertEqual(
            module.resolve_config_relative("../output/value.csv", config_path),
            (ROOT / "config" / "output" / "value.csv").resolve(),
        )

    def test_real_config_resolves_three_common_sections(self) -> None:
        context = module.load_report_context(
            ROOT / "config" / "velocity_error_progress_report.json"
        )
        self.assertEqual(
            [section["name"] for section in context["sections"]],
            ["straight_z10", "upstream_z30", "side_branch"],
        )
        self.assertEqual(context["output_dir"], ROOT / "output" / "progress_report" / "re100_configured")

    def test_summary_values_are_read_from_csv(self) -> None:
        context = module.load_report_context(
            ROOT / "config" / "velocity_error_progress_report.json"
        )
        case_name, case = next(iter(context["cases"].items()))
        record = module.build_section_record(
            case_name, case, context["sections"][0], context["report"]
        )
        source = case["frames"]["fem_star_metrics"]
        expected = float(
            source[source["section_name"] == context["sections"][0]["name"]][
                "relative_l2_error"
            ].iloc[0]
        )
        self.assertAlmostEqual(record["fem_star_relative_l2_percent"], expected)
        self.assertIsNotNone(record["fem_absolute_flow_mm3_s"])
        self.assertIsNotNone(record["steady_threshold_1_continuous_below_s"])

    def test_missing_required_input_is_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sections = json.loads((ROOT / "config" / "sections.json").read_text())
            (base / "sections.json").write_text(json.dumps(sections))
            config = {
                "section_library": "sections.json",
                "section_set": "default_three_sections",
                "cases": {"sample": {"label": "Sample", "fem_star_metrics": "missing.csv"}},
                "report": {"output_directory": "out", "steady_thresholds_percent": []},
                "execution": {"fail_on_missing_required_input": True},
            }
            config_path = base / "report.json"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(module.ReportConfigError, "required input is missing"):
                module.load_report_context(config_path)

    def test_optional_missing_input_is_not_zero_filled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sections = json.loads((ROOT / "config" / "sections.json").read_text())
            (base / "sections.json").write_text(json.dumps(sections))
            config = {
                "section_library": "sections.json",
                "section_set": "default_three_sections",
                "cases": {
                    "sample": {
                        "label": "Sample",
                        "fem_star_metrics": {"path": "missing.csv", "required": False},
                    }
                },
                "report": {"output_directory": "out", "steady_thresholds_percent": []},
                "execution": {"fail_on_missing_required_input": True},
            }
            config_path = base / "report.json"
            config_path.write_text(json.dumps(config))
            context = module.load_report_context(config_path)
            case = context["cases"]["sample"]
            record = module.build_section_record(
                "sample", case, context["sections"][0], context["report"]
            )
            self.assertIsNone(record["fem_star_relative_l2_percent"])

    def test_report_script_has_no_case_or_section_result_constants(self) -> None:
        text = MODULE_PATH.read_text(encoding="utf-8")
        forbidden = [
            "SECTION_" + "FILES",
            "SECTION_" + "NORMALS",
            "Re" + "=10",
            "Re" + "=100",
            "branch_" + "oblique",
        ]
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
