import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_reynolds_profile_comparison import (
    ReynoldsAuditError,
    branch_partition,
    calculate_reynolds,
    classify_reynolds_case,
    json_ready,
    load_json,
    relative_percent,
    safe_amplification,
    validate_case,
    write_json,
)
from section_config import load_section_library
from section_series import generate_section_series


THRESHOLDS = {
    "upstream_flow_difference_percent": 1.0,
    "branch_fraction_difference_percentage_points": 1.0,
    "normalized_profile_l2_percent": 5.0,
    "beta_relative_difference_percent": 3.0,
    "alpha_relative_difference_percent": 5.0,
}


def evidence(profile=1.0, split=0.2, flow=0.2, beta=1.0, alpha=1.0):
    return {
        "profile_l2": profile, "branch_fraction_pp": split,
        "upstream_flow": flow, "beta": beta, "alpha": alpha,
    }


class ReynoldsProfileComparisonTests(unittest.TestCase):
    def test_01_config_loads_two_cases(self):
        config = load_json(ROOT / "config/audit_reynolds_profile_comparison.json")
        self.assertEqual(set(config["cases"]), {"re10", "re100"})

    def test_02_cases_have_distinct_reynolds(self):
        config = load_json(ROOT / "config/audit_reynolds_profile_comparison.json")
        self.assertEqual(
            [config["cases"][name]["nominal_reynolds_number"] for name in ("re10", "re100")],
            [10.0, 100.0],
        )

    def test_03_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved.json"
            write_json(path, {"a": 1, "b": [2, 3]})
            self.assertEqual(load_json(path), {"a": 1, "b": [2, 3]})

    def test_04_reynolds_si_conversion(self):
        self.assertAlmostEqual(calculate_reynolds(1000, 1, 10, 0.001), 10.0)

    def test_05_reynolds_unknown_is_nan(self):
        self.assertTrue(math.isnan(calculate_reynolds(None, 1, 10, None)))

    def test_06_same_seven_sections(self):
        config = load_json(ROOT / "config/audit_reynolds_profile_comparison.json")
        library = load_section_library(ROOT / "config/sections.json")
        sections = generate_section_series(config["section_series"], library)
        self.assertEqual(len(sections), 7)

    def test_07_section_series_positions_preserved(self):
        config = load_json(ROOT / "config/audit_reynolds_profile_comparison.json")
        library = load_section_library(ROOT / "config/sections.json")
        sections = generate_section_series(config["section_series"], library)
        self.assertEqual([item["z_mm"] for item in sections], [49.5,49,48,45,40,35,30])

    def test_08_measurement_key_can_include_case(self):
        keys = {(case, "z30", "fem") for case in ("re10", "re100")}
        self.assertEqual(len(keys), 2)

    def test_09_branch_fraction(self):
        row = branch_partition(100, 60, 40)
        self.assertEqual(row["straight_branch_fraction_percent"], 60.0)
        self.assertEqual(row["side_branch_fraction_percent"], 40.0)

    def test_10_branch_conservation_signed_inputs(self):
        row = branch_partition(-100, -60, 40)
        self.assertAlmostEqual(row["conservation_residual_percent"], 0.0)

    def test_11_percentage_point_difference(self):
        self.assertAlmostEqual(abs(43.4 - 44.1), 0.7)

    def test_12_relative_percent(self):
        self.assertAlmostEqual(relative_percent(99, 100), 1.0)

    def test_13_amplification(self):
        factor, status = safe_amplification(10, 2, 1e-8)
        self.assertEqual((factor, status), (5.0, "ok"))

    def test_14_zero_amplification_denominator_is_safe(self):
        factor, status = safe_amplification(10, 0, 1e-8)
        self.assertTrue(math.isnan(factor))
        self.assertIn("not robust", status)

    def test_15_case_r1(self):
        result = classify_reynolds_case(evidence(), evidence(profile=8, split=3), THRESHOLDS, True)
        self.assertEqual(result["case"], "R1")

    def test_16_case_r2(self):
        result = classify_reynolds_case(evidence(), evidence(), THRESHOLDS, True)
        self.assertEqual(result["case"], "R2")

    def test_17_case_r3(self):
        result = classify_reynolds_case(evidence(profile=8), evidence(profile=9, split=4), THRESHOLDS, True)
        self.assertEqual(result["case"], "R3")

    def test_18_case_r4(self):
        result = classify_reynolds_case(evidence(profile=8, split=2), evidence(), THRESHOLDS, True)
        self.assertEqual(result["case"], "R4")

    def test_19_case_r5(self):
        result = classify_reynolds_case(evidence(profile=8), evidence(profile=9), THRESHOLDS, False)
        self.assertEqual(result["case"], "R5")

    def test_20_r5_retains_conditional_case(self):
        result = classify_reynolds_case(evidence(profile=8), evidence(profile=9), THRESHOLDS, False)
        self.assertEqual(result["conditional_observed_case"], "R3")

    def test_21_json_ready_converts_nan_to_none(self):
        self.assertIsNone(json_ready(float("nan")))

    def test_22_missing_fem_rejected(self):
        with self.assertRaises(ReynoldsAuditError):
            validate_case("x", {"nominal_reynolds_number": 1}, ROOT/"config/x.json")

    def test_23_nonpositive_reynolds_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"a"; path.write_text("x")
            case = {
                "nominal_reynolds_number": 0,
                "fem": {"path": str(path), "velocity_array": "u"},
                "star": {"path": str(path), "velocity_array": "u"},
            }
            with self.assertRaises(ReynoldsAuditError):
                validate_case("x", case, ROOT/"config/x.json")

    def test_24_position_sensitivity_thresholds_present(self):
        config = load_json(ROOT / "config/audit_reynolds_profile_comparison.json")
        thresholds = config["decision_thresholds"]
        self.assertEqual(thresholds["position_sensitivity_flow_percent"], 1.0)
        self.assertEqual(thresholds["position_sensitivity_beta_alpha_percent"], 2.0)


if __name__ == "__main__":
    unittest.main()
