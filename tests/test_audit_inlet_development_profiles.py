from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1];SCRIPTS=ROOT/"scripts"
if str(SCRIPTS) not in sys.path:sys.path.insert(0,str(SCRIPTS))

from audit_inlet_development_profiles import (
    classify_development,
    flow_conservation_rows,
    position_sensitivity_changes,
)
from profile_metrics import common_grid_profile_metrics, compute_profile_metrics
from section_config import load_section_library
from section_series import (
    generate_position_sensitivity_sections,
    generate_section_series,
    section_name_for_position,
)


class SectionSeriesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=ROOT/"config/sections.json";cls.before=hashlib.sha256(cls.path.read_bytes()).hexdigest();cls.library=load_section_library(cls.path)
        cls.config={"name_prefix":"inlet_development","template_section":"upstream_z30","axis":"z","positions_mm":[49.5,49.0,48.0],"center_x_mm":0.0,"center_y_mm":0.0,"normal":[0,0,1],"flow_direction_normal":[0,0,-1],"inlet_position_mm":50.0,"diameter_mm":10.0,"valid_internal_position_range_mm":[30,49.999]}

    def test_positions_generate_series(self):
        sections=generate_section_series(self.config,self.library);self.assertEqual(len(sections),3)
        self.assertEqual([s["z_mm"] for s in sections],[49.5,49.0,48.0])

    def test_decimal_names_are_unique_and_stable(self):
        self.assertEqual(section_name_for_position("inlet_development","z",49.5),"inlet_development_z49p5")
        self.assertEqual(section_name_for_position("inlet_development","z",49.0),"inlet_development_z49p0")

    def test_sections_json_is_not_modified(self):
        generate_section_series(self.config,self.library)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(),self.before)

    def test_template_window_and_axes_are_inherited(self):
        template=self.library["sections"]["upstream_z30"];section=generate_section_series(self.config,self.library)[0]
        self.assertEqual(section["width"],template["width"]);self.assertEqual(section["height"],template["height"])
        np.testing.assert_allclose(section["s_axis"],template["s_axis"]);np.testing.assert_allclose(section["t_axis"],template["t_axis"])

    def test_centers_normals_and_flow_direction(self):
        section=generate_section_series(self.config,self.library)[0]
        np.testing.assert_allclose(section["center"],[0,0,49.5]);np.testing.assert_allclose(section["normalized_normal"],[0,0,1]);np.testing.assert_allclose(section["flow_direction_normal"],[0,0,-1])

    def test_distance_and_x_over_d(self):
        section=generate_section_series(self.config,self.library)[0]
        self.assertAlmostEqual(section["distance_from_inlet_mm"],.5);self.assertAlmostEqual(section["distance_from_inlet_over_diameter"],.05)

    def test_offsets_generate_without_name_collisions(self):
        sections=generate_position_sensitivity_sections(self.config,{"enabled":True,"base_positions_mm":[49.5,49.0,48.0],"offsets_mm":[-.05,0,.05]},self.library)
        self.assertEqual(len(sections),9);self.assertEqual(len({s["name"] for s in sections}),9)
        self.assertIn(49.45,[round(s["z_mm"],2) for s in sections])


class DevelopmentMetricTests(unittest.TestCase):
    @staticmethod
    def calculate(areas,flow,normal=(0,0,1),flow_normal=(0,0,1)):
        flow=np.asarray(flow,float);velocity=np.column_stack((np.zeros_like(flow),np.zeros_like(flow),flow))
        return compute_profile_metrics(areas,velocity,np.zeros_like(flow),np.zeros_like(flow),normal,flow_normal,[1,0,0],[0,1,0],.1)

    def test_uniform_alpha_beta(self):
        result=self.calculate([1,2],[3,3]);self.assertAlmostEqual(result["beta"],1);self.assertAlmostEqual(result["alpha"],1)

    def test_parabolic_profile_matches_fully_developed_factors(self):
        nodes,weights=np.polynomial.legendre.leggauss(80);r=.5*(nodes+1);dr=.5*weights;areas=2*r*dr;u=2*(1-r*r)
        result=self.calculate(areas,u);self.assertAlmostEqual(result["beta"],4/3,places=10);self.assertAlmostEqual(result["alpha"],2,places=10)

    def test_normal_reversal_keeps_flow_direction_positive(self):
        velocity=np.array([[0,0,-2.]])
        result=compute_profile_metrics([1],velocity,[0],[0],[0,0,1],[0,0,-1],[1,0,0],[0,1,0],.1)
        self.assertEqual(result["signed_flow_mm3_s"],-2);self.assertEqual(result["flow_direction_signed_flow_mm3_s"],2)

    def test_flow_conservation_differences(self):
        rows=flow_conservation_rows({"fem":100},[{"solver":"fem","section_name":"a","z_mm":49.5,"distance_from_inlet_mm":.5,"flow_magnitude_mm3_s":99},{"solver":"fem","section_name":"b","z_mm":49,"distance_from_inlet_mm":1,"flow_magnitude_mm3_s":98}])
        self.assertEqual(rows[1]["flow_difference_from_boundary_percent"],-1)
        self.assertAlmostEqual(rows[2]["flow_difference_from_previous_section_percent"],-100/99)

    def test_common_grid_valid_mask_normalized_and_dimensional_l2(self):
        result,valid=common_grid_profile_metrics(np.full(4,1.1),np.ones(4),np.zeros(4),np.zeros(4),np.ones(4,bool),np.ones(4,bool),1,1,1,1,(2,2))
        self.assertTrue(np.all(valid));self.assertAlmostEqual(result["normalized_velocity_relative_l2_percent"],10);self.assertAlmostEqual(result["dimensional_velocity_relative_l2_percent"],10);self.assertAlmostEqual(result["common_valid_area_mm2"],1)

    def test_nonmonotonic_persistent_profile_matches_c_and_e(self):
        comparisons=[
            {"distance_from_inlet_mm":d,"normalized_velocity_relative_l2_percent":v}
            for d,v in zip([.5,1,2,5,10,15,20],[7.4,6.2,5.9,8.4,9.1,8.6,9.9])
        ]
        sensitivity=[{"position_sensitive":False}]
        result=classify_development(
            comparisons,sensitivity,6.2,
            {"large_profile_l2_percent":5,"small_profile_l2_percent":3,"increase_warning_percentage_points":2},
        )
        self.assertEqual(result["matched_cases"],["C","E"])

    def test_position_sensitivity_metrics_and_warning(self):
        base={"solver":"fem","base_position_mm":49.5,"normalized_velocity_relative_l2_percent":5,"beta":1,"alpha":1,"flow_magnitude_mm3_s":100}
        rows=[{**base,"offset_mm":0},{**base,"offset_mm":.05,"flow_magnitude_mm3_s":102,"beta":1.03}]
        result=position_sensitivity_changes(rows,{"flow_change_warning_percent":1,"beta_alpha_change_warning_percent":2,"l2_change_warning_percentage_points":2})
        changed=next(r for r in result if r["offset_mm"]==.05);self.assertTrue(changed["position_sensitive"]);self.assertAlmostEqual(changed["flow_change_from_zero_offset_percent"],2)


if __name__=="__main__":unittest.main()
