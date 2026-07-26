from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from profile_metrics import (
    area_weighted_histogram,
    classify_fem_inlet_nodes,
    common_grid_profile_metrics,
    compute_profile_metrics,
)
from surface_flow import flow_unit_values


def metrics(areas, flow, s=None, t=None, secondary=None, section_normal=(0, 0, 1), flow_normal=(0, 0, 1)):
    areas=np.asarray(areas,float); flow=np.asarray(flow,float)
    secondary=np.zeros_like(flow) if secondary is None else np.asarray(secondary,float)
    velocity=np.column_stack((secondary,np.zeros_like(flow),flow))
    return compute_profile_metrics(
        areas, velocity,
        np.zeros_like(flow) if s is None else s,
        np.zeros_like(flow) if t is None else t,
        section_normal, flow_normal, [1,0,0], [0,1,0], .1,
    )


class ProfileMetricTests(unittest.TestCase):
    def test_uniform_profile_has_unit_alpha_beta(self):
        result=metrics([1,2,3],[4,4,4])
        self.assertAlmostEqual(result["beta"],1.0)
        self.assertAlmostEqual(result["alpha"],1.0)
        self.assertAlmostEqual(result["normalized_velocity_std"],0.0)

    def test_flow_direction_unifies_reversed_section_normal(self):
        velocity=np.array([[0,0,-2.0]])
        result=compute_profile_metrics([1],velocity,[0],[0],[0,0,1],[0,0,-1],[1,0,0],[0,1,0])
        self.assertAlmostEqual(result["signed_flow_mm3_s"],-2.0)
        self.assertAlmostEqual(result["flow_direction_signed_flow_mm3_s"],2.0)
        self.assertAlmostEqual(result["mean_flow_velocity_mm_s"],2.0)

    def test_nonuniform_alpha_beta_match_manual_values(self):
        result=metrics([1,1],[1,3])
        self.assertAlmostEqual(result["beta"],1.25)
        self.assertAlmostEqual(result["alpha"],1.75)

    def test_flux_centroid_matches_manual_value(self):
        result=metrics([1,1],[1,3],s=np.array([-1,1]),t=np.array([0,0]))
        self.assertAlmostEqual(result["flux_centroid_s_mm"],0.5)

    def test_symmetric_profile_centroid_is_geometric_center(self):
        result=metrics([1,1],[2,2],s=np.array([-1,1]),t=np.array([0,0]))
        self.assertAlmostEqual(result["flux_centroid_s_mm"],0.0)
        self.assertAlmostEqual(result["flux_centroid_offset_mm"],0.0)

    def test_secondary_component_decomposition(self):
        result=metrics([1],[4],secondary=[3])
        self.assertAlmostEqual(result["rms_secondary_velocity_mm_s"],3.0)
        self.assertAlmostEqual(result["secondary_velocity_ratio_percent"],75.0)

    def test_area_weighted_histogram(self):
        area,fraction,edges=area_weighted_histogram([0,1],[1,3],bins=[-0.5,0.5,1.5])
        np.testing.assert_allclose(area,[1,3])
        np.testing.assert_allclose(fraction,[.25,.75])
        self.assertEqual(len(edges),3)

    def test_reverse_flow_area(self):
        result=metrics([1,3],[-1,2])
        self.assertAlmostEqual(result["reverse_flow_area_mm2"],1.0)
        self.assertAlmostEqual(result["reverse_flow_area_fraction_percent"],25.0)

    def test_low_velocity_area_uses_configured_mean_fraction(self):
        result=metrics([1,1],[.05,1.95])
        self.assertAlmostEqual(result["mean_flow_velocity_mm_s"],1.0)
        self.assertAlmostEqual(result["low_velocity_area_fraction_percent"],50.0)

    def test_fem_node_classification(self):
        labels,summary=classify_fem_inlet_nodes([10,0,5],[False,True,False],[False,True,False],10,1e-9)
        self.assertEqual(list(labels),["expected","zero","intermediate"])
        self.assertEqual(summary["expected_velocity_node_count"],1)
        self.assertEqual(summary["inlet_wall_shared_node_count"],1)

    def test_common_grid_valid_mask_and_area(self):
        values=np.ones(6)
        metrics_result,valid=common_grid_profile_metrics(values,values,np.zeros(6),np.zeros(6),[1,1,1,1,1,1],[1,1,1,1,1,1],1,1,2,1,(3,2))
        self.assertTrue(np.all(valid))
        self.assertAlmostEqual(metrics_result["common_valid_area_mm2"],2.0)
        self.assertAlmostEqual(metrics_result["common_valid_area_fraction_percent"],100.0)

    def test_normalized_velocity_relative_l2(self):
        fem=np.full(4,1.1);star=np.ones(4)
        result,_=common_grid_profile_metrics(fem,star,np.zeros(4),np.zeros(4),np.ones(4,bool),np.ones(4,bool),1,1,1,1,(2,2))
        self.assertAlmostEqual(result["normalized_velocity_relative_l2_percent"],10.0)

    def test_flow_units(self):
        result=flow_unit_values(1000.0)
        self.assertAlmostEqual(result["signed_flow_rate_ml_s"],1.0)
        self.assertAlmostEqual(result["signed_flow_rate_ml_min"],60.0)
        self.assertAlmostEqual(result["signed_flow_rate_ml_h"],3600.0)


if __name__ == "__main__":
    unittest.main()
