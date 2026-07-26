#!/usr/bin/env python3
"""Area-weighted velocity-profile metrics shared by profile audits."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from common_grid_output import valid_grid_metrics
from surface_flow import normalize_normal


class ProfileMetricsError(ValueError):
    """Raised when profile samples or weights are invalid."""


def _samples(values: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (count,):
        raise ProfileMetricsError(f"{name} must have shape ({count},)")
    if not np.all(np.isfinite(array)):
        raise ProfileMetricsError(f"{name} contains non-finite values")
    return array


def weighted_quantile(values: Any, weights: Any, quantiles: Any) -> np.ndarray:
    """Compute deterministic weighted quantiles using weight-centre interpolation."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape or not len(values):
        raise ProfileMetricsError("weighted quantiles require equal non-empty 1-D arrays")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ProfileMetricsError("weighted quantile inputs must be finite")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ProfileMetricsError("weighted quantile weights must have a positive sum")
    if np.any((quantiles < 0.0) | (quantiles > 1.0)):
        raise ProfileMetricsError("quantiles must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    positions = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / np.sum(sorted_weights)
    return np.interp(quantiles, positions, sorted_values, left=sorted_values[0], right=sorted_values[-1])


def area_weighted_histogram(
    values: Any, areas: Any, bins: int | Any = 30, value_range: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    areas = np.asarray(areas, dtype=float)
    if values.ndim != 1 or areas.shape != values.shape or not len(values):
        raise ProfileMetricsError("histogram values and areas must be equal non-empty 1-D arrays")
    if np.any(areas < 0.0) or not np.all(np.isfinite(values)) or not np.all(np.isfinite(areas)):
        raise ProfileMetricsError("histogram values must be finite and areas non-negative")
    area, edges = np.histogram(values, bins=bins, range=value_range, weights=areas)
    total = float(np.sum(area))
    fraction = area / total if total > 0.0 else np.zeros_like(area, dtype=float)
    return area.astype(float), fraction.astype(float), edges.astype(float)


def compute_profile_metrics(
    areas: Any,
    velocities: Any,
    centroid_s: Any,
    centroid_t: Any,
    section_normal: Any,
    flow_direction_normal: Any,
    s_axis: Any,
    t_axis: Any,
    low_velocity_fraction: float = 0.1,
) -> dict[str, float | int]:
    """Return area-weighted normal, moment, centroid, and secondary-flow metrics."""

    areas = np.asarray(areas, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if areas.ndim != 1 or not len(areas):
        raise ProfileMetricsError("areas must be a non-empty 1-D array")
    count = len(areas)
    if velocities.shape != (count, 3):
        raise ProfileMetricsError(f"velocities must have shape ({count}, 3)")
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ProfileMetricsError("all areas must be finite and positive")
    if not np.all(np.isfinite(velocities)):
        raise ProfileMetricsError("velocities contain non-finite values")
    s = _samples(centroid_s, count, "centroid_s")
    t = _samples(centroid_t, count, "centroid_t")
    n_section = normalize_normal(section_normal)
    n_flow = normalize_normal(flow_direction_normal)
    s_axis = normalize_normal(s_axis)
    t_axis = normalize_normal(t_axis)
    if abs(abs(float(np.dot(n_section, n_flow))) - 1.0) > 1.0e-8:
        raise ProfileMetricsError("section normal and flow-direction normal must be collinear")
    if max(abs(float(np.dot(n_flow, s_axis))), abs(float(np.dot(n_flow, t_axis))), abs(float(np.dot(s_axis, t_axis)))) > 1.0e-8:
        raise ProfileMetricsError("flow normal, s axis, and t axis must be orthogonal")
    if not math.isfinite(low_velocity_fraction) or low_velocity_fraction < 0.0:
        raise ProfileMetricsError("low_velocity_fraction must be finite and non-negative")

    area = float(np.sum(areas))
    u_signed = velocities @ n_section
    u_flow = velocities @ n_flow
    u_s = velocities @ s_axis
    u_t = velocities @ t_axis
    secondary = np.sqrt(u_s * u_s + u_t * u_t)
    signed_flow = float(np.sum(areas * u_signed))
    flow_direction_signed_flow = float(np.sum(areas * u_flow))
    flow_magnitude = abs(signed_flow)
    mean_flow = flow_magnitude / area
    if mean_flow <= 0.0:
        raise ProfileMetricsError("mean flow velocity is zero; normalized metrics are undefined")
    normalized = u_flow / mean_flow
    weighted_mean = lambda values: float(np.sum(areas * values) / area)
    std = math.sqrt(weighted_mean((u_flow - weighted_mean(u_flow)) ** 2))
    rms = math.sqrt(weighted_mean(u_flow * u_flow))
    normalized_mean = weighted_mean(normalized)
    normalized_std = math.sqrt(weighted_mean((normalized - normalized_mean) ** 2))
    q = weighted_quantile(normalized, areas, [0.05, 0.25, 0.5, 0.75, 0.95])
    raw_median = float(weighted_quantile(u_flow, areas, [0.5])[0])
    reverse = u_flow < 0.0
    low = u_flow < low_velocity_fraction * mean_flow
    reverse_area = float(np.sum(areas[reverse]))
    low_area = float(np.sum(areas[low]))
    m2 = float(np.sum(areas * u_flow**2))
    m3 = float(np.sum(areas * u_flow**3))
    positive_velocity = np.maximum(u_flow, 0.0)
    m2_positive = float(np.sum(areas * positive_velocity**2))
    m3_positive = float(np.sum(areas * positive_velocity**3))
    positive_flux = areas * positive_velocity
    positive_total = float(np.sum(positive_flux))
    if positive_total <= 0.0:
        raise ProfileMetricsError("profile has no positive flow contribution")
    flux_s = float(np.sum(s * positive_flux) / positive_total)
    flux_t = float(np.sum(t * positive_flux) / positive_total)
    geometric_s = weighted_mean(s)
    geometric_t = weighted_mean(t)

    def half_fraction(coordinates: np.ndarray, positive_side: bool) -> float:
        mask = coordinates >= 0.0 if positive_side else coordinates < 0.0
        return float(np.sum(positive_flux[mask]) / positive_total)

    secondary_rms = math.sqrt(weighted_mean(secondary**2))
    return {
        "sample_count": count,
        "area_mm2": area,
        "signed_flow_mm3_s": signed_flow,
        "flow_direction_signed_flow_mm3_s": flow_direction_signed_flow,
        "flow_magnitude_mm3_s": flow_magnitude,
        "mean_flow_velocity_mm_s": mean_flow,
        "minimum_flow_velocity_mm_s": float(np.min(u_flow)),
        "maximum_flow_velocity_mm_s": float(np.max(u_flow)),
        "median_flow_velocity_mm_s": raw_median,
        "area_weighted_mean_flow_velocity_mm_s": weighted_mean(u_flow),
        "area_weighted_std_flow_velocity_mm_s": std,
        "velocity_std_mm_s": std,
        "area_weighted_rms_flow_velocity_mm_s": rms,
        "normalized_flow_velocity_area_weighted_mean": normalized_mean,
        "normalized_flow_velocity_area_weighted_std": normalized_std,
        "normalized_velocity_std": normalized_std,
        "normalized_flow_velocity_minimum": float(np.min(normalized)),
        "normalized_flow_velocity_maximum": float(np.max(normalized)),
        "normalized_flow_velocity_q05": float(q[0]),
        "normalized_flow_velocity_q25": float(q[1]),
        "normalized_flow_velocity_median": float(q[2]),
        "normalized_flow_velocity_q75": float(q[3]),
        "normalized_flow_velocity_q95": float(q[4]),
        "reverse_flow_area_mm2": reverse_area,
        "reverse_flow_area_fraction_percent": 100.0 * reverse_area / area,
        "low_velocity_area_mm2": low_area,
        "low_velocity_area_fraction_percent": 100.0 * low_area / area,
        "low_velocity_threshold_fraction_of_mean": low_velocity_fraction,
        "momentum_flux_u2_mm4_s2": m2,
        "velocity_cubed_integral_mm5_s3": m3,
        "momentum_flux_u2_positive_only_mm4_s2": m2_positive,
        "velocity_cubed_integral_positive_only_mm5_s3": m3_positive,
        "momentum_correction_factor_beta": m2 / (area * mean_flow**2),
        "kinetic_energy_correction_factor_alpha": m3 / (area * mean_flow**3),
        "momentum_correction_factor_beta_positive_only": m2_positive / (area * mean_flow**2),
        "kinetic_energy_correction_factor_alpha_positive_only": m3_positive / (area * mean_flow**3),
        "beta": m2 / (area * mean_flow**2),
        "alpha": m3 / (area * mean_flow**3),
        "flux_centroid_s_mm": flux_s,
        "flux_centroid_t_mm": flux_t,
        "flux_centroid_radius_mm": math.hypot(flux_s, flux_t),
        "geometric_centroid_s_mm": geometric_s,
        "geometric_centroid_t_mm": geometric_t,
        "flux_centroid_offset_mm": math.hypot(flux_s - geometric_s, flux_t - geometric_t),
        "positive_s_flow_fraction": half_fraction(s, True),
        "negative_s_flow_fraction": half_fraction(s, False),
        "positive_t_flow_fraction": half_fraction(t, True),
        "negative_t_flow_fraction": half_fraction(t, False),
        "area_weighted_mean_abs_secondary_velocity_mm_s": weighted_mean(secondary),
        "area_weighted_rms_secondary_velocity_mm_s": secondary_rms,
        "rms_secondary_velocity_mm_s": secondary_rms,
        "maximum_secondary_velocity_mm_s": float(np.max(secondary)),
        "secondary_to_mean_flow_velocity_ratio_percent": 100.0 * secondary_rms / mean_flow,
        "secondary_velocity_ratio_percent": 100.0 * secondary_rms / mean_flow,
    }


def classify_fem_inlet_nodes(
    flow_velocity: Any,
    perimeter_mask: Any,
    wall_shared_mask: Any,
    expected_velocity_mm_s: float,
    tolerance_mm_s: float,
) -> tuple[np.ndarray, dict[str, int]]:
    flow_velocity = np.asarray(flow_velocity, dtype=float)
    perimeter = np.asarray(perimeter_mask, dtype=bool)
    shared = np.asarray(wall_shared_mask, dtype=bool)
    if flow_velocity.ndim != 1 or perimeter.shape != flow_velocity.shape or shared.shape != flow_velocity.shape:
        raise ProfileMetricsError("node classification arrays must have equal 1-D shapes")
    if expected_velocity_mm_s <= 0.0 or tolerance_mm_s < 0.0:
        raise ProfileMetricsError("expected velocity must be positive and tolerance non-negative")
    at_expected = np.isclose(flow_velocity, expected_velocity_mm_s, rtol=0.0, atol=tolerance_mm_s)
    at_zero = np.isclose(flow_velocity, 0.0, rtol=0.0, atol=tolerance_mm_s)
    labels = np.full(len(flow_velocity), "intermediate", dtype=object)
    labels[at_zero] = "zero"
    labels[at_expected] = "expected"
    summary = {
        "node_count": int(len(flow_velocity)),
        "expected_velocity_node_count": int(np.sum(at_expected)),
        "zero_velocity_node_count": int(np.sum(at_zero)),
        "intermediate_velocity_node_count": int(np.sum(~at_expected & ~at_zero)),
        "perimeter_node_count": int(np.sum(perimeter)),
        "interior_node_count": int(np.sum(~perimeter)),
        "inlet_wall_shared_node_count": int(np.sum(shared)),
    }
    return labels, summary


def common_grid_profile_metrics(
    fem_flow_velocity: Any,
    star_flow_velocity: Any,
    fem_secondary_speed: Any,
    star_secondary_speed: Any,
    fem_valid: Any,
    star_valid: Any,
    fem_mean_flow_velocity: float,
    star_mean_flow_velocity: float,
    width: float,
    height: float,
    grid_resolution: tuple[int, int],
) -> tuple[dict[str, float | int], np.ndarray]:
    arrays = [np.asarray(value, dtype=float) for value in (
        fem_flow_velocity, star_flow_velocity, fem_secondary_speed, star_secondary_speed
    )]
    size = arrays[0].size
    if any(array.shape != (size,) for array in arrays):
        raise ProfileMetricsError("common-grid value arrays must have equal 1-D shapes")
    fem_valid = np.asarray(fem_valid, dtype=bool)
    star_valid = np.asarray(star_valid, dtype=bool)
    if fem_valid.shape != (size,) or star_valid.shape != (size,):
        raise ProfileMetricsError("common-grid masks must match value arrays")
    if fem_mean_flow_velocity <= 0.0 or star_mean_flow_velocity <= 0.0:
        raise ProfileMetricsError("mean flow velocities must be positive")
    valid = fem_valid & star_valid
    for array in arrays:
        valid &= np.isfinite(array)
    if not np.any(valid):
        raise ProfileMetricsError("common grid has no mutually valid points")
    fem_norm = arrays[0] / fem_mean_flow_velocity
    star_norm = arrays[1] / star_mean_flow_velocity
    difference = fem_norm[valid] - star_norm[valid]
    denominator = float(np.linalg.norm(star_norm[valid]))
    dimensional_difference = arrays[0][valid] - arrays[1][valid]
    dimensional_denominator = float(np.linalg.norm(arrays[1][valid]))
    secondary_difference = arrays[2][valid] - arrays[3][valid]
    secondary_denominator = float(np.linalg.norm(arrays[3][valid]))
    fem_correlation_scale = max(abs(float(np.mean(fem_norm[valid]))), 1.0)
    star_correlation_scale = max(abs(float(np.mean(star_norm[valid]))), 1.0)
    if (
        np.std(fem_norm[valid]) <= 1.0e-12 * fem_correlation_scale
        or np.std(star_norm[valid]) <= 1.0e-12 * star_correlation_scale
    ):
        correlation = math.nan
    else:
        correlation = float(np.corrcoef(fem_norm[valid], star_norm[valid])[0, 1])
    area = valid_grid_metrics(valid, width, height, grid_resolution)
    metrics: dict[str, float | int] = {
        "normalized_velocity_relative_l2_percent": 100.0 * float(np.linalg.norm(difference)) / max(denominator, 1.0e-12),
        "normalized_velocity_mae": float(np.mean(np.abs(difference))),
        "normalized_velocity_rmse": math.sqrt(float(np.mean(difference**2))),
        "normalized_velocity_correlation": correlation,
        "dimensional_velocity_relative_l2_percent": 100.0 * float(np.linalg.norm(dimensional_difference)) / max(dimensional_denominator, 1.0e-12),
        "secondary_velocity_relative_l2_percent": 100.0 * float(np.linalg.norm(secondary_difference)) / max(secondary_denominator, 1.0e-12),
        "common_valid_area_mm2": float(area["common_valid_area"]),
        "common_valid_area_fraction_percent": 100.0 * float(area["valid_cell_fraction"]),
        "common_valid_point_count": int(area["valid_point_count"]),
        "common_valid_cell_count": int(area["valid_cell_count"]),
    }
    return metrics, valid
