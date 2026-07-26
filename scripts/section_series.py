#!/usr/bin/env python3
"""Generate auditable section series from one reusable section template."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

import numpy as np

from section_config import resolve_section, validate_section


class SectionSeriesError(ValueError):
    """Raised when a configured section series is invalid or ambiguous."""


def position_token(value: float, minimum_decimals: int = 1) -> str:
    """Return a stable filename-safe decimal token without losing sign."""

    value = float(value)
    if not math.isfinite(value):
        raise SectionSeriesError("section position must be finite")
    text = f"{abs(value):.9f}".rstrip("0")
    if text.endswith("."):
        text += "0" * minimum_decimals
    else:
        decimals = len(text.split(".", 1)[1])
        if decimals < minimum_decimals:
            text += "0" * (minimum_decimals - decimals)
    return ("m" if value < 0.0 else "") + text.replace(".", "p")


def section_name_for_position(prefix: str, axis: str, position: float) -> str:
    prefix = str(prefix).strip()
    axis = str(axis).strip().lower()
    if not prefix or axis not in {"x", "y", "z"}:
        raise SectionSeriesError("name_prefix is required and axis must be x, y, or z")
    return f"{prefix}_{axis}{position_token(position)}"


def _finite_number(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SectionSeriesError(f"{context} must be numeric") from exc
    if not math.isfinite(result):
        raise SectionSeriesError(f"{context} must be finite")
    return result


def generate_section_series(
    config: Mapping[str, Any], library: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Generate validated sections while inheriting window and plane axes."""

    if not isinstance(config, Mapping):
        raise SectionSeriesError("section_series must be an object")
    prefix = str(config.get("name_prefix", "")).strip()
    axis = str(config.get("axis", "z")).lower()
    if axis not in {"x", "y", "z"}:
        raise SectionSeriesError("section_series.axis must be x, y, or z")
    positions = config.get("positions_mm")
    if not isinstance(positions, list) or not positions:
        raise SectionSeriesError("section_series.positions_mm must be a non-empty array")
    positions = [_finite_number(value, "positions_mm value") for value in positions]
    template_name = str(config.get("template_section", "")).strip()
    template = resolve_section(template_name, library)
    inlet = _finite_number(config.get("inlet_position_mm"), "inlet_position_mm")
    diameter = _finite_number(config.get("diameter_mm"), "diameter_mm")
    if diameter <= 0.0:
        raise SectionSeriesError("diameter_mm must be positive")
    flow = np.asarray(config.get("flow_direction_normal"), dtype=float)
    if flow.shape != (3,) or not np.all(np.isfinite(flow)) or np.linalg.norm(flow) == 0.0:
        raise SectionSeriesError("flow_direction_normal must be a non-zero 3-vector")
    flow /= np.linalg.norm(flow)
    normal = config.get("normal", template["normalized_normal"])
    center_overrides = {
        0: config.get("center_x_mm", template["center"][0]),
        1: config.get("center_y_mm", template["center"][1]),
        2: config.get("center_z_mm", template["center"][2]),
    }
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    valid_range = config.get("valid_internal_position_range_mm")
    if valid_range is not None:
        if not isinstance(valid_range, list) or len(valid_range) != 2:
            raise SectionSeriesError("valid_internal_position_range_mm must be [minimum, maximum]")
        lower, upper = map(float, valid_range)
        if lower > upper:
            raise SectionSeriesError("valid internal position range is reversed")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for position in positions:
        if valid_range is not None and not (lower <= position <= upper):
            raise SectionSeriesError(
                f"position {position} is outside valid internal range [{lower}, {upper}]"
            )
        raw = deepcopy(template)
        center = np.array([float(center_overrides[index]) for index in range(3)])
        center[axis_index] = position
        raw.update({"center": center, "normal": normal, "flip_s": False, "flip_t": False})
        # Preserve the template's resolved display axes explicitly.
        raw["s_axis"] = np.asarray(template["s_axis"], dtype=float)
        raw["t_axis"] = np.asarray(template["t_axis"], dtype=float)
        section = validate_section(raw)
        name = section_name_for_position(prefix, axis, position)
        if name in names:
            raise SectionSeriesError(f"generated section name collision: {name}")
        names.add(name)
        distance = float(np.dot((center - np.eye(3)[axis_index] * inlet), flow))
        # For the configured straight z-series this is inlet_position - z; dot form
        # also remains valid for other collinear axes and flow directions.
        inlet_center = center.copy()
        inlet_center[axis_index] = inlet
        distance = float(np.dot(center - inlet_center, flow))
        if distance <= 0.0:
            raise SectionSeriesError(
                f"position {position} is not downstream of inlet {inlet} along flow direction"
            )
        section.update({
            "name": name,
            "position_mm": position,
            f"{axis}_mm": position,
            "axis": axis,
            "flow_direction_normal": flow.copy(),
            "distance_from_inlet_mm": distance,
            "distance_from_inlet_over_diameter": distance / diameter,
            "diameter_mm": diameter,
            "template_section": template_name,
        })
        result.append(section)
    return result


def generate_position_sensitivity_sections(
    series_config: Mapping[str, Any], sensitivity: Mapping[str, Any], library: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not bool(sensitivity.get("enabled", False)):
        return []
    bases = sensitivity.get("base_positions_mm")
    offsets = sensitivity.get("offsets_mm")
    if not isinstance(bases, list) or not bases or not isinstance(offsets, list) or not offsets:
        raise SectionSeriesError("position sensitivity requires non-empty base_positions_mm and offsets_mm")
    positions: list[float] = []
    metadata: list[tuple[float, float]] = []
    for base_value in bases:
        base = _finite_number(base_value, "base position")
        for offset_value in offsets:
            offset = _finite_number(offset_value, "position offset")
            positions.append(base + offset)
            metadata.append((base, offset))
    generated = generate_section_series({**dict(series_config), "positions_mm": positions}, library)
    names: set[str] = set()
    for section, (base, offset) in zip(generated, metadata):
        name = (
            f"{section['name']}_base{position_token(base)}_offset"
            f"{'p' if offset >= 0 else 'm'}{position_token(abs(offset), minimum_decimals=2)}"
        )
        if name in names:
            raise SectionSeriesError(f"sensitivity section name collision: {name}")
        names.add(name)
        section.update({
            "name": name,
            "base_position_mm": base,
            "offset_mm": offset,
            "is_position_sensitivity": True,
        })
    return generated
