#!/usr/bin/env python3
"""Load, validate, and resolve reusable velocity-analysis section settings."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class SectionConfigError(ValueError):
    """Raised when a section library or section reference is invalid."""


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SectionConfigError(f"{context} must be a JSON object")
    return value


def _vector3(value: Any, context: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SectionConfigError(
            f"{context} must be a finite three-component numeric array"
        ) from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise SectionConfigError(
            f"{context} must be a finite three-component numeric array"
        )
    return vector


def _unit_vector(value: Any, context: str) -> np.ndarray:
    vector = _vector3(value, context)
    magnitude = float(np.linalg.norm(vector))
    if magnitude == 0.0:
        raise SectionConfigError(f"{context} must be a non-zero vector")
    return vector / magnitude


def _positive_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise SectionConfigError(f"{context} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SectionConfigError(
            f"{context} must be a positive finite number"
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise SectionConfigError(f"{context} must be greater than zero")
    return result


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise SectionConfigError(f"{context} must be true or false")
    return value


def make_plane_basis(
    normal: Any,
    s_axis: Any | None = None,
    t_axis: Any | None = None,
    flip_s: bool = False,
    flip_t: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized ``(normal, s_axis, t_axis)`` for a section plane.

    With no explicit axes, this uses the same helper-vector and cross-product
    rule as the existing velocity-distribution comparison scripts. If only one
    in-plane axis is supplied, the other is completed as a right-handed basis.
    """

    flip_s = _boolean(flip_s, "flip_s")
    flip_t = _boolean(flip_t, "flip_t")
    normal_array = _unit_vector(normal, "normal")
    tolerance = 1.0e-8

    if s_axis is None and t_axis is None:
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, normal_array))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        s_array = _unit_vector(
            np.cross(normal_array, helper), "automatically generated s_axis"
        )
        t_array = _unit_vector(
            np.cross(normal_array, s_array), "automatically generated t_axis"
        )
    elif s_axis is not None and t_axis is None:
        s_array = _unit_vector(s_axis, "s_axis")
        if abs(float(np.dot(normal_array, s_array))) > tolerance:
            raise SectionConfigError("s_axis must be perpendicular to normal")
        t_array = _unit_vector(np.cross(normal_array, s_array), "generated t_axis")
    elif s_axis is None and t_axis is not None:
        t_array = _unit_vector(t_axis, "t_axis")
        if abs(float(np.dot(normal_array, t_array))) > tolerance:
            raise SectionConfigError("t_axis must be perpendicular to normal")
        s_array = _unit_vector(np.cross(t_array, normal_array), "generated s_axis")
    else:
        s_array = _unit_vector(s_axis, "s_axis")
        t_array = _unit_vector(t_axis, "t_axis")
        if abs(float(np.dot(normal_array, s_array))) > tolerance:
            raise SectionConfigError("s_axis must be perpendicular to normal")
        if abs(float(np.dot(normal_array, t_array))) > tolerance:
            raise SectionConfigError("t_axis must be perpendicular to normal")
        if abs(float(np.dot(s_array, t_array))) > tolerance:
            raise SectionConfigError("s_axis and t_axis must be perpendicular")

    if flip_s:
        s_array = -s_array
    if flip_t:
        t_array = -t_array
    return normal_array, s_array, t_array


def _validate_section(section: Mapping[str, Any], context: str) -> dict[str, Any]:
    section = _mapping(section, context)
    result = deepcopy(dict(section))

    # Interpret names used by the existing section-comparison JSON files.
    if "grid_resolution" not in result and "resolution" in result:
        result["grid_resolution"] = result["resolution"]
    if "flip_s" not in result and "flip_s_axis" in result:
        result["flip_s"] = result["flip_s_axis"]
    if "flip_t" not in result and "flip_t_axis" in result:
        result["flip_t"] = result["flip_t_axis"]

    required = ("center", "normal", "width", "height", "grid_resolution")
    missing = [key for key in required if key not in result]
    if missing:
        raise SectionConfigError(
            f"{context} is missing required keys: {', '.join(missing)}"
        )

    center = _vector3(result["center"], f"{context}.center")
    flip_s = _boolean(result.get("flip_s", False), f"{context}.flip_s")
    flip_t = _boolean(result.get("flip_t", False), f"{context}.flip_t")
    normal, s_axis, t_axis = make_plane_basis(
        result["normal"],
        result.get("s_axis"),
        result.get("t_axis"),
        flip_s,
        flip_t,
    )

    resolution = result["grid_resolution"]
    if (
        not isinstance(resolution, (list, tuple))
        or len(resolution) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in resolution
        )
    ):
        raise SectionConfigError(
            f"{context}.grid_resolution must be [nx, ny] using integers"
        )
    nx, ny = int(resolution[0]), int(resolution[1])
    if nx < 2 or ny < 2:
        raise SectionConfigError(
            f"{context}.grid_resolution values must both be at least 2"
        )

    label = result.get("label", result.get("name", context))
    if not isinstance(label, str) or not label.strip():
        raise SectionConfigError(f"{context}.label must be a non-empty string")

    result.update(
        {
            "label": label,
            "center": center,
            "normal": normal,
            "normalized_normal": normal.copy(),
            "s_axis": s_axis,
            "t_axis": t_axis,
            "width": _positive_float(result["width"], f"{context}.width"),
            "height": _positive_float(result["height"], f"{context}.height"),
            "grid_resolution": (nx, ny),
            "flip_s": flip_s,
            "flip_t": flip_t,
        }
    )
    return result


def validate_section(section: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one section and return a normalized, canonical copy."""

    return _validate_section(section, "section")


def load_section_library(path: str | Path) -> dict[str, Any]:
    """Read and validate a section library JSON file."""

    library_path = Path(path).expanduser()
    if not library_path.is_file():
        raise SectionConfigError(f"section library does not exist: {library_path}")
    try:
        with library_path.open(encoding="utf-8") as file:
            raw = json.load(file)
    except json.JSONDecodeError as exc:
        raise SectionConfigError(
            f"invalid JSON in section library {library_path}: {exc}"
        ) from exc

    raw = _mapping(raw, f"section library {library_path}")
    sections_raw = _mapping(
        raw.get("sections"), f"section library {library_path}.sections"
    )
    if not sections_raw:
        raise SectionConfigError(
            f"section library {library_path}.sections must not be empty"
        )

    sections: dict[str, dict[str, Any]] = {}
    for name, section in sections_raw.items():
        if not isinstance(name, str) or not name.strip():
            raise SectionConfigError("every section library name must be non-empty")
        resolved = _validate_section(section, f"sections.{name}")
        resolved["name"] = name
        sections[name] = resolved

    sets_raw = raw.get("section_sets", {})
    sets_raw = _mapping(sets_raw, f"section library {library_path}.section_sets")
    section_sets: dict[str, list[str]] = {}
    for set_name, names in sets_raw.items():
        if not isinstance(set_name, str) or not set_name.strip():
            raise SectionConfigError("every section set name must be non-empty")
        if not isinstance(names, list) or not names:
            raise SectionConfigError(
                f"section_sets.{set_name} must be a non-empty array"
            )
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise SectionConfigError(
                f"section_sets.{set_name} must contain non-empty section names"
            )
        unknown = [name for name in names if name not in sections]
        if unknown:
            raise SectionConfigError(
                f"section_sets.{set_name} references unknown sections: "
                + ", ".join(unknown)
            )
        section_sets[set_name] = list(names)

    return {
        "sections": sections,
        "section_sets": section_sets,
        "path": library_path.resolve(),
    }


def resolve_section(name: str, library: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one named section from a loaded library."""

    if not isinstance(name, str) or not name.strip():
        raise SectionConfigError("section name must be a non-empty string")
    library = _mapping(library, "section library")
    sections = _mapping(library.get("sections"), "section library.sections")
    if name not in sections:
        available = ", ".join(sorted(str(key) for key in sections)) or "(none)"
        raise SectionConfigError(
            f"unknown section {name!r}; available sections: {available}"
        )
    section = _validate_section(sections[name], f"sections.{name}")
    section["name"] = name
    return section


def resolve_section_set(
    name: str, library: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Resolve an ordered named set into validated section copies."""

    if not isinstance(name, str) or not name.strip():
        raise SectionConfigError("section set name must be a non-empty string")
    library = _mapping(library, "section library")
    section_sets = _mapping(
        library.get("section_sets", {}), "section library.section_sets"
    )
    if name not in section_sets:
        available = ", ".join(sorted(str(key) for key in section_sets)) or "(none)"
        raise SectionConfigError(
            f"unknown section set {name!r}; available section sets: {available}"
        )
    names = section_sets[name]
    if not isinstance(names, list):
        raise SectionConfigError(f"section set {name!r} must be an array")
    return [resolve_section(section_name, library) for section_name in names]


def _resolve_library_path(value: str | Path, config_path: str | Path | None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if config_path is None:
        return path

    config_file = Path(config_path).expanduser().resolve()
    candidates = [config_file.parent / path, config_file.parent.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _library_from_config(
    config: Mapping[str, Any], config_path: str | Path | None
) -> dict[str, Any] | None:
    reference = config.get("section_library")
    if reference is None:
        return None
    if isinstance(reference, Mapping):
        sections = _mapping(reference.get("sections"), "section_library.sections")
        validated_sections: dict[str, dict[str, Any]] = {}
        for name, section in sections.items():
            validated = _validate_section(section, f"sections.{name}")
            validated["name"] = name
            validated_sections[name] = validated
        return {
            "sections": validated_sections,
            "section_sets": deepcopy(reference.get("section_sets", {})),
        }
    if not isinstance(reference, (str, Path)):
        raise SectionConfigError(
            "section_library must be a path string or a loaded library object"
        )
    return load_section_library(_resolve_library_path(reference, config_path))


def _legacy_section(
    raw: Mapping[str, Any],
    name: str,
    inherited: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    section = deepcopy(dict(raw))
    inherited = inherited or {}
    section.setdefault("name", name)
    section.setdefault("label", str(section.get("name", name)))
    section.setdefault("width", 10.0)
    section.setdefault("height", 10.0)
    if "grid_resolution" not in section and "resolution" not in section:
        section["grid_resolution"] = [201, 201]
    if "flip_s" not in section and "flip_s_axis" not in section:
        section["flip_s"] = inherited.get("flip_s", inherited.get("flip_s_axis", False))
    if "flip_t" not in section and "flip_t_axis" not in section:
        section["flip_t"] = inherited.get("flip_t", inherited.get("flip_t_axis", False))
    resolved = _validate_section(section, f"sections.{name}")
    resolved["name"] = name
    return resolved


def resolve_sections_from_config(
    config: Mapping[str, Any], config_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Resolve section settings using library references or legacy inline forms.

    Resolution priority is:
    ``section_library + section_set``, ``section_library + section_names``,
    inline ``sections`` array, single ``section``, then existing mapping and
    top-level section formats.
    """

    config = _mapping(config, "configuration")
    library = _library_from_config(config, config_path)

    if library is None and "section_set" in config:
        raise SectionConfigError("section_set requires section_library")
    if library is None and "section_names" in config:
        raise SectionConfigError("section_names requires section_library")

    if library is not None and "section_set" in config:
        return resolve_section_set(config["section_set"], library)

    if library is not None and "section_names" in config:
        names = config["section_names"]
        if not isinstance(names, list) or not names:
            raise SectionConfigError("section_names must be a non-empty array")
        return [resolve_section(name, library) for name in names]

    sections = config.get("sections")
    if isinstance(sections, list):
        if not sections:
            raise SectionConfigError("sections must not be an empty array")
        resolved: list[dict[str, Any]] = []
        for index, raw in enumerate(sections):
            if isinstance(raw, str):
                if library is None:
                    raise SectionConfigError(
                        f"sections[{index}] is a name but no section_library is configured"
                    )
                resolved.append(resolve_section(raw, library))
                continue
            raw = _mapping(raw, f"sections[{index}]")
            name = str(raw.get("name", f"section_{index + 1}"))
            resolved.append(_legacy_section(raw, name, config))
        return resolved

    if "section" in config:
        raw = config["section"]
        if isinstance(raw, str):
            if library is None:
                raise SectionConfigError(
                    "section is a name but no section_library is configured"
                )
            return [resolve_section(raw, library)]
        raw = _mapping(raw, "section")
        return [_legacy_section(raw, str(raw.get("name", "section")), config)]

    if isinstance(sections, Mapping):
        if not sections:
            raise SectionConfigError("sections must not be an empty object")
        return [
            _legacy_section(raw, str(name), config)
            for name, raw in sections.items()
        ]

    if "center" in config or "normal" in config:
        return [_legacy_section(config, str(config.get("name", "section")), config)]

    if library is not None:
        raise SectionConfigError(
            "section_library requires section_set, section_names, or a named section reference"
        )
    raise SectionConfigError(
        "no section configuration found; expected section_library references, "
        "sections, section, or top-level center/normal"
    )


__all__ = [
    "SectionConfigError",
    "load_section_library",
    "make_plane_basis",
    "resolve_section",
    "resolve_section_set",
    "resolve_sections_from_config",
    "validate_section",
]
