from __future__ import annotations

import argparse
from pathlib import Path

import pyvista as pv

from streamlines_common import (
    ROOT,
    add_time_arrays,
    load_config,
    load_geometry_from_config,
    make_seed_source,
    PathlineTracker,
    physical_time_value,
    read_star_mesh,
    resolve_output_dir,
    resolve_path,
    save_streamline_outputs,
    time_points_from_config,
)


DEFAULT_CONFIG = ROOT / "config" / "streamlines_star_ccm.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw time-varying streamlines from a Star-CCM Ensight case.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    output_dir = resolve_output_dir(config, "output/streamlines/star_ccm")

    case_file = resolve_path(config.get("case_file"))
    if case_file is None or not case_file.exists():
        raise FileNotFoundError(f"Case file not found: {case_file}")

    velocity_name = config.get("velocity_name", "Velocity")
    source = make_seed_source(config.get("seed_source", {}))
    streamline_options = config.get("streamlines", {})

    print(f"Reading Star-CCM case: {case_file.relative_to(ROOT)}")
    reader = pv.get_reader(case_file)
    time_points = time_points_from_config(config, reader.number_time_points)

    pathlines_by_time = []
    first_mesh = None
    display_offset = int(config.get("display_time_index_offset", 1))
    initial_time_index = max(time_points[0] + display_offset - 1, 0)
    initial_time_value = max(physical_time_value(initial_time_index, config), 0.0)
    tracker = PathlineTracker(source, initial_time_index=initial_time_index, initial_time_value=initial_time_value)
    for reader_time_point in time_points:
        mesh, _reader_time_value = read_star_mesh(reader, reader_time_point, config)
        display_time_index = reader_time_point + display_offset
        time_value = physical_time_value(display_time_index, config)
        if first_mesh is None:
            first_mesh = mesh
        print(f"Star-CCM time point {reader_time_point} -> index {display_time_index}: time={time_value:g} s")
        pathlines = tracker.advance(mesh, velocity_name, display_time_index, time_value)
        pathlines = add_time_arrays(pathlines, display_time_index, time_value)
        print(f"  pathlines: {pathlines.n_cells} lines, {pathlines.n_points} points")
        pathlines_by_time.append((display_time_index, time_value, pathlines))

    geometry = load_geometry_from_config(config, first_mesh)
    save_streamline_outputs(
        pathlines_by_time,
        output_dir,
        prefix=config.get("output_prefix", "star_ccm_streamlines"),
        screenshot_config=config.get("screenshot", {}),
        config=config,
        geometry=geometry,
    )
    print(f"Saved outputs to: {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
