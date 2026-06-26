from __future__ import annotations

import argparse
from pathlib import Path

from streamlines_common import (
    ROOT,
    iter_solver_vtu_files,
    load_config,
    load_geometry_from_config,
    make_seed_source,
    PathlineTracker,
    physical_time_value,
    read_solver_mesh,
    resolve_output_dir,
    save_streamline_outputs,
    add_time_arrays,
)


DEFAULT_CONFIG = ROOT / "config" / "streamlines_solver.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw time-varying streamlines from solver VTU files.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    output_dir = resolve_output_dir(config, "output/streamlines/solver")

    velocity_name = config.get("velocity_name", "solution_velocity")
    source = make_seed_source(config.get("seed_source", {}))
    streamline_options = config.get("streamlines", {})
    files = iter_solver_vtu_files(config)
    if not files:
        raise ValueError("No solver VTU files were selected.")

    pathlines_by_time = []
    first_mesh = None
    initial_time_index = max(files[0][0] - 1, 0)
    initial_time_value = max(physical_time_value(initial_time_index, config), 0.0)
    tracker = PathlineTracker(source, initial_time_index=initial_time_index, initial_time_value=initial_time_value)
    for time_index, vtu_file in files:
        print(f"Reading solver time {time_index}: {vtu_file.relative_to(ROOT)}")
        mesh = read_solver_mesh(vtu_file, config)
        if first_mesh is None:
            first_mesh = mesh
        time_value = physical_time_value(time_index, config)
        pathlines = tracker.advance(mesh, velocity_name, time_index, time_value)
        pathlines = add_time_arrays(pathlines, time_index, time_value)
        print(f"  time={time_value:g} s / pathlines: {pathlines.n_cells} lines, {pathlines.n_points} points")
        pathlines_by_time.append((time_index, time_value, pathlines))

    geometry = load_geometry_from_config(config, first_mesh)
    save_streamline_outputs(
        pathlines_by_time,
        output_dir,
        prefix=config.get("output_prefix", "solver_streamlines"),
        screenshot_config=config.get("screenshot", {}),
        config=config,
        geometry=geometry,
    )
    print(f"Saved outputs to: {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
