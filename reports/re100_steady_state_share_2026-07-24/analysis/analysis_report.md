# Re=100 velocity steady-state analysis

Generated: 2026-07-24T04:55:58.612944+00:00

## Inputs and provenance

- Result directory: `/workspace/data/260629_solver`
- Solver log: `/workspace/data/260629_solver/result.txt`
- Reference configurations:
- `/workspace/config/compare_streamlines_solver_star_ccm.json`
- `/workspace/config/relative_error_colormap_solver_star_ccm.json`
- VTU files: 300 (`solution_000001.vtu` through `solution_000300.vtu`)
- Step/time range: 1 / 0.125 s through 300 / 37.5 s
- Output interval: 0.125 s (one VTU per solver step)
- Log dt values: 0.125 s; log nsteps: 300
- Velocity array: point data `solution_velocity`
- Inlet-boundary data check: 245 points at z=50 mm; unique speed values [0.0, 10.0] mm/s; maximum 10 mm/s
- Mesh SHA-256 (points, offsets, connectivity, cell types): `7da5bce95cd2c6069111f1a50f8875365ede9ae776309032540caab1a35d7307`
- Mesh identity: exact hash equality was confirmed for every VTU file.
- Original solver input/configuration: not present in this repository. The run log verifies dt/nsteps,
  while existing post-processing configurations establish the Re=100 case mapping and output interval.

## Method

At every velocity point, speed is `||u||`. Whole-field mean, maximum, and 95th-percentile speeds are
unweighted point statistics. The primary change is
`100 ||u(t)-u(t-1 s)||_2 / ||u(t)||_2`; the maximum pointwise vector difference is
also reported. The actual lag is 8 output steps (1 s). One-step changes
are reported separately. Thresholds use strict `<` comparisons. “Sustained” means that every later
available primary comparison also remains below the threshold.

Section analysis: center=[10.0, 0.0, 15.0], normal=[1.0, 0.0, -1.0], width=12 mm, height=10 mm. Section 0.5%: first 10.25 s, sustained 10.25 s; section 0.1%: first 13.75 s, sustained 13.75 s.

## Whole-field threshold results

- 0.5%: first below at **8.25 s**; sustained from **8.25 s**.
- 0.1%: first below at **11.75 s**; sustained from **11.75 s**.

## Requested check times

| time [s] | step | relative L2 / 1 s [%] | max difference [mm/s] | mean [mm/s] | max [mm/s] | p95 [mm/s] |
|---:|---:|---:|---:|---:|---:|---:|
| 2.500 | 20 | 11.532 | 4.01308 | 3.41377 | 18.1958 | 12.7863 |
| 5.000 | 40 | 2.16801 | 0.782714 | 3.40169 | 18.2125 | 13.2662 |
| 7.500 | 60 | 0.680145 | 0.19054 | 3.39592 | 18.2066 | 13.307 |
| 10.000 | 80 | 0.216676 | 0.0671937 | 3.39411 | 18.2047 | 13.3197 |
| 12.500 | 100 | 0.0672607 | 0.021297 | 3.39358 | 18.2042 | 13.3205 |
| 15.000 | 120 | 0.0205764 | 0.0065857 | 3.39342 | 18.204 | 13.3202 |
| 20.000 | 160 | 0.00190436 | 0.000614339 | 3.39336 | 18.2039 | 13.32 |
| 25.000 | 200 | 0.000175792 | 5.67516e-05 | 3.39335 | 18.2039 | 13.32 |
| 30.000 | 240 | 1.62235e-05 | 5.23785e-06 | 3.39335 | 18.2039 | 13.32 |
| 35.000 | 280 | 1.49729e-06 | 4.83449e-07 | 3.39335 | 18.2039 | 13.32 |
| 37.500 | 300 | 4.54927e-07 | 1.46873e-07 | 3.39335 | 18.2039 | 13.32 |

## Output files

- `velocity_steady_state_metrics.csv`
- `relative_l2_change.png`
- `speed_statistics.png`
- `section_metrics.png`
