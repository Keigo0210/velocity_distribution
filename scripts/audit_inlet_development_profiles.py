#!/usr/bin/env python3
"""Audit near-inlet internal-profile development from configured section series."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from audit_boundary_flow import _find_named_block, _scaled_copy, load_source
from audit_inlet_upstream_profile import (
    common_grid_for_location,
    compute_location_profile,
    fem_internal_section_samples,
    star_internal_section_samples,
)
from section_config import load_section_library
from section_series import (
    SectionSeriesError,
    generate_position_sensitivity_sections,
    generate_section_series,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "audit_inlet_development_profiles.json"


class DevelopmentAuditError(ValueError):
    """Raised for invalid 4B-3 inputs or configuration."""


def resolve_config_path(value: str | Path, config_path: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config_path).resolve().parent / path).resolve()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DevelopmentAuditError(f"{context} must be a JSON object")
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DevelopmentAuditError(f"required existing result does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def profile_row(
    solver: str,
    section: Mapping[str, Any],
    samples: Mapping[str, Any],
    profile: Mapping[str, Any],
    extraction: Mapping[str, Any],
    beta_reference: float,
    alpha_reference: float,
) -> dict[str, Any]:
    beta = float(profile["beta"])
    alpha = float(profile["alpha"])
    return {
        **dict(profile),
        "solver": solver,
        "section_name": section["name"],
        "z_mm": float(section["z_mm"]),
        "distance_from_inlet_mm": float(section["distance_from_inlet_mm"]),
        "distance_from_inlet_over_diameter": float(section["distance_from_inlet_over_diameter"]),
        "normalized_velocity_mean": profile["normalized_flow_velocity_area_weighted_mean"],
        "normalized_velocity_std": profile["normalized_flow_velocity_area_weighted_std"],
        "normalized_velocity_min": profile["normalized_flow_velocity_minimum"],
        "normalized_velocity_max": profile["normalized_flow_velocity_maximum"],
        "normalized_velocity_q05": profile["normalized_flow_velocity_q05"],
        "normalized_velocity_q25": profile["normalized_flow_velocity_q25"],
        "normalized_velocity_median": profile["normalized_flow_velocity_median"],
        "normalized_velocity_q75": profile["normalized_flow_velocity_q75"],
        "normalized_velocity_q95": profile["normalized_flow_velocity_q95"],
        "intersected_cell_count": int(extraction["intersected_cell_count"]),
        "generated_polygon_count": int(extraction["generated_polygon_count"]),
        "valid_polygon_count": int(extraction["valid_polygon_count"]),
        "invalid_polygon_count": int(extraction["invalid_polygon_count"]),
        "duplicate_original_cell_count": int(extraction.get("duplicate_original_cell_count", 0)),
        "multiple_polygon_original_cell_count": int(extraction.get("multiple_polygon_original_cell_count", 0)),
        "unmapped_polygon_count": int(extraction["unmapped_polygon_count"]),
        "intersection_area_mm2": float(extraction["intersection_area_mm2"]),
        "beta_difference_from_fully_developed": beta - beta_reference,
        "alpha_difference_from_fully_developed": alpha - alpha_reference,
        "beta_relative_difference_from_fully_developed_percent": 100.0 * abs(beta - beta_reference) / beta_reference,
        "alpha_relative_difference_from_fully_developed_percent": 100.0 * abs(alpha - alpha_reference) / alpha_reference,
        "status": "success",
        "warning": str(extraction.get("warning", "")),
    }


def comparison_row(
    section: Mapping[str, Any],
    fem: Mapping[str, Any],
    star: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "section_name": section["name"],
        "z_mm": section["z_mm"],
        "distance_from_inlet_mm": section["distance_from_inlet_mm"],
        "distance_from_inlet_over_diameter": section["distance_from_inlet_over_diameter"],
        **dict(grid),
    }
    fields = [
        "flow_magnitude_mm3_s", "mean_flow_velocity_mm_s", "beta", "alpha",
        "normalized_velocity_std", "flux_centroid_s_mm", "flux_centroid_t_mm",
        "flux_centroid_offset_mm", "secondary_velocity_ratio_percent",
    ]
    for field in fields:
        first = float(fem[field]); second = float(star[field])
        row[f"fem_{field}"] = first
        row[f"star_{field}"] = second
        row[f"signed_difference_{field}"] = first - second
        row[f"absolute_difference_{field}"] = abs(first - second)
        row[f"relative_difference_percent_{field}"] = 100.0 * abs(first - second) / max(abs(second), 1.0e-12)
    row["flux_centroid_distance_mm"] = math.hypot(
        float(fem["flux_centroid_s_mm"]) - float(star["flux_centroid_s_mm"]),
        float(fem["flux_centroid_t_mm"]) - float(star["flux_centroid_t_mm"]),
    )
    row["secondary_velocity_ratio_difference_percentage_points"] = abs(
        float(fem["secondary_velocity_ratio_percent"])
        - float(star["secondary_velocity_ratio_percent"])
    )
    return row


def flow_conservation_rows(
    boundary_flow_by_solver: Mapping[str, float], profile_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for solver in sorted(boundary_flow_by_solver):
        boundary = abs(float(boundary_flow_by_solver[solver]))
        result.append({
            "solver": solver, "location_type": "inlet_boundary", "section_name": "inlet_boundary_z50",
            "z_mm": 50.0, "distance_from_inlet_mm": 0.0,
            "flow_magnitude_mm3_s": boundary,
            "flow_difference_from_boundary_mm3_s": 0.0,
            "flow_difference_from_boundary_percent": 0.0,
            "flow_difference_from_previous_section_mm3_s": 0.0,
            "flow_difference_from_previous_section_percent": 0.0,
        })
        previous = boundary
        rows = sorted(
            (row for row in profile_rows if row["solver"] == solver),
            key=lambda row: float(row["distance_from_inlet_mm"]),
        )
        for row in rows:
            value = abs(float(row["flow_magnitude_mm3_s"]))
            result.append({
                "solver": solver, "location_type": "near_inlet_internal_section",
                "section_name": row["section_name"], "z_mm": row["z_mm"],
                "distance_from_inlet_mm": row["distance_from_inlet_mm"],
                "flow_magnitude_mm3_s": value,
                "flow_difference_from_boundary_mm3_s": value - boundary,
                "flow_difference_from_boundary_percent": 100.0 * (value - boundary) / max(boundary, 1.0e-12),
                "flow_difference_from_previous_section_mm3_s": value - previous,
                "flow_difference_from_previous_section_percent": 100.0 * (value - previous) / max(previous, 1.0e-12),
            })
            previous = value
    return result


def position_sensitivity_changes(
    rows: list[dict[str, Any]], settings: Mapping[str, Any]
) -> list[dict[str, Any]]:
    flow_limit = float(settings["flow_change_warning_percent"])
    moment_limit = float(settings["beta_alpha_change_warning_percent"])
    l2_limit = float(settings["l2_change_warning_percentage_points"])
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["solver"]), float(row["base_position_mm"])), []).append(row)
    result: list[dict[str, Any]] = []
    for (solver, base), group in groups.items():
        zero = next((row for row in group if abs(float(row["offset_mm"])) < 1.0e-12), None)
        if zero is None:
            raise DevelopmentAuditError(f"sensitivity group {solver}/{base} lacks zero offset")
        for row in group:
            current = dict(row)
            flow_change = 100.0 * abs(float(row["flow_magnitude_mm3_s"]) - float(zero["flow_magnitude_mm3_s"])) / max(abs(float(zero["flow_magnitude_mm3_s"])), 1.0e-12)
            beta_change = 100.0 * abs(float(row["beta"]) - float(zero["beta"])) / max(abs(float(zero["beta"])), 1.0e-12)
            alpha_change = 100.0 * abs(float(row["alpha"]) - float(zero["alpha"])) / max(abs(float(zero["alpha"])), 1.0e-12)
            l2_change = abs(float(row["normalized_velocity_relative_l2_percent"]) - float(zero["normalized_velocity_relative_l2_percent"]))
            warnings = []
            if flow_change > flow_limit: warnings.append("flow position sensitivity exceeds threshold")
            if max(beta_change, alpha_change) > moment_limit: warnings.append("beta/alpha position sensitivity exceeds threshold")
            if l2_change > l2_limit: warnings.append("profile L2 position sensitivity exceeds threshold")
            current.update({
                "flow_change_from_zero_offset_percent": flow_change,
                "beta_change_from_zero_offset_percent": beta_change,
                "alpha_change_from_zero_offset_percent": alpha_change,
                "l2_change_from_zero_offset_percentage_points": l2_change,
                "position_sensitive": bool(warnings),
                "warning": "; ".join(warnings),
            })
            result.append(current)
    return sorted(result, key=lambda row: (float(row["base_position_mm"]), str(row["solver"]), float(row["offset_mm"])))


def classify_development(
    comparisons: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    boundary_l2: float,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(comparisons, key=lambda row: float(row["distance_from_inlet_mm"]))
    values = [float(row["normalized_velocity_relative_l2_percent"]) for row in ordered]
    distances = [float(row["distance_from_inlet_mm"]) for row in ordered]
    large = float(thresholds["large_profile_l2_percent"])
    small = float(thresholds["small_profile_l2_percent"])
    increase = float(thresholds["increase_warning_percentage_points"])
    position_sensitive = any(bool(row["position_sensitive"]) for row in sensitivity)
    near = values[:3]
    later = values[3:]
    increases = [
        {"from_distance_mm": distances[index - 1], "to_distance_mm": distances[index], "increase_percentage_points": values[index] - values[index - 1]}
        for index in range(1, len(values)) if values[index] - values[index - 1] > increase
    ]
    matched: list[str] = []
    statements: list[str] = []
    if position_sensitive:
        matched.append("D")
        statements.append("少なくとも一つの指標が±0.05 mm位置感度基準を超え、単一cut位置を代表値にできません。")
    else:
        if max(near) < small:
            matched.append("B")
            statements.append("境界差は最初の内部断面群で急速に小さくなりました。")
        if later and min(near) < boundary_l2 and max(later) > min(near) + increase:
            matched.append("C")
            statements.append("入口近傍で境界面より一度小さくなったprofile差が、0.5D以降で再増大しました。")
        if values[0] >= large and values[-1] < values[0] - increase:
            matched.append("A")
            statements.append("最初の内部断面に大きな差があり、下流で減衰する傾向もあります。")
        if values[-1] >= large:
            matched.append("E")
            statements.append("位置感度は設定基準内ですが、profile差がz=30 mmまで残りました。")
    if not matched:
        matched = ["unclassified"]
        statements = ["設定ケースへ一意に分類できない非単調な挙動です。"]
    return {
        "case": "+".join(matched),
        "matched_cases": matched,
        "interpretation": " ".join(statements) + " 原因は断定しません。",
        "boundary_normalized_l2_percent": boundary_l2,
        "internal_distances_mm": distances,
        "internal_normalized_l2_percent": values,
        "large_intermediate_increases": increases,
        "position_sensitivity_warning": position_sensitive,
        "causation_determined": False,
    }


def _masked(values: np.ndarray, valid: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.array(values, mask=~valid)


def plot_section(section: Mapping[str, Any], arrays: Mapping[str, np.ndarray], output: Path, plot: Mapping[str, Any]) -> None:
    profiles = output / "figures/profiles"; comparisons = output / "figures/comparisons"
    profiles.mkdir(parents=True, exist_ok=True); comparisons.mkdir(parents=True, exist_ok=True)
    name = str(section["name"]); ss=np.asarray(arrays["s_grid"]);tt=np.asarray(arrays["t_grid"]);valid=np.asarray(arrays["common_valid"],bool)
    fem=np.asarray(arrays["fem_normalized_flow_velocity"]);star=np.asarray(arrays["star_normalized_flow_velocity"])
    dpi=int(plot.get("dpi",180));cmap=str(plot.get("cmap","viridis"));dcmap=str(plot.get("difference_cmap","coolwarm"))
    minimum=float(np.nanmin(np.concatenate((fem[valid],star[valid]))));maximum=float(np.nanmax(np.concatenate((fem[valid],star[valid]))))
    def single(values,path,title,vmin,vmax,color):
        fig,ax=plt.subplots(figsize=(6,4));image=ax.pcolormesh(ss,tt,_masked(values,valid),shading="auto",cmap=color,vmin=vmin,vmax=vmax);fig.colorbar(image,ax=ax);ax.set(xlabel="s [mm]",ylabel="t [mm]",title=title,aspect="equal");fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)
    single(fem,profiles/f"fem_{name}_normalized_velocity.png",f"FEM {name}",minimum,maximum,cmap)
    single(star,profiles/f"star_{name}_normalized_velocity.png",f"STAR {name}",minimum,maximum,cmap)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,data,title in zip(axes,(fem,star),("FEM","STAR")):
        image=ax.pcolormesh(ss,tt,_masked(data,valid),shading="auto",cmap=cmap,vmin=minimum,vmax=maximum);ax.set(xlabel="s [mm]",ylabel="t [mm]",title=title,aspect="equal")
    fig.colorbar(image,ax=axes.tolist());fig.suptitle(name);fig.savefig(comparisons/f"{name}_normalized_velocity_comparison.png",dpi=dpi,bbox_inches="tight");plt.close(fig)
    difference=fem-star;limit=float(np.nanmax(np.abs(difference[valid])));single(difference,comparisons/f"{name}_normalized_velocity_difference.png",f"{name}: FEM - STAR",-limit,limit,dcmap)


def plot_development(
    profiles: list[dict[str, Any]], comparisons: list[dict[str, Any]],
    boundary_profiles: Mapping[str, Mapping[str, Any]], boundary_grid: Mapping[str, Any],
    output: Path, plot: Mapping[str, Any]
) -> None:
    target=output/"figures/development";target.mkdir(parents=True,exist_ok=True);dpi=int(plot.get("dpi",180));axis_mode=str(plot.get("x_axis","distance_mm"));xkey="distance_from_inlet_over_diameter" if axis_mode=="x_over_d" else "distance_from_inlet_mm";xlabel="distance from inlet / D" if axis_mode=="x_over_d" else "distance from inlet [mm]"
    by_solver={solver:sorted([r for r in profiles if r["solver"]==solver],key=lambda r:r[xkey]) for solver in ("fem","star")};comp=sorted(comparisons,key=lambda r:r[xkey])
    boundary_x=0.0
    def solver_plot(field,filename,ylabel,reference=None):
        fig,ax=plt.subplots(figsize=(6,4))
        for solver,marker in (("fem","o"),("star","s")):
            rows=by_solver[solver];x=[boundary_x]+[r[xkey] for r in rows];y=[float(boundary_profiles[solver][field])]+[float(r[field]) for r in rows];ax.plot(x,y,marker=marker,label=solver.upper());ax.scatter([boundary_x],[y[0]],s=70,facecolors="none",edgecolors=ax.lines[-1].get_color())
        if reference is not None:ax.axhline(reference,color="gray",ls="--",label="fully developed reference")
        ax.set(xlabel=xlabel,ylabel=ylabel);ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();fig.savefig(target/filename,dpi=dpi);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));ax.plot([0]+[r[xkey] for r in comp],[float(boundary_grid["normalized_velocity_relative_l2_percent"])]+[float(r["normalized_velocity_relative_l2_percent"]) for r in comp],marker="o");ax.scatter([0],[float(boundary_grid["normalized_velocity_relative_l2_percent"])],s=80,facecolors="none",edgecolors="C0",label="inlet boundary");ax.set(xlabel=xlabel,ylabel="normalized velocity L2 [%]");ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();fig.savefig(target/"normalized_velocity_l2_vs_distance.png",dpi=dpi);plt.close(fig)
    solver_plot("beta","beta_vs_distance.png","beta",4/3);solver_plot("alpha","alpha_vs_distance.png","alpha",2.0)
    for field,filename,ylabel in (("absolute_difference_beta","beta_difference_vs_distance.png","|FEM - STAR| beta"),("absolute_difference_alpha","alpha_difference_vs_distance.png","|FEM - STAR| alpha"),("absolute_difference_flow_magnitude_mm3_s","flow_difference_vs_distance.png","|FEM - STAR| flow [mm3/s]")):
        fig,ax=plt.subplots(figsize=(6,4));ax.plot([r[xkey] for r in comp],[r[field] for r in comp],marker="o");ax.set(xlabel=xlabel,ylabel=ylabel);ax.grid(True,alpha=.3);fig.tight_layout();fig.savefig(target/filename,dpi=dpi);plt.close(fig)
    solver_plot("flow_magnitude_mm3_s","flow_vs_distance.png","flow magnitude [mm3/s]");solver_plot("mean_flow_velocity_mm_s","mean_velocity_vs_distance.png","mean velocity [mm/s]");solver_plot("normalized_velocity_std","normalized_velocity_std_vs_distance.png","normalized velocity std");solver_plot("secondary_velocity_ratio_percent","secondary_velocity_ratio_vs_distance.png","secondary / mean [%]");solver_plot("flux_centroid_offset_mm","flux_centroid_offset_vs_distance.png","flux centroid offset [mm]")


def plot_sensitivity(rows: list[dict[str, Any]], output: Path, dpi: int) -> None:
    target=output/"figures/position_sensitivity";target.mkdir(parents=True,exist_ok=True)
    for field,filename,ylabel in (("flow_magnitude_mm3_s","near_inlet_position_sensitivity_flow.png","flow [mm3/s]"),("beta","near_inlet_position_sensitivity_beta.png","beta"),("alpha","near_inlet_position_sensitivity_alpha.png","alpha")):
        fig,ax=plt.subplots(figsize=(7,4))
        for base in sorted({float(r["base_position_mm"]) for r in rows}):
            for solver in ("fem","star"):
                group=sorted([r for r in rows if float(r["base_position_mm"])==base and r["solver"]==solver],key=lambda r:r["offset_mm"]);ax.plot([r["offset_mm"] for r in group],[r[field] for r in group],marker="o",label=f"{solver} z={base:g}")
        ax.set(xlabel="z offset [mm]",ylabel=ylabel);ax.grid(True,alpha=.3);ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(target/filename,dpi=dpi);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));unique={}
    for row in rows:unique[(float(row["base_position_mm"]),float(row["offset_mm"]))]=row
    for base in sorted({key[0] for key in unique}):
        group=sorted([r for key,r in unique.items() if key[0]==base],key=lambda r:r["offset_mm"]);ax.plot([r["offset_mm"] for r in group],[r["normalized_velocity_relative_l2_percent"] for r in group],marker="o",label=f"z={base:g}")
    ax.set(xlabel="z offset [mm]",ylabel="normalized velocity L2 [%]");ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();fig.savefig(target/"near_inlet_position_sensitivity_l2.png",dpi=dpi);plt.close(fig)


def build_markdown(audit: Mapping[str, Any]) -> str:
    decision=audit["decision"]
    lines=["# 入口近傍から分岐前までの速度profile発達監査","",f"作成日: {date.today().isoformat()}","","z=50 mmは実入口境界であり、境界条件の離散表現監査に限定する。z<50 mmはnear-inlet internal sectionで、正式な内部profile比較に使用する。z=49.5 mmを入口境界流量とは呼ばない。","","STARの正式流量・alpha・betaはnative volume-cell intersection、FEMは線形point速度の三角形求積。共通gridは可視化・profile差専用で、STAR cell-to-point平滑化を正式定量値へ混ぜない。","","## 内部断面profile","","| z | x/D | FEM Q | STAR Q | L2 normalized | L2 dimensional | FEM beta/alpha | STAR beta/alpha |","|---:|---:|---:|---:|---:|---:|---:|---:|"]
    profiles={(r["solver"],r["section_name"]):r for r in audit["profile_development_summary"]}
    for c in audit["comparisons"]:
        f=profiles[("fem",c["section_name"])];s=profiles[("star",c["section_name"])]
        lines.append(f"| {c['z_mm']:.3f} | {c['distance_from_inlet_over_diameter']:.3f} | {f['flow_magnitude_mm3_s']:.8g} | {s['flow_magnitude_mm3_s']:.8g} | {c['normalized_velocity_relative_l2_percent']:.6g}% | {c['dimensional_velocity_relative_l2_percent']:.6g}% | {f['beta']:.6g}/{f['alpha']:.6g} | {s['beta']:.6g}/{s['alpha']:.6g} |")
    lines.extend(["","## 判定","",f"ケース **{decision['case']}**: {decision['interpretation']}","",f"大きな区間増加: `{decision['large_intermediate_increases']}`。位置感度warning: `{decision['position_sensitivity_warning']}`。","","完全発達円管層流のbeta=4/3、alpha=2は参考値であり、各断面が完全発達しているとは仮定しない。FEM–STAR差と理論profileとの差を混同しない。","","原因は自動的に断定しない。新規CFD計算、境界条件変更、分岐後解析は行っていない。","","## 実行","","```bash","cd /workspace","python scripts/audit_inlet_development_profiles.py --config config/audit_inlet_development_profiles.json","```",""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG);args=parser.parse_args(argv)
    config_path=args.config if args.config.is_absolute() else ROOT/args.config;config_path=config_path.resolve();config=dict(_mapping(json.loads(config_path.read_text(encoding="utf-8")),"configuration"))
    output_cfg=_mapping(config["output"],"output");output=resolve_config_path(output_cfg["directory"],config_path);execution=_mapping(config.get("execution",{}),"execution")
    if execution.get("refuse_nonempty_output_directory",True) and output.exists() and any(output.iterdir()):raise DevelopmentAuditError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True,exist_ok=True)
    library_path=resolve_config_path(config["section_library"],config_path);library=load_section_library(library_path);series_cfg=_mapping(config["section_series"],"section_series");sections=generate_section_series(series_cfg,library);sensitivity_sections=generate_position_sensitivity_sections(series_cfg,_mapping(config["position_sensitivity"],"position_sensitivity"),library)
    base_path=resolve_config_path(config["base_profile_audit_config"],config_path);base=dict(_mapping(json.loads(base_path.read_text(encoding="utf-8")),"base profile config"));boundary_cfg_path=resolve_config_path(base["boundary_audit_config"],base_path);boundary_cfg=dict(_mapping(json.loads(boundary_cfg_path.read_text(encoding="utf-8")),"boundary config"));solvers=_mapping(base["solvers"],"solvers")
    source_names={key:str(_mapping(value,f"solver {key}")["source_name"]) for key,value in solvers.items()};sources={key:dict(_mapping(boundary_cfg["data_sources"][name],f"source {name}")) for key,name in source_names.items()};loaded={key:load_source(source_names[key],sources[key],boundary_cfg_path) for key in ("fem","star")};fem_mesh=loaded["fem"]["primary"];part_name=str(_mapping(solvers["star"],"star solver")["volume_part_name"]);star_mesh=_scaled_copy(_find_named_block(loaded["star"]["raw"],part_name),sources["star"]);fem_velocity=str(sources["fem"]["velocity_array"]);star_velocity=str(sources["star"]["velocity_array"])
    profile_cfg=_mapping(config["profile"],"profile");low=float(profile_cfg["low_velocity_fraction_of_mean"]);beta_ref=float(profile_cfg["fully_developed_beta"]);alpha_ref=float(profile_cfg["fully_developed_alpha"]);minimum_area=float(_mapping(config["integration"],"integration")["minimum_polygon_area_mm2"])
    profile_rows=[];comparison_rows=[];common_rows=[];primary_cache={}
    def analyze(section,save_grid):
        fem_samples=fem_internal_section_samples(fem_mesh,fem_velocity,section);star_samples,star_extract,star_records,star_vtp,star_diag=star_internal_section_samples(star_mesh,star_velocity,section,minimum_area)
        fem_profile=compute_location_profile("fem",section["name"],fem_samples,section,low);star_profile=compute_location_profile("star",section["name"],star_samples,section,low)
        fem_ex={"intersected_cell_count":fem_samples["element_count"],"generated_polygon_count":fem_samples["element_count"],"valid_polygon_count":fem_samples["element_count"],"invalid_polygon_count":0,"unmapped_polygon_count":0,"intersection_area_mm2":fem_profile["area_mm2"]}
        star_ex={"intersected_cell_count":star_extract["intersected_volume_cell_count"],"generated_polygon_count":star_extract["generated_polygon_count"],"valid_polygon_count":star_extract["valid_polygon_count"],"invalid_polygon_count":star_extract["invalid_polygon_count"],"duplicate_original_cell_count":star_extract["duplicate_original_cell_count"],"multiple_polygon_original_cell_count":star_extract.get("multiple_polygon_original_cell_count",0),"unmapped_polygon_count":star_extract["unmapped_polygon_count"],"intersection_area_mm2":star_extract["native_intersection_area_mm2"],"warning":star_extract["warning"]}
        fem_row=profile_row("fem",section,fem_samples,fem_profile,fem_ex,beta_ref,alpha_ref);star_row=profile_row("star",section,star_samples,star_profile,star_ex,beta_ref,alpha_ref);profiles={("fem",section["name"]):fem_row,("star",section["name"]):star_row}
        grid,arrays=common_grid_for_location(section["name"],section,fem_mesh,pv.PolyData(),fem_velocity,star_mesh,pv.PolyData(),star_velocity,profiles);comparison=comparison_row(section,fem_row,star_row,grid)
        if save_grid and output_cfg.get("save_common_grid_npz",True):
            path=output/"grids"/f"{section['name']}_common_grid.npz";path.parent.mkdir(parents=True,exist_ok=True);metadata={"section_name":section["name"],"z_mm":section["z_mm"],"distance_from_inlet_mm":section["distance_from_inlet_mm"],"formal_integration_uses_common_grid":False,"interpolation_method":"linear_tri_point_visualization"};np.savez_compressed(path,**arrays,metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
        if save_grid and output_cfg.get("save_section_figures",True):plot_section(section,arrays,output,_mapping(config["plot"],"plot"))
        return fem_row,star_row,comparison,arrays,star_diag
    for section in sections:
        fem_row,star_row,comparison,arrays,diagnostic=analyze(section,True);profile_rows.extend((fem_row,star_row));comparison_rows.append(comparison);common_rows.append({key:value for key,value in comparison.items() if key in {"section_name","z_mm","distance_from_inlet_mm","distance_from_inlet_over_diameter","normalized_velocity_relative_l2_percent","normalized_velocity_mae","normalized_velocity_rmse","normalized_velocity_correlation","dimensional_velocity_relative_l2_percent","secondary_velocity_relative_l2_percent","common_valid_area_mm2","common_valid_area_fraction_percent","common_valid_point_count","common_valid_cell_count","interpolation_method","status","warning"}});primary_cache[float(section["z_mm"])]=(fem_row,star_row,comparison)
    sensitivity_raw=[]
    for section in sensitivity_sections:
        position=float(section["z_mm"])
        if abs(float(section["offset_mm"]))<1e-12 and position in primary_cache:fem_row,star_row,comparison=primary_cache[position]
        else:fem_row,star_row,comparison,_,_=analyze(section,False)
        for solver,row in (("fem",fem_row),("star",star_row)):
            sensitivity_raw.append({"solver":solver,"base_position_mm":section["base_position_mm"],"offset_mm":section["offset_mm"],"z_mm":position,"section_name":section["name"],"area_mm2":row["area_mm2"],"flow_magnitude_mm3_s":row["flow_magnitude_mm3_s"],"mean_flow_velocity_mm_s":row["mean_flow_velocity_mm_s"],"beta":row["beta"],"alpha":row["alpha"],"normalized_velocity_std":row["normalized_velocity_std"],"normalized_velocity_relative_l2_percent":comparison["normalized_velocity_relative_l2_percent"],"intersected_cell_count":row["intersected_cell_count"],"generated_polygon_count":row["generated_polygon_count"],"valid_polygon_count":row["valid_polygon_count"],"unmapped_polygon_count":row["unmapped_polygon_count"],"duplicate_original_cell_count":row["duplicate_original_cell_count"],"multiple_polygon_original_cell_count":row.get("multiple_polygon_original_cell_count",0)})
    sensitivity_rows=position_sensitivity_changes(sensitivity_raw,_mapping(config["position_sensitivity"],"position_sensitivity"))
    existing=_mapping(config["existing_boundary_audit"],"existing_boundary_audit");boundary_profiles_raw=_read_csv(resolve_config_path(existing["profile_metrics_summary"],config_path));boundary_profiles={solver:next(row for row in boundary_profiles_raw if row["solver"]==solver and row["location"]=="inlet") for solver in ("fem","star")};boundary_grid=next(row for row in _read_csv(resolve_config_path(existing["common_grid_metrics"],config_path)) if row["location"]=="inlet")
    flow_rows=flow_conservation_rows({solver:float(row["flow_magnitude_mm3_s"]) for solver,row in boundary_profiles.items()},profile_rows);decision=classify_development(comparison_rows,sensitivity_rows,float(boundary_grid["normalized_velocity_relative_l2_percent"]),_mapping(config["decision_thresholds"],"decision_thresholds"))
    generated={"section_series_config":series_cfg,"primary_sections":[_json_ready(section) for section in sections],"position_sensitivity_config":config["position_sensitivity"],"position_sensitivity_sections":[_json_ready(section) for section in sensitivity_sections],"section_library":str(library_path),"sections_json_modified":False}
    (output/"generated_sections.json").write_text(json.dumps(generated,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    _write_csv(output/"profile_development_summary.csv",profile_rows);_write_csv(output/"fem_star_profile_comparison.csv",comparison_rows);_write_csv(output/"flow_conservation_by_section.csv",flow_rows);_write_csv(output/"position_sensitivity_summary.csv",sensitivity_rows);_write_csv(output/"common_grid_metrics_by_section.csv",common_rows)
    plot_cfg=_mapping(config["plot"],"plot")
    if output_cfg.get("save_development_figures",True):plot_development(profile_rows,comparison_rows,boundary_profiles,boundary_grid,output,plot_cfg)
    if output_cfg.get("save_position_sensitivity_figures",True):plot_sensitivity(sensitivity_rows,output,int(plot_cfg.get("dpi",180)))
    audit={"configuration":str(config_path),"data":{"fem":{"path":str(loaded["fem"]["path"]),"time_s":sources["fem"].get("time"),"velocity_array":fem_velocity},"star":{"path":str(loaded["star"]["path"]),"time_index":loaded["star"]["reader_time_point"],"time_s":sources["star"].get("time"),"volume_part":part_name,"velocity_array":star_velocity}},"terminology":{"inlet_boundary":"z=50 mm native boundary used only for boundary-condition discretization audit","near_inlet_internal_section":"configured z<50 mm plane used for internal-profile comparison"},"generated_sections":generated,"methods":{"fem_formal":"existing FEM slice/window plus positive degree-4 triangle quadrature","star_formal":"native volume-cell intersection polygon","common_grid":"linear triangular interpolation with STAR cell-to-point only for visualization/profile comparison","fully_developed_reference":{"beta":beta_ref,"alpha":alpha_ref,"assumed_fully_developed":False}},"inlet_boundary_profiles":boundary_profiles,"inlet_boundary_common_grid":boundary_grid,"profile_development_summary":profile_rows,"comparisons":comparison_rows,"flow_conservation":flow_rows,"position_sensitivity":sensitivity_rows,"decision":decision,"limitations":["Common-grid interpolation may smooth STAR cell values and is not used for formal flow, beta, or alpha.","The fully developed laminar values are references, not assumptions about these sections.","Non-monotonic differences are not interpreted automatically as continuous propagation from the inlet.","No downstream-of-branch section or new CFD calculation was included."]}
    (output/"profile_development_audit.json").write_text(json.dumps(_json_ready(audit),ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(output/"profile_development_audit.md").write_text(build_markdown(audit),encoding="utf-8")
    print(f"Configuration: {config_path}");print(f"Output: {output}")
    for row in comparison_rows:print(f"z={float(row['z_mm']):.3f} mm x/D={float(row['distance_from_inlet_over_diameter']):.3f}: FEM Q={float(row['fem_flow_magnitude_mm3_s']):.9g}, STAR Q={float(row['star_flow_magnitude_mm3_s']):.9g}, normalized L2={float(row['normalized_velocity_relative_l2_percent']):.6g}%, dimensional L2={float(row['dimensional_velocity_relative_l2_percent']):.6g}%")
    print(f"Position sensitivity warnings: {sum(bool(row['position_sensitive']) for row in sensitivity_rows)} / {len(sensitivity_rows)} rows");print(f"Decision: case {decision['case']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
