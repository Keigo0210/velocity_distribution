#!/usr/bin/env python3
"""Build a configuration-driven velocity-comparison progress report.

Only compact summary CSV files are consumed. Section geometry is resolved via
``section_config.py``; no case, section coordinate, or numerical result is
embedded in this module.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
from html import escape
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from section_config import (  # noqa: E402
    SectionConfigError,
    load_section_library,
    resolve_section,
    resolve_section_set,
)

INPUT_KEYS = (
    "fem_star_metrics",
    "fem_case_metrics",
    "steady_state_metrics",
    "flow_metrics",
)


class ReportConfigError(ValueError):
    """Raised for an invalid report configuration or summary input."""


def resolve_config_relative(value: str | Path, config_path: str | Path) -> Path:
    """Resolve a path exactly relative to the JSON file containing it."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config_path).resolve().parent / path).resolve()


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportConfigError(f"{context} must be a JSON object")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReportConfigError(f"configuration file does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ReportConfigError(f"invalid JSON in {path}: {exc}") from exc
    return dict(_require_mapping(value, "configuration"))


def _input_spec(value: Any, context: str) -> tuple[str, bool]:
    if isinstance(value, str) and value.strip():
        return value, True
    if isinstance(value, Mapping):
        path = value.get("path")
        required = value.get("required", True)
        if not isinstance(path, str) or not path.strip():
            raise ReportConfigError(f"{context}.path must be a non-empty string")
        if not isinstance(required, bool):
            raise ReportConfigError(f"{context}.required must be true or false")
        return path, required
    raise ReportConfigError(
        f"{context} must be a path string or an object with path/required"
    )


def _load_csv(path: Path, context: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ReportConfigError(f"could not read {context} CSV {path}: {exc}") from exc
    if frame.empty:
        raise ReportConfigError(f"{context} CSV is empty: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ReportConfigError(
            f"{context} is missing required columns: {', '.join(missing)}"
        )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_value(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    for value in frame[column]:
        number = _finite(value)
        if number is not None:
            return number
    return None


def _text(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "未評価"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "未評価"
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_text(value) for value in row) + " |")
    return "\n".join(lines)


def load_report_context(config_path: Path) -> dict[str, Any]:
    """Validate configuration, resolve sections, and load configured summaries."""

    config = _load_json(config_path)
    cases = _require_mapping(config.get("cases"), "cases")
    if not cases:
        raise ReportConfigError("cases must not be empty")
    report = _require_mapping(config.get("report"), "report")
    execution = _require_mapping(config.get("execution", {}), "execution")
    fail_missing = execution.get("fail_on_missing_required_input", True)
    if not isinstance(fail_missing, bool):
        raise ReportConfigError(
            "execution.fail_on_missing_required_input must be true or false"
        )

    library_value = config.get("section_library")
    if not isinstance(library_value, str) or not library_value.strip():
        raise ReportConfigError("section_library must be a non-empty path")
    library_path = resolve_config_relative(library_value, config_path)
    try:
        library = load_section_library(library_path)
    except SectionConfigError as exc:
        raise ReportConfigError(str(exc)) from exc

    include = report.get("include_sections")
    if include is None:
        set_name = config.get("section_set")
        if not isinstance(set_name, str) or not set_name.strip():
            raise ReportConfigError(
                "report.include_sections or section_set must be provided"
            )
        try:
            sections = resolve_section_set(set_name, library)
        except SectionConfigError as exc:
            raise ReportConfigError(str(exc)) from exc
    else:
        if not isinstance(include, list) or not include or not all(
            isinstance(name, str) and name.strip() for name in include
        ):
            raise ReportConfigError(
                "report.include_sections must be a non-empty array of names"
            )
        try:
            sections = [resolve_section(name, library) for name in include]
        except SectionConfigError as exc:
            raise ReportConfigError(str(exc)) from exc

    output_value = report.get("output_directory")
    if not isinstance(output_value, str) or not output_value.strip():
        raise ReportConfigError("report.output_directory must be a non-empty path")
    output_dir = resolve_config_relative(output_value, config_path)

    loaded_cases: dict[str, Any] = {}
    input_records: list[dict[str, Any]] = []
    for case_name, raw_case in cases.items():
        if not isinstance(case_name, str) or not case_name.strip():
            raise ReportConfigError("case names must be non-empty strings")
        case = dict(_require_mapping(raw_case, f"cases.{case_name}"))
        label = case.get("label", case_name)
        if not isinstance(label, str) or not label.strip():
            raise ReportConfigError(f"cases.{case_name}.label must be non-empty")
        frames: dict[str, pd.DataFrame | None] = {}
        paths: dict[str, Path | None] = {}
        for key in INPUT_KEYS:
            if key not in case:
                frames[key] = None
                paths[key] = None
                input_records.append(
                    {
                        "case_name": case_name,
                        "input_type": key,
                        "configured": False,
                        "required": False,
                        "path": None,
                        "exists": False,
                        "row_count": None,
                        "status": "not_configured",
                    }
                )
                continue
            raw_path, required = _input_spec(case[key], f"cases.{case_name}.{key}")
            path = resolve_config_relative(raw_path, config_path)
            paths[key] = path
            if not path.is_file():
                record = {
                    "case_name": case_name,
                    "input_type": key,
                    "configured": True,
                    "required": required,
                    "path": str(path),
                    "exists": False,
                    "row_count": None,
                    "status": "missing",
                }
                input_records.append(record)
                if required and fail_missing:
                    raise ReportConfigError(
                        f"required input is missing for case '{case_name}': {key} = {path}"
                    )
                frames[key] = None
                continue
            frame = _load_csv(path, f"{case_name}.{key}")
            frames[key] = frame
            input_records.append(
                {
                    "case_name": case_name,
                    "input_type": key,
                    "configured": True,
                    "required": required,
                    "path": str(path),
                    "exists": True,
                    "row_count": int(len(frame)),
                    "status": "loaded",
                }
            )
        loaded_cases[case_name] = {
            "label": label,
            "config": case,
            "frames": frames,
            "paths": paths,
        }

    primary = report.get("steady_thresholds_percent", [])
    if not isinstance(primary, list) or any(_finite(value) is None for value in primary):
        raise ReportConfigError("report.steady_thresholds_percent must be numeric array")

    return {
        "config": config,
        "config_path": config_path.resolve(),
        "library_path": library_path,
        "sections": sections,
        "cases": loaded_cases,
        "report": dict(report),
        "output_dir": output_dir,
        "input_records": input_records,
    }


def _section_rows(frame: pd.DataFrame | None, section_name: str) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    for column in ("section_name", "section"):
        if column in frame.columns:
            return frame[frame[column].astype(str) == section_name].copy()
    raise ReportConfigError("summary CSV has neither section_name nor section column")


def _steady_rows(
    frame: pd.DataFrame | None, section_name: str, source_name: str | None
) -> pd.DataFrame:
    rows = _section_rows(frame, section_name)
    if not rows.empty and source_name is not None:
        _require_columns(rows, ["source_name"], "steady-state summary")
        rows = rows[rows["source_name"].astype(str) == source_name].copy()
    return rows


def _threshold_value(rows: pd.DataFrame, threshold: float, column: str) -> float | None:
    if rows.empty or "threshold" not in rows.columns or column not in rows.columns:
        return None
    values = pd.to_numeric(rows["threshold"], errors="coerce")
    matches = rows[np.isclose(values, threshold, rtol=0.0, atol=1.0e-9)]
    return _first_value(matches, column)


def _flow_row(
    frame: pd.DataFrame | None, section_name: str, source_name: str | None
) -> pd.DataFrame:
    rows = _section_rows(frame, section_name)
    if not rows.empty and source_name is not None:
        _require_columns(rows, ["source_name"], "flow summary")
        rows = rows[rows["source_name"].astype(str) == source_name].copy()
    return rows.head(1)


def build_section_record(
    case_name: str,
    case: Mapping[str, Any],
    section: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one machine-readable case/section row from summary CSV data."""

    frames = _require_mapping(case.get("frames"), "case.frames")
    case_config = _require_mapping(case.get("config"), "case.config")
    section_name = str(section["name"])
    fem_star = _section_rows(frames.get("fem_star_metrics"), section_name)
    fem_cases = _section_rows(frames.get("fem_case_metrics"), section_name)
    steady_source = case_config.get("steady_source")
    if steady_source is not None and not isinstance(steady_source, str):
        raise ReportConfigError("steady_source must be a string")
    steady = _steady_rows(frames.get("steady_state_metrics"), section_name, steady_source)

    flow_sources = _require_mapping(case_config.get("flow_sources", {}), "flow_sources")
    fem_flow = _flow_row(frames.get("flow_metrics"), section_name, flow_sources.get("fem"))
    star_flow = _flow_row(frames.get("flow_metrics"), section_name, flow_sources.get("star"))

    thresholds = [float(value) for value in report.get("steady_thresholds_percent", [])]
    record: dict[str, Any] = {
        "case_name": case_name,
        "case_label": case["label"],
        "section_name": section_name,
        "section_label": section["label"],
        "center_x_mm": float(section["center"][0]),
        "center_y_mm": float(section["center"][1]),
        "center_z_mm": float(section["center"][2]),
        "normal_x": float(section["normal"][0]),
        "normal_y": float(section["normal"][1]),
        "normal_z": float(section["normal"][2]),
        "fem_star_relative_l2_percent": _first_value(fem_star, "relative_l2_error"),
        "fem_star_common_valid_area_mm2": _first_value(fem_star, "common_valid_area"),
        "fem_star_valid_point_fraction": _first_value(fem_star, "valid_point_fraction"),
        "fem_star_valid_cell_fraction": _first_value(fem_star, "valid_cell_fraction"),
        "fem_case_comparison_count": int(len(fem_cases)),
        "fem_case_relative_l2_min_percent": None,
        "fem_case_relative_l2_mean_percent": None,
        "fem_case_relative_l2_max_percent": None,
    }
    if not fem_cases.empty:
        _require_columns(fem_cases, ["relative_l2_error"], "FEM-case summary")
        values = pd.to_numeric(fem_cases["relative_l2_error"], errors="coerce").dropna()
        if not values.empty:
            record.update(
                {
                    "fem_case_relative_l2_min_percent": float(values.min()),
                    "fem_case_relative_l2_mean_percent": float(values.mean()),
                    "fem_case_relative_l2_max_percent": float(values.max()),
                }
            )
    star_error = record["fem_star_relative_l2_percent"]
    fem_error = record["fem_case_relative_l2_max_percent"]
    record["fem_case_max_to_fem_star_ratio"] = (
        fem_error / star_error
        if fem_error is not None and star_error is not None and star_error != 0.0
        else None
    )

    for index, threshold in enumerate(thresholds, start=1):
        prefix = f"steady_threshold_{index}"
        record[f"{prefix}_percent"] = threshold
        record[f"{prefix}_first_below_s"] = _threshold_value(
            steady, threshold, "first_below_threshold_time"
        )
        record[f"{prefix}_continuous_below_s"] = _threshold_value(
            steady, threshold, "continuous_below_threshold_time"
        )

    for role, rows in (("fem", fem_flow), ("star", star_flow)):
        record[f"{role}_flow_source"] = flow_sources.get(role)
        for output_name, source_column in (
            ("section_area_mm2", "section_area_mm2"),
            ("signed_flow_mm3_s", "signed_flow_rate_mm3_s"),
            ("absolute_flow_mm3_s", "absolute_flow_rate_mm3_s"),
            ("area_mean_normal_velocity_mm_s", "area_mean_normal_velocity_mm_s"),
            ("balance_error_percent", "flow_balance_error_percent"),
        ):
            record[f"{role}_{output_name}"] = _first_value(rows, source_column)

    fraction_column = f"{section_name}_fraction"
    record["fem_branch_fraction"] = _first_value(fem_flow, fraction_column)
    record["star_branch_fraction"] = _first_value(star_flow, fraction_column)
    return record


def _comparison_details(context: Mapping[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    section_names = {str(section["name"]) for section in context["sections"]}
    for case_name, case in context["cases"].items():
        frame = case["frames"].get("fem_case_metrics")
        if frame is None:
            continue
        _require_columns(
            frame,
            ["relative_l2_error"],
            f"{case_name}.fem_case_metrics",
        )
        section_column = "section_name" if "section_name" in frame.columns else "section"
        comparison_column = "comparison_name" if "comparison_name" in frame.columns else "comparison"
        _require_columns(frame, [section_column, comparison_column], "FEM-case summary")
        for row in frame.itertuples(index=False):
            values = row._asdict()
            section_name = str(values[section_column])
            if section_name not in section_names:
                continue
            records.append(
                {
                    "case_name": case_name,
                    "case_label": case["label"],
                    "comparison": values[comparison_column],
                    "section_name": section_name,
                    "source_1": values.get("source_1"),
                    "source_2": values.get("source_2"),
                    "time_source_1_s": values.get("time_source_1"),
                    "time_source_2_s": values.get("time_source_2"),
                    "relative_l2_percent": values["relative_l2_error"],
                    "common_valid_area_mm2": values.get("common_valid_area"),
                    "valid_point_fraction": values.get("valid_point_fraction"),
                }
            )
    return pd.DataFrame(records)


def _source_table(context: Mapping[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for case_name, case in context["cases"].items():
        for input_type, path in case["paths"].items():
            records.append(
                {
                    "case": case["label"],
                    "summary type": input_type,
                    "path": str(path) if path is not None else "未設定",
                    "status": "読込済み" if case["frames"].get(input_type) is not None else "未評価",
                }
            )
    return pd.DataFrame(records)


def _data_source_table(context: Mapping[str, Any]) -> pd.DataFrame:
    """Collect configured/observed data sources without assuming source names."""

    records: list[dict[str, Any]] = []
    for case_name, case in context["cases"].items():
        case_label = case["label"]
        fem_cases = case["frames"].get("fem_case_metrics")
        if fem_cases is not None:
            for role in ("source_1", "source_2"):
                if role not in fem_cases.columns:
                    continue
                for row in fem_cases.drop_duplicates(subset=[role]).to_dict("records"):
                    records.append(
                        {
                            "case": case_label,
                            "source": row.get(role),
                            "label": row.get(f"{role}_label"),
                            "path": row.get(f"{role}_path"),
                            "dt [s]": row.get(f"{role}_dt_s", row.get(f"dt_{role}")),
                            "time [s]": row.get(f"{role}_time_s", row.get(f"time_{role}")),
                            "observed in": "fem_case_metrics",
                        }
                    )
        fem_star = case["frames"].get("fem_star_metrics")
        if fem_star is not None:
            for role in ("source_1", "source_2"):
                if role not in fem_star.columns:
                    continue
                for row in fem_star.drop_duplicates(subset=[role]).to_dict("records"):
                    records.append(
                        {
                            "case": case_label,
                            "source": row.get(role),
                            "label": None,
                            "path": None,
                            "dt [s]": None,
                            "time [s]": row.get("time_value"),
                            "observed in": "fem_star_metrics",
                        }
                    )
        flow = case["frames"].get("flow_metrics")
        if flow is not None and "source_name" in flow.columns:
            for row in flow.drop_duplicates(subset=["source_name"]).to_dict("records"):
                records.append(
                    {
                        "case": case_label,
                        "source": row.get("source_name"),
                        "label": row.get("source_label"),
                        "path": row.get("input_path"),
                        "dt [s]": None,
                        "time [s]": row.get("time_s"),
                        "observed in": "flow_metrics",
                    }
                )
    if not records:
        return pd.DataFrame(columns=["case", "source", "label", "path", "dt [s]", "time [s]", "observed in"])
    return pd.DataFrame(records).drop_duplicates().reset_index(drop=True)


def _section_table(context: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "section": section["name"],
                "label": section["label"],
                "center [mm]": ", ".join(_text(float(x)) for x in section["center"]),
                "normalized normal": ", ".join(_text(float(x)) for x in section["normal"]),
            }
            for section in context["sections"]
        ]
    )


def _build_interpretation(summary: pd.DataFrame, comparisons: pd.DataFrame) -> str:
    star_values = pd.to_numeric(
        summary.get("fem_star_relative_l2_percent", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    fem_values = pd.to_numeric(
        comparisons.get("relative_l2_percent", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if star_values.empty or fem_values.empty:
        return "FEM内差とFEM対外部ソルバー差の定量比較は、必要なsummaryがないため未評価です。"
    largest_fem = float(fem_values.max())
    smallest_star = float(star_values.min())
    ratio = largest_fem / smallest_star if smallest_star != 0.0 else math.inf
    if largest_fem < smallest_star:
        return (
            f"入力summaryでは、評価対象のFEM同士の最大相対ベクトルL2差は"
            f"{largest_fem:.6g}%で、FEM対外部ソルバーの最小値{smallest_star:.6g}%の"
            f"{ratio:.3g}倍です。評価した断面・時刻では時間刻み依存性は相対的に小さく、"
            "FEM対外部ソルバー差を時間刻みだけで説明することは難しいと考えられます。"
            "ただし、これは全領域・全時刻の厳密な時間収束証明ではありません。"
        )
    return (
        f"入力summaryではFEM同士の最大差{largest_fem:.6g}%がFEM対外部ソルバーの"
        f"最小差{smallest_star:.6g}%以上でした。評価範囲だけから時間刻み依存性が小さいとは"
        "判断できません。全領域・全時刻の厳密な時間収束証明でもありません。"
    )


def build_markdown(
    context: Mapping[str, Any], summary: pd.DataFrame, comparisons: pd.DataFrame
) -> str:
    title = str(context["report"].get("title", "Velocity comparison progress report"))
    sections = _section_table(context)
    sources = _source_table(context)
    data_sources = _data_source_table(context)
    interpretation = _build_interpretation(summary, comparisons)

    fem_star_columns = [
        "case_label", "section_name", "fem_star_relative_l2_percent",
        "fem_star_common_valid_area_mm2", "fem_star_valid_point_fraction",
        "fem_star_valid_cell_fraction",
    ]
    steady_columns = ["case_label", "section_name"] + [
        column for column in summary.columns if column.startswith("steady_threshold_")
    ]
    flow_columns = [
        "case_label", "section_name", "fem_flow_source", "fem_signed_flow_mm3_s",
        "fem_absolute_flow_mm3_s", "star_flow_source", "star_signed_flow_mm3_s",
        "star_absolute_flow_mm3_s", "fem_branch_fraction", "star_branch_fraction",
        "fem_balance_error_percent", "star_balance_error_percent",
    ]
    missing = [record for record in context["input_records"] if record["status"] != "loaded"]
    missing_text = (
        markdown_table(pd.DataFrame(missing)) if missing else "全設定入力を読み込みました。"
    )
    partition_mode = context["report"].get("flow_partition_mode", "not specified")
    return f"""# {title}

作成日: {date.today().isoformat()}

## 1. 入力ケース・summary一覧

{markdown_table(sources)}

本レポートは巨大な共通格子CSVではなく、設定されたsummary CSVだけを入力にします。

### summaryから確認したデータソース

{markdown_table(data_sources)}

同じデータソースが複数summaryに現れる場合は、どのsummaryで確認した情報かを`observed in`に示します。値がsummaryに記録されていない項目は「未評価」です。

## 2. 使用断面

{markdown_table(sections)}

断面の位置・正規化法線は共通断面ライブラリ `{context['library_path']}` から解決しました。

## 3. FEM対外部ソルバーの断面別誤差

{markdown_table(summary[fem_star_columns])}

`common_valid_area`は共通格子セルの4頂点がすべて有効なセル面積の総和です。`valid_point_fraction`は格子点率、`valid_cell_fraction`はセル率であり、点数比に矩形全面積を掛けた旧近似とは区別します。

## 4. FEM対FEMの時間刻み依存性

{markdown_table(comparisons)}

{interpretation}

## 5. 定常到達時刻

{markdown_table(summary[steady_columns])}

`first_below`は最初にしきい値を下回った時刻、`continuous_below`はその時刻以降解析末尾まで下回り続けた最初の時刻です。断面値は共通断面上の評価、全領域値は別途steady-state summaryの`whole_domain`行で確認します。

## 6. 断面積分流量・分岐配分・保存誤差

{markdown_table(summary[flow_columns])}

流量は断面三角形ごとに `Q = area * mean(vertex normal velocity)` を積算した体積流量です。signed flowの符号は設定断面法線に対する向きを示し、absolute flowはその絶対値です。分岐配分モードは `{partition_mode}` で、表のfractionは面積積分流量から得ています。保存誤差は設定した入口総流量と出口総流量の差です。

旧方式の平均法線速度比は体積流量ではなく、断面積が異なる場合は流量比と一致しないため、本レポートの配分には使用していません。

## 7. 注意事項・制約

- 結論は設定された断面、時刻、summaryに限定されます。
- FEM対FEM差が小さいことは、全領域・全時刻の厳密な時間収束証明ではありません。
- セルデータを点データへ変換した入力では、その平均化が急勾配を平滑化する可能性があります。
- 断面法線の向きによりsigned flowの符号が変わります。収支・配分でabsoluteまたはorientation-adjustedを使う選択は設定に従います。
- 入力不足をゼロとして補うことはありません。未取得値は「未評価」として保持します。

## 8. 入力欠損状態

{missing_text}
"""


def _save_figures(summary: pd.DataFrame, comparisons: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    valid = summary.dropna(subset=["fem_star_relative_l2_percent"])
    if not valid.empty:
        labels = [f"{row.case_label}\n{row.section_name}" for row in valid.itertuples()]
        fig, ax = plt.subplots(figsize=(max(7.0, 1.25 * len(valid)), 4.5), constrained_layout=True)
        ax.bar(np.arange(len(valid)), valid["fem_star_relative_l2_percent"], color="#4C78A8")
        ax.set_xticks(np.arange(len(valid)), labels, rotation=25, ha="right")
        ax.set_ylabel("Relative vector L2 difference [%]")
        ax.set_title("Section-wise FEM versus external solver")
        ax.grid(axis="y", alpha=0.25)
        path = figures_dir / "fem_star_relative_l2.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)

    if not comparisons.empty:
        fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(comparisons)), 4.7), constrained_layout=True)
        labels = [f"{row.comparison}\n{row.section_name}" for row in comparisons.itertuples()]
        ax.bar(np.arange(len(comparisons)), comparisons["relative_l2_percent"], color="#F58518")
        ax.set_xticks(np.arange(len(comparisons)), labels, rotation=35, ha="right")
        ax.set_ylabel("Relative vector L2 difference [%]")
        ax.set_title("FEM time-step comparison")
        ax.grid(axis="y", alpha=0.25)
        path = figures_dir / "fem_case_relative_l2.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)

    flow_rows: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        for role in ("fem", "star"):
            value = row.get(f"{role}_absolute_flow_mm3_s")
            if _finite(value) is not None:
                flow_rows.append(
                    {
                        "label": f"{row['case_label']}\n{row['section_name']}\n{role}",
                        "value": float(value),
                    }
                )
    if flow_rows:
        fig, ax = plt.subplots(figsize=(max(7.0, 0.75 * len(flow_rows)), 4.8), constrained_layout=True)
        ax.bar(np.arange(len(flow_rows)), [row["value"] for row in flow_rows], color="#54A24B")
        ax.set_xticks(np.arange(len(flow_rows)), [row["label"] for row in flow_rows], rotation=35, ha="right")
        ax.set_ylabel("Absolute integrated flow [mm³/s]")
        ax.set_title("Section-integrated flow rates")
        ax.grid(axis="y", alpha=0.25)
        path = figures_dir / "section_flow_rates.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
    return paths


def _build_html(title: str, markdown: str, figures: list[Path], output_dir: Path) -> str:
    images = "\n".join(
        f'<figure><img src="{escape(str(path.relative_to(output_dir)))}" alt="{escape(path.stem)}"></figure>'
        for path in figures
    )
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:auto;padding:2rem;line-height:1.6}}pre{{white-space:pre-wrap;background:#f6f8fa;padding:1rem}}img{{max-width:100%;height:auto}}figure{{margin:2rem 0}}</style></head>
<body><h1>{escape(title)}</h1>{images}<h2>Markdown report</h2><pre>{escape(markdown)}</pre></body></html>"""


def generate_report(config_path: str | Path) -> dict[str, Any]:
    """Generate configured report files and return paths/data for tests and callers."""

    context = load_report_context(Path(config_path))
    records = [
        build_section_record(case_name, case, section, context["report"])
        for case_name, case in context["cases"].items()
        for section in context["sections"]
    ]
    summary = pd.DataFrame(records)
    comparisons = _comparison_details(context)
    output_dir: Path = context["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "report_summary.csv"
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    if not comparisons.empty:
        comparisons.to_csv(output_dir / "fem_case_details.csv", index=False)

    inputs_payload = {
        "config_path": str(context["config_path"]),
        "section_library": str(context["library_path"]),
        "output_directory": str(output_dir),
        "sections": [section["name"] for section in context["sections"]],
        "inputs": context["input_records"],
    }
    inputs_path = output_dir / "report_inputs.json"
    inputs_path.write_text(
        json.dumps(inputs_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    markdown = build_markdown(context, summary, comparisons)
    report_path = output_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")

    dpi = context["report"].get("dpi", 180)
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ReportConfigError("report.dpi must be a positive integer")
    figures = _save_figures(summary, comparisons, output_dir, dpi)
    title = str(context["report"].get("title", "Velocity comparison progress report"))
    html_path = output_dir / "report.html"
    html_path.write_text(_build_html(title, markdown, figures, output_dir), encoding="utf-8")

    return {
        "output_dir": output_dir,
        "report": report_path,
        "summary": summary_path,
        "inputs": inputs_path,
        "html": html_path,
        "figures": figures,
        "summary_frame": summary,
        "comparison_frame": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a config-driven velocity-comparison progress report."
    )
    parser.add_argument("--config", type=Path, required=True, help="Report JSON configuration")
    args = parser.parse_args()
    try:
        result = generate_report(args.config)
    except (ReportConfigError, SectionConfigError) as exc:
        parser.error(str(exc))
    print(f"Report directory: {result['output_dir']}")
    print(f"Markdown: {result['report']}")
    print(f"Summary: {result['summary']}")
    print(f"Inputs: {result['inputs']}")
    print(f"HTML: {result['html']}")


if __name__ == "__main__":
    main()
