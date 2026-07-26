# 実行手順

## 1. 前提

実データを用いるときはリポジトリルートから実行する。

```bash
cd /workspace
python -m pip install numpy pandas pyvista vtk matplotlib pillow
```

共有ZIPはレビュー用スナップショットで、`data/`と`output/`を含まない。ZIP展開物だけでは
数値結果を再現できない。STARのEnsight `.case`は参照先geometry/velocityファイルも必要。

## 2. 共通断面の確認

```bash
python -m unittest tests.test_section_config tests.test_section_config_integration -v
```

`config/sections.json`の`default_three_sections`がstraight、upstream、side branchを解決する。

## 3. FEM単独断面

```bash
python scripts/slice_velocity.py --config config/slice_velocity.json
```

point data `solution_velocity`を共通3断面でsliceし、断面別CSV、PNG、overview、metadataを出す。

## 4. STAR単独断面

```bash
python scripts/slice_velocity_star-ccm.py \
  --config config/slice_velocity_star_ccm.json
```

cell data `Velocity`をpointへ変換し、座標・速度をm系からmm系へscaleして同じ断面を出す。

## 5. FEM対STAR共通格子比較

```bash
python scripts/relative_error_colormap_solver_star_ccm.py \
  --config config/relative_error_colormap_solver_star_ccm.json
```

現在の設定は共通3断面、501×401格子、slice後の三角形線形補間、
`grid_output_format=none`で、出力先は
`output/relative_error_colormap/solver_vs_star_ccm_Re100_v2/`。
`summary_metrics.csv`に相対ベクトルL2、common_valid_area、有効点・セル率等を出す。

## 6. Baseline・Case A・Case Bの15秒比較

```bash
python scripts/compare_fem_cases.py --config config/compare_fem_cases.json
```

| case | file | dt [s] | time [s] |
|---|---|---:|---:|
| Baseline | `data/260629_solver/solution_000120.vtu` | 0.125 | 15 |
| Case A | `data/branch_duct_re100_dt003125/solution_000480.vtu` | 0.03125 | 15 |
| Case B | `data/branch_duct_re100_dt00125_vtu20/solution_001200.vtu` | 0.0125 | 15 |

JSONの3比較×3断面を順に処理し、`output/fem_case_comparison_v2/comparison_metrics.csv`へまとめる。
ケースや断面を増やす場合はdata_sources、sections/library、comparisonsだけを追加する。

## 7. 共通格子保存形式

通常（格子なし）:

```bash
python scripts/compare_fem_cases.py --config config/compare_fem_cases.json
```

NPZ例:

```bash
python scripts/compare_fem_cases.py \
  --config config/examples/compare_fem_cases_grid_npz.json
```

CSV.GZ例:

```bash
python scripts/compare_fem_cases.py \
  --config config/examples/compare_fem_cases_grid_csv_gz.json
```

`npz`は`s_grid`、`t_grid`、両速度ベクトル・速度絶対値、差、mask、metadata JSONを含む。
旧`save_grid_csv`設定は互換解釈される。

## 8. 複数sourceの定常判定

```bash
python scripts/check_velocity_steady_state.py \
  --config config/check_velocity_steady_state.json
```

Baseline、Case A、Case Bの`whole_domain`と共通3断面を1秒程度の差で評価する。
出力は`output/steady_state/configured_re100/`。判定は0.5%と0.1%について最初と末尾までの
継続時刻を分ける。

従来CLIの回帰例:

```bash
python scripts/check_velocity_steady_state.py data/260629_solver \
  --dt 0.125 --velocity-name solution_velocity --lag-seconds 1.0 \
  --section-center 10 0 15 --section-normal 1 0 -1 \
  --section-width 12 --section-height 10 \
  --output-dir output/steady_state/legacy_cli_check
```

## 9. FEM/STAR断面積分流量

```bash
python scripts/section_flow.py --config config/section_flow.json
```

Baseline/Case A/Case Bの15秒、FEM/STARの37.5秒について共通3断面を三角形面積積分する。

```text
output/section_flow/re100/flow_summary.csv
output/section_flow/re100/flow_balance_summary.csv
```

signed flowの符号は断面法線に依存する。標準設定はabsolute flowで入口・出口収支と分岐配分を求める。

## 10. 進捗レポート

```bash
python scripts/build_velocity_error_progress_report.py \
  --config config/velocity_error_progress_report.json
```

入力は上記4種類のsummaryだけ。出力:

```text
output/progress_report/re100_configured/
├── report.md
├── report.html
├── report_summary.csv
├── report_inputs.json
├── fem_case_details.csv
└── figures/
```

別ケースはJSONの`cases`へlabel、summaryパス、steady/flow sourceを追加する。Python変更は不要。

## 11. 流跡線

```bash
python scripts/streamlines_solver.py --config config/streamlines_solver.json
python scripts/streamlines_star_ccm.py --config config/streamlines_star_ccm.json
python scripts/compare_streamlines_solver_star_ccm.py \
  --config config/compare_streamlines_solver_star_ccm.json
```

最後はFEM流跡線上へ両速度をsampleする非対称比較で、固定断面比較とは異なる。

## 12. 全テスト・形式検査

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall -q scripts tests
```

追加検査例:

```bash
python -m json.tool config/check_velocity_steady_state.json >/dev/null
python -m json.tool config/section_flow.json >/dev/null
python -m json.tool config/velocity_error_progress_report.json >/dev/null
```

PNGはPillow `Image.verify()`、NPZは`numpy.load(..., allow_pickle=False)`で読込確認する。

## 13. 共有ZIP検査

```bash
unzip -t flow_velocity_analysis_code.zip
sha256sum flow_velocity_analysis_code.zip
```

ZIPにはコード、設定、テスト、マニュアル、analysis summaryのみを含め、大容量data/outputを含めない。


## 14. 境界流量監査

```bash
cd /workspace
python scripts/audit_boundary_flow.py --config config/audit_boundary_flow.json
```

現在の設定はFEM `solution_000300.vtu`とSTAR `duct_test.case` time point 299を37.5秒で比較する。
FEMはGmsh boundary ID、STARはEnsight surface partを使用する。

出力先は`output/boundary_flow_audit/re100_v2/`。既存ファイルがある出力先は上書きしないため、
再実行時はJSONの`output.directory`を未使用名へ変更する。

単体・全体テスト:

```bash
python -m unittest tests.test_audit_boundary_flow tests.test_section_flow -v
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall -q scripts tests
```

確認順序:

1. `boundary_audit.json`のinventoryとselection evidence。
2. `boundary_flow_summary.csv`の面積、signed Q、法線。
3. `boundary_balance_summary.csv`のソルバー内収支。
4. `boundary_vs_internal_sections.csv`のsigned/magnitude差。
5. `star_cell_to_point_comparison.csv`のnative cellと変換point差。
6. `boundary_condition_audit.csv`の確認不能項目。

## 15. STAR native volume-cell-wise内部断面積分

```bash
cd /workspace
python scripts/integrate_volume_cell_sections.py \
  --config config/integrate_volume_cell_sections.json
```

標準出力先`output/native_volume_cell_sections/star_re100/`は既存ファイルがあると停止する。
再実行する場合は既存結果を削除せず、JSONの`output.directory`を新しい名前へ変更する。

確認:

```bash
python -m unittest tests.test_integrate_volume_cell_sections -v
python -m unittest discover -s tests -p 'test_*.py' -v
python -m py_compile \
  scripts/surface_flow.py \
  scripts/integrate_volume_cell_sections.py \
  tests/test_integrate_volume_cell_sections.py
python -m json.tool config/integrate_volume_cell_sections.json >/dev/null
```

主な確認順序:

1. `native_cell_section_flow_summary.csv`の面積、signed flow、polygon/ID count。
2. `polygons/*.csv.gz`が再読込でき、全valid IDが整数・範囲内・一意対応で面積が正であること。
3. `vtp/*.vtp`の元cell IDがpolygon CSV.GZと一致すること。
4. `native_vs_point_vs_boundary.csv`のsignedとmagnitude差、法線符号。
5. `native_cell_section_audit.json/.md`の時刻、part、観測patternと未解決事項。

既存`section_flow.py`と`audit_boundary_flow.py`は変更せず、全テストで回帰確認する。

## 16. 入口・分岐前profile監査

```bash
cd /workspace
python scripts/audit_inlet_upstream_profile.py \
  --config config/audit_inlet_upstream_profile.json

python -m unittest tests.test_audit_inlet_upstream_profile -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

標準出力は`output/inlet_upstream_profile_audit/re100/`。非空出力先は上書きしないため、再実行時は
JSONの`output.directory`を未使用名へ変更する。`profile_metrics_summary.csv`を正式profile値、
`common_grid_metrics.csv`と`grids/*.npz`を補間形状比較として区別する。入口STAR相関は完全一様profile
では数学的に未定義なので判定へ使わず、L2、MAE、alpha/betaを確認する。

## 17. 入口近傍profile発達監査

```bash
cd /workspace
python scripts/audit_inlet_development_profiles.py \
  --config config/audit_inlet_development_profiles.json
python -m unittest tests.test_audit_inlet_development_profiles -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

断面位置は`config/audit_inlet_development_profiles.json`の`section_series.positions_mm`だけを変更する。
直径は`diameter_mm`、感度位置は`position_sensitivity.base_positions_mm`と`offsets_mm`、横軸は
`plot.x_axis`の`distance_mm`または`x_over_d`で指定する。`sections.json`へ系列を追加しない。

標準出力`output/inlet_development_profile_audit/re100/`が非空なら停止する。再実行は新しい出力名を使う。
`profile_development_summary.csv`とflow conservationは正式積分、`common_grid_metrics_by_section.csv`と
NPZは補間形状比較として区別する。z=50境界とz<50内部を用語・location typeで混同しない。



## 18. Re=10 / Re=100 profile・分岐配分比較

```bash
cd /workspace
python scripts/audit_reynolds_profile_comparison.py \
  --config config/audit_reynolds_profile_comparison.json
python -m unittest tests.test_audit_reynolds_profile_comparison -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

標準出力先が非空なら上書きせず停止する。再実行時はJSONで未使用の出力先を指定する。
