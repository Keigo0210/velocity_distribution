# 流速分布解析コード構成

## 1. 対象と共有物の目的

- 実リポジトリルート: `/workspace`
- 実データ: `/workspace/data`
- 共有ZIP: ChatGPTへコード、設定、テスト、文書の実装状況を渡すスナップショット

共有ZIPは解析環境・大容量入力・解析出力を含む完全再現パッケージではない。実データ検証は
サーバー上の`/workspace/data`を使う。ZIPから`data/`、`output/`、`.git`、cache、仮想環境、
生成物、秘密情報を除外する。STAR Ensight `.case`は参照するgeometry/velocityファイルが
なければ単独再現できない。

## 2. 解析系統

1. FEM/STARの単独断面抽出と可視化
2. FEM対FEM、FEM対STARの共通格子比較
3. 時系列FEMの全領域・共通断面定常判定
4. FEM/STARの実断面三角形による体積流量積分
5. summary CSVからの設定駆動進捗レポート
6. FEM/STAR時系列の流跡線と流跡線上誤差

## 3. 共通断面ライブラリ

`config/sections.json`が断面を一元管理し、`scripts/section_config.py`が読込・検証・解決する。
標準セット`default_three_sections`は次の順である。

| name | center [mm] | input normal | normalized normal |
|---|---|---|---|
| straight_z10 | (0, 0, 10) | (0, 0, 1) | (0, 0, 1) |
| upstream_z30 | (0, 0, 30) | (0, 0, 1) | (0, 0, 1) |
| side_branch | (10, 0, 15) | (1, 0, -1) | (0.707107, 0, -0.707107) |

各断面はlabel、center、normal、width、height、grid_resolution、flip_s、flip_tを持ち、
任意で`s_axis`と`t_axis`を明示できる。省略時は共通規則で自動生成する。side_branchは
`s=(0,-1,0)`、`t=(-0.707107,0,-0.707107)`となり、既存表示方向を維持する。

解決優先順位は`section_library + section_set`、`section_library + section_names`、
JSON内`sections`、単一`section`、解釈可能な旧形式である。ゼロ法線、非正寸法、2未満の格子数、
非直交な明示軸、不明な参照名は処理前にエラーになる。

## 4. 基準ケースと15秒比較

| case | directory | dt [s] | file | time [s] |
|---|---|---:|---|---:|
| Baseline | `data/260629_solver` | 0.125 | `solution_000120.vtu` | 15 |
| Case A | `data/branch_duct_re100_dt003125` | 0.03125 | `solution_000480.vtu` | 15 |
| Case B | `data/branch_duct_re100_dt00125_vtu20` | 0.0125 | `solution_001200.vtu` | 15 |

VTUには物理時刻FieldDataがないため、step番号×dtと各`result.txt`を照合した。3つとも
19,275点、48,934セルで、速度はpoint dataの`solution_velocity`、座標mm、速度mm/sである。
Case Bは20 solver stepごとのVTU出力、すなわち0.25秒間隔である。

## 5. 処理フロー

### 5.1 共通格子比較

```text
source 1 mesh ─ slice ─ project(s,t) ─ triangulate ─ linear interpolate ┐
                                                                       ├ compare on one grid
source 2 mesh ─ slice ─ project(s,t) ─ triangulate ─ linear interpolate ┘
```

point dataを使用し、設定がcell associationならpointへ変換する。単位scaleを適用し、重複断面点を
許容差内で統合する。各速度成分を`matplotlib.tri.LinearTriInterpolator`で同一格子へ補間し、
共通有効点だけでベクトルL2、MAE、最大差、U95正規化値等を求める。

### 5.2 定常判定

```text
solution_<step>.vtu + configured dt
  -> physical time
  -> whole-domain mesh identity check
  -> whole-domain direct vector difference
  -> common section slices at every time
  -> vector relative L2 over configured interval
  -> first/continuous threshold times
```

`whole_domain`と3断面を1回で処理する。断面はnative slice点を時刻間で照合する。既存side_branch
単独方式と同じ数値定義を維持している。

### 5.3 断面流量

```text
mesh -> slice -> width/height clip -> triangulate
     -> Σ triangle_area × mean(vertex velocity · normal)
```

signed flowは法線向きを保持し、absolute flowは絶対値。標準分岐配分は入口・出口のabsolute flowで
計算する。平均法線速度比ではなく真の面積積分体積流量である。

### 5.4 レポート

4つのsummary CSVだけを読み、設定された全ケース・全断面を順に処理する。断面別機械可読CSV、
Markdown、HTML、入力manifest、PNGを作る。ケース名・座標・結果数値はPythonへ固定しない。

## 6. 主要スクリプト

- `section_config.py`: 共通断面library、set、軸、検証。
- `slice_velocity.py`: FEM native sliceのCSV/PNG/overview。
- `slice_velocity_star-ccm.py`: Ensight読込、cell-to-point、単位変換後の同等出力。
- `compare_fem_cases.py`: JSONの任意data sources/comparisons/sectionsによるFEM対FEM比較。
- `relative_error_colormap_solver_star_ccm.py`: FEM対STARの複数断面共通格子比較。
- `common_grid_output.py`: 有効セル面積とnone/csv/csv.gz/npz出力の共通処理。
- `check_velocity_steady_state.py`: 複数sourceの全領域・複数断面定常判定。旧CLIも維持。
- `section_flow.py`: VTU/Ensight断面の三角形面積積分流量、配分、収支。
- `build_velocity_error_progress_report.py`: summaryのみを読む設定駆動レポート。
- `streamlines_common.py`: 時系列読込、単位変換、sample、pathline、各種出力。
- `streamlines_solver.py` / `streamlines_star_ccm.py`: 各速度場の流跡線。
- `compare_streamlines_solver_star_ccm.py`: FEM流跡線上の非対称なFEM/STAR比較。
- `identify_mesh_boundaries.py`: Gmsh境界physical groupの対話設定。

## 7. 入力とdata association

- FEM VTU: `solution_velocity`、point data、mm、mm/s。
- STAR Ensight: `Velocity`、cell data、m、m/s。比較前にpointへ変換し1000倍する。
- すべての比較data sourceはpath、velocity_array、data_association、length/velocity scaleをJSONから読む。
- cell指定は将来のFEMや別sourceでもpointへ変換できる。

## 8. 出力と面積

FEM対FEMは`comparison_metrics.csv`、FEM対STARは`summary_metrics.csv`をルートへ出す。
断面別にmetrics、PNG、任意VTP/格子を出す。

`common_valid_area`は4頂点すべて有効な格子セルの`ds×dt`総和である。
`legacy_valid_area_estimate`は有効点率×矩形全面積の旧近似で、数値差の追跡用に残す。

`grid_output_format`:

- `none`: 通常実行。格子を保存しない。
- `csv`: 非圧縮で可読だが巨大。
- `csv.gz`: CSV互換の圧縮。
- `npz`: 配列＋metadataの圧縮形式。

旧`save_grid_csv`は後方互換で解釈する。

## 9. 定常判定の定義

```text
relative_L2 [%] = ||u(t)-u(t-ΔT)||₂ / ||u(t)||₂ × 100
```

- first: 最初にしきい値未満となる時刻。
- continuous: その後解析末尾までしきい値未満を維持する最初の時刻。

標準設定は約1秒差、0.5%と0.1%を評価する。出力はsource/region別CSV・PNGと
`steady_state_summary.csv`。

## 10. 流量の符号、配分、保存誤差

現共通法線ではupstream/straightのsigned Qは負、side branchは正である。これは物理流向と
法線方向の関係である。標準`absolute`モードでは符号を配分から外す。

```text
outlet_fraction = abs(Q_outlet) / Σ abs(Q_outlets)
balance_error = Σ abs(Q_outlets) - Σ abs(Q_inlets)
```

orientation-adjustedモードの係数は`config/section_flow.json`で設定する。

## 11. レポート入力・出力

`config/velocity_error_progress_report.json`はcases、4summary、steady/flow source、断面、しきい値、
出力先を定義する。相対パスはこのJSON基準。必須欠損は設定に応じてエラー、任意欠損は未評価。

出力:

- `report.md`
- `report.html`
- `report_summary.csv`
- `report_inputs.json`
- `fem_case_details.csv`
- `figures/*.png`

## 12. 現在コードで設定に移した値

ケース名、VTU/CASEパス、dt、比較時刻、速度配列、association、単位、断面中心・法線・範囲、
格子数、s/t軸、flip、比較pair、reference、出力先、レポートsummaryパスはJSON側にある。
Pythonロジックは`baseline`等の名称や特定Reを前提に分岐しない。

## 13. 既知の制約・今後

- VTU時刻はFieldDataでなくstep×dtから復元する。
- 定常判定断面はnative slice点一致を要求し、変形メッシュには固定格子化が必要。
- STAR cell-to-point変換が壁近傍勾配を平均化する可能性がある。
- 単独slice CSVはnative点配置なので、異種メッシュ間の直接減算には使えない。
- 流跡線比較はFEM流跡線上という非対称評価。
- ZIPに実データとoutputがないため単体で数値再現できない。
- 次の改善候補は、VTU FieldData時刻の保存、時刻指定によるファイル自動選択、変形メッシュ断面定常判定。


## 14. ステップ4A境界流量監査

`scripts/surface_flow.py`がpoint三角形積分、native surface cell積分、法線正規化、単位換算、
既存section収支を共通提供する。`section_flow.py`は同じ関数をimportし、積分式を重複させない。

`scripts/audit_boundary_flow.py`は`config/audit_boundary_flow.json`に定義された任意source/boundaryを
順に処理する。対応selection methodは`boundary_id`、`part_name`、`block_name`、
`explicit_surface_file`、`geometric_plane`。今回の実データはnative tag/partを持つためplane抽出不要。

FEMの根拠:

- 結果: `solution_000300.vtu`、37.5 s、point `solution_velocity`、mm/mm/s。
- 19,275点、48,934 volume cells（33,283 tetra、15,651 wedge）、boundary tagなし。
- 元`branch_duct.msh`に10 inlet、20 outlet_main、30 outlet_branch、40 wall。
- 元meshのvolume cell数・型は結果VTUと一致し、3 boundaryの全点が結果点へ距離0で対応。

STARの根拠:

- `duct_test.case` time point 299、37.5 s、cell `Velocity`、m/m/s。
- volume part `領域`とsurface part wall/inlet1/outlet1/outlet2。
- inlet/outlet partsはPOLYGONとQUADで、native cell velocityを直接積分可能。

監査は外向き法線をJSONへ明示し、raw signed Qを保存する。STARはnative cell積分を主値とし、
surface cell-to-point後のpoint積分を影響確認値として併記する。内部断面summaryとはsignedとmagnitudeを
別々に比較する。出力先にファイルがある場合は上書きしない。

主出力は`boundary_flow_summary.csv`、`boundary_balance_summary.csv`、
`boundary_vs_internal_sections.csv`、`star_cell_to_point_comparison.csv`、
`boundary_condition_audit.csv`、`boundary_audit.json/.md`である。

## 15. ステップ4B-1 native volume-cell-wise内部断面流量

`scripts/integrate_volume_cell_sections.py`は`config/integrate_volume_cell_sections.json`を読み、
STAR Ensight volume partのcell `Velocity`をpoint dataへ変換せず、共通3断面と各volume cellの
交差polygonを積分する。`surface_flow.py`に元cell ID付与、矩形polygon clip、shoelace面積、
native cell積分を共通化し、`section_flow.py`と`audit_boundary_flow.py`の既存式・出力は変更しない。

処理フロー:

```text
STAR volume part + native cell Velocity
  -> attach exact int64 original cell ID
  -> plane cut from section library
  -> verify inherited ID and look up original cell Velocity
  -> project polygon to section s,t
  -> clip to configured width/height rectangle
  -> polygon area × dot(cell Velocity, normalized normal)
  -> summary + polygon CSV.GZ + VTP
  -> compare with current point-converted internal + native boundary
```

STAR sourceは`duct_test.case`、time index 299 = 37.5 s、volume part `領域`、m/m/s入力を
mm/mm/sへ換算する。断面座標はPythonへ固定せず`sections.json`の`upstream_z30`、
`straight_z10`、`side_branch`を解決する。比較対応もJSONの`section_to_boundary`に置く。

定量的なSTAR内部断面流量にはnative volume-cell-wise値を優先し、point-converted値は可視化・
既存比較の追跡値として扱う。native境界は独立参照である。今回native内部は3断面すべてで既存
point値より境界値へ近づいたが、残差は0ではなく断面差もあるため、cell-to-pointだけが唯一の
原因とは断定できない。volume cell値の有限体積的意味、cut/clip品質、面積差等は未解決である。

出力は`native_cell_section_flow_summary.csv`、`native_vs_point_vs_boundary.csv`、audit JSON/MD、
`polygons/*.csv.gz`、`vtp/*.vtp`。共有ZIPにはこれらの`output`実データを含めず、コード、設定、
テスト、文書だけを収録する。

## 16. ステップ4B-2 profile監査

`scripts/profile_metrics.py`は面積加重分位点、histogram、M2/M3、alpha/beta、正方向流量重心、
逆流/低流速面積、二次流れ、共通grid L2を共通提供する。`audit_inlet_upstream_profile.py`は
`audit_boundary_flow.py`の明示boundary選択、`surface_flow.py`の面積積分・native polygon、
`section_config.py`の共通断面、既存の線形三角形補間を再利用する。

正式な定量profileはFEM三角形求積、STAR入口native surface-cell、STAR上流native volume-cellを使用。
cell-to-point値は共通grid画像だけに限定する。入口法線、flow-direction normal、低流速閾値、
一様速度許容差、grid、判定基準、出力先は`config/audit_inlet_upstream_profile.json`で指定する。

FEM入口の外周35点はGmsh wall ID 40との共有点で全て0、内部173点は全て10 mm/s。共有点ゼロモデルが
実流量を完全再現する。STAR入口170 native cellsは一様10 mm/s相当。分岐前には正規化profileと
alpha/beta差が残るためケースAだが、重心・二次流れ差は設定基準内で、因果は未確定である。

共有ZIPにはコード、設定、テスト、文書だけを含め、この監査の`output`、NPZ、PNG、実データは含めない。

## 17. ステップ4B-3 断面系列と発達監査

`scripts/section_series.py`は`positions_mm`と既存template sectionから、断面中心、width/height、s/t軸、
法線、flow direction、入口距離、x/D、安定名を生成する。`config/sections.json`を編集せず、生成実体を
`generated_sections.json`へ保存する。位置感度系列もbase positionsとoffsetsから自動生成する。

`scripts/audit_inlet_development_profiles.py`は4B-2のFEM三角形求積、profile_metrics、共通gridと、
4B-1のSTAR native volume-cell polygonを再利用する。入口境界z=50 mmは既存4B-2 CSVを基準値として
読み、内部7断面とは`location_type`を分ける。主出力はprofile summary、FEM–STAR comparison、
flow conservation、position sensitivity、common-grid metrics、7 NPZ、43 PNG、audit JSON/Markdown。

正規化L2は0.05Dから2Dで7.463, 6.228, 5.928, 8.384, 9.111, 8.566, 9.888%。
位置感度warningは0で、判定C+E。profile差は単純な単調減衰でなく、cut位置感度だけでもない。
完全発達beta=4/3、alpha=2は参考で、完全発達や因果は仮定しない。共有ZIPにはoutputを含めない。



## 18. ステップ4B-4 Reynolds比較監査

`scripts/audit_reynolds_profile_comparison.py`は新しい数式実装を複製せず、4B-2/4B-3、
`section_series.py`、FEM三角形求積、STAR native volume-cell積分、profile metricsを設定駆動で
再利用する。Re=10は`260625_r1_solver/solution_000300.vtu`と`260626star-ccm/duct_test.case`
(time index 299)、Re=100は既存4B-3の最終37.5 sデータを使用する。同じ7内部断面と3分岐断面で、
formal/native値とcommon-grid profile値を分離する。判定R1〜R5のうち、条件付き数値観測はR3、
rho/mu未確認を含む正式判定はR5。既存出力やCFD結果は変更しない。
