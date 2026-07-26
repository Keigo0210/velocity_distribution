# CFD Visualization Workflow Manual

このドキュメントは、現在の `/workspace` にある Python スクリプト群の使い方と処理内容をまとめたものです。特に、solver と Star-CCM+ の流速比較を流線上で行う処理について詳しく説明します。

## 1. 想定ディレクトリ構成

主に以下のディレクトリを使用します。

```text
/workspace
├── config
│   ├── compare_streamlines_solver_star_ccm.json
│   ├── streamlines_solver.json
│   ├── streamlines_star_ccm.json
│   ├── slice_velocity.json
│   └── slice_velocity_star_ccm.json
├── data
│   ├── mesh
│   │   ├── branch_duct.msh
│   │   └── *_identified.msh
│   ├── 260629_solver
│   │   └── solution_*.vtu
│   └── 260629star-ccm
│       └── duct_test.case
├── output
├── scripts
└── manual
```

`config/*.json` で入力データ、出力先、seed 点、時間範囲、可視化設定を指定します。基本的には `.py` の中身を直接書き換えず、`.json` を編集してケースを切り替えます。

## 2. 環境構築

Docker コンテナ内で実行する前提です。最低限、以下の Python パッケージを使います。

```bash
pip install numpy pandas pyvista vtk matplotlib pillow
```

主な依存関係は以下です。

- `numpy`: 配列計算、速度ベクトル、誤差計算
- `pandas`: summary CSV の出力
- `pyvista`: VTU/CASE/MSH 読み込み、補間、スクリーンショット出力
- `vtk`: PyVista の内部処理、オフスクリーン描画
- `matplotlib`: 断面速度分布の静的画像
- `Pillow`: GIF アニメーション作成

Docker 環境では X11 関連の警告が出やすいため、`scripts/streamlines_common.py` で VTK の警告表示を抑制しています。GUI を表示する処理ではなく、オフスクリーンで PNG/GIF/HTML を出す用途なら、このまま実行できます。

## 3. 基本的な実行方法

作業ディレクトリは `/workspace` を想定します。

```bash
cd /workspace
```

solver の流線を作る場合:

```bash
python scripts/streamlines_solver.py
```

Star-CCM+ の流線を作る場合:

```bash
python scripts/streamlines_star_ccm.py
```

solver と Star-CCM+ の流速誤差を比較する場合:

```bash
python scripts/compare_streamlines_solver_star_ccm.py
```

solver の断面速度分布を見る場合:

```bash
python scripts/slice_velocity.py
```

Star-CCM+ の断面速度分布を見る場合:

```bash
python scripts/slice_velocity_star-ccm.py
```

mesh の inlet/outlet/wall を対話的に指定する場合:

```bash
python scripts/identify_mesh_boundaries.py
```

別の設定ファイルを使う場合は `--config` を指定します。

```bash
python scripts/compare_streamlines_solver_star_ccm.py --config config/compare_streamlines_solver_star_ccm.json
```

## 4. 主要な設定ファイル

### 4.1 `config/streamlines_solver.json`

solver の `solution_*.vtu` を連続的に読み込み、solver 速度場で pathline を作成します。

重要な項目:

- `output_dir`: 出力先
- `input_dir`: solver の `.vtu` があるディレクトリ
- `file_pattern`: 読み込むファイル名パターン
- `velocity_name`: 速度ベクトル配列名。現在は `solution_velocity`
- `seed_source`: 流線開始点の配置
- `start`, `stop`, `stride`: 読み込む time index
- `time_step_seconds`: 1 step あたりの物理時間。現在は `0.125`
- `screenshot.animation`: GIF 作成設定

### 4.2 `config/streamlines_star_ccm.json`

Star-CCM+ の Ensight `.case` を読み込み、Star-CCM+ 速度場で pathline を作成します。

重要な項目:

- `case_file`: Star-CCM+ の `.case`
- `velocity_name`: 速度ベクトル配列名。現在は `Velocity`
- `coordinate_scale`: Star-CCM+ 座標のスケール。現在は `1000.0`
- `velocity_scale`: Star-CCM+ 速度のスケール。現在は `1000.0`
- `start`, `stop`: Star-CCM+ reader の time point。現在は `0..299`
- `display_time_index_offset`: 表示 index を `1..300` にするための offset

### 4.3 `config/compare_streamlines_solver_star_ccm.json`

solver と Star-CCM+ の流速を比較する設定です。現在は `branch_duct.msh` を共通表示 geometry として使用します。

重要な項目:

- `solver.vtu_template`: solver の `.vtu` パス
- `star_ccm.case_file`: Star-CCM+ の `.case` パス
- `seed_source`: 比較用 pathline の開始点
- `time_step_seconds`: 物理時間刻み
- `geometry_file`: 表示形状。現在は `data/mesh/branch_duct.msh`
- `use_mesh_surface_as_geometry`: `false` にして、solver mesh 表面ではなく共通 geometry を使う
- `error_visualization.display_max_percent`: 表示用の誤差上限。現在は `10.0`
- `error_visualization.over_limit_label`: 表示上限を超えた箇所の凡例

現在の誤差表示は、真の誤差値を保持したまま、見やすさのために色表示だけ `0..10%` にクリップします。

### 4.4 `config/slice_velocity.json`

solver の指定断面で速度分布を見る設定です。

重要な項目:

- `vtu_file`: 読み込む solver `.vtu`
- `section_library`, `section_set`または`section_names`: 共通断面の選択
- `geometry_file`: 参照形状。`null` の場合は自動探索または解データから表示
- `plot_title_suffix`: 図タイトルの suffix

### 4.5 `config/slice_velocity_star_ccm.json`

Star-CCM+ の指定断面で速度分布を見る設定です。

重要な項目:

- `case_file`: Star-CCM+ の `.case`
- `time_point`: reader の time point
- `coordinate_scale`, `velocity_scale`: Star-CCM+ データの単位変換
- `section_library`, `section_set`または`section_names`: FEM版と共通の断面指定

## 5. 各 Python スクリプトの概要

### 5.1 `scripts/streamlines_common.py`

流線処理で共通に使う関数群です。

主な役割:

- JSON 設定の読み込み
- 入出力パスの解決
- seed 点群の生成
- solver `.vtu` と Star-CCM+ `.case` の読み込み
- 座標・速度のスケール変換
- cell data の速度を point data へ変換
- pathline の時間発展
- `.vtp`, `.vtm`, `.png`, `.gif`, `.csv` の出力

`PathlineTracker` が時間発展する流跡線を管理します。各 seed 点について、現在位置、過去の点列、速度、time index、物理時刻を保持します。

### 5.2 `scripts/streamlines_solver.py`

solver の `.vtu` 群を連続的に読み込み、solver 速度場で pathline を作ります。

処理の流れ:

1. `config/streamlines_solver.json` を読む
2. `input_dir` と `file_pattern` から `.vtu` を選ぶ
3. `seed_source` から入口付近の seed 点を作る
4. 各時刻の solver mesh を読み込む
5. `PathlineTracker.advance()` で pathline を進める
6. `.vtp`, `.vtm`, summary CSV, GIF などを出力する

### 5.3 `scripts/streamlines_star_ccm.py`

Star-CCM+ の `.case` を読み込み、Star-CCM+ 速度場で pathline を作ります。

solver 版との主な違い:

- `.case` reader の time point を使う
- `display_time_index_offset` で表示 index を `1..300` に合わせる
- `coordinate_scale` と `velocity_scale` で Star-CCM+ 側の単位を solver 側に合わせる

### 5.4 `scripts/compare_streamlines_solver_star_ccm.py`

solver と Star-CCM+ の速度場を比較し、流線上の相対誤差を可視化します。最も重要なスクリプトです。

出力:

- `vtp/streamline_error_*.vtp`: 各時刻の誤差付き pathline
- `streamline_error_all_times.vtm`: 全時刻のまとめ
- `streamline_error_summary.csv`: 時刻ごとの誤差統計
- `png/streamline_error_*.png`: 各時刻の静止画
- `streamline_error_time_animation.gif`: 時間変化 GIF
- `html/streamline_error_interactive.html`: ブラウザで自由視点確認できるHTML

### 5.5 `scripts/slice_velocity.py`

solver の速度場を任意断面で切り、速度分布を可視化します。

`section_config.py`で解決した平面に対して速度をsliceし、断面別CSV、静止画像、概要HTMLを出します。

### 5.6 `scripts/slice_velocity_star-ccm.py`

Star-CCM+ の速度場を任意断面で切り、速度分布を可視化します。基本構造は solver 版と同じですが、`.case` の time point と Star-CCM+ の単位変換を扱います。

### 5.7 `scripts/identify_mesh_boundaries.py`

`.msh` から境界面を抽出し、クリック操作で `inlet`, `outlet`, `wall` などの Physical Group を付ける対話ツールです。

主な機能:

- `data/mesh` 内の `.msh` を読み込む
- hex8 volume element から外部境界 face を抽出する
- クリックした面を seed として、近い法線方向の連結 patch を選択する
- 選択面を `inlet`, `outlet`, `wall`, 任意名に割り当てる
- 未指定面をまとめて wall にできる
- 元 mesh とは別に `*_identified.msh` として保存する

## 6. 流線による誤差表示の詳細

### 6.1 比較の基本方針

この比較は、solver mesh の節点と Star-CCM+ mesh の節点を一対一で対応させる方法ではありません。メッシュ形状や節点位置が完全には一致しないため、直接節点同士を引き算するのではなく、共通の評価点を作って、その点で両方の速度場を補間して比較します。

現在の共通評価点は、solver の速度場で作った pathline 上の点です。

つまり、比較している量は以下です。

```text
solver pathline 上の評価点 x において

u_solver(x) = solver mesh 上で補間した速度
u_star(x)   = Star-CCM+ mesh 上で補間した速度

relative_error(x) = |u_solver(x) - u_star(x)| / |u_star(x)|
relative_error_percent(x) = relative_error(x) * 100
```

### 6.2 評価点の作成

`seed_source` で入口付近に複数の seed 点を作ります。

```json
"seed_source": {
  "center": [0.0, 0.0, 49.5],
  "normal": [0.0, 0.0, -1.0],
  "radius": 4.0,
  "rings": 4,
  "points_per_ring": 16,
  "include_center": true
}
```

この設定では、中心点と同心円状の点群を入口断面に配置します。`rings` と `points_per_ring` を増やすと、流線本数が増え、分岐後の誤差分布も見やすくなります。ただしHTMLやGIFのファイルサイズと処理時間も増えます。

### 6.3 solver pathline の作成

`PathlineTracker.advance()` で、solver 速度場を使って seed 点を時間発展させます。

単純化すると以下の更新です。

```text
x_next = x_current + u_solver(x_current) * dt
```

`dt` は `time_step_seconds` です。現在は `0.125 s` です。

速度 `u_solver(x_current)` は、solver mesh 上で `sample()` により補間して取得します。点が mesh 外に出た場合や補間できない場合、その seed は inactive になります。

### 6.4 空間補間の方法

補間は PyVista/VTK の `sample()` を使います。

```python
sampled = solver_pathlines.sample(star_mesh)
solver_sampled = solver_pathlines.sample(solver_mesh)
```

`sample()` は評価点がどのセル内にあるかを探し、そのセル内で point data を補間します。たとえば線形四面体セルなら、4つの節点速度 `u1..u4` とセル内の重み `N1..N4` から次のように速度を作ります。

```text
u(x) = N1*u1 + N2*u2 + N3*u3 + N4*u4
```

六面体なら、そのセルの補間に必要な節点値とセル内座標から同様に補間されます。最近傍点の値をそのまま拾っているわけではありません。

Star-CCM+ の `Velocity` が cell data に入っている場合は、読み込み時に `cell_data_to_point_data(pass_cell_data=True)` で point data へ変換します。この変換は、節点に接するセル値を平均するような処理です。その後、その point data を使ってセル内補間を行います。

### 6.5 誤差計算

solver pathline 上の各点で、solver と Star-CCM+ の速度ベクトルを補間取得し、ベクトル差のノルムを Star-CCM+ の速度ノルムで割ります。

```text
error_vector = u_solver - u_star
velocity_error_magnitude = |error_vector|
relative_velocity_error = |u_solver - u_star| / |u_star|
relative_velocity_error_percent = relative_velocity_error * 100
```

Star-CCM+ 側の速度がほぼゼロの点では分母が小さくなるため、相対誤差が極端に大きくなることがあります。`zero_speed_tolerance` より小さい場合は有効な相対誤差として扱いません。

### 6.6 現在の誤差配列

比較後の `.vtp` には主に以下の配列が入ります。

- `velocity_error`: 速度差ベクトル
- `velocity_error_magnitude`: 速度差ベクトルの大きさ
- `solver_speed`: solver 側速度の大きさ
- `star_speed`: Star-CCM+ 側速度の大きさ
- `speed_error`: `solver_speed - star_speed`
- `relative_velocity_error`: 相対誤差の比
- `relative_velocity_error_percent`: 相対誤差の百分率
- `relative_velocity_error_display_percent`: 表示用にクリップした百分率
- `relative_velocity_error_over_limit`: 表示上限を超えた点のフラグ

解析や統計には `relative_velocity_error_percent` を使います。可視化には `relative_velocity_error_display_percent` を使います。

### 6.7 0-10% クリップ表示

誤差は局所的に非常に大きくなることがあります。その最大値にカラーバーを合わせると、多くの領域が同じような色になり、小さい誤差の違いが見えなくなります。

そのため、現在は表示用に以下の設定を使っています。

```json
"error_visualization": {
  "display_min_percent": 0.0,
  "display_max_percent": 10.0,
  "show_over_limit_points": true,
  "over_limit_color": "black",
  "over_limit_html_color": "#111111",
  "over_limit_point_size": 7.0,
  "over_limit_width_multiplier": 1.45,
  "over_limit_label": "> 10 %"
}
```

この設定では、色表示は `0..10%` に固定されます。10%を超えた点は、真値を捨てるのではなく、`relative_velocity_error_over_limit` でフラグ付けし、HTMLでは黒線、PNG/GIFでは黒い点として別表示します。

つまり、可視化上は以下の役割分担になります。

- 色: 0-10% の範囲で細かい違いを見る
- 黒表示: 10%を超える箇所を別途見つける
- CSV/VTP の真値: 最大誤差や平均誤差などの定量評価に使う

### 6.8 共通 geometry の使用

Re=10 と Re=100 で solver mesh 表面をそれぞれ使うと、同じ形状でもメッシュ線の密度や表示点数が変わります。そのため、現在は `branch_duct.msh` を共通表示 geometry として使います。

```json
"geometry_file": "data/mesh/branch_duct.msh",
"use_mesh_surface_as_geometry": false
```

`branch_duct.msh` は bounds が以下で、現在の流線座標系と一致しています。

```text
x = -5..25
y = -5..5
z = 0..50
```

これにより、Re が変わってもHTML上の形状表示を揃えられます。

### 6.9 HTML ビューア

`html/streamline_error_interactive.html` では、ブラウザで以下の操作ができます。

- ドラッグ: 視点回転
- ホイール: ズーム
- スライダー: frame 移動
- `Jump to time index`: time index を直接入力して移動

HTML内の表示中心とズームは、流線点の平均ではなく geometry bounds から決めています。これにより、Re=10 と Re=100 で流線の伸び方が違っても、初期表示の位置とスケールが揃いやすくなります。

## 7. 出力ファイルの見方

比較スクリプトの主な出力は以下です。

```text
output/streamlines/solver_vs_star_ccm/solver_vs_star_ccm_Re=100
├── html
│   └── streamline_error_interactive.html
├── png
│   └── streamline_error_*.png
├── vtp
│   └── streamline_error_*.vtp
├── streamline_error_all_times.vtm
├── streamline_error_summary.csv
└── streamline_error_time_animation.gif
```

`streamline_error_summary.csv` には時刻ごとの平均誤差、最大誤差、95パーセンタイル、表示上限超過率などが入ります。局所的な最大値だけで判断すると外れ値に引っ張られるため、平均値、95パーセンタイル、`over_limit_point_ratio_percent` を合わせて見るのがよいです。

## 8. 注意点と改善余地

現在の比較は、solver の pathline 上で Star-CCM+ を評価する非対称な比較です。solver の流れに沿った位置で、Star-CCM+ 速度場との差を見る、という意味になります。

より中立的な比較をしたい場合は、以下の方法も考えられます。

- 固定断面上の共通格子に両方の速度場をサンプリングして比較する
- 管内全体に共通の点群を作って比較する
- solver pathline と Star-CCM+ pathline を別々に作り、到達位置や分岐先も比較する
- Star-CCM+ 速度が小さい場所では相対誤差ではなく絶対誤差も併記する

今の流線誤差表示は、流れに沿って「どこで solver と Star-CCM+ の速度差が大きくなるか」を視覚的に見るための方法です。定量比較では、HTMLの色だけでなく、CSVの統計値と `.vtp` 内の真の誤差配列も合わせて確認してください。

## 9. 共通断面ライブラリ

断面定義は `config/sections.json` で一元管理する。現在のライブラリには
`straight_z10`、`upstream_z30`、`side_branch` と、これらをこの順で処理する
`default_three_sections` がある。法線の正規化、断面内の s/t 軸生成、軸反転、
幅・高さ・格子数の検証は `scripts/section_config.py` が共通して行う。

設定JSONから3断面を一括指定する例:

```json
{
  "section_library": "sections.json",
  "section_set": "default_three_sections"
}
```

一部だけを指定する例:

```json
{
  "section_library": "sections.json",
  "section_names": ["side_branch"]
}
```

`section_library` の相対パスは、まず解析設定JSONが置かれたディレクトリを基準に
解決する。たとえば `config/compare_fem_cases.json` 内の `sections.json` は
`config/sections.json` になる。互換用に `config/sections.json` のような
リポジトリルート基準の記述も、設定JSONの親ディレクトリのさらに親を基準とする
候補として解決する。絶対パスはそのまま使用する。現在の作業ディレクトリには依存しない。

一般設定の解決優先順位は、`section_library + section_set`、
`section_library + section_names`、直接 `sections` 配列、単一 `section`、
名前付き `sections` オブジェクト、旧トップレベル形式の順である。
`resolution`、`flip_s_axis`、`flip_t_axis` も互換キーとして受け付ける。

FEM対FEMの各comparisonでは、`section_set`、`sections`、`section`、トップレベル指定
の順で解決する。複数指定がある場合は採用した指定を警告表示する。

### 9.1 実行コマンド

```bash
cd /workspace
python scripts/slice_velocity.py --config config/slice_velocity.json
python scripts/slice_velocity_star-ccm.py --config config/slice_velocity_star_ccm.json
python scripts/relative_error_colormap_solver_star_ccm.py \
  --config config/relative_error_colormap_solver_star_ccm.json
python scripts/compare_fem_cases.py --config config/compare_fem_cases.json
```

単一断面だけを実行する場合は、使用するJSONで `section_set`を削除し、
`"section_names": ["side_branch"]` を指定する。

### 9.2 出力構造

断面出力は `output/fem_solver/260629/<section_name>/` および
`output/star_ccm/260629/<section_name>/` にCSV、PNG、overview、HTML、metadata JSONを出す。

FEM対STARは
`output/relative_error_colormap/solver_vs_star_ccm_Re=100/<section_name>/` に
`metrics.csv`、`comparison_grid.csv`、速度場・差分PNG、VTPを出し、ルートに
`summary_metrics.csv`を出す。

FEM対FEMは `output/fem_case_comparison/<comparison_name>/<section_name>/` に
`metrics.csv`、`comparison_grid.csv`、速度場・差分PNG、任意のVTPを出し、ルートに
`comparison_metrics.csv`を出す。

旧形式の回帰用JSONは `config/examples/*_legacy_*.json` に保存している。
`execution.fail_fast` が `true` の場合は最初の断面エラーで停止し、`false`の場合は
失敗をCSVへ記録して残りの断面処理を継続する。

## 10. 定常判定・断面流量・進捗レポート（現行）

### 10.1 全領域と共通3断面の定常判定

```bash
cd /workspace
python scripts/check_velocity_steady_state.py \
  --config config/check_velocity_steady_state.json
```

1回で設定内の全data sourceについて`whole_domain`と共通3断面を評価する。VTUに物理時刻が
ない場合は`solution_<step>.vtu`のstepと設定`dt`から時刻を復元する。Case Bのように
20 solver stepごとの出力でも、step差から1秒に相当する4出力差を選ぶ。

```text
relative_L2 [%] = ||u(t) - u(t-ΔT)||₂ / ||u(t)||₂ × 100
```

`first_below_threshold_time`は最初にしきい値未満になった時刻、
`continuous_below_threshold_time`はそこから解析末尾まで未満が続く最初の時刻である。
`whole_domain`はメッシュ座標・接続・点順序の同一性をhashで検査して節点ベクトルを直接比較する。
断面は各時刻を同一平面でsliceし、対応点の一致を確認して比較する。従来の位置を直接渡すCLIも維持する。

### 10.2 三角形面積積分による流量

```bash
python scripts/section_flow.py --config config/section_flow.json
```

VTUまたはEnsight CASEを共通断面でslice・clip・triangulateし、実三角形を積分する。

```text
Q_triangle = triangle_area × mean(u_vertex · normalized_normal)
Q_section  = Σ Q_triangle
```

`signed_flow`は設定法線方向を正とし、`absolute_flow`はその絶対値である。現法線では
`upstream_z30`と`straight_z10`の流れが法線と逆なので負、`side_branch`は正になる。
標準の`absolute`配分では入口`upstream_z30`、出口`straight_z10`と`side_branch`の
絶対流量を使用する。

```text
balance_error = outlet_total - inlet_total
balance_error_percent = abs(balance_error) / inlet_total × 100
outlet_fraction = outlet_absolute_Q / Σ outlet_absolute_Q
```

`orientation_adjusted`を使う場合の符号係数はJSONに置き、コードへ断面別係数を固定しない。
旧レポートの平均法線速度比は体積流量ではなく、断面積が違えば流量比と一致しないため、
新しい流量・レポートの配分には使わない。

### 10.3 common_valid_area

- `valid_point_fraction`: 両速度が有効な共通格子点の割合。
- `valid_cell_fraction`: 4頂点すべてが有効な格子セルの割合。
- `common_valid_area`: 有効セルごとの`ds × dt`の総和。
- `legacy_valid_area_estimate`: 有効点率 × 断面矩形全面積という旧近似。

`common_valid_area`は境界の欠損形状をセル単位で反映する。旧近似は後方比較用に残すが、
新しい面積評価には使わない。

### 10.4 共通格子出力形式

`output.grid_output_format`は`none`、`csv`、`csv.gz`、`npz`を受け付ける。
通常は`none`、表計算互換には`csv.gz`、Pythonでの再解析には圧縮`npz`を推奨する。
非圧縮`csv`は可読だが非常に大きい。旧`save_grid_csv: true/false`は`csv/none`として
互換解釈し、新旧両方がある場合は新キーを優先して警告する。

```bash
python scripts/compare_fem_cases.py --config config/compare_fem_cases.json
python scripts/compare_fem_cases.py --config config/examples/compare_fem_cases_grid_npz.json
python scripts/compare_fem_cases.py --config config/examples/compare_fem_cases_grid_csv_gz.json
```

### 10.5 設定駆動レポート

```bash
python scripts/build_velocity_error_progress_report.py \
  --config config/velocity_error_progress_report.json
```

入力はFEM対STAR、FEM対FEM、定常判定、面積積分流量のsummary CSVだけで、巨大な格子CSVを
必須にしない。相対パスはレポートJSONの場所を基準に解決する。ケース、summary、断面、
steady source、flow source、しきい値、出力先はJSON指定で、新ケースは`cases`への追加だけで
処理できる。必須入力欠損は明示エラー、任意入力欠損はゼロにせず「未評価」とする。

出力は`report.md`、`report.html`、`report_summary.csv`、`report_inputs.json`、
`fem_case_details.csv`とPNGである。解釈は評価断面・時刻に限定し、FEM内差が小さくても
全領域・全時刻の厳密な時間収束証明とは断定しない。

## 11. 検証

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall -q scripts tests
```

JSON構文、CSVヘッダー、PillowによるPNG検査、`numpy.load(..., allow_pickle=False)`による
NPZ検査も実施する。

## 12. 共有ZIP

`flow_velocity_analysis_code.zip`はChatGPTへ現在のコード、設定、テスト、文書構成を共有する
スナップショットであり、解析環境や実験結果の完全再現を目的としない。実データは
`/workspace/data`にあり、実解析はサーバー上のデータで行う。ZIPには`scripts`、`config`、
`tests`、`manual`、`analysis_code_summary`を含め、大容量の`data`、`output`、`.git`、仮想環境、
cache、生成物、秘密情報を含めない。形式例のSTAR `.case`だけでは参照データ不足で読込不能である。

## 13. 現在の制約

- 時系列断面定常判定はnative slice点の一致を要求し、変形メッシュには固定格子方式が必要。
- STAR cell dataからpoint dataへの変換は急勾配を平均化する可能性がある。
- 流跡線比較はFEM流跡線上でSTARを評価する非対称比較。
- VTU物理時刻は現データではファイルstepと設定dtから復元する。
- ZIPは実データと解析出力を含まないため、ZIP単体で数値結果を再現できない。


## 14. ステップ4A：native境界流量監査

目的は、FEMとSTAR-CCM+の総流量差が実際の入口・出口境界ですでに存在するかを、
新規計算を行わず既存37.5秒結果だけで調べることである。

```bash
cd /workspace
python scripts/audit_boundary_flow.py --config config/audit_boundary_flow.json
```

### 14.1 境界選択

選択優先順位はnative surface part、boundary tag、明示surface file、最後にgeometric planeである。
現在の設定では次を使い、geometric planeは使用しない。

- FEM: `data/mesh/branch_duct.msh`のPhysical Surface ID 10/20/30。
- STAR: Ensightの`branch_duct_test.inlet1/outlet1/outlet2` surface part。

FEM結果VTUにはboundary tagがないため、元Gmsh boundary節点を結果VTU節点へ座標一致で対応させる。
体積cell数・型が一致し、今回の全boundary点は距離0で対応した。同一座標に複数VTU点がある場合は
速度を平均し、multiplicityと最大速度spreadを`boundary_audit.json`へ記録する。

`geometric_plane`を使う場合はcenter、normal、tolerance、抽出点/cell数、選択根拠を保存し、
native boundaryとは呼ばない。

### 14.2 point data積分

FEM point速度はsurfaceを三角形化し、各三角形で線形速度の面積積分を行う。

```text
Q_triangle = area × mean(u_vertex · configured_normal)
```

### 14.3 native cell data積分

STAR surface partのcell速度は元polygon/quad cellの面積を使う。

```text
Q_cell = cell_area × (u_cell · configured_normal)
```

STARは同じsurface partをcell-to-point変換後に三角形point積分し、native cell値との差も出す。
体積cell値を根拠なく境界へ投影しない。

### 14.4 法線と符号

外向き法線をJSONへ明示する。入口は`+z`なので流入Qは負、直進出口は`-z`なので流出Qは正、
側枝出口は`(1,0,-1)`方向なので流出Qは正となる。mesh faceの向きは混在し得るため、
暗黙のcell orientationを外向き法線とは見なさない。

### 14.5 内部断面との比較

`output/section_flow/re100/flow_summary.csv`を読み、入口対`upstream_z30`、直進出口対
`straight_z10`、側枝出口対`side_branch`を比較する。境界外向き法線と内部断面法線が逆の場合が
あるため、signed differenceと流量絶対値のdifferenceを分離する。

### 14.6 出力

```text
output/boundary_flow_audit/re100_v2/
├── boundary_flow_summary.csv
├── boundary_balance_summary.csv
├── boundary_vs_internal_sections.csv
├── star_cell_to_point_comparison.csv
├── solver_boundary_comparison.csv
├── boundary_condition_audit.csv
├── boundary_audit.json
└── boundary_audit.md
```

スクリプトは出力先に既存ファイルがある場合、上書きせず明確なエラーで停止する。

## 15. ステップ4B-1：STAR内部断面のnative volume-cell-wise積分

目的は、STAR内部断面の定量流量からcell-to-point変換を除き、元の体積cell `Velocity`を
各cellと断面平面の交差polygonへ対応付けて積分することである。可視化用point dataと定量積分を
分離し、今回のSTAR内部流量ではnative volume-cell-wise値を優先する。境界native流量は独立した
保存性参照値として併記する。

```bash
cd /workspace
python scripts/integrate_volume_cell_sections.py \
  --config config/integrate_volume_cell_sections.json
```

対象は`duct_test.case`のtime index 299（37.5 s）、volume part `領域`、cell data `Velocity`と、
共通libraryの`upstream_z30`、`straight_z10`、`side_branch`である。パス、part、時刻、配列、単位、
断面名、比較CSV、対応boundary、しきい値、出力先はすべてJSONで指定する。

### 15.1 元volume cellとintersection polygon

切断前のvolume meshへ`int64`の`__original_volume_cell_id`を0から付与し、VTK cutter後も整数型・
範囲・shapeを検証する。定量速度はcutter出力の値を使わず、保持IDで元volume cellの`Velocity`を
直接参照する。合成1-cell/2-cell meshでpolygonとID対応を検証する。浮動小数IDの丸めは許可しない。

### 15.2 section window clippingと面積

cutter polygonを直交正規断面座標`s,t`へ射影し、`[-width/2,width/2] × [-height/2,height/2]`へ
Sutherland–Hodgman法でclipする。clip後の順序付きpolygon面積・重心は2次元shoelace式で求める。
`s,t`が直交正規なのでこの面積は3次元平面面積と同じである。断面外polygon、非正面積、重複、
未対応IDを追跡し、未対応は標準設定で失敗させる。

```text
Q_polygon = polygon_area × (u_original_cell · normalized_section_normal)
Q_section = Σ Q_polygon
```

`signed_flow`はlibrary法線を正とする。境界の外向き法線と内部断面法線が逆の場合があるため、
signed differenceと流量の大きさのdifferenceを別々に保存する。`absolute_flow`はnet signed flowの
絶対値である。

### 15.3 出力

```text
output/native_volume_cell_sections/star_re100/
├── native_cell_section_flow_summary.csv
├── native_vs_point_vs_boundary.csv
├── native_cell_section_audit.json
├── native_cell_section_audit.md
├── polygons/*.csv.gz
└── vtp/*.vtp
```

polygon CSV.GZには元cell ID、面積、元cell速度、法線速度、流量寄与、重心、validを保存する。
VTPにも元cell IDと流量寄与をcell dataとして保持する。比較CSVは今回のnative内部、従来の
cell-to-point内部、ステップ4Aのnative境界をsignedとmagnitudeの両方で比較する。

今回、native内部値は全3断面で従来point-converted値よりnative境界の流量絶対値へ近づいた。
ただしnative境界との差は断面ごとに残り、1%基準では上流だけが範囲内だった。したがって
cell-to-point後処理の寄与は示されるが、残差の唯一の原因とは断定しない。volume cell出力値の
有限体積的意味、cut/clip、内部断面と境界の面積差、part/time対応等は引き続き候補である。

## 16. ステップ4B-2：入口・分岐前速度プロファイル監査

```bash
cd /workspace
python scripts/audit_inlet_upstream_profile.py \
  --config config/audit_inlet_upstream_profile.json
```

対象は37.5 sのFEM/STAR入口native boundaryと`upstream_z30`だけで、追加断面や分岐後の詳細比較は
行わない。断面法線に対するsigned normal velocityと、設定したflow-direction normalに対する
正方向速度を分離する。

正式profile指標は、FEM point速度では正の6点・4次三角形求積（線形速度のM2/M3を厳密積分）、
STAR入口ではnative surface-cell polygon、STAR上流では元volume-cell交差polygonを使用する。
共通gridは形状可視化専用で、STAR cell-to-point変換による平滑化値を正式流量、alpha、betaへ混ぜない。

主指標は面積、signed/absolute flow、面積加重平均・標準偏差・RMS・分位点、逆流/低流速面積、
`M2=∫u_flow²dA`、`M3=∫u_flow³dA`、beta、alpha、正方向流量重心、s/t半面流量、二次流れである。
共通gridでは正規化速度L2/MAE/RMSE/correlation、二次流れL2、共通valid面積を出す。

FEM入口は208点中、外周かつwall共有の35点が0 mm/s、内部173点が10 mm/sで、中間値はない。
全点10 mm/s理論流量781.186294 mm³/sに対し、共有35点だけ0とする線形三角形モデルは
実流量764.350330 mm³/sを差0で再現した。STAR入口native 170 cellsは許容差1e-6 mm/s内で
全て9.999999776 mm/s、二次流れ0、alpha=beta=1である。

入口の正規化速度L2は6.195%、分岐前は9.888%。分岐前のFEM beta/alphaは1.2897/1.8383、
STARは1.2278/1.6368で、主速度profile差が残る。一方、流量重心差0.0199 mm、二次流れ比の差
0.463 percentage pointsは設定基準内だった。判定はケースAだが、入口・発達差が分岐配分差の
唯一の原因とは断定しない。

出力は`output/inlet_upstream_profile_audit/re100/`のCSV、audit JSON/Markdown、共通grid NPZ、
16 PNGである。出力先が非空なら停止する。

## 17. ステップ4B-3：入口近傍内部profile発達監査

z=50 mmは実入口境界で、境界条件の離散表現監査だけに使う。z<50 mmはnear-inlet internal sectionで、
正式な内部profile比較に使う。z=49.5 mmの流量を入口境界流量とは呼ばない。

```bash
cd /workspace
python scripts/audit_inlet_development_profiles.py \
  --config config/audit_inlet_development_profiles.json
```

`section_series.positions_mm`からz=49.5, 49, 48, 45, 40, 35, 30 mmを自動生成する。
width、height、grid、s/t軸は`upstream_z30` templateから継承し、`sections.json`は変更しない。
断面名は`inlet_development_z49p5`等の安定名で、`generated_sections.json`へ実定義を保存する。
位置、入口z、直径10 mm、法線+z、flow direction -zはJSON設定である。positionsの変更だけで系列を変更できる。

FEM正式値は既存slice/windowと線形point速度の正6点三角形求積、STAR正式値は元volume-cell ID付き
intersection polygonを使用する。共通gridは形状比較と画像専用で、cell-to-point平滑化値から流量、
beta、alphaを再計算しない。z=35 mmでは2つの非凸元cellが各2連結polygonを作るが、完全重複は0で、
全成分を同じ元cell速度へ正確に追跡する。

入口から0.05, 0.1, 0.2, 0.5, 1, 1.5, 2Dで、正規化L2差は7.463, 6.228, 5.928,
8.384, 9.111, 8.566, 9.888%。0.2Dまで一度低下後、0.5Dで2.456ポイント再増大し、単調減衰しない。
±0.05 mm位置感度は、最大でも流量0.255%、beta 0.347%、alpha 0.776%、L2 0.478ポイントで
設定警告閾値内。判定はケースC+E（再増大し、cut位置感度ではない差が2Dまで残存）である。

完全発達円管層流beta=4/3、alpha=2は参考値だけに用いる。2Dでも両solverとも参考値へ未到達であり、
完全発達を仮定しない。完全発達との差とFEM–STAR間差は別々に保存する。入口差がz=30 mmへそのまま
連続伝播したとは断定せず、0.2D以降の再増大には離散化、メッシュ、発達過程など他の候補が残る。



## 18. ステップ4B-4：Reynolds数によるprofile・分岐配分比較

既存データだけを使い、Re=10とRe=100へ同じ4B-2/4B-3監査を適用する。
Re=10はFEM `data/260625_r1_solver/solution_000300.vtu`（step 300、37.5 s、point
`solution_velocity`、mm/mm/s）とSTAR `data/260626star-ccm/duct_test.case`（time index 299、
37.5 s、volume part `領域`、native cell `Velocity`、m/m/s）を使用する。Re=100は4B-3と同じ
`data/260629_solver/solution_000300.vtu`と`data/260629star-ccm/duct_test.case`を使用する。

```bash
cd /workspace
python scripts/audit_reynolds_profile_comparison.py \
  --config config/audit_reynolds_profile_comparison.json
python -m unittest discover -s tests -p 'test_*.py' -v
```

入口z=50 mmは境界条件離散表現の監査用で、内部profileはz=49.5, 49, 48, 45, 40, 35,
30 mmの同一7断面を使う。FEM正式値はpoint-linear三角形求積、STAR正式値はnative volume-cell
intersection polygon積分である。共通gridはprofile L2・centerline診断・可視化だけに使い、正式な
流量、beta、alpha、分岐配分へ使わない。分岐配分は`upstream_z30`, `straight_z10`,
`side_branch`の正式流量絶対値から計算する。

判定ケースR1〜R5と閾値はJSONに置く。現在は数値パターンだけならR3（Re=10でもprofile差は5%を
超えるが、分岐配分差は1 percentage point以内）である。一方、元のrhoとdynamic viscosity設定が
未収録でReを独立再計算できないため、正式判定はR5とする。力学的課題（慣性・運動量配分感度）と
数値的課題（メッシュ、離散化、数値拡散、dt/CFL、field表現）を分け、今回だけで原因を断定しない。
出力は`output/reynolds_profile_comparison/re10_vs_re100/`で、非空なら停止する。
