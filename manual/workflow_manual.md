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
- `sections`: 断面の `center`, `normal`, `width`, `height`
- `geometry_file`: 参照形状。`null` の場合は自動探索または解データから表示
- `plot_title_suffix`: 図タイトルの suffix

### 4.5 `config/slice_velocity_star_ccm.json`

Star-CCM+ の指定断面で速度分布を見る設定です。

重要な項目:

- `case_file`: Star-CCM+ の `.case`
- `time_point`: reader の time point
- `coordinate_scale`, `velocity_scale`: Star-CCM+ データの単位変換
- `sections`: 断面指定

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

`sections` で指定した平面に対して、速度をサンプリングし、静止画像や概要HTMLを出します。

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
