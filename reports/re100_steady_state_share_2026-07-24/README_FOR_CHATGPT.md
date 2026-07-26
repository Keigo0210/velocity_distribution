# Re=100 非定常流れ・定常化時刻の確認パッケージ

このZIPは、FEMによるbranch duct系のRe=100非定常流れ計算について、速度場がほぼ定常になる時刻を確認するための解析成果物です。

## ChatGPTへの確認依頼

次の点をレビューしてください。

1. `analysis_report.md` の結論が `velocity_steady_state_metrics.csv` と整合しているか。
2. 1秒差の相対L2変化率について、0.5%および0.1%を初めて下回る時刻と、その後も継続して満たす時刻が正しいか。
3. 全領域と比較断面の結果から、今後のRe=100計算を15 sまでとする判断が妥当か。
4. 平均・最大・95パーセンタイル流速が定常化判定を支持しているか。
5. 解析スクリプトの計算方法や閾値判定に問題がないか。

## 主要な結論

- 全領域、1秒差:
  - 0.5%未満: 8.25 sから継続
  - 0.1%未満: 11.75 sから継続
- 比較断面、1秒差:
  - 0.5%未満: 10.25 sから継続
  - 0.1%未満: 13.75 sから継続
- 推奨終了時刻: 15 s

## 計算・解析条件

- 結果ディレクトリ: `data/260629_solver`
- VTU: `solution_000001.vtu`から`solution_000300.vtu`までの300ファイル
- dtおよび出力間隔: 0.125 s
- 最終時刻: 37.5 s
- 速度配列: point dataの`solution_velocity`
- 全領域メッシュ: 19,275点、48,934セル
- 比較断面:
  - center = (10, 0, 15) mm
  - normal = (1, 0, -1)
  - width = 12 mm
  - height = 10 mm
- 主評価: 8ステップ＝1秒前との差
- 相対L2変化率:
  - `100 * ||u(t) - u(t-1 s)||_2 / ||u(t)||_2`

## 収録ファイル

- `analysis/analysis_report.md`: 詳細な解析結果
- `analysis/velocity_steady_state_metrics.csv`: 全300時刻の指標
- `analysis/relative_l2_change.png`: 相対L2変化率
- `analysis/speed_statistics.png`: 全領域の代表流速
- `analysis/section_metrics.png`: 比較断面の指標
- `source/check_velocity_steady_state.py`: 再利用可能な解析スクリプト
- `provenance/result.txt`: 元計算ログ
- `provenance/*.json`: Re=100との対応と比較断面を示す既存設定
- `SHA256SUMS.txt`: 収録ファイルのチェックサム

## 収録していないデータ

元VTU 300ファイルは合計容量が大きいため、この共有ZIPには含めていません。そのため、このZIPだけで速度場を最初から再計算することはできませんが、全時刻の計算済み指標、解析コード、ログ、設定、図を使った結果レビューは可能です。

また、元のソルバ入力設定ファイルはリポジトリ内に存在しません。dtとnstepsは`result.txt`、入口速度10 mm/sは元VTUの入口境界値、Re=100との対応は既存後処理設定から確認されています。
