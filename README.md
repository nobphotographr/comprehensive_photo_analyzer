# Comprehensive Photo Analysis Tool

写真・画像総合解析ツール - LUT効果の科学的分析システム

## 概要

このツールは、写真の色彩・質感・印象を包括的に解析し、LUT（Look-Up Table）適用効果を科学的に定量化するローカル実行ツールです。段階的拡張により、業界初の総合的画像解析プラットフォームを構築します。

## 主要機能

### Phase 1-3: 基本機能（現在実装中）
- **基本色彩解析**: RGB/HSV/LAB色空間での統計分析
- **ヒストグラム比較**: 色分布の詳細比較
- **色変化解析**: 色相・彩度・明度の変化量定量化
- **主要色抽出**: k-meansクラスタリングによる主要色分析
- **HTMLレポート**: 包括的な解析結果の自動生成

### 将来実装予定
- **Phase 4-6**: 質感・光学解析（グレイン、ハレーション、レンズ特性）
- **Phase 7-9**: 印象・感情解析、周波数解析、高度統合機能

## 動作環境

- **OS**: macOS (Apple Silicon / Intel)
- **Python**: 3.11以上
- **開発環境**: VSCode + Claude Code推奨
- **メモリ**: 8GB以上推奨
- **ストレージ**: 5GB以上の空き容量

## インストール

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd comprehensive_photo_analyzer
```

### 2. 仮想環境の作成・有効化
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 依存関係のインストール
```bash
# 基本依存関係
pip install -r requirements.txt

# 開発用依存関係（オプション）
pip install -r requirements-dev.txt
```

### 4. VSCode設定（推奨）
```bash
# 設定ファイルのコピー
cp .vscode/settings.json.template .vscode/settings.json
cp .vscode/launch.json.template .vscode/launch.json
```

## 使用方法

### 基本的な使用例

#### 単一画像ペアの解析
```bash
# 基本解析（Phase 1-3）
python main.py --mode single --original original.jpg --processed processed.jpg --phase basic

# 総合解析（全機能）
python main.py --mode comprehensive --original original.jpg --processed processed.jpg --phase all
```

#### バッチ処理
```bash
# 複数画像ペアの一括解析
python main.py --mode batch --input-dir data/input/ --output-dir data/output/

# 特定のPhaseのみ
python main.py --mode batch --input-dir data/input/ --phase basic
```

#### 設定ファイル使用
```bash
# カスタム設定での実行
python main.py --config config/custom_analysis.yaml
```

### コマンドライン引数

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--mode` | 動作モード (single/comprehensive/batch) | single |
| `--original` | 元画像ファイルパス | - |
| `--processed` | 処理済み画像ファイルパス | - |
| `--input-dir` | 入力ディレクトリ（バッチ用） | - |
| `--output-dir` | 出力ディレクトリ | - |
| `--phase` | 解析フェーズ (basic/intermediate/advanced/all) | basic |
| `--config` | 設定ファイルパス | - |
| `--analyzers` | 使用解析器 (color,texture,impression) | - |
| `--output-format` | 出力形式 (html/json/csv/all) | html |
| `--log-level` | ログレベル (DEBUG/INFO/WARNING/ERROR) | INFO |
| `--benchmark` | ベンチマークモード | false |

### VSCodeでの実行

VSCode + Claude Code環境では、以下の起動設定が利用できます：

1. **Phase 1 Basic Analysis**: 基本色彩解析
2. **Comprehensive Analysis**: 総合解析
3. **Batch Processing**: バッチ処理
4. **Custom Config**: カスタム設定

F5キーで選択した設定での実行が可能です。

## 出力結果

### HTMLレポート
- 包括的な解析結果を視覚的に表示
- インタラクティブなグラフと統計情報
- 画像比較表示
- 主要指標のサマリー

### データ形式
- **JSON**: 詳細な数値データ
- **CSV**: 統計データの表形式
- **PNG**: 可視化グラフ・チャート

## ディレクトリ構造

```
comprehensive_photo_analyzer/
├── main.py                    # メインエントリポイント
├── config/                    # 設定ファイル
├── src/                       # ソースコード
│   ├── core/                  # コア解析エンジン
│   ├── analyzers/             # 専門解析モジュール
│   ├── visualization/         # 可視化・レポート
│   └── utils/                 # ユーティリティ
├── data/                      # データディレクトリ
│   ├── input/                 # 入力画像
│   ├── output/                # 解析結果
│   └── sample/                # サンプルデータ
├── tests/                     # テストスイート
└── docs/                      # ドキュメント
```

## 設定

### 基本設定ファイル
`config/default_config.yaml`で以下の設定が可能：

- 解析フェーズとアナライザー選択
- 画像処理パラメーター
- 出力形式とディレクトリ
- パフォーマンス設定
- ログレベル

### カスタム設定例
```yaml
analysis:
  phase: "basic"
  color_spaces: ["RGB", "HSV"]
  precision: "high"

output:
  format: "all"  # HTML, JSON, CSV すべて出力
  include_visualizations: true

performance:
  max_workers: 8  # 並列処理数を増加
```

## 開発情報

### 段階的開発計画
- **Phase 1-3** (基盤): 基本色彩解析 ✅ 実装中
- **Phase 4-6** (中級): 質感・光学解析
- **Phase 7-9** (高級): 印象・周波数解析

### 技術スタック
- **画像処理**: OpenCV, Pillow, scikit-image
- **数値計算**: NumPy, SciPy
- **統計・可視化**: pandas, matplotlib, seaborn
- **色彩科学**: colour-science (Phase 4以降)
- **機械学習**: scikit-learn (印象分析用)

### テスト実行
```bash
# 単体テスト
pytest tests/unit/

# 結合テスト
pytest tests/integration/

# カバレッジ測定
pytest --cov=src tests/
```

## トラブルシューティング

### よくある問題

#### 1. 画像読み込みエラー
```
エラー: 画像の読み込みに失敗しました
```
**解決策**: 対応形式（JPEG, PNG, TIFF, BMP）を確認、ファイルパスを確認

#### 2. メモリ不足
```
エラー: メモリ使用量が上限を超えています
```
**解決策**: 設定で`analysis_size`を小さくする、`max_workers`を減らす

#### 3. 依存関係エラー
```
ModuleNotFoundError: No module named 'cv2'
```
**解決策**: `pip install -r requirements.txt`で依存関係を再インストール

### ログファイル
詳細なエラー情報は`logs/`ディレクトリのログファイルを確認してください。

## ライセンス

MIT License

## 作者

Photo Analysis Team

## 更新履歴

### v0.1.0 (2025-06-04)
- 初期バージョンリリース
- Phase 1-3 基本色彩解析機能実装
- HTMLレポート生成機能
- バッチ処理対応
- VSCode + Claude Code統合