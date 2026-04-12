# 1.5 眉メイク

眉消し + 新しい眉の描画。形状はトレーサーツールで定義する。

## ファイル構成

| ファイル | 役割 |
|----------|------|
| `main.py` | 眉消し・描画のメインロジック |
| `eyebrow_tracer.py` | 眉形状トレーサー（GUIツール） |
| `eyebrow_shapes.json` | トレース済み形状データ（5タイプ） |
| `reference_medicalbrows.jpg` | リファレンス画像（medicalbrows.jp） |
| `run_all.py` | 全サンプル画像に眉消しを一括適用 |

## 形状定義の手順

### 1. トレーサーでリファレンス画像から形状をトレース

```bash
python eyebrow_tracer.py [画像パス] [-o 保存先JSON]
```

**GUIの操作:**
1. 眉頭(HEAD)をクリック
2. 眉尻(TAIL)をクリック
3. 眉の輪郭に沿ってクリックで1周トレース（4点以上）
4. **[Save]ボタン** で保存（即座にJSONに書き出し）
5. **[Next]** で次のタイプへ
6. 右クリックで直前のポイント削除

保存時に HEAD-TAIL 基準軸で正規化され、自動的に上辺/下辺に分割される。

### 2. 眉の適用

```bash
# 単一タイプ
python main.py <入力画像> -t natural
python main.py <入力画像> -t arch --zoom -o result.png

# 全トレース済みタイプを一括生成（eyebrow_{type}.png として出力）
python main.py <入力画像> --all-types
```

### オプション一覧

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-t, --type` | straight | 眉タイプ (natural/straight/arch/parallel/corner) |
| `-o, --output` | eyebrow_{type}.png | 出力ファイル名 |
| `--color R G B` | 85 60 45 | 眉の色 (RGB) |
| `--intensity` | 0.75 | 描画強度 (0-1) |
| `--thickness` | 1.0 | 厚み倍率 |
| `--zoom` | - | 眉元ズーム比較画像で出力 |
| `--imgonly` | - | 結果画像のみ（比較なし） |
| `--no-draw` | - | 眉消しのみ |
| `--all-types` | - | 全タイプ一括生成 |

## 処理フロー

1. **Phase 1: 眉消し** - ランドマークからポリゴンマスク生成 → cv2.inpaint TELEA で補完
2. **Phase 2: 眉描画** - `eyebrow_shapes.json` の形状をスプライン補間 → 顔ランドマークにマッピング → アルファブレンド

## 座標系（eyebrow_shapes.json）

- `t`: HEAD→TAIL 軸上の位置 (0=眉頭, 1=眉尻)
- `offset`: 軸に垂直な距離 / 眉長さ (正=上方向)
- JSONがない場合はパラメータベースのフォールバック形状を使用
