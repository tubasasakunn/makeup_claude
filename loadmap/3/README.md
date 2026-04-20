# Phase 3 化粧選択

Phase 2 (顔判定) の結果を受け取り、Phase 1 (化粧反映) に渡す「化粧処方 (Prescription)」を生成・適用するレイヤ。

## 構成
- `3.0-schema/prescription.py` — 処方のデータ構造 (dataclass)。`Prescription` が base/highlight/shadow/eye/eyebrow + source/rationale を束ねる。
- `3.1-rules/main.py` — 画像 → Phase 2.2.8 → ルール適用 → 処方 JSON を出力する CLI。
- `3.1-rules/verify_visual.py` — 5 顔型の処方適用 (眉のみ) before/after グリッド。
- `3.2-pipeline/main.py` — 処方 → Phase 1 の各ステップを順次適用してフル仕上げ画像を生成。`--auto` で判定→処方→適用まで一気通貫。
- `3.2-pipeline/verify_visual.py` — 5 顔型のフルメイク before/after グリッド。
- `outputs/` — 生成物 (処方 JSON, verify_grid.png, pipeline_grid.png)。

## M1 (完了)
**ルール**: 骨格タイプ (2.1) → 眉タイプ (1.5)

| skeletal | brow_type | 理由 |
|---|---|---|
| oval              | natural  | バランス型、標準の緩やかカーブ |
| round             | arch     | 頬のふっくら感を引き締める縦ライン |
| long              | parallel | 縦長を抑える水平・太め |
| inverted_triangle | straight | シャープさを和らげる |
| base              | natural  | 角張りを和らげる自然カーブ |

## M2 (完了)
**処方 → Phase 1 適用パイプライン**: base → highlight → shadow → eye(5 areas) → eyebrow の順で合成。
デフォルト値は Phase 1 側の値をそのまま引き継ぐので、M2 時点では顔型によって変わるのは「眉の形状」のみ。
他の色・強度・ON/OFF の顔型別差分は M3 で 3.1 ルール拡張する予定。

## 使い方
```bash
# 処方 JSON を 1 枚に対して生成
python loadmap/3/3.1-rules/main.py imgs/卵.png -o /tmp/rx.json

# 処方を適用してフルメイク画像を生成
python loadmap/3/3.2-pipeline/main.py imgs/卵.png --rx /tmp/rx.json -o /tmp/out.png

# 判定→処方→適用まで一気通貫
python loadmap/3/3.2-pipeline/main.py imgs/卵.png --auto -o /tmp/out.png

# 5 顔型の検証グリッドを生成
python loadmap/3/3.1-rules/verify_visual.py      # outputs/verify_grid.png (眉のみ)
python loadmap/3/3.2-pipeline/verify_visual.py   # outputs/pipeline_grid.png (フル)
```

## 次 (M3)
- 3.1 ルール拡張 (顔比率/目/口/鼻 → 色・強度・ON/OFF)
- 3.3 E2E + レポート可視化
