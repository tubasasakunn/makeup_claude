# Phase 3 化粧選択

Phase 2 (顔判定) の結果を受け取り、Phase 1 (化粧反映) に渡す「化粧処方 (Prescription)」を生成・適用するレイヤ。

## 構成
- `3.0-schema/prescription.py` — 処方のデータ構造 (dataclass)。`Prescription` が base/highlight/shadow/eye/eyebrow + source/rationale を束ねる。
- `3.1-rules/main.py` — 画像 → Phase 2.2.8 → ルール適用 → 処方 JSON を出力する最小版 CLI。
- `3.1-rules/verify_visual.py` — 5 顔型のテスト画像で処方を適用し before/after グリッドを生成 (目視確認用)。
- `outputs/` — 生成物 (サンプル処方 JSON, verify_grid.png)。

## M1 (完了)
**ルール**: 骨格タイプ (2.1) → 眉タイプ (1.5)

| skeletal | brow_type | 理由 |
|---|---|---|
| oval              | natural  | バランス型、標準の緩やかカーブ |
| round             | arch     | 頬のふっくら感を引き締める縦ライン |
| long              | parallel | 縦長を抑える水平・太め |
| inverted_triangle | straight | シャープさを和らげる |
| base              | natural  | 角張りを和らげる自然カーブ |

## 使い方
```bash
# 処方 JSON を 1 枚に対して生成
python loadmap/3/3.1-rules/main.py imgs/卵.png -o /tmp/rx.json

# 5 顔型の検証グリッドを生成 (outputs/verify_grid.png)
python loadmap/3/3.1-rules/verify_visual.py
```

## 次 (M2 以降)
- 3.2 処方 → Phase 1 適用パイプライン (base→highlight→shadow→eye→eyebrow を順次合成)
- 3.1 ルール拡張 (顔比率/目/口/鼻 → 色・強度・ON/OFF)
- 3.3 E2E + レポート可視化
