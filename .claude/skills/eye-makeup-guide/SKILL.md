---
name: eye-makeup-guide
description: アイメイク部位のメッシュID特定・ガイド画像生成のリファレンス。目のメイク領域を追加・修正するとき、新しい顔パーツのメッシュを特定するとき、ガイド画像を再生成するときに参照する。
user-invocable: false
---

# アイメイク メッシュID特定 リファレンス

このスキルは `loadmap/1-virtual-makeup/1-4-eye/` の実装で得た知見をまとめたもの。
新しい顔パーツのメッシュ特定や、既存エリアの修正時に参照すること。

## アーキテクチャ概要

```
imgs/base.png                 → 入力画像
loadmap/shared/facemesh.py    → FaceMesh クラス（MediaPipe + 中点分割）
loadmap/1-virtual-makeup/1-4-eye/find_eye_meshes.py → メッシュ特定 + ガイド画像生成
target.json                   → 全エリアのメッシュID保存先
```

- `FaceMesh(subdivision_level=1)` で 478 ランドマーク → 1800 頂点 / 3416 三角形
- `target.json` に `highlight`, `shadow`, `eye` セクションとして保存
- 各エリアは `name` + `mesh_id`（リスト）の形式。eyeliner のみ `type: "polyline"` で別形式

## 5つのアイメイクエリア

| エリア | 方式 | 概要 |
|--------|------|------|
| eyeshadow_base | メッシュ | 眉下〜上まぶた（アイホール全体） |
| eyeshadow_crease | メッシュ | アイホール下半分の帯状（二重幅） |
| eyeliner | ポリライン | 上下まぶたのラッシュライン（6分割可能） |
| tear_bag | メッシュ | 下まぶた〜その下の帯（涙袋） |
| lower_outer | メッシュ | 涙袋の目尻側40% |

## 重要な実装パターン

### 1. ランドマーク点列のソート

**必ず x 座標昇順でソートする**（`sort_by_x=True`）。

MediaPipe のランドマーク ID 順は左→右とは限らない。
例: `RIGHT_EYEBROW_LOWER = [55, 65, 52, 53, 46]` は x が右→左。
`RIGHT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173]` は左→右。

ソートせずに `make_polygon()` すると**自己交差ポリゴン**（蝶ネクタイ型）になり、
メッシュ選択が不正確になる。

**例外**: 目の輪郭（`RIGHT_EYE_OUTLINE`）は周回順序を維持する必要があるため
`sort_by_x=False` を使う。

### 2. 左右対称のミラー処理

**右目のみ計算** → `fm.find_mirror_meshes()` で左目を生成する。

```python
r_eyehole_set = set(r_eyehole)
l_eyehole = sorted(fm.find_mirror_meshes(r_eyehole_set))
areas["eyeshadow_base"] = sorted(set(r_eyehole) | set(l_eyehole))
```

**注意**: ミラー後に追加フィルタが必要な場合がある。
例: `lower_outer` はミラー後にも x 座標フィルタを適用しないと目頭側のメッシュが混入する。

```python
l_outer_candidates = sorted(fm.find_mirror_meshes(set(r_outer)))
# ミラー後にも目尻側フィルタを再適用
l_outer = [m for m in l_outer_candidates if centroid_x(m) <= l_threshold_x]
```

### 3. 眼球メッシュの除外

目の開口部は非常に狭く（特に一重・細い目）、ポリゴン内に重心が入るメッシュは 0 個の場合がある。
しかし、目の輪郭に接するメッシュの三角形は視覚的に眼球に重なる。

**対策**: 目の輪郭ポリゴンを重心から 20% 拡大して除外ゾーンとする。

```python
r_eye_outline_arr = np.array(outline_pts, dtype=np.float64)
centroid = r_eye_outline_arr.mean(axis=0)
r_eye_expanded = centroid + (r_eye_outline_arr - centroid) * 1.2
r_eyeball_meshes = set(find_meshes_in_polygon(fm, r_eye_expanded))
```

- 15% → まだ少し侵入する
- 20% → 適切（eyeshadow_base, tear_bag に適用）
- 30% → 除外しすぎ（eyeliner が 2 meshes しか残らない）
- **eyeshadow_crease には適用しない**（二重幅は目のキワに近いため削れすぎる）

### 4. ポリラインの外側オフセット

eyeliner のようにメッシュではなくポリラインで描画する場合、
太さが内側（眼球側）にも広がる問題がある。

**対策**: 各点を目の中心から外側にオフセットする。

```python
# 目の中心を算出
eye_center_x = np.mean([fm.points[lid]["x"] * w for lid in all_ids])
eye_center_y = np.mean([fm.points[lid]["y"] * h for lid in all_ids])

# 各点を外側にオフセット
offset_px = thickness // 2 + 1
dx = px - eye_center_x
dy = py - eye_center_y
dist = max(np.sqrt(dx*dx + dy*dy), 1e-6)
px += dx / dist * offset_px
py += dy / dist * offset_px
```

### 5. ポリゴン作成の基本パターン

```python
def make_polygon(upper_pts, lower_pts):
    """上辺（左→右ソート済み）と下辺で閉じたポリゴンを作成"""
    return np.array(upper_pts + lower_pts[::-1], dtype=np.float64)
```

上辺: 左→右、下辺: 右→左 でつなぐと正しい非自己交差ポリゴンになる。

### 6. ガイド画像の生成

- 左: 元画像、右: オーバーレイ の比較画像（`np.hstack`）
- 半透明カラーオーバーレイ（alpha 0.5〜0.55）
- 輪郭線を `cv2.findContours` + `cv2.drawContours` で描画
- 凡例は右上に背景ボックス付きで配置
- メッシュ方式は `fm.build_mask(mesh_ids, w, h)` でマスク生成
- ポリライン方式は `cv2.polylines` でマスク生成

## MediaPipe ランドマーク ID 早見表（目周辺）

```
右目:
  上まぶた:  [246, 161, 160, 159, 158, 157, 173]
  下まぶた:  [33, 7, 163, 144, 145, 153, 154, 155, 133]
  目頭: 33,  目尻: 133
  眉下端:    [55, 65, 52, 53, 46]
  中間ライン: [156, 28, 27, 29, 30, 247]
  涙袋下端:  [243, 112, 26, 22, 23, 24, 110, 25]

左目:
  上まぶた:  [466, 388, 387, 386, 385, 384, 398]
  下まぶた:  [263, 249, 390, 373, 374, 380, 381, 382, 362]
  目頭: 263, 目尻: 362
  眉下端:    [285, 295, 282, 283, 276]
  中間ライン: [383, 258, 257, 259, 260, 467]
  涙袋下端:  [463, 341, 256, 252, 253, 254, 339, 255]

顔幅参照: landmarks 234（右耳） ↔ 454（左耳）
```

## PR 画像ルール（CLAUDE.md 準拠）

- PR 作成時は必ず変更内容がわかる画像を含める
- push するたびに画像を最新状態に更新する
- 画像 URL はコミット SHA 指定にする: `https://raw.githubusercontent.com/tubasasakunn/makeup_claude/<SHA>/path/to/image.png`
- push 後に `mcp__github__update_pull_request` で body の SHA を差し替える

## 今後の拡張ポイント

- `1.5 眉` も同じパターン（眉のランドマーク → メッシュ特定 → ミラー）で実装可能
- 新しい顔パーツを追加する場合: ランドマーク ID を特定 → `landmarks_to_points` + `make_polygon` + `find_meshes_in_polygon` の組み合わせ
- 線状のメイク（眉毛のアウトラインなど）はポリライン方式を採用する
