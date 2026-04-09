# 眉タイプと理想配置ルール

男性向けメイクで使う眉のタイプと、黄金比・顔のランドマークに基づく理想的な配置ルールをまとめる。各ルールは MediaPipe FaceLandmarker の478点ランドマークで実装可能な形で記述。

---

## 1. MediaPipe ランドマーク早見表（眉配置に使う主要点）

実装時に参照する主要ランドマーク。

### 眉
| 名称 | 右 | 左 |
|------|---|---|
| 眉頭（内端） | 55 | 285 |
| 眉山候補（中〜外） | 52, 65 | 282, 295 |
| 眉尻（外端） | 46 | 276 |
| 眉上端ライン | 70, 63, 105, 66, 107 | 300, 293, 334, 296, 336 |
| 眉下端ライン | 46, 53, 52, 65, 55 | 276, 283, 282, 295, 285 |

### 目
| 名称 | 右 | 左 |
|------|---|---|
| 目頭（鼻側） | 133 | 362 |
| 目尻（外側） | 33 | 263 |
| 上まぶた中央 | 159 | 386 |
| 下まぶた中央 | 145 | 374 |
| 黒目中心（近似） | 159と145の中点 | 386と374の中点 |

### 鼻
| 名称 | ID |
|------|---|
| 鼻先 | 1 |
| 鼻根（眉間） | 168 |
| 小鼻（右） | 64, 98, 129 |
| 小鼻（左） | 294, 327, 358 |
| 鼻孔外縁（右） | 219 |
| 鼻孔外縁（左） | 439 |
| 小鼻のくぼみ右（頬との接点） | 102 or 129 |
| 小鼻のくぼみ左 | 331 or 358 |

### 顔輪郭
| 名称 | ID |
|------|---|
| 額頂上 | 10 |
| あご先 | 152 |
| 右こめかみ | 234 |
| 左こめかみ | 454 |

---

## 2. 理想の眉配置（3大ポイントルール）

すべての眉タイプに共通する「眉頭・眉山・眉尻」の配置ルール。これらは日本の黄金比理論・化粧理論の標準。

### 2.1 眉頭 (Eyebrow Head) の位置

| ルール名 | 内容 | 実装 |
|---------|------|------|
| **小鼻外側真上ルール** | 眉頭のX座標 ≒ 小鼻外縁(64/294)のX座標 | `brow_head.x == nose_wing.x` |
| **目頭垂直ルール** | 眉頭は目頭(133/362)の真上より少し内側〜同じ | `brow_head.x ∈ [inner_eye.x - δ, inner_eye.x]` |

**採用**: 小鼻外側真上ルール（黄金比理論で最も一般的）

```
右眉頭候補X = landmarks[64].x    # 右小鼻外縁
左眉頭候補X = landmarks[294].x   # 左小鼻外縁
```

### 2.2 眉山 (Eyebrow Peak/Arch) の位置

| ルール名 | 内容 | 実装 |
|---------|------|------|
| **黒目外側ルール** | 眉山のX座標 = 黒目中心のX 〜 黒目外縁のX | `peak.x ∈ [iris.x, iris.x + r]` |
| **小鼻-黒目外側ライン延長** | 小鼻(64/294)と黒目外側を結ぶ線の延長上 | 直線の方程式で求める |
| **目尻1/3ルール（令和版）** | 目頭から目尻までの 2/3 〜 3/4 あたり | `peak.x = inner + (outer - inner) * 0.7` |

**採用**: 令和版比率（2/3 〜 3/4）— 現代的なバランス

```
右眉山候補X = lerp(landmarks[133].x, landmarks[33].x, 0.7)    # 目頭→目尻の70%
左眉山候補X = lerp(landmarks[362].x, landmarks[263].x, 0.7)
```

### 2.3 眉尻 (Eyebrow Tail) の位置

| ルール名 | 内容 | 実装 |
|---------|------|------|
| **小鼻-目尻延長ルール** ⭐ | 小鼻(64/294)と目尻(33/263)を結ぶ直線の延長上 | `tail = small_nose + t * (outer_eye - small_nose)` |
| **眉頭より上ルール** | 眉尻のY ≤ 眉頭のY（下がり眉を避ける） | `tail.y <= head.y` |
| **こめかみライン** | 眉尻は目尻の外側、こめかみ方向に | `tail.x > outer_eye.x` |

**採用**: 小鼻-目尻延長ルール（最も有名で信頼性高い）

```
# 直線: P(t) = nose_wing + t * (outer_eye - nose_wing)
# 眉尻はこの直線上で目尻の外側
右眉尻 = ray(landmarks[64], landmarks[33], t_out)    # t_out ≈ 1.3〜1.5
左眉尻 = ray(landmarks[294], landmarks[263], t_out)
```

### 2.4 眉の縦位置・高さ

| ルール名 | 内容 |
|---------|------|
| **目と眉の距離** | 目の縦幅 ×  1.0〜1.5 分、目から離す（男性はやや近め） |
| **眉の太さ** | 目の縦幅 × 1.0〜1.3（男性の基本） |

```
eye_height = |landmarks[159].y - landmarks[145].y|
brow_bottom_y = eye_top_y - eye_height * 1.0    # 目から1眼分上
brow_thickness = eye_height * 1.2               # 男性基本の太さ
```

---

## 3. 眉タイプ一覧

男性メイクで使う代表的な眉タイプ。各タイプは「眉頭Y、眉山の相対位置、眉尻Y、太さ、カーブ度」で定義できる。

### 3.1 タイプ定義表

| タイプ | 眉山位置 | 眉山の高さ | 眉尻の高さ | 太さ | カーブ | 印象 | 似合う顔型 |
|--------|---------|----------|----------|------|-------|------|----------|
| **1. ストレート眉** | なし（直線） | 眉頭と同じ | 眉頭と同じ | 太 | 0 (直線) | 若々しい・優しい・韓流 | 面長、大人顔 |
| **2. 並行眉** | 緩やか（中央） | 眉頭より少し上 | 眉頭と同じ | 太 | 0.1 (弱) | ナチュラル・男らしい | 万能 |
| **3. ナチュラルアーチ** | 目尻側2/3 | 眉頭より上 | 眉頭と同じ〜少し下 | 中 | 0.3 (中) | 知的・バランス型 | 卵型、丸顔 |
| **4. アーチ眉（標準）** | 目尻側2/3 | 眉頭より明確に上 | 眉頭と同じ〜少し下 | 中 | 0.5 (強) | 女性的・上品 | 丸顔、ベース型 |
| **5. 角度眉（シャープ）** | 黒目外側 | 眉頭より大きく上 | 眉頭より下 | 中 | 0.7 (角) | 強い・シャープ・クール | 逆三角、丸顔 |
| **6. 短め太眉** | 緩やか | 眉頭と同じ | 眉頭と同じ | 極太 | 0.1 | 強い・ワイルド | 面長 |
| **7. 長めアーチ** | 目尻側3/4 | 眉頭より上 | 眉頭より下 | 細 | 0.4 | クール・大人 | 丸顔、ベース型 |

### 3.2 タイプのパラメータ定義（実装用）

各タイプを数値パラメータで表現。描画時はこれを元に制御点を配置。

```python
EYEBROW_TYPES = {
    "straight": {               # 1. ストレート眉
        "peak_position": None,          # 眉山なし
        "peak_height_ratio": 0.0,       # 眉頭からの上昇量（眉高さ比）
        "tail_height_ratio": 0.0,       # 眉頭からの下降量（眉高さ比）
        "thickness_ratio": 1.3,         # 目の縦幅基準の太さ倍率
        "curve_strength": 0.0,          # 曲線の強さ (0=直線, 1=強カーブ)
        "length_ratio": 1.0,            # 標準長さ (小鼻-目尻延長点まで)
    },
    "parallel_thick": {         # 2. 並行眉
        "peak_position": 0.5,           # 眉の中央あたり
        "peak_height_ratio": 0.1,
        "tail_height_ratio": 0.0,
        "thickness_ratio": 1.3,
        "curve_strength": 0.15,
        "length_ratio": 1.0,
    },
    "natural_arch": {           # 3. ナチュラルアーチ
        "peak_position": 0.65,
        "peak_height_ratio": 0.25,
        "tail_height_ratio": 0.1,
        "thickness_ratio": 1.1,
        "curve_strength": 0.3,
        "length_ratio": 1.0,
    },
    "arch": {                   # 4. アーチ眉
        "peak_position": 0.65,
        "peak_height_ratio": 0.4,
        "tail_height_ratio": 0.15,
        "thickness_ratio": 1.0,
        "curve_strength": 0.5,
        "length_ratio": 1.0,
    },
    "angular": {                # 5. 角度眉
        "peak_position": 0.6,           # 黒目外側あたり
        "peak_height_ratio": 0.5,
        "tail_height_ratio": 0.25,
        "thickness_ratio": 1.0,
        "curve_strength": 0.75,         # シャープに
        "length_ratio": 1.0,
    },
    "short_thick": {            # 6. 短め太眉
        "peak_position": 0.5,
        "peak_height_ratio": 0.05,
        "tail_height_ratio": 0.0,
        "thickness_ratio": 1.5,
        "curve_strength": 0.1,
        "length_ratio": 0.85,           # 標準より短く
    },
    "long_arch": {              # 7. 長めアーチ
        "peak_position": 0.7,
        "peak_height_ratio": 0.3,
        "tail_height_ratio": 0.2,
        "thickness_ratio": 0.85,
        "curve_strength": 0.4,
        "length_ratio": 1.1,            # 標準より長く
    },
}
```

---

## 4. 眉の描画計算式

タイプパラメータと顔ランドマークから、眉のポリゴン/ポリラインを計算する手順。

### 4.1 基準点の計算

```python
def compute_brow_anchors(fm, side="right"):
    """眉の3大ポイント（頭・山・尻）と基準値を計算"""
    if side == "right":
        nose_wing = fm.landmarks_px[64]       # 右小鼻
        inner_eye = fm.landmarks_px[133]      # 右目頭
        outer_eye = fm.landmarks_px[33]       # 右目尻
        eye_top = fm.landmarks_px[159]        # 右上まぶた中央
        eye_bot = fm.landmarks_px[145]        # 右下まぶた中央
    else:
        nose_wing = fm.landmarks_px[294]
        inner_eye = fm.landmarks_px[362]
        outer_eye = fm.landmarks_px[263]
        eye_top = fm.landmarks_px[386]
        eye_bot = fm.landmarks_px[374]

    eye_height = abs(eye_top[1] - eye_bot[1])

    # 眉頭: 小鼻外側の真上、目の縦1倍分上
    head_x = nose_wing[0]
    head_y = eye_top[1] - eye_height * 1.0

    # 眉尻: 小鼻-目尻の延長線上、眉頭と同じY基準で
    # 直線の方程式: P(t) = nose_wing + t * (outer_eye - nose_wing)
    dx = outer_eye[0] - nose_wing[0]
    dy = outer_eye[1] - nose_wing[1]
    # 眉尻のY(head_y)を通るtを求める
    t_tail = (head_y - nose_wing[1]) / dy if dy != 0 else 1.3
    tail_x = nose_wing[0] + t_tail * dx
    tail_y = head_y

    # 眉山: 眉頭→眉尻を peak_position で補間したX、眉頭より上のY
    # これはタイプごとに異なる計算

    return {
        "head": (head_x, head_y),
        "tail": (tail_x, tail_y),
        "eye_height": eye_height,
        "brow_length": abs(tail_x - head_x),
    }
```

### 4.2 眉の上下ラインをBezier曲線で生成

```python
def generate_brow_polyline(anchors, eyebrow_type):
    """タイプパラメータから眉のポリゴン頂点列を生成"""
    head = anchors["head"]
    tail = anchors["tail"]
    eye_h = anchors["eye_height"]
    brow_len = anchors["brow_length"]

    params = EYEBROW_TYPES[eyebrow_type]

    # 眉山のX/Y
    peak_t = params["peak_position"] or 0.5
    peak_x = head[0] + (tail[0] - head[0]) * peak_t
    peak_y = head[1] - eye_h * params["peak_height_ratio"]

    # 眉尻のY補正（タイプによっては眉頭より下げる）
    tail_y_adj = tail[1] + eye_h * params["tail_height_ratio"]

    # 太さ
    thickness = eye_h * params["thickness_ratio"]

    # 上辺の制御点（Bezier）
    upper_points = bezier_curve([head, (peak_x, peak_y), (tail[0], tail_y_adj)], n=20)

    # 下辺は上辺を thickness 分下にオフセット
    lower_points = [(p[0], p[1] + thickness) for p in upper_points]

    # ポリゴン
    return upper_points + lower_points[::-1]
```

---

## 5. 顔型別おすすめ眉タイプ

LOADMAP.md の 2.1 骨格判定（卵型/丸型/面長/逆三角形/ベース型）に基づく推奨。

| 顔型 | 特徴 | 推奨タイプ | 理由 |
|------|------|----------|------|
| **卵型** | 理想形 | 3. ナチュラルアーチ / 4. アーチ眉 | どれでも似合う、バランス基準 |
| **丸型** | 丸くふっくら | 4. アーチ眉 / 5. 角度眉 | 縦ラインを強調してシャープに見せる |
| **面長** | 縦に長い | 1. ストレート眉 / 2. 並行眉 / 6. 短め太眉 | 横ラインを強調して顔を短く見せる |
| **逆三角形** | あご細 | 3. ナチュラルアーチ / 7. 長めアーチ | 柔らかく、あごの細さを中和 |
| **ベース型** | エラ張り | 4. アーチ眉 / 7. 長めアーチ | 曲線で横幅の強さを緩和 |

---

## 6. 実装ロードマップ

この研究結果を元にした実装順序：

1. **眉ランドマーク基準点計算** (`compute_brow_anchors`)
   - 小鼻-目尻の直線で眉尻位置計算
   - 眉頭・眉尻の標準位置を算出

2. **眉タイプ定数定義** (`EYEBROW_TYPES` dict)
   - 7タイプのパラメータを main.py に追加

3. **眉ポリライン生成関数** (`generate_brow_polyline`)
   - Bezier曲線で上下ラインを作成
   - ポリゴン化して描画準備

4. **眉描画関数** (`draw_eyebrow`)
   - 眉消し済み画像にポリゴンを描画
   - ソフトエッジ、微細な毛流れテクスチャ追加

5. **顔型別推奨の自動選択** (Phase 2 の顔判定と連携)

---

## 参考文献・引用

- 日本黄金比理論: 小鼻-目尻延長ルール
- 令和版バランス: 眉山は目の2/3〜3/4位置
- MediaPipe FaceLandmarker 478点モデル
