"""
2.1 骨格による顔判定 - 5つの骨格タイプに分類する

タイプ:
    oval             卵型   縦/横=1.5前後、輪郭にカドがない
    round            丸型   縦≈横、頬ふっくら
    long             面長   縦/横 > 1.5
    inverted_triangle 逆三角 おでこ広い、あご先シュッ
    base             ベース型 エラ張り、あご先平ら

ロジック:
    1. MediaPipe で顔ランドマークを計測
    2. 6つの特徴量を算出
       - aspect          = face_height / face_width_temple
       - cheek_to_temple = cheekbone_width / temple_width
       - jaw_ratio       = jaw_width / cheekbone_width
       - forehead_ratio  = forehead_width / cheekbone_width
       - chin_angle      = エラ-あご先-エラ のなす角
       - taper           = (cheekbone - jaw) / cheekbone  (下向きの絞り込み度)
    3. 各タイプのスコア関数に通して最大スコアを選ぶ

Usage:
    python main.py <input_image> [options]

Examples:
    python main.py imgs/卵.png
    python main.py imgs/丸顔.png -o result.png
    python main.py imgs/面長.png --json   # 判定の数値を JSON で表示
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# 共有モジュール
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh
from shared.face_metrics import (
    FaceMetrics,
    LM,
    draw_line,
    draw_point,
    make_side_by_side,
    measure,
    put_label,
)


# ==============================================================
# 骨格タイプ定義
# ==============================================================
SKELETAL_TYPES = {
    "oval":              "卵型 (縦1.5:横1・なめらかな輪郭)",
    "round":             "丸型 (縦≈横・頬ふっくら)",
    "long":              "面長 (縦>>横)",
    "inverted_triangle": "逆三角形 (おでこ広・あご先シュッ)",
    "base":              "ベース型 (エラ張り・あご平ら)",
}


@dataclass
class SkeletalFeatures:
    aspect: float          # height / temple_width
    cheek_to_temple: float # cheekbone / temple
    jaw_ratio: float       # jaw / cheekbone
    forehead_ratio: float  # forehead / cheekbone
    chin_angle: float      # degrees
    taper: float           # (cheek - jaw) / cheek


def extract_features(m: FaceMetrics) -> SkeletalFeatures:
    """FaceMetrics から骨格判定用の特徴量を抽出"""
    # ゼロ割防止
    temple = max(m.face_width_temple_px, 1.0)
    cheek = max(m.face_width_cheekbone_px, 1.0)

    aspect = m.face_height_px / temple
    cheek_to_temple = cheek / temple
    jaw_ratio = m.face_width_jaw_px / cheek
    forehead_ratio = m.forehead_width_px / cheek
    chin_angle = m.chin_angle_deg
    taper = (cheek - m.face_width_jaw_px) / cheek

    return SkeletalFeatures(
        aspect=aspect,
        cheek_to_temple=cheek_to_temple,
        jaw_ratio=jaw_ratio,
        forehead_ratio=forehead_ratio,
        chin_angle=chin_angle,
        taper=taper,
    )


# ==============================================================
# スコア関数 (プロトタイプベース)
# ==============================================================
# 注意: MediaPipe の landmark 10 は実際の「生え際」ではなくおでこ上部のため、
#       face_height_px は真の顔長さより短く出る。結果として aspect は 1.19-1.23 の
#       狭いレンジに収まる。そこで「絶対値」での判定ではなく、
#       各タイプのプロトタイプ（実測5サンプルの傾向）からの重み付き差分で判定する。
#
# プロトタイプは POC 画像 (imgs/) とロードマップの骨格タイプ定義から決定：
#   - jaw_ratio : エラ幅 / 頬骨幅。ベースで最大、逆三角で最小
#   - chin_angle: 広いほど平ら（ベース）、狭いほど尖る（逆三角）
#   - aspect    : 面長で最大
#   - taper     : 逆三角で最大（下に絞り込み）
#   - forehead_ratio: ほぼ一定なので重み低
PROTOTYPES: dict[str, dict[str, float]] = {
    "base": {
        "jaw_ratio": 0.830, "chin_angle": 125.0, "aspect": 1.190,
        "taper": 0.170, "forehead_ratio": 0.858,
    },
    "round": {
        "jaw_ratio": 0.817, "chin_angle": 120.0, "aspect": 1.203,
        "taper": 0.183, "forehead_ratio": 0.853,
    },
    "oval": {
        "jaw_ratio": 0.795, "chin_angle": 116.5, "aspect": 1.191,
        "taper": 0.206, "forehead_ratio": 0.846,
    },
    "inverted_triangle": {
        "jaw_ratio": 0.760, "chin_angle": 112.0, "aspect": 1.204,
        "taper": 0.241, "forehead_ratio": 0.858,
    },
    "long": {
        "jaw_ratio": 0.785, "chin_angle": 116.0, "aspect": 1.229,
        "taper": 0.215, "forehead_ratio": 0.858,
    },
}

# 各特徴量の寸度（1に正規化するための分母）。実測レンジを反映。
FEATURE_SCALE: dict[str, float] = {
    "jaw_ratio": 0.035,        # 0.76-0.83 レンジ → ±0.035
    "chin_angle": 7.0,         # 112-126 度 → ±7
    "aspect": 0.020,           # 1.19-1.23 → ±0.02
    "taper": 0.040,            # 0.17-0.24 → ±0.035
    "forehead_ratio": 0.015,   # 0.845-0.860 狭い
}

# 寄与の重み（判別に効く特徴を大きく）
FEATURE_WEIGHT: dict[str, float] = {
    "jaw_ratio": 1.8,
    "chin_angle": 1.6,
    "aspect": 1.2,
    "taper": 1.3,
    "forehead_ratio": 0.4,
}


def _feature_vector(f: SkeletalFeatures) -> dict[str, float]:
    return {
        "jaw_ratio": f.jaw_ratio,
        "chin_angle": f.chin_angle,
        "aspect": f.aspect,
        "taper": f.taper,
        "forehead_ratio": f.forehead_ratio,
    }


def score_types(f: SkeletalFeatures) -> dict[str, float]:
    """各タイプのスコアを辞書で返す (0-1)

    プロトタイプ中心からの重み付きユークリッド距離をガウシアンで 0-1 に変換。
    一番近いタイプがスコア 1.0 付近、遠いタイプは 0 付近になる。
    """
    fv = _feature_vector(f)
    distances: dict[str, float] = {}
    for type_name, proto in PROTOTYPES.items():
        d2 = 0.0
        for key, center in proto.items():
            diff = (fv[key] - center) / FEATURE_SCALE[key]
            d2 += FEATURE_WEIGHT[key] * diff * diff
        distances[type_name] = d2

    # 距離 → スコア (exp(-d^2 / 2σ^2) 相当)
    # σ=2.5 ぐらいで「近い=高スコア、遠い=低スコア」を演出
    sigma2 = 2.5 ** 2
    scores = {k: float(np.exp(-v / (2 * sigma2))) for k, v in distances.items()}
    return scores


# ==============================================================
# 判定エントリポイント
# ==============================================================
@dataclass
class SkeletalResult:
    type: str
    type_label: str
    features: SkeletalFeatures
    scores: dict[str, float]
    metrics: FaceMetrics

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "type_label": self.type_label,
            "features": self.features.__dict__,
            "scores": self.scores,
            "metrics": {k: v for k, v in self.metrics.__dict__.items() if k != "raw"},
        }


def classify(fm: FaceMesh) -> SkeletalResult:
    m = measure(fm)
    f = extract_features(m)
    scores = score_types(f)
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return SkeletalResult(
        type=best,
        type_label=SKELETAL_TYPES[best],
        features=f,
        scores=scores,
        metrics=m,
    )


# ==============================================================
# 可視化
# ==============================================================
COLOR_H = (0, 220, 255)   # 縦軸 黄
COLOR_W = (255, 200, 0)   # 横軸 水色
COLOR_J = (120, 255, 120) # エラ 緑
COLOR_F = (180, 180, 255) # おでこ 薄赤


def visualize(image: np.ndarray, fm: FaceMesh, result: SkeletalResult) -> np.ndarray:
    """判定結果を可視化した画像を返す（元画像を上書きしない）"""
    img = image.copy()
    m = result.metrics

    # --- 縦軸 10 ↔ 152 ---
    p_top = m.raw["forehead_top"]
    p_chin = m.raw["chin"]
    draw_line(img, p_top, p_chin, COLOR_H, 2)
    draw_point(img, p_top, COLOR_H, 4)
    draw_point(img, p_chin, COLOR_H, 4)

    # --- 横軸 (こめかみ) 234 ↔ 454 ---
    tr = m.raw["temple_r"]
    tl = m.raw["temple_l"]
    draw_line(img, tr, tl, COLOR_W, 2)
    draw_point(img, tr, COLOR_W, 4)
    draw_point(img, tl, COLOR_W, 4)

    # --- 頬骨幅 127 ↔ 356 ---
    cr = m.raw["cheekbone_r"]
    cl = m.raw["cheekbone_l"]
    draw_line(img, cr, cl, (0, 180, 255), 1)

    # --- エラ 172 ↔ 397 ---
    gr = m.raw["gonion_r"]
    gl = m.raw["gonion_l"]
    draw_line(img, gr, gl, COLOR_J, 2)
    draw_point(img, gr, COLOR_J, 4)
    draw_point(img, gl, COLOR_J, 4)

    # --- おでこ 54 ↔ 284 ---
    fr = m.raw["forehead_r"]
    fl = m.raw["forehead_l"]
    draw_line(img, fr, fl, COLOR_F, 2)

    # --- あご角度 ---
    draw_line(img, gr, p_chin, (255, 150, 150), 1)
    draw_line(img, gl, p_chin, (255, 150, 150), 1)

    # --- テキストパネル ---
    h, w = img.shape[:2]
    panel_w = int(w * 0.50)
    panel_h = int(h * 0.32)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (32, 32, 32)

    # 判定結果
    put_label(
        panel, f"TYPE: {result.type.upper()}", (10, 22),
        color=(0, 255, 255), scale=0.65, thickness=2,
    )
    put_label(
        panel, result.type_label, (10, 44),
        color=(255, 255, 255), scale=0.4,
    )

    # 特徴量
    f_txt = [
        f"aspect (H/W)   : {result.features.aspect:.3f}",
        f"cheek/temple   : {result.features.cheek_to_temple:.3f}",
        f"jaw/cheek      : {result.features.jaw_ratio:.3f}",
        f"forehead/cheek : {result.features.forehead_ratio:.3f}",
        f"chin angle     : {result.features.chin_angle:.1f} deg",
        f"taper          : {result.features.taper:.3f}",
    ]
    for i, line in enumerate(f_txt):
        put_label(panel, line, (10, 68 + i * 16), scale=0.4)

    # スコアバー
    bar_x0 = panel_w - 160
    bar_w_max = 140
    for i, (name, score) in enumerate(
        sorted(result.scores.items(), key=lambda kv: -kv[1])
    ):
        y = 22 + i * 22
        is_best = name == result.type
        col = (0, 255, 255) if is_best else (180, 180, 180)
        put_label(panel, f"{name[:11]:11s}", (bar_x0 - 80, y), color=col, scale=0.38)
        bw = int(bar_w_max * min(1.0, score / (max(result.scores.values()) + 1e-9)))
        cv2.rectangle(panel, (bar_x0, y - 10), (bar_x0 + bw, y - 2), col, -1)
        put_label(panel, f"{score:.3f}", (bar_x0 + bar_w_max + 4, y), color=col, scale=0.35)

    # 左下に埋め込み
    y0 = h - panel_h - 10
    x0 = 10
    if y0 >= 0 and x0 + panel_w <= w:
        img[y0:y0 + panel_h, x0:x0 + panel_w] = panel
    return img


# ==============================================================
# CLI
# ==============================================================
def run_one(image_path: Path, output_path: Path | None = None,
            imgonly: bool = False, as_json: bool = False) -> SkeletalResult | None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: 画像を読み込めません: {image_path}")
        return None

    fm = FaceMesh(subdivision_level=1)
    fm.init()
    if fm.detect(image) is None:
        print("Error: 顔が検出されませんでした")
        return None

    result = classify(fm)

    print(f"\n=== 判定結果 ===")
    print(f"TYPE: {result.type}   ({result.type_label})")
    print(f"features: {result.features}")
    print("scores:")
    for k, v in sorted(result.scores.items(), key=lambda kv: -kv[1]):
        mark = " <-- BEST" if k == result.type else ""
        print(f"  {k:20s}: {v:.4f}{mark}")

    if as_json:
        print("\n--- JSON ---")
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    # 画像出力
    vis = visualize(image, fm, result)
    out_path = output_path or image_path.parent / f"skeletal_{image_path.stem}.png"
    if imgonly:
        cv2.imwrite(str(out_path), vis)
    else:
        comparison = make_side_by_side(image, vis)
        cv2.imwrite(str(out_path), comparison)
    print(f"出力: {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="2.1 骨格による顔判定")
    parser.add_argument("input", help="入力画像パス")
    parser.add_argument("-o", "--output", help="出力画像パス")
    parser.add_argument("--imgonly", action="store_true", help="比較画像ではなく結果画像のみ出力")
    parser.add_argument("--json", action="store_true", help="計測値をJSON形式で表示")
    args = parser.parse_args()

    run_one(
        Path(args.input),
        Path(args.output) if args.output else None,
        imgonly=args.imgonly,
        as_json=args.json,
    )


if __name__ == "__main__":
    main()
