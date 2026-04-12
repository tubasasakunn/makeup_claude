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
    compose_report,
    draw_line_outlined,
    draw_pil_pill,
    draw_pil_text,
    draw_point_outlined,
    draw_radar_chart,
    draw_text_outlined,
    make_side_by_side,
    measure,
    render_report_panel,
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
# カラーパレット (BGR)
C_H = (80, 220, 255)         # 縦軸 (おでこ→あご) - アンバー
C_W = (255, 200, 80)         # 横軸 (左右こめかみ) - シアン
C_CHEEK = (255, 180, 120)    # 頬骨 - 明るい青
C_JAW = (140, 255, 140)      # エラ - 緑
C_FOREHEAD = (220, 160, 220) # おでこ - マゼンタ淡
C_CHIN = (180, 140, 255)     # あご角度 - ラベンダー

# タイプごとの表示色 (BGR)
TYPE_COLORS = {
    "oval": (80, 220, 255),            # アンバー
    "round": (180, 220, 90),           # ミント
    "long": (240, 180, 120),           # ブルー
    "inverted_triangle": (120, 180, 255),  # オレンジレッド
    "base": (200, 140, 255),           # マゼンタ
}


def annotate_face(image: np.ndarray, fm: FaceMesh, result: SkeletalResult,
                  scale: float = 1.0) -> np.ndarray:
    """顔画像にランドマーク線・計測線を重ねる

    scale > 1 を指定すると、まず画像を scale 倍にして、座標も同倍して描画する
    (cv2.resize の後でテキストを描くのでテキストがボケない)。
    """
    if scale != 1.0:
        img = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
        )
    else:
        img = image.copy()
    m = result.metrics
    h, w = img.shape[:2]

    def S(p):
        return (float(p[0]) * scale, float(p[1]) * scale)

    # ---- 計測線 ----
    p_top = S(m.raw["forehead_top"])
    p_chin = S(m.raw["chin"])
    tr = S(m.raw["temple_r"])
    tl = S(m.raw["temple_l"])
    cr = S(m.raw["cheekbone_r"])
    cl = S(m.raw["cheekbone_l"])
    gr = S(m.raw["gonion_r"])
    gl = S(m.raw["gonion_l"])
    fr = S(m.raw["forehead_r"])
    fl = S(m.raw["forehead_l"])

    lw = max(2, int(2 * scale))
    lw_strong = max(3, int(3 * scale))

    draw_line_outlined(img, p_top, p_chin, C_H, thickness=lw)
    draw_line_outlined(img, tr, tl, C_W, thickness=lw)
    draw_line_outlined(img, cr, cl, C_CHEEK, thickness=lw)
    draw_line_outlined(img, gr, gl, C_JAW, thickness=lw_strong)
    draw_line_outlined(img, fr, fl, C_FOREHEAD, thickness=lw)
    draw_line_outlined(img, gr, p_chin, C_CHIN, thickness=lw)
    draw_line_outlined(img, gl, p_chin, C_CHIN, thickness=lw)

    # ---- ポイント ----
    pt_r = max(3, int(4 * scale))
    for p, col in [
        (p_top, C_H), (p_chin, C_H),
        (tr, C_W), (tl, C_W),
        (gr, C_JAW), (gl, C_JAW),
        (cr, C_CHEEK), (cl, C_CHEEK),
        (fr, C_FOREHEAD), (fl, C_FOREHEAD),
    ]:
        draw_point_outlined(img, p, col, r=pt_r)

    # ---- 寸法ラベル (半透明背景付き, PIL で鮮明) ----
    label_bg = (20, 20, 25)
    label_size = max(16, int(18 * scale))
    small_label_size = max(14, int(16 * scale))

    tx_right = max(tr[0], tl[0])
    draw_pil_text(
        img, f"{m.face_width_temple_px:.0f}",
        (tx_right + 12, (tr[1] + tl[1]) / 2 - 12),
        color=C_W, size=label_size, bg=label_bg, bg_alpha=0.78, bg_pad=5,
    )
    gx_right = max(gr[0], gl[0])
    draw_pil_text(
        img, f"{m.face_width_jaw_px:.0f}",
        (gx_right + 12, (gr[1] + gl[1]) / 2 - 12),
        color=C_JAW, size=label_size, bg=label_bg, bg_alpha=0.78, bg_pad=5,
    )
    # 縦: 顎の右下
    draw_pil_text(
        img, f"{m.face_height_px:.0f}",
        (p_chin[0] + 14, p_chin[1] - 12),
        color=C_H, size=label_size, bg=label_bg, bg_alpha=0.78, bg_pad=5,
    )
    # 頬骨幅
    draw_pil_text(
        img, f"{m.face_width_cheekbone_px:.0f}",
        ((cr[0] + cl[0]) / 2 - 18, (cr[1] + cl[1]) / 2 - 26),
        color=C_CHEEK, size=small_label_size,
        bg=label_bg, bg_alpha=0.78, bg_pad=4,
    )

    # ---- 左上に判定ピルバッジ ----
    label_color = TYPE_COLORS.get(result.type, (255, 255, 255))
    pill_size = int(26 * max(1.0, scale * 0.9))
    draw_pil_pill(
        img, result.type.upper(), (18, 18),
        text_color=(20, 20, 30), pill_color=label_color, size=pill_size,
        pad_x=18, pad_y=10, radius=22,
    )

    # ---- 右上に凡例 ----
    legend_items = [
        ("縦 H",   C_H),
        ("横 W",   C_W),
        ("頬骨",   C_CHEEK),
        ("エラ",   C_JAW),
        ("あご角", C_CHIN),
    ]
    legend_font = max(14, int(15 * scale))
    swatch_w = int(18 * scale)
    swatch_h = int(14 * scale)
    row_h = int(26 * scale)
    lx = w - int(140 * scale)
    ly = int(20 * scale)
    # 背景
    cv2.rectangle(
        img,
        (lx - 8, ly - 6),
        (w - 14, ly + row_h * len(legend_items) + 4),
        (0, 0, 0),
        -1,
    )
    for i, (name, col) in enumerate(legend_items):
        y_item = ly + i * row_h
        cv2.rectangle(
            img,
            (lx, y_item + 4),
            (lx + swatch_w, y_item + 4 + swatch_h),
            col, -1,
        )
        draw_pil_text(
            img, name,
            (lx + swatch_w + 8, y_item),
            color=(235, 235, 240), size=legend_font,
        )

    return img


TYPE_JP = {
    "oval": "卵型",
    "round": "丸型",
    "long": "面長",
    "inverted_triangle": "逆三角",
    "base": "ベース",
}


def build_panel(result: SkeletalResult, width: int, height: int) -> np.ndarray:
    """2.1 用のレポートパネルを構築

    - 上段: 判定結果 + 特徴量 KV テーブル
    - 下段: 5 タイプスコアのレーダーチャート (差を相対表示)
    """
    label_color = TYPE_COLORS.get(result.type, (255, 255, 255))

    best_score = result.scores[result.type]
    second = sorted(
        ((k, v) for k, v in result.scores.items() if k != result.type),
        key=lambda kv: -kv[1],
    )[0]
    gap_pct = (best_score - second[1]) / max(best_score, 1e-6) * 100

    spec = [
        ("title", "2.1  骨格タイプ判定", (230, 230, 230)),
        ("subtitle", "Skeletal Type (prototype distance)"),
        ("divider",),
        ("spacer", 4),
        ("section", "判定結果"),
        ("big", result.type.upper(), label_color),
        ("text", result.type_label, (210, 210, 215)),
        ("spacer", 6),
        ("section", "特徴量"),
        ("kv", "縦横比 (H/W)",   f"{result.features.aspect:.3f}"),
        ("kv", "頬 / こめかみ",  f"{result.features.cheek_to_temple:.3f}"),
        ("kv", "エラ / 頬",      f"{result.features.jaw_ratio:.3f}"),
        ("kv", "おでこ / 頬",    f"{result.features.forehead_ratio:.3f}"),
        ("kv", "あご角度",       f"{result.features.chin_angle:.1f} °"),
        ("kv", "下方絞り",       f"{result.features.taper:.3f}"),
        ("spacer", 6),
        ("section", "信頼度"),
        ("kv", f"{TYPE_JP.get(result.type, result.type)} (best)",
            f"{best_score:.3f}", label_color),
        ("kv", f"{TYPE_JP.get(second[0], second[0])} (2nd)",
            f"{second[1]:.3f}"),
        ("kv", "1-2位の差", f"{gap_pct:+.1f} %", (150, 255, 180)),
        ("spacer", 8),
    ]

    # レーダー: 軸ラベルを日本語で、差を強調
    radar_labels = ["卵型", "丸型", "面長", "逆三角", "ベース"]
    radar_keys = ["oval", "round", "long", "inverted_triangle", "base"]
    radar_vals = [result.scores[k] for k in radar_keys]
    rv_min = min(radar_vals)
    rv_max = max(radar_vals)
    rv_range = max(rv_max - rv_min, 1e-6)
    radar_norm = [0.25 + 0.75 * (v - rv_min) / rv_range for v in radar_vals]
    best_idx = radar_keys.index(result.type)
    spec.append(("radar", radar_labels, radar_norm, best_idx, label_color))

    return render_report_panel(spec, width, height)


def visualize(image: np.ndarray, fm: FaceMesh, result: SkeletalResult) -> np.ndarray:
    """後方互換: annotated 画像のみを返す"""
    return annotate_face(image, fm, result)


def build_report(image: np.ndarray, fm: FaceMesh, result: SkeletalResult) -> np.ndarray:
    """Before | Annotated | Panel の横連結画像を返す

    scale 倍に拡大した画像上で annotate するので テキストがボケない。
    """
    scale = 1.5
    src_big = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
    )
    # scale 済みの解像度で annotate (線・テキストがネイティブ解像度で描画される)
    ann_big = annotate_face(image, fm, result, scale=scale)
    h_big = src_big.shape[0]
    panel_w = 620
    panel = build_panel(result, panel_w, h_big)
    return compose_report(src_big, ann_big, panel)


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
    out_path = output_path or image_path.parent / f"skeletal_{image_path.stem}.png"
    if imgonly:
        cv2.imwrite(str(out_path), annotate_face(image, fm, result))
    else:
        cv2.imwrite(str(out_path), build_report(image, fm, result))
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
