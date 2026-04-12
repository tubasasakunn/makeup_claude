"""
2.2.1 顔全体の縦横バランス判定

計測項目:
    - 顔の縦幅（全頭高の近似）: landmark 10→152 × 1.25 補正（生え際まで）
    - 顔の横幅: 234↔454（耳付け根間, こめかみ）
    - 縦横比 ratio = height / width
    - 3つの理想比との乖離度:
        * 黄金比      1.618
        * 白銀比      1.414
        * 日本人理想比 1.460
    - 実寸換算（虹彩径 11.7mm 基準で mm/pixel を算出）
    - 産総研データ比較 (男性平均 縦23.2cm / 横14.5cm)
    - 小顔度スコア（基準: 縦20cm以下, 横14cm以下）

注意:
    MediaPipe の landmark 10 は実際の生え際ではなくおでこ上部。
    このため raw の face_height_px は真の全頭高より短く、
    aspect は 1.2 付近に偏ってしまう。
    面積補正係数 FOREHEAD_EXTEND = 1.25 を掛けて生え際までを推定する
    （平均的な日本人顔における統計値）。

Usage:
    python main.py <input_image> [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh
from shared.face_metrics import (
    LM,
    draw_line,
    draw_point,
    make_side_by_side,
    measure,
    put_label,
)


# ==============================================================
# 定数
# ==============================================================
# MediaPipe landmark 10 → 真の生え際までの補正係数
FOREHEAD_EXTEND = 1.25

# 理想比率
RATIOS = {
    "golden": 1.618,   # 黄金比 シャープ・クール
    "silver": 1.414,   # 白銀比 安定・親しみ
    "japanese": 1.460, # 日本人理想比 現代的
}

# 虹彩径の標準値（mm）: 成人平均 11.7mm
IRIS_DIAMETER_MM = 11.7

# 産総研男性平均（cm）
AIST_MALE_HEIGHT_CM = 23.2
AIST_MALE_WIDTH_CM = 14.5

# 小顔基準（cm）
KOGAO_HEIGHT_CM = 20.0
KOGAO_WIDTH_CM = 14.0


# ==============================================================
# 計測
# ==============================================================
@dataclass
class FaceRatioResult:
    # pixel 値
    face_height_px_raw: float = 0.0
    face_height_px: float = 0.0       # 補正済み
    face_width_px: float = 0.0
    aspect_raw: float = 0.0
    aspect: float = 0.0               # 補正済み縦/横

    # 理想比との乖離
    losses: dict[str, float] = field(default_factory=dict)  # 小さいほど近い
    closest_ratio: str = ""           # golden/silver/japanese
    impression: str = ""

    # 実寸換算
    mm_per_pixel: float = 0.0
    face_height_cm: float = 0.0
    face_width_cm: float = 0.0

    # 小顔度
    kogao_score: float = 0.0          # 0-100 の複合スコア
    kogao_label: str = ""

    # 産総研比較
    vs_aist_height: float = 0.0       # 平均に対する割合
    vs_aist_width: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _iris_diameter_px(fm) -> float:
    """虹彩径 (ピクセル) を左右平均で返す"""
    ir = np.array([fm.landmarks_px[i].astype(np.float64) for i in LM.IRIS_R])
    il = np.array([fm.landmarks_px[i].astype(np.float64) for i in LM.IRIS_L])
    # ir[1] と ir[3] が左右の端で距離 = 直径
    d_r = float(np.linalg.norm(ir[1] - ir[3]))
    d_l = float(np.linalg.norm(il[1] - il[3]))
    return (d_r + d_l) / 2.0


def _impression_for(ratio_name: str) -> str:
    return {
        "golden": "Sharp / Cool / Refined",
        "silver": "Stable / Harmonious / Friendly",
        "japanese": "Modern Balance",
    }.get(ratio_name, "")


def analyze(fm: FaceMesh) -> FaceRatioResult:
    m = measure(fm)
    r = FaceRatioResult()

    r.face_height_px_raw = m.face_height_px
    r.face_height_px = m.face_height_px * FOREHEAD_EXTEND
    r.face_width_px = m.face_width_temple_px
    r.aspect_raw = m.face_height_px / max(m.face_width_temple_px, 1.0)
    r.aspect = r.face_height_px / max(m.face_width_temple_px, 1.0)

    # 理想比との乖離 (相対%)
    for name, target in RATIOS.items():
        r.losses[name] = abs(r.aspect - target) / target

    r.closest_ratio = min(r.losses.items(), key=lambda kv: kv[1])[0]
    r.impression = _impression_for(r.closest_ratio)

    # 実寸換算
    iris_px = _iris_diameter_px(fm)
    if iris_px > 1e-3:
        r.mm_per_pixel = IRIS_DIAMETER_MM / iris_px
        r.face_height_cm = r.face_height_px * r.mm_per_pixel / 10.0
        r.face_width_cm = r.face_width_px * r.mm_per_pixel / 10.0

    # 小顔度 (100% = 基準値以下)
    if r.face_height_cm > 0 and r.face_width_cm > 0:
        h_score = max(0.0, min(100.0, (KOGAO_HEIGHT_CM / r.face_height_cm) * 100.0))
        w_score = max(0.0, min(100.0, (KOGAO_WIDTH_CM / r.face_width_cm) * 100.0))
        r.kogao_score = (h_score + w_score) / 2.0

        if r.kogao_score >= 95:
            r.kogao_label = "Very Small (+)"
        elif r.kogao_score >= 85:
            r.kogao_label = "Small"
        elif r.kogao_score >= 75:
            r.kogao_label = "Average"
        else:
            r.kogao_label = "Large"

        # 産総研平均比
        r.vs_aist_height = r.face_height_cm / AIST_MALE_HEIGHT_CM
        r.vs_aist_width = r.face_width_cm / AIST_MALE_WIDTH_CM

    return r


# ==============================================================
# 可視化
# ==============================================================
COLOR_H = (0, 220, 255)   # 縦軸 黄
COLOR_W = (255, 200, 0)   # 横軸 水色
COLOR_G = (180, 255, 180) # 参考補助線


def visualize(image: np.ndarray, fm: FaceMesh, r: FaceRatioResult) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]

    p_top = fm.landmarks_px[LM.FOREHEAD_TOP].astype(np.float64)
    p_chin = fm.landmarks_px[LM.CHIN_BOTTOM].astype(np.float64)
    p_tr = fm.landmarks_px[LM.TEMPLE_R].astype(np.float64)
    p_tl = fm.landmarks_px[LM.TEMPLE_L].astype(np.float64)

    # 補正した生え際位置（おでこ上部から face_height_px_raw*0.25 上）
    ext = (r.face_height_px - r.face_height_px_raw)
    p_top_ext = p_top + (p_top - p_chin) / max(r.face_height_px_raw, 1.0) * ext

    # --- 縦軸 ---
    draw_line(img, p_top_ext, p_chin, COLOR_H, 2)
    draw_point(img, p_top_ext, COLOR_H, 4)
    draw_point(img, p_top, (0, 180, 220), 3)  # raw
    draw_point(img, p_chin, COLOR_H, 4)

    # --- 横軸 ---
    draw_line(img, p_tr, p_tl, COLOR_W, 2)
    draw_point(img, p_tr, COLOR_W, 4)
    draw_point(img, p_tl, COLOR_W, 4)

    # --- 理想比率の補助線 (同じ幅で3本 縦の参考点) ---
    for name, target in RATIOS.items():
        ideal_h = r.face_width_px * target
        # 顔中央を基準にして縦線を引く
        cx = (p_tr[0] + p_tl[0]) / 2
        cy = (p_top_ext[1] + p_chin[1]) / 2
        y0 = cy - ideal_h / 2
        y1 = cy + ideal_h / 2
        draw_line(img, (cx + 20, y0), (cx + 20, y1), COLOR_G, 1)
        put_label(
            img, f"{name}:{target}", (int(cx + 25), int(y0 + 10)),
            color=COLOR_G, scale=0.38,
        )

    # --- テキストパネル ---
    panel_w = int(w * 0.50)
    panel_h = int(h * 0.42)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (32, 32, 32)

    put_label(
        panel, "2.2.1 FACE RATIO", (10, 22),
        color=(0, 255, 255), scale=0.55, thickness=2,
    )

    lines = [
        f"height(raw)  : {r.face_height_px_raw:.1f}px",
        f"height(ext)  : {r.face_height_px:.1f}px  (x{FOREHEAD_EXTEND})",
        f"width        : {r.face_width_px:.1f}px",
        f"aspect(raw)  : {r.aspect_raw:.3f}",
        f"aspect(ext)  : {r.aspect:.3f}",
        f"closest ideal: {r.closest_ratio} ({RATIOS[r.closest_ratio]})",
        f"  -> {r.impression}",
    ]
    for i, line in enumerate(lines):
        put_label(panel, line, (10, 44 + i * 16), scale=0.4)

    # 損失バー
    y_bar = 44 + len(lines) * 16 + 8
    put_label(panel, "deviation from ideal:", (10, y_bar), scale=0.4, color=(255, 255, 255))
    y_bar += 14
    for name, loss in r.losses.items():
        best = name == r.closest_ratio
        col = (0, 255, 255) if best else (180, 180, 180)
        put_label(panel, f"{name:9s} {loss*100:5.2f}%", (12, y_bar + 12), scale=0.38, color=col)
        bw = int(200 * min(1.0, loss * 10))  # 10% で バー full
        cv2.rectangle(panel, (140, y_bar + 2), (140 + bw, y_bar + 10), col, -1)
        y_bar += 16

    # mm/cm
    if r.face_height_cm > 0:
        y_bar += 6
        mm_lines = [
            f"mm/pixel     : {r.mm_per_pixel:.3f}mm (iris={IRIS_DIAMETER_MM}mm)",
            f"height       : {r.face_height_cm:.1f}cm  (AIST avg {AIST_MALE_HEIGHT_CM}cm)",
            f"width        : {r.face_width_cm:.1f}cm  (AIST avg {AIST_MALE_WIDTH_CM}cm)",
            f"kogao score  : {r.kogao_score:.1f}  [{r.kogao_label}]",
        ]
        for i, line in enumerate(mm_lines):
            put_label(panel, line, (10, y_bar + 12 + i * 16), scale=0.4,
                      color=(200, 255, 200))

    # 埋め込み
    y0 = h - panel_h - 10
    x0 = 10
    if y0 >= 0 and x0 + panel_w <= w:
        img[y0:y0 + panel_h, x0:x0 + panel_w] = panel
    return img


# ==============================================================
# CLI
# ==============================================================
def run_one(image_path: Path, output_path: Path | None = None,
            imgonly: bool = False, as_json: bool = False):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: 画像を読み込めません: {image_path}")
        return None

    fm = FaceMesh(subdivision_level=1)
    fm.init()
    if fm.detect(image) is None:
        print("Error: 顔が検出されませんでした")
        return None

    r = analyze(fm)

    print(f"\n=== 2.2.1 顔全体の縦横バランス ===")
    print(f"aspect(raw={r.aspect_raw:.3f}, ext={r.aspect:.3f})")
    print(f"closest: {r.closest_ratio} ({RATIOS[r.closest_ratio]}) -> {r.impression}")
    print(f"losses: " + ", ".join(f"{k}={v*100:.2f}%" for k, v in r.losses.items()))
    if r.face_height_cm > 0:
        print(f"size: {r.face_height_cm:.1f}cm x {r.face_width_cm:.1f}cm")
        print(f"kogao: {r.kogao_score:.1f} [{r.kogao_label}]")

    if as_json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))

    vis = visualize(image, fm, r)
    out_path = output_path or image_path.parent / f"face_ratio_{image_path.stem}.png"
    if imgonly:
        cv2.imwrite(str(out_path), vis)
    else:
        cv2.imwrite(str(out_path), make_side_by_side(image, vis))
    print(f"出力: {out_path}")
    return r


def main():
    parser = argparse.ArgumentParser(description="2.2.1 顔全体の縦横バランス判定")
    parser.add_argument("input", help="入力画像パス")
    parser.add_argument("-o", "--output", help="出力画像パス")
    parser.add_argument("--imgonly", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    run_one(
        Path(args.input),
        Path(args.output) if args.output else None,
        imgonly=args.imgonly,
        as_json=args.json,
    )


if __name__ == "__main__":
    main()
