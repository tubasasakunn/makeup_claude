"""
2.2.7 眉の幾何学的判定 (一直線ルール)

計測項目:
    - 眉尻の位置: 小鼻外側 → 目尻 を結んだ延長線上
    - 眉頭の位置: 目頭の真上 / 小鼻の延長線 (垂直)
    - 眉山の位置: 小鼻外側 → 黒目外側 の延長線上
    - 眉の内部比率: 眉頭→眉山 : 眉山→眉尻 = 2:1
    - 水平ラインの判定: 眉頭と眉尻の Y が水平
    - 眉の角度 (眉頭→眉山)
    - 左右対称性

ランドマーク:
    右眉頭 55, 右眉山 105, 右眉尻 46
    左眉頭 285, 左眉山 334, 左眉尻 276
    右目頭 133, 右目尻 33, 左目頭 362, 左目尻 263
    右虹彩外 (IRIS_R[3]), 左虹彩外 (IRIS_L[3])
    右小鼻 64, 左小鼻 294

Usage:
    python main.py <input_image>
"""

from __future__ import annotations

import argparse
import json
import math
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
    put_label,
)


@dataclass
class BrowSide:
    head: tuple = (0.0, 0.0)
    peak: tuple = (0.0, 0.0)
    tail: tuple = (0.0, 0.0)

    head_deviation_px: float = 0.0       # 眉頭が目頭垂直線からずれた量
    tail_deviation_px: float = 0.0       # 眉尻が小鼻-目尻線からずれた量
    peak_deviation_px: float = 0.0       # 眉山が小鼻-虹彩外線からずれた量

    head_to_peak_px: float = 0.0
    peak_to_tail_px: float = 0.0
    peak_ratio: float = 0.0              # head_to_peak / peak_to_tail (ideal 2.0)

    horizontal_tilt_deg: float = 0.0     # 眉頭→眉尻 の Y 差 / 長さ の角度
    angle_head_to_peak_deg: float = 0.0  # 眉頭→眉山 の上昇角 (+は上)


@dataclass
class EyebrowResult:
    right: BrowSide = field(default_factory=BrowSide)
    left: BrowSide = field(default_factory=BrowSide)
    symmetry_score: float = 0.0
    category: str = ""

    def to_dict(self) -> dict:
        return {
            "right": self.right.__dict__,
            "left": self.left.__dict__,
            "symmetry_score": self.symmetry_score,
            "category": self.category,
        }


def _p(fm, idx) -> np.ndarray:
    return fm.landmarks_px[idx].astype(np.float64)


def _signed_distance(line_a: np.ndarray, line_b: np.ndarray, pt: np.ndarray) -> float:
    """点 pt の、直線 ab からの符号付き垂直距離"""
    ab = line_b - line_a
    ap = pt - line_a
    denom = np.linalg.norm(ab)
    if denom < 1e-6:
        return 0.0
    return float((ab[0] * ap[1] - ab[1] * ap[0]) / denom)


def _measure_side(fm, head_id, peak_id, tail_id,
                  eye_in_id, eye_out_id, iris_outer_id, nose_wing_id) -> BrowSide:
    s = BrowSide()
    head = _p(fm, head_id)
    peak = _p(fm, peak_id)
    tail = _p(fm, tail_id)
    eye_in = _p(fm, eye_in_id)
    eye_out = _p(fm, eye_out_id)
    iris_out = _p(fm, iris_outer_id)
    nose = _p(fm, nose_wing_id)

    s.head = tuple(head)
    s.peak = tuple(peak)
    s.tail = tuple(tail)

    # 眉頭位置: 目頭の真上 = eye_in の X 一致
    s.head_deviation_px = abs(head[0] - eye_in[0])
    # 眉山位置: 小鼻外側 → 黒目外側 を結んだ線上
    s.peak_deviation_px = abs(_signed_distance(nose, iris_out, peak))
    # 眉尻位置: 小鼻外側 → 目尻 を結んだ線の延長
    s.tail_deviation_px = abs(_signed_distance(nose, eye_out, tail))

    s.head_to_peak_px = float(np.linalg.norm(peak - head))
    s.peak_to_tail_px = float(np.linalg.norm(tail - peak))
    if s.peak_to_tail_px > 1e-3:
        s.peak_ratio = s.head_to_peak_px / s.peak_to_tail_px

    # 水平傾き (Y差 / X差)
    dx = tail[0] - head[0]
    dy = tail[1] - head[1]
    length = max(np.hypot(dx, dy), 1e-6)
    # 負角 = 眉頭より眉尻が上がっている
    s.horizontal_tilt_deg = math.degrees(math.asin(dy / length))

    dx2 = peak[0] - head[0]
    dy2 = peak[1] - head[1]
    len2 = max(np.hypot(dx2, dy2), 1e-6)
    s.angle_head_to_peak_deg = -math.degrees(math.asin(dy2 / len2))  # 上がり+
    return s


def analyze(fm: FaceMesh) -> EyebrowResult:
    r = EyebrowResult()
    r.right = _measure_side(
        fm,
        head_id=LM.BROW_HEAD_R, peak_id=LM.BROW_PEAK_R, tail_id=LM.BROW_TAIL_R,
        eye_in_id=LM.EYE_INNER_R, eye_out_id=LM.EYE_OUTER_R,
        iris_outer_id=LM.IRIS_R[3], nose_wing_id=LM.NOSE_WING_R,
    )
    r.left = _measure_side(
        fm,
        head_id=LM.BROW_HEAD_L, peak_id=LM.BROW_PEAK_L, tail_id=LM.BROW_TAIL_L,
        eye_in_id=LM.EYE_INNER_L, eye_out_id=LM.EYE_OUTER_L,
        iris_outer_id=LM.IRIS_L[3], nose_wing_id=LM.NOSE_WING_L,
    )

    # 対称性
    def sym(a, b):
        m = (a + b) / 2
        return 1.0 - (abs(a - b) / m if m > 1e-3 else 0.0)

    sims = [
        sym(r.right.head_to_peak_px, r.left.head_to_peak_px),
        sym(r.right.peak_to_tail_px, r.left.peak_to_tail_px),
        sym(
            abs(r.right.horizontal_tilt_deg),
            abs(r.left.horizontal_tilt_deg),
        ),
    ]
    r.symmetry_score = max(0.0, min(1.0, sum(sims) / len(sims)))

    # カテゴリ: 眉頭→眉山 の角度平均
    avg_peak_angle = (r.right.angle_head_to_peak_deg + r.left.angle_head_to_peak_deg) / 2
    if avg_peak_angle >= 8:
        r.category = "Masculine Rising (>=8 deg)"
    elif avg_peak_angle >= 3:
        r.category = "Natural"
    else:
        r.category = "Flat / Parallel"
    return r


# ==============================================================
# 可視化
# ==============================================================
def visualize(image: np.ndarray, fm: FaceMesh, r: EyebrowResult) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]

    # 右眉の基準線を描画
    for side, head_id, peak_id, tail_id, eye_in_id, eye_out_id, iris_out_id, nose_id in [
        (r.right, LM.BROW_HEAD_R, LM.BROW_PEAK_R, LM.BROW_TAIL_R,
         LM.EYE_INNER_R, LM.EYE_OUTER_R, LM.IRIS_R[3], LM.NOSE_WING_R),
        (r.left, LM.BROW_HEAD_L, LM.BROW_PEAK_L, LM.BROW_TAIL_L,
         LM.EYE_INNER_L, LM.EYE_OUTER_L, LM.IRIS_L[3], LM.NOSE_WING_L),
    ]:
        head = fm.landmarks_px[head_id]
        peak = fm.landmarks_px[peak_id]
        tail = fm.landmarks_px[tail_id]
        eye_in = fm.landmarks_px[eye_in_id]
        eye_out = fm.landmarks_px[eye_out_id]
        iris_out = fm.landmarks_px[iris_out_id]
        nose = fm.landmarks_px[nose_id]

        # 眉の3点を結ぶ
        draw_line(img, head, peak, (0, 255, 255), 2)
        draw_line(img, peak, tail, (0, 255, 255), 2)

        # 基準線（薄め）
        # 眉頭垂直: eye_in の x で上下
        draw_line(img, (int(eye_in[0]), int(head[1]) - 30), (int(eye_in[0]), int(head[1]) + 10),
                  (255, 100, 100), 1)
        # 小鼻 → 目尻 の延長 (眉尻基準)
        vec = tail - nose
        ext = nose + vec * 1.3
        draw_line(img, nose, ext, (100, 255, 100), 1)
        # 小鼻 → 虹彩外 の延長 (眉山基準)
        vec2 = peak - nose
        ext2 = nose + vec2 * 1.1
        draw_line(img, nose, ext2, (255, 200, 100), 1)

        draw_point(img, head, (255, 255, 0), 4)
        draw_point(img, peak, (0, 255, 0), 4)
        draw_point(img, tail, (255, 100, 100), 4)

    # パネル
    panel_w = int(w * 0.52)
    panel_h = int(h * 0.40)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (32, 32, 32)
    put_label(panel, "2.2.7 EYEBROW GEOMETRY", (10, 22), color=(0, 255, 255), scale=0.55, thickness=2)

    def _fmt(side: BrowSide, label: str) -> list[str]:
        return [
            f"{label} head dev   : {side.head_deviation_px:4.1f}px",
            f"{label} peak dev   : {side.peak_deviation_px:4.1f}px",
            f"{label} tail dev   : {side.tail_deviation_px:4.1f}px",
            f"{label} h2p/p2t    : {side.peak_ratio:.2f} (ideal 2.0)",
            f"{label} peak angle : {side.angle_head_to_peak_deg:+.1f} deg",
        ]

    lines = _fmt(r.right, "R") + _fmt(r.left, "L") + [
        f"symmetry : {r.symmetry_score*100:.1f}%",
        f"category : {r.category}",
    ]
    for i, line in enumerate(lines):
        put_label(panel, line, (10, 42 + i * 16), scale=0.38)

    y0 = h - panel_h - 10
    x0 = 10
    if y0 >= 0 and x0 + panel_w <= w:
        img[y0:y0 + panel_h, x0:x0 + panel_w] = panel
    return img


def run_one(image_path: Path, output_path=None, imgonly=False, as_json=False):
    image = cv2.imread(str(image_path))
    if image is None: return None
    fm = FaceMesh(subdivision_level=1); fm.init()
    if fm.detect(image) is None:
        print("顔未検出"); return None
    r = analyze(fm)
    print(f"\n=== 2.2.7 眉 ===")
    print(f"R peak_ratio={r.right.peak_ratio:.2f}  angle={r.right.angle_head_to_peak_deg:.1f}")
    print(f"L peak_ratio={r.left.peak_ratio:.2f}  angle={r.left.angle_head_to_peak_deg:.1f}")
    print(f"sym={r.symmetry_score*100:.1f}%  category={r.category}")
    if as_json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
    vis = visualize(image, fm, r)
    out = output_path or image_path.parent / f"brow_{image_path.stem}.png"
    cv2.imwrite(str(out), vis if imgonly else make_side_by_side(image, vis))
    print(f"出力: {out}")
    return r


def main():
    parser = argparse.ArgumentParser(description="2.2.7 眉の幾何学的判定")
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--imgonly", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    run_one(Path(args.input), Path(args.output) if args.output else None,
            imgonly=args.imgonly, as_json=args.json)


if __name__ == "__main__":
    main()
