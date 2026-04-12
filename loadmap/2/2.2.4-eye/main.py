"""
2.2.4 目の判定

計測項目:
    - 目の縦横比 (理想 1:3)
    - 瞳の黄金比 (白目:黒目:白目 = 1:2:1) 左右別
    - 左右対称性スコア
    - 目の大きさが顔全体に対して占める比率

ランドマーク:
    右目外角 33 / 右目内角 133 / 右目上 159 / 右目下 145
    左目外角 263 / 左目内角 362 / 左目上 386 / 左目下 374
    右虹彩 468-472 / 左虹彩 473-477

Usage:
    python main.py <input_image>
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


@dataclass
class EyeSide:
    width_px: float = 0.0
    height_px: float = 0.0
    ratio: float = 0.0               # height/width
    iris_diameter_px: float = 0.0
    white_left_px: float = 0.0
    black_px: float = 0.0
    white_right_px: float = 0.0
    iris_norm: list[float] = field(default_factory=list)  # white:black:white を 1:? で
    iris_loss: float = 0.0           # 1:2:1 との差


@dataclass
class EyeResult:
    right: EyeSide = field(default_factory=EyeSide)
    left: EyeSide = field(default_factory=EyeSide)
    mean_width_ratio: float = 0.0    # height/width の左右平均
    ideal_ratio_loss: float = 0.0    # 理想 1/3 ≒ 0.333 との差
    symmetry_score: float = 0.0      # 0-1 (1=完全対称)
    eye_to_face_ratio: float = 0.0   # 目の横幅 / 顔横幅
    category: str = ""

    def to_dict(self) -> dict:
        return {
            "right": self.right.__dict__,
            "left": self.left.__dict__,
            "mean_width_ratio": self.mean_width_ratio,
            "ideal_ratio_loss": self.ideal_ratio_loss,
            "symmetry_score": self.symmetry_score,
            "eye_to_face_ratio": self.eye_to_face_ratio,
            "category": self.category,
        }


def _iris_horizontal_span(fm, iris_ids) -> tuple[float, float]:
    xs = [fm.landmarks_px[i][0] for i in iris_ids]
    return float(min(xs)), float(max(xs))


def _measure_side(fm, outer, inner, top, bot, iris_ids) -> EyeSide:
    s = EyeSide()
    po = fm.landmarks_px[outer].astype(np.float64)
    pi = fm.landmarks_px[inner].astype(np.float64)
    pt = fm.landmarks_px[top].astype(np.float64)
    pb = fm.landmarks_px[bot].astype(np.float64)

    s.width_px = float(np.linalg.norm(po - pi))
    s.height_px = float(np.linalg.norm(pt - pb))
    if s.width_px > 1e-3:
        s.ratio = s.height_px / s.width_px

    iris_left_x, iris_right_x = _iris_horizontal_span(fm, iris_ids)
    s.iris_diameter_px = iris_right_x - iris_left_x

    # 目の左右端 X
    eye_left_x = min(po[0], pi[0])
    eye_right_x = max(po[0], pi[0])

    s.white_left_px = max(0.0, iris_left_x - eye_left_x)
    s.black_px = s.iris_diameter_px
    s.white_right_px = max(0.0, eye_right_x - iris_right_x)

    if s.black_px > 1e-3:
        s.iris_norm = [
            s.white_left_px / s.black_px,
            1.0,                              # black を 1 とする (理想 0.5)
            s.white_right_px / s.black_px,
        ]
        # 1:2:1 は "white:black:white = 1:2:1" → 正規化後 white=0.5, black=1
        target = np.array([0.5, 1.0, 0.5])
        vec = np.array(s.iris_norm)
        s.iris_loss = float(np.mean((vec - target) ** 2) ** 0.5)
    return s


def analyze(fm: FaceMesh) -> EyeResult:
    r = EyeResult()
    r.right = _measure_side(
        fm, LM.EYE_OUTER_R, LM.EYE_INNER_R, LM.EYE_TOP_R, LM.EYE_BOT_R, LM.IRIS_R
    )
    r.left = _measure_side(
        fm, LM.EYE_OUTER_L, LM.EYE_INNER_L, LM.EYE_TOP_L, LM.EYE_BOT_L, LM.IRIS_L
    )

    r.mean_width_ratio = (r.right.ratio + r.left.ratio) / 2
    r.ideal_ratio_loss = abs(r.mean_width_ratio - (1 / 3))

    # 対称性 (1 - |r - l| / mean)
    def _sym(a, b):
        m = (a + b) / 2
        return 1.0 - (abs(a - b) / m if m > 1e-3 else 0.0)

    widths_sym = _sym(r.right.width_px, r.left.width_px)
    heights_sym = _sym(r.right.height_px, r.left.height_px)
    r.symmetry_score = max(0.0, min(1.0, (widths_sym + heights_sym) / 2))

    # 目 vs 顔
    m = measure(fm)
    mean_eye_w = (r.right.width_px + r.left.width_px) / 2
    if m.face_width_temple_px > 1e-3:
        r.eye_to_face_ratio = mean_eye_w / m.face_width_temple_px

    # カテゴリ
    if r.mean_width_ratio > 0.40:
        r.category = "Big-Round Eyes"
    elif r.mean_width_ratio < 0.28:
        r.category = "Narrow Eyes"
    else:
        r.category = "Balanced"
    return r


# ==============================================================
# 可視化
# ==============================================================
def visualize(image: np.ndarray, fm: FaceMesh, r: EyeResult) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]

    for outer, inner, top, bot, iris_ids, side in [
        (LM.EYE_OUTER_R, LM.EYE_INNER_R, LM.EYE_TOP_R, LM.EYE_BOT_R, LM.IRIS_R, r.right),
        (LM.EYE_OUTER_L, LM.EYE_INNER_L, LM.EYE_TOP_L, LM.EYE_BOT_L, LM.IRIS_L, r.left),
    ]:
        po = fm.landmarks_px[outer]
        pi = fm.landmarks_px[inner]
        pt = fm.landmarks_px[top]
        pb = fm.landmarks_px[bot]
        # 横幅
        draw_line(img, po, pi, (0, 255, 255), 2)
        # 縦幅
        draw_line(img, pt, pb, (255, 200, 0), 2)
        for p in (po, pi, pt, pb):
            draw_point(img, p, (0, 255, 255), 3)
        # 虹彩の円
        ixs = [fm.landmarks_px[i] for i in iris_ids]
        cx = int(np.mean([p[0] for p in ixs]))
        cy = int(np.mean([p[1] for p in ixs]))
        cv2.circle(img, (cx, cy), max(1, int(side.iris_diameter_px / 2)),
                   (100, 255, 100), 1, cv2.LINE_AA)

    # パネル
    panel_w = int(w * 0.50)
    panel_h = int(h * 0.42)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (32, 32, 32)
    put_label(panel, "2.2.4 EYES", (10, 22), color=(0, 255, 255), scale=0.55, thickness=2)

    lines = [
        f"R: w={r.right.width_px:5.1f} h={r.right.height_px:5.1f} "
        f"ratio={r.right.ratio:.3f}",
        f"   iris={r.right.iris_diameter_px:5.1f}  "
        f"1:2:1 loss={r.right.iris_loss:.3f}",
        f"L: w={r.left.width_px:5.1f} h={r.left.height_px:5.1f} "
        f"ratio={r.left.ratio:.3f}",
        f"   iris={r.left.iris_diameter_px:5.1f}  "
        f"1:2:1 loss={r.left.iris_loss:.3f}",
        f"mean h/w: {r.mean_width_ratio:.3f}  (ideal 0.333)",
        f"ideal loss: {r.ideal_ratio_loss:.3f}",
        f"symmetry : {r.symmetry_score*100:.1f}%",
        f"eye/face : {r.eye_to_face_ratio*100:.1f}%",
        f"category : {r.category}",
    ]
    for i, line in enumerate(lines):
        put_label(panel, line, (10, 44 + i * 18), scale=0.42)

    y0 = h - panel_h - 10
    x0 = 10
    if y0 >= 0 and x0 + panel_w <= w:
        img[y0:y0 + panel_h, x0:x0 + panel_w] = panel
    return img


def run_one(image_path: Path, output_path: Path | None = None,
            imgonly: bool = False, as_json: bool = False):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    fm = FaceMesh(subdivision_level=1); fm.init()
    if fm.detect(image) is None:
        print("顔未検出")
        return None
    r = analyze(fm)
    print(f"\n=== 2.2.4 目 ===")
    print(f"R: ratio={r.right.ratio:.3f}  iris_loss={r.right.iris_loss:.3f}")
    print(f"L: ratio={r.left.ratio:.3f}  iris_loss={r.left.iris_loss:.3f}")
    print(f"mean h/w={r.mean_width_ratio:.3f}  sym={r.symmetry_score:.3f}  eye/face={r.eye_to_face_ratio:.3f}")
    print(f"category: {r.category}")
    if as_json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
    vis = visualize(image, fm, r)
    out = output_path or image_path.parent / f"eye_{image_path.stem}.png"
    cv2.imwrite(str(out), vis if imgonly else make_side_by_side(image, vis))
    print(f"出力: {out}")
    return r


def main():
    parser = argparse.ArgumentParser(description="2.2.4 目の判定")
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--imgonly", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    run_one(Path(args.input), Path(args.output) if args.output else None,
            imgonly=args.imgonly, as_json=args.json)


if __name__ == "__main__":
    main()
