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
    compose_report,
    draw_line_outlined,
    draw_pil_pill,
    draw_pil_text,
    draw_point_outlined,
    make_side_by_side,
    render_report_panel,
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
# 可視化 (UI/UX 改善版: レポートパネル + annotate 分離)
# ==============================================================
C_BROW = (80, 220, 255)         # 眉本体 アンバー
C_REF_HEAD = (200, 140, 255)    # 眉頭垂直 マゼンタ
C_REF_TAIL = (140, 255, 160)    # 小鼻→目尻 緑
C_REF_PEAK = (255, 200, 80)     # 小鼻→虹彩外 シアン
C_HEAD_PT = (255, 220, 80)
C_PEAK_PT = (140, 255, 160)
C_TAIL_PT = (200, 140, 255)

CATEGORY_COLOR = {
    "Masculine Rising (>=8 deg)": (200, 140, 255),
    "Natural": (140, 255, 160),
    "Flat / Parallel": (80, 180, 255),
}


def annotate_face(image: np.ndarray, fm: FaceMesh, r: EyebrowResult,
                  scale: float = 1.0) -> np.ndarray:
    """眉本体の線 (頭→山→尻) のみ + 眉山比ラベル 1 つ"""
    if scale != 1.0:
        img = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
        )
    else:
        img = image.copy()

    def S(p):
        return (float(p[0]) * scale, float(p[1]) * scale)

    lw = max(2, int(2 * scale))
    pt_r = max(3, int(4 * scale))

    for head_id, peak_id, tail_id in [
        (LM.BROW_HEAD_R, LM.BROW_PEAK_R, LM.BROW_TAIL_R),
        (LM.BROW_HEAD_L, LM.BROW_PEAK_L, LM.BROW_TAIL_L),
    ]:
        head = S(fm.landmarks_px[head_id])
        peak = S(fm.landmarks_px[peak_id])
        tail = S(fm.landmarks_px[tail_id])

        draw_line_outlined(img, head, peak, C_BROW, thickness=lw)
        draw_line_outlined(img, peak, tail, C_BROW, thickness=lw)

        draw_point_outlined(img, head, C_HEAD_PT, r=pt_r)
        draw_point_outlined(img, peak, C_PEAK_PT, r=pt_r)
        draw_point_outlined(img, tail, C_TAIL_PT, r=pt_r)

    # ---- ラベル: 平均眉山比 1 つだけ ----
    avg_peak_ratio = (r.right.peak_ratio + r.left.peak_ratio) / 2
    label_bg = (20, 20, 25)
    label_size = max(18, int(20 * scale))
    # 右眉の上に表示
    head_r = S(fm.landmarks_px[LM.BROW_HEAD_R])
    draw_pil_text(
        img, f"眉山比 = {avg_peak_ratio:.2f}",
        (head_r[0] - int(80 * scale), head_r[1] - int(36 * scale)),
        color=(240, 240, 245), size=label_size,
        bg=label_bg, bg_alpha=0.78, bg_pad=6,
    )

    # 左上ピル
    pill_color = CATEGORY_COLOR.get(r.category, (255, 255, 255))
    pill_text = r.category.split(" ")[0].upper()
    pill_size = int(24 * max(1.0, scale * 0.9))
    draw_pil_pill(
        img, pill_text, (18, 18),
        text_color=(20, 20, 30), pill_color=pill_color, size=pill_size,
        pad_x=18, pad_y=10, radius=22,
    )

    return img


def build_panel(r: EyebrowResult, width: int, height: int) -> np.ndarray:
    pill_color = CATEGORY_COLOR.get(r.category, (255, 255, 255))
    sym_color = (140, 255, 160) if r.symmetry_score >= 0.9 else (230, 230, 235)
    IDEAL_COL = (200, 200, 205)

    avg_peak_ratio = (r.right.peak_ratio + r.left.peak_ratio) / 2
    compare_items = [
        ("右眉", r.right.peak_ratio, f"{r.right.peak_ratio:.2f}", pill_color),
        ("左眉", r.left.peak_ratio, f"{r.left.peak_ratio:.2f}", pill_color),
        ("理想", 2.0, "2.0", IDEAL_COL),
    ]
    diff_pct = (avg_peak_ratio - 2.0) / 2.0 * 100

    spec = [
        ("title", "2.2.7  眉", (230, 230, 230)),
        ("subtitle", "Eyebrow Geometry"),
        ("divider",),
        ("spacer", 4),
        ("section", "判定結果"),
        ("big", r.category.split(" ")[0].upper(), pill_color),
        ("text", r.category, (210, 210, 215)),
        ("spacer", 8),
        ("section", "眉山比 (頭→山:山→尻, 理想 2.0)"),
        ("ratio_compare", compare_items, "右眉"),
        ("spacer", 4),
        ("section", "理想からの差"),
        ("diff_bar", "", diff_pct, 50.0, pill_color),
        ("spacer", 6),
        ("section", "対称性"),
        ("kv", "symmetry", f"{r.symmetry_score*100:.1f} %", sym_color),
    ]

    return render_report_panel(spec, width, height)


def build_report(image: np.ndarray, fm: FaceMesh,
                 r: EyebrowResult) -> np.ndarray:
    scale = 1.5
    src_big = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
    )
    ann_big = annotate_face(image, fm, r, scale=scale)
    panel = build_panel(r, 620, src_big.shape[0])
    return compose_report(src_big, ann_big, panel)


# 後方互換
def visualize(image: np.ndarray, fm: FaceMesh, r: EyebrowResult) -> np.ndarray:
    return annotate_face(image, fm, r, scale=1.0)


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
    out = output_path or image_path.parent / f"brow_{image_path.stem}.png"
    if imgonly:
        cv2.imwrite(str(out), annotate_face(image, fm, r))
    else:
        cv2.imwrite(str(out), build_report(image, fm, r))
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
