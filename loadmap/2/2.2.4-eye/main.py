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
    compose_report,
    draw_line_outlined,
    draw_pil_pill,
    draw_pil_text,
    draw_point_outlined,
    make_side_by_side,
    measure,
    render_report_panel,
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
# 可視化 (UI/UX 改善版: レポートパネル + annotate 分離)
# ==============================================================
C_W = (80, 220, 255)       # 横幅 アンバー
C_H = (255, 200, 80)       # 縦幅 シアン
C_IRIS = (140, 255, 160)   # 虹彩 緑

CATEGORY_COLOR = {
    "Big-Round Eyes": (200, 140, 255),
    "Narrow Eyes": (80, 180, 255),
    "Balanced": (140, 255, 160),
}


def annotate_face(image: np.ndarray, fm: FaceMesh, r: EyeResult,
                  scale: float = 1.0) -> np.ndarray:
    """目の幅・高さ線と虹彩円のみ + 縦/横ラベル 1 つ"""
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

    for outer, inner, top, bot, iris_ids, side in [
        (LM.EYE_OUTER_R, LM.EYE_INNER_R, LM.EYE_TOP_R, LM.EYE_BOT_R, LM.IRIS_R, r.right),
        (LM.EYE_OUTER_L, LM.EYE_INNER_L, LM.EYE_TOP_L, LM.EYE_BOT_L, LM.IRIS_L, r.left),
    ]:
        po = S(fm.landmarks_px[outer])
        pi = S(fm.landmarks_px[inner])
        pt = S(fm.landmarks_px[top])
        pb = S(fm.landmarks_px[bot])

        draw_line_outlined(img, po, pi, C_W, thickness=lw)
        draw_line_outlined(img, pt, pb, C_H, thickness=lw)
        for p in (po, pi, pt, pb):
            draw_point_outlined(img, p, C_W, r=pt_r)

        ixs = [fm.landmarks_px[i] for i in iris_ids]
        cx = float(np.mean([p[0] for p in ixs])) * scale
        cy = float(np.mean([p[1] for p in ixs])) * scale
        rad = max(1, int(side.iris_diameter_px * scale / 2))
        cv2.circle(
            img, (int(cx), int(cy)), rad + 1, (0, 0, 0), 2, cv2.LINE_AA,
        )
        cv2.circle(
            img, (int(cx), int(cy)), rad, C_IRIS, 1, cv2.LINE_AA,
        )

    # ---- ラベル: 縦/横 1 つだけ (両目の中央下) ----
    label_bg = (20, 20, 25)
    label_size = max(18, int(20 * scale))
    po_r = S(fm.landmarks_px[LM.EYE_OUTER_R])
    pi_l = S(fm.landmarks_px[LM.EYE_OUTER_L])
    cx_mid = (po_r[0] + pi_l[0]) / 2
    cy_mid = (po_r[1] + pi_l[1]) / 2
    draw_pil_text(
        img, f"縦/横 = {r.mean_width_ratio:.3f}",
        (cx_mid - int(70 * scale), cy_mid + int(20 * scale)),
        color=(240, 240, 245), size=label_size,
        bg=label_bg, bg_alpha=0.78, bg_pad=6,
    )

    # 左上ピル
    pill_color = CATEGORY_COLOR.get(r.category, (255, 255, 255))
    pill_text = r.category.upper()
    pill_size = int(22 * max(1.0, scale * 0.9))
    draw_pil_pill(
        img, pill_text, (18, 18),
        text_color=(20, 20, 30), pill_color=pill_color, size=pill_size,
        pad_x=18, pad_y=10, radius=22,
    )

    return img


def build_panel(r: EyeResult, width: int, height: int) -> np.ndarray:
    pill_color = CATEGORY_COLOR.get(r.category, (255, 255, 255))
    sym_color = (140, 255, 160) if r.symmetry_score >= 0.9 else (230, 230, 235)
    IDEAL_COL = (200, 200, 205)

    compare_items = [
        ("あなた", r.mean_width_ratio, f"{r.mean_width_ratio:.2f}", pill_color),
        ("理想",   0.333, "0.33", IDEAL_COL),
    ]
    diff_pct = (r.mean_width_ratio - 0.333) / 0.333 * 100

    spec = [
        ("title", "2.2.4  目", (230, 230, 230)),
        ("subtitle", "Eye Ratio / Symmetry"),
        ("divider",),
        ("spacer", 4),
        ("section", "判定結果"),
        ("big", r.category.upper(), pill_color),
        ("spacer", 8),
        ("section", "縦/横 比率 (理想 0.333 = 1:3)"),
        ("ratio_compare", compare_items, "あなた"),
        ("spacer", 4),
        ("section", "理想からの差"),
        ("diff_bar", "", diff_pct, 50.0, pill_color),
        ("spacer", 6),
        ("section", "対称性"),
        ("kv", "symmetry", f"{r.symmetry_score*100:.1f} %", sym_color),
    ]

    return render_report_panel(spec, width, height)


def build_report(image: np.ndarray, fm: FaceMesh,
                 r: EyeResult) -> np.ndarray:
    scale = 1.5
    src_big = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
    )
    ann_big = annotate_face(image, fm, r, scale=scale)
    panel = build_panel(r, 620, src_big.shape[0])
    return compose_report(src_big, ann_big, panel)


# 後方互換
def visualize(image: np.ndarray, fm: FaceMesh, r: EyeResult) -> np.ndarray:
    return annotate_face(image, fm, r, scale=1.0)


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
    out = output_path or image_path.parent / f"eye_{image_path.stem}.png"
    if imgonly:
        cv2.imwrite(str(out), annotate_face(image, fm, r))
    else:
        cv2.imwrite(str(out), build_report(image, fm, r))
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
