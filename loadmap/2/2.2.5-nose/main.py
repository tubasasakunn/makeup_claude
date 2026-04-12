"""
2.2.5 鼻の判定

計測項目:
    - 小鼻幅 : 目間距離 比 (理想 1:1)
    - 鼻の縦幅 : 小鼻幅 比 (理想 1.5:1)
    - 鼻唇角 (鼻柱と上唇の角度, 理想 90-100 度)
    - 鼻額角 (おでこと鼻筋の角度, 理想 120-130 度)
    - E ライン (鼻先とあご先の直線に対する唇の位置)

ランドマーク:
    鼻根 168, 鼻先 1, 鼻下点 2
    小鼻 右 64 左 294  (右外 129 左外 358)
    上唇中央外 0
    目頭 133, 362 → 目間の基準

E ライン判定:
    鼻先(1)→あご先(152) の直線を引き、上唇中央外(0) の位置をその直線からの
    符号付き距離で評価。+方向(画像右) で 顔面の外側になるかどうかは顔の向き依存。
    ここでは符号=0 で一致、|dist| < lip_ref_px*0.05 を「Ideal」とする。

Usage:
    python main.py <input_image>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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
class NoseResult:
    nose_length_px: float = 0.0
    nose_wing_width_px: float = 0.0
    eye_gap_px: float = 0.0            # 内角距離

    wing_to_gap_ratio: float = 0.0     # 小鼻幅 / 目間
    length_to_wing_ratio: float = 0.0  # 鼻の縦 / 小鼻幅

    nose_lip_angle_deg: float = 0.0    # 鼻柱と上唇 (2 - 1 と 2 - 0)
    # Eライン: 鼻先-あご先 の線に対する上唇の垂直距離
    eline_offset_px: float = 0.0
    eline_norm: float = 0.0            # 負なら直線より内側(顔の中心側)

    wing_loss: float = 0.0             # 1.0 からの乖離
    length_loss: float = 0.0           # 1.5 からの乖離
    angle_status: str = ""
    eline_status: str = ""
    overall: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def analyze(fm: FaceMesh) -> NoseResult:
    r = NoseResult()

    nose_root = fm.landmarks_px[LM.NOSE_ROOT].astype(np.float64)
    nose_tip = fm.landmarks_px[LM.NOSE_TIP].astype(np.float64)
    subnasal = fm.landmarks_px[LM.SUBNASAL].astype(np.float64)
    wing_r = fm.landmarks_px[LM.NOSE_WING_R].astype(np.float64)
    wing_l = fm.landmarks_px[LM.NOSE_WING_L].astype(np.float64)
    eye_inner_r = fm.landmarks_px[LM.EYE_INNER_R].astype(np.float64)
    eye_inner_l = fm.landmarks_px[LM.EYE_INNER_L].astype(np.float64)
    upper_lip = fm.landmarks_px[LM.UPPER_LIP_TOP].astype(np.float64)
    chin = fm.landmarks_px[LM.CHIN_BOTTOM].astype(np.float64)

    r.nose_length_px = float(np.linalg.norm(nose_tip - nose_root))
    r.nose_wing_width_px = float(np.linalg.norm(wing_r - wing_l))
    r.eye_gap_px = float(np.linalg.norm(eye_inner_r - eye_inner_l))

    if r.eye_gap_px > 1e-3:
        r.wing_to_gap_ratio = r.nose_wing_width_px / r.eye_gap_px
    if r.nose_wing_width_px > 1e-3:
        r.length_to_wing_ratio = r.nose_length_px / r.nose_wing_width_px

    # 鼻唇角: subnasal(2) を頂点とした nose_tip(1) と upper_lip(0) のなす角
    v1 = nose_tip - subnasal
    v2 = upper_lip - subnasal
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 > 1e-3 and n2 > 1e-3:
        cos_a = float(np.dot(v1, v2) / (n1 * n2))
        cos_a = max(-1.0, min(1.0, cos_a))
        r.nose_lip_angle_deg = math.degrees(math.acos(cos_a))

    # E ライン: 鼻先 → あご先 の線上に上唇があるか
    ab = chin - nose_tip
    ap = upper_lip - nose_tip
    ab_len = np.linalg.norm(ab)
    if ab_len > 1e-3:
        # 2D 外積 (符号付き距離)
        cross = ab[0] * ap[1] - ab[1] * ap[0]
        r.eline_offset_px = float(cross / ab_len)
        r.eline_norm = r.eline_offset_px / ab_len

    # 乖離度
    r.wing_loss = abs(r.wing_to_gap_ratio - 1.0)
    r.length_loss = abs(r.length_to_wing_ratio - 1.5) / 1.5

    # 角度評価
    if 90 <= r.nose_lip_angle_deg <= 100:
        r.angle_status = "Ideal (90-100)"
    elif r.nose_lip_angle_deg < 90:
        r.angle_status = "Acute (chinless)"
    else:
        r.angle_status = "Obtuse"

    # Eライン
    face_h = float(np.linalg.norm(chin - nose_tip))
    tol = face_h * 0.03
    if abs(r.eline_offset_px) < tol:
        r.eline_status = "Ideal (on line)"
    elif r.eline_offset_px > 0:
        r.eline_status = "Lip ahead of line"
    else:
        r.eline_status = "Lip behind line"

    # 全体評価
    ok_wing = r.wing_loss < 0.20
    ok_length = r.length_loss < 0.20
    ok_angle = "Ideal" in r.angle_status
    ok_eline = "Ideal" in r.eline_status
    hits = sum([ok_wing, ok_length, ok_angle, ok_eline])
    r.overall = f"{hits}/4 ideal"

    return r


# ==============================================================
# 可視化
# ==============================================================
def visualize(image: np.ndarray, fm: FaceMesh, r: NoseResult) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]

    nose_root = fm.landmarks_px[LM.NOSE_ROOT]
    nose_tip = fm.landmarks_px[LM.NOSE_TIP]
    subnasal = fm.landmarks_px[LM.SUBNASAL]
    wing_r = fm.landmarks_px[LM.NOSE_WING_R]
    wing_l = fm.landmarks_px[LM.NOSE_WING_L]
    eye_r = fm.landmarks_px[LM.EYE_INNER_R]
    eye_l = fm.landmarks_px[LM.EYE_INNER_L]
    upper_lip = fm.landmarks_px[LM.UPPER_LIP_TOP]
    chin = fm.landmarks_px[LM.CHIN_BOTTOM]

    # 鼻の縦ライン
    draw_line(img, nose_root, nose_tip, (0, 220, 255), 2)
    # 小鼻幅
    draw_line(img, wing_r, wing_l, (255, 200, 0), 2)
    # 目間
    draw_line(img, eye_r, eye_l, (255, 100, 100), 1)
    # 鼻唇角ライン
    draw_line(img, subnasal, nose_tip, (100, 255, 100), 1)
    draw_line(img, subnasal, upper_lip, (100, 255, 100), 1)
    # E ライン
    draw_line(img, nose_tip, chin, (0, 255, 255), 1)
    draw_point(img, upper_lip, (255, 100, 255), 4)

    for p in (nose_root, nose_tip, subnasal, wing_r, wing_l, eye_r, eye_l, upper_lip, chin):
        draw_point(img, p, (255, 255, 255), 2)

    # パネル
    panel_w = int(w * 0.50)
    panel_h = int(h * 0.42)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (32, 32, 32)
    put_label(panel, "2.2.5 NOSE", (10, 22), color=(0, 255, 255), scale=0.55, thickness=2)

    lines = [
        f"nose length : {r.nose_length_px:.1f}px",
        f"wing width  : {r.nose_wing_width_px:.1f}px",
        f"eye gap     : {r.eye_gap_px:.1f}px",
        f"wing/gap    : {r.wing_to_gap_ratio:.3f}  (ideal 1.0)",
        f"length/wing : {r.length_to_wing_ratio:.3f}  (ideal 1.5)",
        f"nose-lip angle: {r.nose_lip_angle_deg:.1f} deg",
        f"  -> {r.angle_status}",
        f"E-line offset: {r.eline_offset_px:+.1f}px ({r.eline_norm*100:+.2f}%)",
        f"  -> {r.eline_status}",
        f"overall: {r.overall}",
    ]
    for i, line in enumerate(lines):
        put_label(panel, line, (10, 44 + i * 17), scale=0.4)

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
    print(f"\n=== 2.2.5 鼻 ===")
    print(f"wing/gap={r.wing_to_gap_ratio:.3f}  length/wing={r.length_to_wing_ratio:.3f}")
    print(f"nose-lip angle={r.nose_lip_angle_deg:.1f} -> {r.angle_status}")
    print(f"E-line: {r.eline_offset_px:+.1f}px -> {r.eline_status}")
    print(f"overall: {r.overall}")
    if as_json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
    vis = visualize(image, fm, r)
    out = output_path or image_path.parent / f"nose_{image_path.stem}.png"
    cv2.imwrite(str(out), vis if imgonly else make_side_by_side(image, vis))
    print(f"出力: {out}")
    return r


def main():
    parser = argparse.ArgumentParser(description="2.2.5 鼻の判定")
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--imgonly", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    run_one(Path(args.input), Path(args.output) if args.output else None,
            imgonly=args.imgonly, as_json=args.json)


if __name__ == "__main__":
    main()
