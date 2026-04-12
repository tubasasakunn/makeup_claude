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
    compose_report,
    draw_line_outlined,
    draw_pil_pill,
    draw_pil_text,
    draw_point_outlined,
    make_side_by_side,
    render_report_panel,
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
# 可視化 (UI/UX 改善版: レポートパネル + annotate 分離)
# ==============================================================
C_VERT = (80, 220, 255)    # 鼻縦軸 アンバー
C_WING = (255, 200, 80)    # 小鼻幅 シアン
C_GAP = (200, 140, 255)    # 目間 マゼンタ
C_ANGLE = (140, 255, 160)  # 鼻唇角 緑
C_ELINE = (80, 180, 255)   # E ライン オレンジ


def annotate_face(image: np.ndarray, fm: FaceMesh, r: NoseResult,
                  scale: float = 1.0) -> np.ndarray:
    if scale != 1.0:
        img = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
        )
    else:
        img = image.copy()
    h, w = img.shape[:2]

    def S(p):
        return (float(p[0]) * scale, float(p[1]) * scale)

    nose_root = S(fm.landmarks_px[LM.NOSE_ROOT])
    nose_tip = S(fm.landmarks_px[LM.NOSE_TIP])
    subnasal = S(fm.landmarks_px[LM.SUBNASAL])
    wing_r = S(fm.landmarks_px[LM.NOSE_WING_R])
    wing_l = S(fm.landmarks_px[LM.NOSE_WING_L])
    eye_r = S(fm.landmarks_px[LM.EYE_INNER_R])
    eye_l = S(fm.landmarks_px[LM.EYE_INNER_L])
    upper_lip = S(fm.landmarks_px[LM.UPPER_LIP_TOP])
    chin = S(fm.landmarks_px[LM.CHIN_BOTTOM])

    lw = max(2, int(2 * scale))
    pt_r = max(3, int(4 * scale))

    draw_line_outlined(img, nose_root, nose_tip, C_VERT, thickness=lw)
    draw_line_outlined(img, wing_r, wing_l, C_WING, thickness=lw)
    draw_line_outlined(img, eye_r, eye_l, C_GAP, thickness=lw)
    draw_line_outlined(img, subnasal, nose_tip, C_ANGLE, thickness=lw)
    draw_line_outlined(img, subnasal, upper_lip, C_ANGLE, thickness=lw)
    draw_line_outlined(img, nose_tip, chin, C_ELINE, thickness=lw)

    for p, col in [
        (nose_root, C_VERT), (nose_tip, C_VERT), (subnasal, C_ANGLE),
        (wing_r, C_WING), (wing_l, C_WING),
        (eye_r, C_GAP), (eye_l, C_GAP),
        (upper_lip, C_ELINE), (chin, C_ELINE),
    ]:
        draw_point_outlined(img, p, col, r=pt_r)

    label_bg = (20, 20, 25)
    label_size = max(14, int(15 * scale))
    draw_pil_text(
        img, f"len {r.nose_length_px:.0f}",
        (nose_tip[0] + 12, (nose_root[1] + nose_tip[1]) / 2),
        color=C_VERT, size=label_size, bg=label_bg, bg_alpha=0.78, bg_pad=4,
    )
    draw_pil_text(
        img, f"wing {r.nose_wing_width_px:.0f}",
        (max(wing_r[0], wing_l[0]) + 10, (wing_r[1] + wing_l[1]) / 2 - 8),
        color=C_WING, size=label_size, bg=label_bg, bg_alpha=0.78, bg_pad=4,
    )
    draw_pil_text(
        img, f"angle {r.nose_lip_angle_deg:.0f}°",
        (subnasal[0] + 12, subnasal[1] + 4),
        color=C_ANGLE, size=label_size, bg=label_bg, bg_alpha=0.78, bg_pad=4,
    )

    # 左上ピル
    pill_color = (
        (140, 255, 160) if "4/4" in r.overall or "3/4" in r.overall
        else (255, 200, 80)
    )
    pill_text = r.overall.upper()
    pill_size = int(24 * max(1.0, scale * 0.9))
    draw_pil_pill(
        img, pill_text, (18, 18),
        text_color=(20, 20, 30), pill_color=pill_color, size=pill_size,
        pad_x=18, pad_y=10, radius=22,
    )

    # 右上凡例
    legend_items = [
        ("鼻縦", C_VERT),
        ("小鼻", C_WING),
        ("目間", C_GAP),
        ("鼻唇角", C_ANGLE),
        ("Eライン", C_ELINE),
    ]
    legend_font = max(14, int(15 * scale))
    sw_w = int(18 * scale)
    sw_h = int(14 * scale)
    row_h = int(26 * scale)
    lx = w - int(160 * scale)
    ly = int(20 * scale)
    cv2.rectangle(
        img, (lx - 8, ly - 6),
        (w - 14, ly + row_h * len(legend_items) + 4),
        (0, 0, 0), -1,
    )
    for i, (name, col) in enumerate(legend_items):
        y_item = ly + i * row_h
        cv2.rectangle(
            img, (lx, y_item + 4), (lx + sw_w, y_item + 4 + sw_h), col, -1,
        )
        draw_pil_text(
            img, name, (lx + sw_w + 8, y_item),
            color=(235, 235, 240), size=legend_font,
        )

    return img


def build_panel(r: NoseResult, width: int, height: int) -> np.ndarray:
    pill_color = (
        (140, 255, 160) if "4/4" in r.overall or "3/4" in r.overall
        else (255, 200, 80)
    )

    def col_for(ok: bool):
        return (140, 255, 160) if ok else (200, 200, 205)

    ok_wing = r.wing_loss < 0.20
    ok_length = r.length_loss < 0.20
    ok_angle = "Ideal" in r.angle_status
    ok_eline = "Ideal" in r.eline_status

    spec = [
        ("title", "2.2.5  鼻", (230, 230, 230)),
        ("subtitle", "Nose Wings / Length / Angle / E-line"),
        ("divider",),
        ("spacer", 4),
        ("section", "判定結果"),
        ("big", r.overall.upper(), pill_color),
        ("spacer", 6),
        ("section", "比率"),
        ("kv", "wing / gap (理想 1.0)",
            f"{r.wing_to_gap_ratio:.3f}", col_for(ok_wing)),
        ("kv", "length / wing (理想 1.5)",
            f"{r.length_to_wing_ratio:.3f}", col_for(ok_length)),
        ("spacer", 6),
        ("section", "鼻唇角 (理想 90-100°)"),
        ("kv", "angle", f"{r.nose_lip_angle_deg:.1f} °", col_for(ok_angle)),
        ("text", r.angle_status, (200, 200, 205)),
        ("spacer", 6),
        ("section", "E ライン"),
        ("kv", "offset", f"{r.eline_offset_px:+.1f} px", col_for(ok_eline)),
        ("kv", "norm", f"{r.eline_norm*100:+.2f} %"),
        ("text", r.eline_status, (200, 200, 205)),
        ("spacer", 6),
        ("section", "ピクセル長"),
        ("kv", "鼻縦", f"{r.nose_length_px:.1f} px"),
        ("kv", "小鼻幅", f"{r.nose_wing_width_px:.1f} px"),
        ("kv", "目間", f"{r.eye_gap_px:.1f} px"),
    ]

    return render_report_panel(spec, width, height)


def build_report(image: np.ndarray, fm: FaceMesh,
                 r: NoseResult) -> np.ndarray:
    scale = 1.5
    src_big = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
    )
    ann_big = annotate_face(image, fm, r, scale=scale)
    panel = build_panel(r, 620, src_big.shape[0])
    return compose_report(src_big, ann_big, panel)


# 後方互換
def visualize(image: np.ndarray, fm: FaceMesh, r: NoseResult) -> np.ndarray:
    return annotate_face(image, fm, r, scale=1.0)


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
    out = output_path or image_path.parent / f"nose_{image_path.stem}.png"
    if imgonly:
        cv2.imwrite(str(out), annotate_face(image, fm, r))
    else:
        cv2.imwrite(str(out), build_report(image, fm, r))
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
