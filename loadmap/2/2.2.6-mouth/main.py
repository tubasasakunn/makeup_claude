"""
2.2.6 口の判定

計測項目:
    - 口の幅: 瞳の内側から垂直に下ろした線との一致度
    - 上唇 : 下唇 の厚み比 (理想 1:1.5-2)
    - 人中の比率: 鼻下→唇中央 : 唇中央→あご先 (理想 1:2)
    - 口の幅 / 顔横幅 の調和度

ランドマーク:
    口右角 61, 口左角 291
    上唇中央外 0, 上唇中央内 13
    下唇中央内 14, 下唇中央外 17
    鼻下 2, あご先 152
    右虹彩中心 (468-472 平均), 左虹彩中心 (473-477 平均)

Usage:
    python main.py <input_image>
"""

from __future__ import annotations

import argparse
import json
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
class MouthResult:
    mouth_width_px: float = 0.0
    upper_lip_thickness_px: float = 0.0
    lower_lip_thickness_px: float = 0.0

    lip_ratio: float = 0.0            # lower / upper (理想 1.5-2)
    philtrum_top_px: float = 0.0      # 鼻下 → 唇中央
    philtrum_bot_px: float = 0.0      # 唇中央 → あご先
    philtrum_ratio: float = 0.0       # philtrum_bot / philtrum_top (理想 2.0)

    iris_edge_align_px: float = 0.0   # 口角 と 虹彩内縁 X の差
    mouth_to_face_ratio: float = 0.0  # 口幅 / 顔横幅

    lip_status: str = ""
    philtrum_status: str = ""
    alignment_status: str = ""
    overall: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _iris_center(fm, iris_ids) -> np.ndarray:
    return np.mean(
        [fm.landmarks_px[i].astype(np.float64) for i in iris_ids], axis=0
    )


def analyze(fm: FaceMesh) -> MouthResult:
    r = MouthResult()

    m_r = fm.landmarks_px[LM.MOUTH_R].astype(np.float64)
    m_l = fm.landmarks_px[LM.MOUTH_L].astype(np.float64)
    u_out = fm.landmarks_px[LM.UPPER_LIP_TOP].astype(np.float64)
    u_in = fm.landmarks_px[LM.UPPER_LIP_IN].astype(np.float64)
    l_in = fm.landmarks_px[LM.LOWER_LIP_IN].astype(np.float64)
    l_out = fm.landmarks_px[LM.LOWER_LIP_BOT].astype(np.float64)
    subnasal = fm.landmarks_px[LM.SUBNASAL].astype(np.float64)
    chin = fm.landmarks_px[LM.CHIN_BOTTOM].astype(np.float64)
    iris_r = _iris_center(fm, LM.IRIS_R)
    iris_l = _iris_center(fm, LM.IRIS_L)
    temple_r = fm.landmarks_px[LM.TEMPLE_R].astype(np.float64)
    temple_l = fm.landmarks_px[LM.TEMPLE_L].astype(np.float64)

    r.mouth_width_px = float(np.linalg.norm(m_r - m_l))
    r.upper_lip_thickness_px = float(np.linalg.norm(u_out - u_in))
    r.lower_lip_thickness_px = float(np.linalg.norm(l_in - l_out))

    if r.upper_lip_thickness_px > 1e-3:
        r.lip_ratio = r.lower_lip_thickness_px / r.upper_lip_thickness_px

    # 人中 (上)・人中 (下)  Y のみで評価
    mouth_mid_y = (u_out[1] + l_out[1]) / 2
    r.philtrum_top_px = float(mouth_mid_y - subnasal[1])
    r.philtrum_bot_px = float(chin[1] - mouth_mid_y)
    if r.philtrum_top_px > 1e-3:
        r.philtrum_ratio = r.philtrum_bot_px / r.philtrum_top_px

    # 虹彩内縁ライン: 口角 vs 瞳の内側端 (iris_center ± iris_radius)
    # 正面想定で、瞳の X 位置が 口角の真上にある理想
    iris_inner_r = iris_r[0] + 0  # 実際の iris 内側は iris_center に近い
    iris_inner_l = iris_l[0] + 0
    diff_r = m_r[0] - iris_r[0]
    diff_l = m_l[0] - iris_l[0]
    r.iris_edge_align_px = float((abs(diff_r) + abs(diff_l)) / 2)

    # 顔横幅との比
    face_w = float(np.linalg.norm(temple_r - temple_l))
    if face_w > 1e-3:
        r.mouth_to_face_ratio = r.mouth_width_px / face_w

    # ステータス
    if 1.2 <= r.lip_ratio <= 2.2:
        r.lip_status = "Ideal (1.5-2 range)"
    elif r.lip_ratio < 1.2:
        r.lip_status = "Upper-heavy"
    else:
        r.lip_status = "Lower-heavy"

    if 1.6 <= r.philtrum_ratio <= 2.4:
        r.philtrum_status = "Ideal (~2.0)"
    elif r.philtrum_ratio < 1.6:
        r.philtrum_status = "Philtrum-long"
    else:
        r.philtrum_status = "Chin-long"

    tol = r.mouth_width_px * 0.12
    if r.iris_edge_align_px < tol:
        r.alignment_status = "Ideal (align)"
    else:
        r.alignment_status = "Off (diff>12%)"

    ok = [
        "Ideal" in r.lip_status,
        "Ideal" in r.philtrum_status,
        "Ideal" in r.alignment_status,
        0.32 <= r.mouth_to_face_ratio <= 0.40,
    ]
    r.overall = f"{sum(ok)}/4 ideal"
    return r


# ==============================================================
# 可視化 (UI/UX 改善版: レポートパネル + annotate 分離)
# ==============================================================
C_W = (80, 220, 255)        # 口幅 アンバー
C_LIP = (255, 200, 80)      # 唇厚 シアン
C_PHIL = (140, 255, 160)    # 人中 緑
C_IRIS = (200, 140, 255)    # 虹彩垂直 マゼンタ


def annotate_face(image: np.ndarray, fm: FaceMesh, r: MouthResult,
                  scale: float = 1.0) -> np.ndarray:
    """口幅・唇厚・人中線のみ + 下唇/上唇ラベル 1 つ"""
    if scale != 1.0:
        img = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
        )
    else:
        img = image.copy()

    def S(p):
        return (float(p[0]) * scale, float(p[1]) * scale)

    m_r = S(fm.landmarks_px[LM.MOUTH_R])
    m_l = S(fm.landmarks_px[LM.MOUTH_L])
    u_out = S(fm.landmarks_px[LM.UPPER_LIP_TOP])
    l_out = S(fm.landmarks_px[LM.LOWER_LIP_BOT])
    subnasal = S(fm.landmarks_px[LM.SUBNASAL])
    chin = S(fm.landmarks_px[LM.CHIN_BOTTOM])

    lw = max(2, int(2 * scale))
    pt_r = max(3, int(4 * scale))

    draw_line_outlined(img, m_r, m_l, C_W, thickness=lw)
    draw_line_outlined(img, u_out, l_out, C_LIP, thickness=lw)

    mouth_mid_y = (u_out[1] + l_out[1]) / 2
    cx_mouth = (m_r[0] + m_l[0]) / 2
    draw_line_outlined(
        img, (cx_mouth, subnasal[1]), (cx_mouth, mouth_mid_y),
        C_PHIL, thickness=lw,
    )
    draw_line_outlined(
        img, (cx_mouth, mouth_mid_y), (cx_mouth, chin[1]),
        C_PHIL, thickness=lw,
    )

    for p, col in [
        (m_r, C_W), (m_l, C_W),
        (u_out, C_LIP), (l_out, C_LIP),
        (subnasal, C_PHIL), (chin, C_PHIL),
    ]:
        draw_point_outlined(img, p, col, r=pt_r)

    # ---- ラベル: 下唇/上唇 1 つだけ ----
    label_bg = (20, 20, 25)
    label_size = max(18, int(20 * scale))
    draw_pil_text(
        img, f"下唇/上唇 = {r.lip_ratio:.2f}",
        (max(m_r[0], m_l[0]) + int(14 * scale), mouth_mid_y - 12),
        color=(240, 240, 245), size=label_size,
        bg=label_bg, bg_alpha=0.78, bg_pad=6,
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

    return img


def build_panel(r: MouthResult, width: int, height: int) -> np.ndarray:
    pill_color = (
        (140, 255, 160) if "4/4" in r.overall or "3/4" in r.overall
        else (255, 200, 80)
    )
    IDEAL_COL = (200, 200, 205)

    lip_items = [
        ("あなた", r.lip_ratio, f"{r.lip_ratio:.2f}", pill_color),
        ("理想", 1.75, "1.75", IDEAL_COL),
    ]
    lip_diff_pct = (r.lip_ratio - 1.75) / 1.75 * 100

    phil_items = [
        ("あなた", r.philtrum_ratio, f"{r.philtrum_ratio:.2f}", pill_color),
        ("理想", 2.0, "2.0", IDEAL_COL),
    ]
    phil_diff_pct = (r.philtrum_ratio - 2.0) / 2.0 * 100

    spec = [
        ("title", "2.2.6  口", (230, 230, 230)),
        ("subtitle", "Lip Ratio / Philtrum"),
        ("divider",),
        ("spacer", 4),
        ("section", "判定結果"),
        ("big", r.overall.upper(), pill_color),
        ("spacer", 8),
        ("section", "下唇 / 上唇 (理想 1.5-2.0)"),
        ("ratio_compare", lip_items, "あなた"),
        ("spacer", 2),
        ("diff_bar", "", lip_diff_pct, 50.0, pill_color),
        ("spacer", 6),
        ("section", "人中比 (理想 2.0)"),
        ("ratio_compare", phil_items, "あなた"),
        ("spacer", 2),
        ("diff_bar", "", phil_diff_pct, 30.0, pill_color),
    ]

    return render_report_panel(spec, width, height)


def build_report(image: np.ndarray, fm: FaceMesh,
                 r: MouthResult) -> np.ndarray:
    scale = 1.5
    src_big = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
    )
    ann_big = annotate_face(image, fm, r, scale=scale)
    panel = build_panel(r, 620, src_big.shape[0])
    return compose_report(src_big, ann_big, panel)


# 後方互換
def visualize(image: np.ndarray, fm: FaceMesh, r: MouthResult) -> np.ndarray:
    return annotate_face(image, fm, r, scale=1.0)


def run_one(image_path: Path, output_path=None, imgonly=False, as_json=False):
    image = cv2.imread(str(image_path))
    if image is None: return None
    fm = FaceMesh(subdivision_level=1); fm.init()
    if fm.detect(image) is None:
        print("顔未検出"); return None
    r = analyze(fm)
    print(f"\n=== 2.2.6 口 ===")
    print(f"lip lower/upper={r.lip_ratio:.2f}  philtrum bot/top={r.philtrum_ratio:.2f}")
    print(f"mouth/face={r.mouth_to_face_ratio*100:.1f}%  {r.alignment_status}")
    print(f"overall: {r.overall}")
    if as_json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
    out = output_path or image_path.parent / f"mouth_{image_path.stem}.png"
    if imgonly:
        cv2.imwrite(str(out), annotate_face(image, fm, r))
    else:
        cv2.imwrite(str(out), build_report(image, fm, r))
    print(f"出力: {out}")
    return r


def main():
    parser = argparse.ArgumentParser(description="2.2.6 口の判定")
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--imgonly", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    run_one(Path(args.input), Path(args.output) if args.output else None,
            imgonly=args.imgonly, as_json=args.json)


if __name__ == "__main__":
    main()
