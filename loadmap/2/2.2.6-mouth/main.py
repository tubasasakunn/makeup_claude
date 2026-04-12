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
    draw_line,
    draw_point,
    make_side_by_side,
    put_label,
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
# 可視化
# ==============================================================
def visualize(image: np.ndarray, fm: FaceMesh, r: MouthResult) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]

    m_r = fm.landmarks_px[LM.MOUTH_R]
    m_l = fm.landmarks_px[LM.MOUTH_L]
    u_out = fm.landmarks_px[LM.UPPER_LIP_TOP]
    l_out = fm.landmarks_px[LM.LOWER_LIP_BOT]
    subnasal = fm.landmarks_px[LM.SUBNASAL]
    chin = fm.landmarks_px[LM.CHIN_BOTTOM]
    iris_r = _iris_center(fm, LM.IRIS_R)
    iris_l = _iris_center(fm, LM.IRIS_L)

    draw_line(img, m_r, m_l, (0, 220, 255), 2)  # 口幅
    draw_line(img, u_out, l_out, (255, 200, 0), 2)  # 唇厚
    # 人中・下人中
    mouth_mid_y = int((u_out[1] + l_out[1]) / 2)
    cx_mouth = int((m_r[0] + m_l[0]) / 2)
    draw_line(img, (cx_mouth, int(subnasal[1])), (cx_mouth, mouth_mid_y), (100, 255, 100), 1)
    draw_line(img, (cx_mouth, mouth_mid_y), (cx_mouth, int(chin[1])), (100, 255, 100), 1)

    # 虹彩内縁の垂直線
    draw_line(img, (int(iris_r[0]), int(iris_r[1])), (int(iris_r[0]), int(chin[1])),
              (255, 100, 255), 1)
    draw_line(img, (int(iris_l[0]), int(iris_l[1])), (int(iris_l[0]), int(chin[1])),
              (255, 100, 255), 1)

    for p in (m_r, m_l, u_out, l_out, subnasal, chin):
        draw_point(img, p, (255, 255, 255), 2)

    # パネル
    panel_w = int(w * 0.50)
    panel_h = int(h * 0.42)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (32, 32, 32)
    put_label(panel, "2.2.6 MOUTH", (10, 22), color=(0, 255, 255), scale=0.55, thickness=2)

    lines = [
        f"mouth width  : {r.mouth_width_px:.1f}px",
        f"upper lip    : {r.upper_lip_thickness_px:.1f}px",
        f"lower lip    : {r.lower_lip_thickness_px:.1f}px",
        f"lower/upper  : {r.lip_ratio:.2f}  ({r.lip_status})",
        f"philtrum top : {r.philtrum_top_px:.1f}px",
        f"philtrum bot : {r.philtrum_bot_px:.1f}px",
        f"ratio bot/top: {r.philtrum_ratio:.2f}  ({r.philtrum_status})",
        f"iris-edge diff: {r.iris_edge_align_px:.1f}px ({r.alignment_status})",
        f"mouth/face   : {r.mouth_to_face_ratio*100:.1f}% (ideal 32-40)",
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
    print(f"\n=== 2.2.6 口 ===")
    print(f"lip lower/upper={r.lip_ratio:.2f}  philtrum bot/top={r.philtrum_ratio:.2f}")
    print(f"mouth/face={r.mouth_to_face_ratio*100:.1f}%  {r.alignment_status}")
    print(f"overall: {r.overall}")
    if as_json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
    vis = visualize(image, fm, r)
    out = output_path or image_path.parent / f"mouth_{image_path.stem}.png"
    cv2.imwrite(str(out), vis if imgonly else make_side_by_side(image, vis))
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
