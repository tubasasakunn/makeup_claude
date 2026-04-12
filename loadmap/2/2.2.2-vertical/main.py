"""
2.2.2 顔の垂直三分割判定

4つのランドマークで顔を縦に3区画に分割し、各区間長の比率を算出する。

    区間1: 上顔面 (生え際 → 眉頭下端)
    区間2: 中顔面 (眉頭下端 → 鼻下点)
    区間3: 下顔面 (鼻下点 → あご先)

    伝統的理想比: 1 : 1 : 1
    令和版理想比 : 1 : 1 : 0.8  (下顔面を短めに)

ランドマーク対応:
    生え際近似    : landmark 10 から chin 方向と逆に 0.25*(10→152) 延長
    眉頭下端      : 55, 285 (右・左眉頭) のY平均
    鼻下点        : landmark 2
    あご先        : landmark 152

分類:
    伝統的バランス型  : traditional_loss < reiwa_loss and both < 8%
    令和小顔型       : reiwa_loss < traditional_loss and reiwa_loss < 8%
    上顔面優位       : 上顔面 >> 他
    中顔面優位       : 中顔面 >> 他
    下顔面優位       : 下顔面 >> 他

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
    put_label,
)


# 10 番 → 真の生え際までの補正率 (raw face height に対する追加分)
# MediaPipe の landmark 10 は既に「額上部」付近にあり、そこから眉頭までの距離が
# 実測で ≒ 0.23*raw_h しかない。伝統 1:1:1 を満たすには上顔面を
# 中顔面 (≒0.37*raw_h) まで伸ばす必要があり、その差分は ≒0.13*raw_h。
FOREHEAD_EXTEND = 0.13


@dataclass
class VerticalResult:
    hairline_y: float = 0.0
    brow_y: float = 0.0
    subnasal_y: float = 0.0
    chin_y: float = 0.0

    upper_px: float = 0.0    # 上顔面
    middle_px: float = 0.0   # 中顔面
    lower_px: float = 0.0    # 下顔面

    # 中顔面を1として正規化
    upper_norm: float = 0.0
    middle_norm: float = 1.0
    lower_norm: float = 0.0

    traditional_loss: float = 0.0  # (1,1,1) との差の二乗平均
    reiwa_loss: float = 0.0        # (1,1,0.8) との差
    closest: str = ""              # traditional / reiwa
    category: str = ""             # 伝統/令和/上/中/下 優位

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _y(fm, idx: int) -> float:
    return float(fm.landmarks_px[idx][1])


def analyze(fm: FaceMesh) -> VerticalResult:
    r = VerticalResult()

    top_y = _y(fm, LM.FOREHEAD_TOP)
    chin_y = _y(fm, LM.CHIN_BOTTOM)
    raw_h = chin_y - top_y

    # 生え際は 10 より上に raw_h * 0.25 延長
    r.hairline_y = top_y - raw_h * FOREHEAD_EXTEND
    r.brow_y = (_y(fm, LM.BROW_HEAD_R) + _y(fm, LM.BROW_HEAD_L)) / 2
    r.subnasal_y = _y(fm, LM.SUBNASAL)
    r.chin_y = chin_y

    r.upper_px = r.brow_y - r.hairline_y
    r.middle_px = r.subnasal_y - r.brow_y
    r.lower_px = r.chin_y - r.subnasal_y

    # 中顔面を 1.0 として正規化
    if r.middle_px > 1e-3:
        r.upper_norm = r.upper_px / r.middle_px
        r.middle_norm = 1.0
        r.lower_norm = r.lower_px / r.middle_px

    # 理想比との二乗平均誤差
    trad = np.array([1.0, 1.0, 1.0])
    reiwa = np.array([1.0, 1.0, 0.8])
    vec = np.array([r.upper_norm, r.middle_norm, r.lower_norm])
    r.traditional_loss = float(np.mean((vec - trad) ** 2) ** 0.5)
    r.reiwa_loss = float(np.mean((vec - reiwa) ** 2) ** 0.5)

    r.closest = "traditional" if r.traditional_loss < r.reiwa_loss else "reiwa"

    # カテゴリ分類
    # どれか1区間が 15% 以上ズレていたら優位型
    max_idx = int(np.argmax(vec))
    min_idx = int(np.argmin(vec))
    max_val = float(vec[max_idx])
    min_val = float(vec[min_idx])

    if (max_val - min_val) / max(min_val, 1e-3) < 0.15:
        # バランス良い
        if r.closest == "traditional":
            r.category = "Traditional Balanced"
        else:
            r.category = "Reiwa Small-Face"
    else:
        labels = ["Upper", "Middle", "Lower"]
        r.category = f"{labels[max_idx]}-Dominant"

    return r


# ==============================================================
# 可視化
# ==============================================================
def visualize(image: np.ndarray, fm: FaceMesh, r: VerticalResult) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]

    # 顔の左右端 x
    x_l = int(fm.landmarks_px[LM.TEMPLE_R][0])
    x_r = int(fm.landmarks_px[LM.TEMPLE_L][0])

    # 3区間を色で塗る
    overlay = img.copy()
    regions = [
        (r.hairline_y, r.brow_y, (0, 220, 255), "Upper"),
        (r.brow_y, r.subnasal_y, (0, 180, 100), "Middle"),
        (r.subnasal_y, r.chin_y, (255, 140, 0), "Lower"),
    ]
    for y0, y1, col, name in regions:
        cv2.rectangle(
            overlay, (x_l, int(y0)), (x_r, int(y1)),
            col, -1
        )
    cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)

    # 境界線を引く
    for y in [r.hairline_y, r.brow_y, r.subnasal_y, r.chin_y]:
        draw_line(img, (x_l - 5, y), (x_r + 5, y), (255, 255, 255), 1)

    # 各区間のラベル
    center_x = (x_l + x_r) // 2
    for (y0, y1, col, name), norm in zip(
        regions, [r.upper_norm, r.middle_norm, r.lower_norm]
    ):
        cy = int((y0 + y1) / 2)
        put_label(
            img, f"{name} {norm:.2f}",
            (center_x - 30, cy + 4),
            color=col, scale=0.5, thickness=1,
        )

    # --- テキストパネル ---
    panel_w = int(w * 0.50)
    panel_h = int(h * 0.36)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (32, 32, 32)

    put_label(panel, "2.2.2 VERTICAL THIRDS", (10, 22),
              color=(0, 255, 255), scale=0.55, thickness=2)

    lines = [
        f"upper  : {r.upper_px:6.1f}px  ({r.upper_norm:.3f})",
        f"middle : {r.middle_px:6.1f}px  ({r.middle_norm:.3f})",
        f"lower  : {r.lower_px:6.1f}px  ({r.lower_norm:.3f})",
        f"traditional(1:1:1)  loss={r.traditional_loss:.3f}",
        f"reiwa      (1:1:0.8) loss={r.reiwa_loss:.3f}",
        f"closest    : {r.closest}",
        f"category   : {r.category}",
    ]
    for i, line in enumerate(lines):
        put_label(panel, line, (10, 44 + i * 18), scale=0.42)

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
        print("Error: 顔未検出")
        return None

    r = analyze(fm)
    print(f"\n=== 2.2.2 垂直三分割 ===")
    print(f"upper:middle:lower = {r.upper_norm:.2f} : {r.middle_norm:.2f} : {r.lower_norm:.2f}")
    print(f"traditional loss = {r.traditional_loss:.3f}  reiwa loss = {r.reiwa_loss:.3f}")
    print(f"closest: {r.closest}   category: {r.category}")
    if as_json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))

    vis = visualize(image, fm, r)
    out_path = output_path or image_path.parent / f"vertical_{image_path.stem}.png"
    if imgonly:
        cv2.imwrite(str(out_path), vis)
    else:
        cv2.imwrite(str(out_path), make_side_by_side(image, vis))
    print(f"出力: {out_path}")
    return r


def main():
    parser = argparse.ArgumentParser(description="2.2.2 顔の垂直三分割判定")
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--imgonly", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    run_one(
        Path(args.input),
        Path(args.output) if args.output else None,
        imgonly=args.imgonly, as_json=args.json,
    )


if __name__ == "__main__":
    main()
