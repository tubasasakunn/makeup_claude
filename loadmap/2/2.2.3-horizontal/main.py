"""
2.2.3 顔の水平五分割判定

顔の横幅を 5 つのセグメントに分割する：

    [右余白] [右目幅] [目間距離] [左目幅] [左余白]

    左右端  = temple (234, 454)
    右目外角 = 33, 右目内角 = 133
    左目内角 = 362, 左目外角 = 263

評価:
    - 理想比率 1:1:1:1:1 との乖離度
    - 目間距離 / 目幅 の比 (理想 1.0、日本人向け 1.15)
    - 分類:
        * center-converged (求心顔)   目間 < 目幅
        * center-diverged  (遠心顔)   目間 > 目幅 * 1.2
        * ideal            (理想配置) それ以外

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


# 日本人向け理想 (目幅 : 目間 : 目幅)
JP_IDEAL = (1.0, 1.15, 1.0)


@dataclass
class HorizontalResult:
    seg_px: list[float] = field(default_factory=list)     # [右余白, 右目幅, 目間, 左目幅, 左余白]
    seg_norm: list[float] = field(default_factory=list)   # 目幅平均を1に正規化
    eye_gap_ratio: float = 0.0       # 目間/目幅 平均
    left_right_balance: float = 0.0  # 右余白 - 左余白 の差の割合（左右非対称性）
    ideal_loss_1: float = 0.0        # 1:1:1:1:1 との差
    jp_loss: float = 0.0             # 日本人向け 1:1.15:1 との差
    closest: str = ""
    category: str = ""               # ideal / center-converged / center-diverged

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _x(fm, idx: int) -> float:
    return float(fm.landmarks_px[idx][0])


def analyze(fm: FaceMesh) -> HorizontalResult:
    r = HorizontalResult()

    # ランドマーク (MediaPipe では右が画像左側なので座標的には小さい)
    x_temple_r = _x(fm, LM.TEMPLE_R)
    x_temple_l = _x(fm, LM.TEMPLE_L)
    x_eye_r_out = _x(fm, LM.EYE_OUTER_R)
    x_eye_r_in = _x(fm, LM.EYE_INNER_R)
    x_eye_l_in = _x(fm, LM.EYE_INNER_L)
    x_eye_l_out = _x(fm, LM.EYE_OUTER_L)

    # 方向正規化: 必ず小→大
    xs = sorted([x_temple_r, x_temple_l])
    x_left_edge, x_right_edge = xs
    # 右目(画像の左側)が x_eye_r_out < x_eye_r_in < 中央 < x_eye_l_in < x_eye_l_out
    r_out, r_in = min(x_eye_r_out, x_eye_r_in), max(x_eye_r_out, x_eye_r_in)
    l_in, l_out = min(x_eye_l_in, x_eye_l_out), max(x_eye_l_in, x_eye_l_out)

    seg1 = r_out - x_left_edge          # 左余白 (画像左端 → 右目外角)
    seg2 = r_in - r_out                  # 右目幅
    seg3 = l_in - r_in                   # 目間
    seg4 = l_out - l_in                  # 左目幅
    seg5 = x_right_edge - l_out          # 右余白
    r.seg_px = [seg1, seg2, seg3, seg4, seg5]

    # 正規化: 目幅の平均を 1 とする
    eye_mean = (seg2 + seg4) / 2
    if eye_mean > 1e-3:
        r.seg_norm = [s / eye_mean for s in r.seg_px]
        r.eye_gap_ratio = seg3 / eye_mean

    # 1:1:1:1:1 との差
    target1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    v = np.array(r.seg_norm)
    r.ideal_loss_1 = float(np.mean((v - target1) ** 2) ** 0.5)

    # 日本人理想: 目幅:目間:目幅 = 1:1.15:1 (余白は評価外)
    jp_v = np.array([r.seg_norm[1], r.seg_norm[2], r.seg_norm[3]])
    jp_target = np.array(JP_IDEAL)
    r.jp_loss = float(np.mean((jp_v - jp_target) ** 2) ** 0.5)

    r.closest = "ideal_11111" if r.ideal_loss_1 < r.jp_loss else "japanese_1_115_1"

    # 左右余白バランス
    if seg1 + seg5 > 1e-3:
        r.left_right_balance = (seg1 - seg5) / ((seg1 + seg5) / 2)

    # 分類 (MediaPipe の内角ランドマークは涙丘よりやや内側のため、
    # 実測での gap/eye は 1.3 - 1.5 に偏る。日本人サンプル基準で閾値調整)
    if r.eye_gap_ratio < 1.30:
        r.category = "Center-Converged (yose-me)"
    elif r.eye_gap_ratio > 1.55:
        r.category = "Center-Diverged (hanare-me)"
    else:
        r.category = "Ideal"

    return r


# ==============================================================
# 可視化
# ==============================================================
SEG_COLORS = [
    (180, 180, 180),  # 右余白
    (0, 200, 255),    # 右目幅
    (100, 255, 100),  # 目間
    (0, 200, 255),    # 左目幅
    (180, 180, 180),  # 左余白
]
SEG_NAMES = ["mgn_R", "eye_R", "gap", "eye_L", "mgn_L"]


def visualize(image: np.ndarray, fm: FaceMesh, r: HorizontalResult) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]

    # 水平ラインの Y: 目の中央
    y_r = (fm.landmarks_px[LM.EYE_TOP_R][1] + fm.landmarks_px[LM.EYE_BOT_R][1]) / 2
    y_l = (fm.landmarks_px[LM.EYE_TOP_L][1] + fm.landmarks_px[LM.EYE_BOT_L][1]) / 2
    y_line = int((y_r + y_l) / 2)

    x_temple_r = min(
        fm.landmarks_px[LM.TEMPLE_R][0], fm.landmarks_px[LM.TEMPLE_L][0]
    )
    x_temple_l = max(
        fm.landmarks_px[LM.TEMPLE_R][0], fm.landmarks_px[LM.TEMPLE_L][0]
    )
    r_out_x = min(fm.landmarks_px[LM.EYE_OUTER_R][0], fm.landmarks_px[LM.EYE_INNER_R][0])
    r_in_x = max(fm.landmarks_px[LM.EYE_OUTER_R][0], fm.landmarks_px[LM.EYE_INNER_R][0])
    l_in_x = min(fm.landmarks_px[LM.EYE_INNER_L][0], fm.landmarks_px[LM.EYE_OUTER_L][0])
    l_out_x = max(fm.landmarks_px[LM.EYE_INNER_L][0], fm.landmarks_px[LM.EYE_OUTER_L][0])
    xs = [x_temple_r, r_out_x, r_in_x, l_in_x, l_out_x, x_temple_l]

    # 5セグメントの帯を描画
    overlay = img.copy()
    for i in range(5):
        x0 = int(xs[i])
        x1 = int(xs[i + 1])
        cv2.rectangle(
            overlay, (x0, y_line - 18), (x1, y_line + 18),
            SEG_COLORS[i], -1
        )
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    for x in xs:
        draw_line(img, (int(x), y_line - 22), (int(x), y_line + 22), (255, 255, 255), 1)

    # 各セグメントのノーマライズ値ラベル
    for i in range(5):
        cx = (int(xs[i]) + int(xs[i + 1])) // 2
        if r.seg_norm:
            put_label(
                img, f"{r.seg_norm[i]:.2f}",
                (cx - 14, y_line + 36),
                color=SEG_COLORS[i], scale=0.42,
            )
            put_label(
                img, SEG_NAMES[i],
                (cx - 18, y_line - 26),
                color=SEG_COLORS[i], scale=0.38,
            )

    # --- パネル ---
    panel_w = int(w * 0.50)
    panel_h = int(h * 0.38)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (32, 32, 32)
    put_label(panel, "2.2.3 HORIZONTAL FIFTHS", (10, 22),
              color=(0, 255, 255), scale=0.55, thickness=2)
    lines = [
        f"seg norm (eye=1): "
        + " : ".join(f"{v:.2f}" for v in r.seg_norm),
        f"eye_gap / eye  : {r.eye_gap_ratio:.3f}",
        f"1:1:1:1:1 loss : {r.ideal_loss_1:.3f}",
        f"JP 1:1.15:1 loss: {r.jp_loss:.3f}",
        f"closest        : {r.closest}",
        f"balance (L-R)  : {r.left_right_balance:+.3f}",
        f"category       : {r.category}",
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
        return None
    fm = FaceMesh(subdivision_level=1); fm.init()
    if fm.detect(image) is None:
        print("顔未検出")
        return None
    r = analyze(fm)
    print(f"\n=== 2.2.3 水平五分割 ===")
    print("seg_norm: " + " : ".join(f"{v:.2f}" for v in r.seg_norm))
    print(f"eye_gap/eye={r.eye_gap_ratio:.3f}  closest={r.closest}  category={r.category}")
    if as_json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
    vis = visualize(image, fm, r)
    out_path = output_path or image_path.parent / f"horizontal_{image_path.stem}.png"
    if imgonly:
        cv2.imwrite(str(out_path), vis)
    else:
        cv2.imwrite(str(out_path), make_side_by_side(image, vis))
    print(f"出力: {out_path}")
    return r


def main():
    parser = argparse.ArgumentParser(description="2.2.3 水平五分割判定")
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--imgonly", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    run_one(Path(args.input), Path(args.output) if args.output else None,
            imgonly=args.imgonly, as_json=args.json)


if __name__ == "__main__":
    main()
