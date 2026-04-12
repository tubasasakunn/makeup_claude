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
    compose_report,
    draw_line_outlined,
    draw_pil_pill,
    draw_pil_text,
    draw_point_outlined,
    make_side_by_side,
    render_report_panel,
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
# 可視化 (UI/UX 改善版: レポートパネル + annotate 分離)
# ==============================================================
SEG_COLORS = [
    (180, 180, 180),  # 右余白
    (80, 220, 255),   # 右目幅 アンバー
    (140, 255, 160),  # 目間 緑
    (80, 220, 255),   # 左目幅 アンバー
    (180, 180, 180),  # 左余白
]
SEG_NAMES = ["余白R", "目R", "目間", "目L", "余白L"]

CATEGORY_COLOR = {
    "Ideal": (140, 255, 160),
    "Center-Converged (yose-me)": (80, 180, 255),
    "Center-Diverged (hanare-me)": (200, 140, 255),
}


def _segment_xs(fm: FaceMesh):
    x_temple_r = min(
        fm.landmarks_px[LM.TEMPLE_R][0], fm.landmarks_px[LM.TEMPLE_L][0]
    )
    x_temple_l = max(
        fm.landmarks_px[LM.TEMPLE_R][0], fm.landmarks_px[LM.TEMPLE_L][0]
    )
    r_out_x = min(
        fm.landmarks_px[LM.EYE_OUTER_R][0], fm.landmarks_px[LM.EYE_INNER_R][0]
    )
    r_in_x = max(
        fm.landmarks_px[LM.EYE_OUTER_R][0], fm.landmarks_px[LM.EYE_INNER_R][0]
    )
    l_in_x = min(
        fm.landmarks_px[LM.EYE_INNER_L][0], fm.landmarks_px[LM.EYE_OUTER_L][0]
    )
    l_out_x = max(
        fm.landmarks_px[LM.EYE_INNER_L][0], fm.landmarks_px[LM.EYE_OUTER_L][0]
    )
    return [x_temple_r, r_out_x, r_in_x, l_in_x, l_out_x, x_temple_l]


def annotate_face(image: np.ndarray, fm: FaceMesh, r: HorizontalResult,
                  scale: float = 1.0) -> np.ndarray:
    if scale != 1.0:
        img = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
        )
    else:
        img = image.copy()
    h, w = img.shape[:2]

    y_r = (fm.landmarks_px[LM.EYE_TOP_R][1] + fm.landmarks_px[LM.EYE_BOT_R][1]) / 2
    y_l = (fm.landmarks_px[LM.EYE_TOP_L][1] + fm.landmarks_px[LM.EYE_BOT_L][1]) / 2
    y_line = int(((y_r + y_l) / 2) * scale)

    xs = [x * scale for x in _segment_xs(fm)]

    band_h = int(22 * scale)
    overlay = img.copy()
    for i in range(5):
        x0 = int(xs[i])
        x1 = int(xs[i + 1])
        cv2.rectangle(
            overlay, (x0, y_line - band_h), (x1, y_line + band_h),
            SEG_COLORS[i], -1,
        )
    cv2.addWeighted(overlay, 0.32, img, 0.68, 0, img)

    lw = max(2, int(2 * scale))
    for x in xs:
        draw_line_outlined(
            img, (int(x), y_line - band_h - 4),
            (int(x), y_line + band_h + 4),
            (255, 255, 255), thickness=lw,
        )

    label_bg = (20, 20, 25)
    label_size = max(14, int(15 * scale))
    for i in range(5):
        cx = (int(xs[i]) + int(xs[i + 1])) // 2
        if r.seg_norm:
            draw_pil_text(
                img, f"{r.seg_norm[i]:.2f}",
                (cx - int(20 * scale), y_line + band_h + int(6 * scale)),
                color=SEG_COLORS[i], size=label_size,
                bg=label_bg, bg_alpha=0.78, bg_pad=4,
            )
            draw_pil_text(
                img, SEG_NAMES[i],
                (cx - int(22 * scale), y_line - band_h - int(28 * scale)),
                color=SEG_COLORS[i], size=label_size,
                bg=label_bg, bg_alpha=0.78, bg_pad=4,
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

    # 右上凡例
    legend_items = [
        ("余白", SEG_COLORS[0]),
        ("目幅", SEG_COLORS[1]),
        ("目間", SEG_COLORS[2]),
    ]
    legend_font = max(14, int(15 * scale))
    sw_w = int(18 * scale)
    sw_h = int(14 * scale)
    row_h = int(26 * scale)
    lx = w - int(140 * scale)
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


def build_panel(r: HorizontalResult, width: int, height: int) -> np.ndarray:
    pill_color = CATEGORY_COLOR.get(r.category, (255, 255, 255))
    seg = r.seg_norm if r.seg_norm else [0.0] * 5

    spec = [
        ("title", "2.2.3  水平五分割", (230, 230, 230)),
        ("subtitle", "Horizontal Fifths (eye = 1)"),
        ("divider",),
        ("spacer", 4),
        ("section", "判定結果"),
        ("big", r.category.split(" ")[0].upper(), pill_color),
        ("text", r.category, (210, 210, 215)),
        ("spacer", 6),
        ("section", "5 セグメント (目幅=1)"),
        ("kv", "余白R",  f"{seg[0]:.2f}", SEG_COLORS[0]),
        ("kv", "目R",    f"{seg[1]:.2f}", SEG_COLORS[1]),
        ("kv", "目間",   f"{seg[2]:.2f}", SEG_COLORS[2]),
        ("kv", "目L",    f"{seg[3]:.2f}", SEG_COLORS[3]),
        ("kv", "余白L",  f"{seg[4]:.2f}", SEG_COLORS[4]),
        ("spacer", 6),
        ("section", "目間 / 目幅"),
        ("kv", "eye_gap_ratio", f"{r.eye_gap_ratio:.3f}", pill_color),
        ("text", "ideal 1.00 - 1.55", (180, 180, 185)),
        ("spacer", 6),
        ("section", "乖離度"),
        ("kv", "1:1:1:1:1 loss", f"{r.ideal_loss_1:.3f}"),
        ("kv", "JP 1:1.15:1 loss", f"{r.jp_loss:.3f}"),
        ("kv", "balance (L-R)", f"{r.left_right_balance:+.3f}"),
    ]

    return render_report_panel(spec, width, height)


def build_report(image: np.ndarray, fm: FaceMesh,
                 r: HorizontalResult) -> np.ndarray:
    scale = 1.5
    src_big = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
    )
    ann_big = annotate_face(image, fm, r, scale=scale)
    panel = build_panel(r, 620, src_big.shape[0])
    return compose_report(src_big, ann_big, panel)


# 後方互換
def visualize(image: np.ndarray, fm: FaceMesh, r: HorizontalResult) -> np.ndarray:
    return annotate_face(image, fm, r, scale=1.0)


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
    out_path = output_path or image_path.parent / f"horizontal_{image_path.stem}.png"
    if imgonly:
        cv2.imwrite(str(out_path), annotate_face(image, fm, r))
    else:
        cv2.imwrite(str(out_path), build_report(image, fm, r))
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
