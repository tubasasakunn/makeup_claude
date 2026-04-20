"""
3.1 の処方が「見た目として妥当か」を目視確認するための検証スクリプト。

5つのサンプル画像それぞれに、ルールが生成した処方(眉タイプのみ)を適用し、
Before | After | ラベル付きタイル を並べたグリッド画像を1枚作る。

Usage:
    python verify_visual.py
    python verify_visual.py -o custom_output.png
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOADMAP = ROOT / "loadmap"
sys.path.insert(0, str(LOADMAP))
sys.path.insert(0, str(LOADMAP / "3" / "3.0-schema"))
sys.path.insert(0, str(LOADMAP / "3" / "3.1-rules"))

from shared.facemesh import FaceMesh  # noqa: E402
from shared.face_metrics import draw_pil_text  # noqa: E402
from prescription import Prescription  # noqa: E402
import importlib

rules_main = importlib.import_module("main")  # 3.1 ルール


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eyebrow_mod = _load_module(
    LOADMAP / "1-virtual-makeup" / "1-5-eyebrow" / "main.py", "eyebrow_apply"
)


SAMPLES = [
    ("oval",              "卵.png"),
    ("round",             "丸顔.png"),
    ("long",              "面長.png"),
    ("inverted_triangle", "逆三角.png"),
    ("base",              "ベース.png"),
]

TILE_SIZE = 420  # 各タイルの幅(正方形)


def _crop_face(image: np.ndarray, fm: FaceMesh, margin: float = 0.35) -> np.ndarray:
    h, w = image.shape[:2]
    lm = fm.landmarks_px
    xs = lm[:, 0]; ys = lm[:, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    bw = x2 - x1; bh = y2 - y1
    mx, my = bw * margin, bh * margin
    x1 = max(0, int(x1 - mx)); x2 = min(w, int(x2 + mx))
    y1 = max(0, int(y1 - my)); y2 = min(h, int(y2 + my))
    crop = image[y1:y2, x1:x2]
    return crop


def _resize_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w = int(w * scale); new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 25, dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _draw_label_bar(width: int, height: int, text: str,
                    bg=(30, 30, 35), fg=(240, 240, 245),
                    size: int = 18) -> np.ndarray:
    bar = np.full((height, width, 3), bg, dtype=np.uint8)
    # 中央寄せのためサイズを概算 (size * 文字数 * 0.6 で大まかに中心を取る)
    tw_est = int(size * 0.6 * len(text))
    x = max(6, (width - tw_est) // 2)
    y = max(4, (height - size) // 2 - 2)
    draw_pil_text(bar, text, (x, y), color=fg, size=size, outline=1,
                  outline_color=(0, 0, 0))
    return bar


def _crop_eyebrow(image: np.ndarray, fm: FaceMesh, margin: float = 0.4) -> np.ndarray:
    h, w = image.shape[:2]
    brow_lms = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55,
                300, 293, 334, 296, 336, 276, 283, 282, 295, 285]
    xs = [float(fm.landmarks_px[l][0]) for l in brow_lms]
    ys = [float(fm.landmarks_px[l][1]) for l in brow_lms]
    m = int((max(xs) - min(xs)) * margin)
    x1 = max(0, int(min(xs)) - m); x2 = min(w, int(max(xs)) + m)
    y1 = max(0, int(min(ys)) - m); y2 = min(h, int(max(ys)) + m)
    return image[y1:y2, x1:x2]


def _process_one(image_path: Path):
    image = cv2.imread(str(image_path))
    fm = FaceMesh(subdivision_level=1)
    fm.init()
    fm.detect(image)

    rx = rules_main.build_prescription(rules_main.symmetry_mod.analyze(fm))

    after = eyebrow_mod.apply_eyebrow_makeup(
        image, fm,
        brow_type=rx.eyebrow.brow_type,
        color_rgb=rx.eyebrow.color_rgb,
        intensity=rx.eyebrow.intensity,
    )
    return {
        "before_face": _crop_face(image, fm),
        "after_face":  _crop_face(after, fm),
        "before_brow": _crop_eyebrow(image, fm),
        "after_brow":  _crop_eyebrow(after, fm),
        "rx": rx,
    }


def _info_tile(size: int, fname: str, rx: Prescription,
               expected: str) -> np.ndarray:
    tile = np.full((size, size, 3), 20, dtype=np.uint8)
    ok = rx.source["skeletal_type"] == expected
    color = (140, 255, 140) if ok else (90, 90, 255)
    mark = "PASS" if ok else "MISMATCH"

    lines = [
        (f"{fname}", 16, (230, 230, 235)),
        (f"骨格判定: {rx.source['skeletal_type']}", 15, color),
        (f"(期待値: {expected})", 14, (180, 180, 185)),
        (f"→ 眉タイプ: {rx.eyebrow.brow_type}", 16, (200, 230, 255)),
        (f"黄金比: {rx.source['golden_score']:.1f}  "
         f"({rx.source['golden_label']})", 14, (200, 200, 200)),
    ]
    y = 18
    for text, fsize, col in lines:
        draw_pil_text(tile, text, (14, y), color=col, size=fsize,
                      outline=1, outline_color=(0, 0, 0))
        y += fsize + 8

    draw_pil_text(tile, mark, (14, size - 36), color=color, size=22,
                  outline=2, outline_color=(0, 0, 0))
    return tile


def _fit_aspect(img: np.ndarray, w: int, h: int,
                bg=(30, 30, 30)) -> np.ndarray:
    """アスペクト保持で (w, h) のキャンバスに収める"""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    new_w = max(1, int(iw * scale)); new_h = max(1, int(ih * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)
    y_off = (h - new_h) // 2
    x_off = (w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def build_grid(output_path: Path) -> Path:
    """横長レイアウト: 1 行 = 1 顔 (face-before | face-after | brow-before | brow-after | info)"""
    face_w = 300              # 顔サムネ幅
    brow_w = 360              # 眉ズーム幅
    info_w = 360              # 情報パネル幅
    row_h = 300               # 1 行の高さ
    label_h = 40

    col_widths = [face_w, face_w, brow_w, brow_w, info_w]
    total_w = sum(col_widths)

    # ヘッダ
    header_cells = [
        _draw_label_bar(face_w, label_h, "Before 顔", bg=(50, 50, 58), fg=(255, 255, 200)),
        _draw_label_bar(face_w, label_h, "After 顔",  bg=(50, 50, 58), fg=(255, 255, 200)),
        _draw_label_bar(brow_w, label_h, "Before 眉ズーム", bg=(50, 50, 58), fg=(255, 255, 200)),
        _draw_label_bar(brow_w, label_h, "After 眉ズーム",  bg=(50, 50, 58), fg=(255, 255, 200)),
        _draw_label_bar(info_w, label_h, "判定結果・処方",  bg=(50, 50, 58), fg=(255, 255, 200)),
    ]
    rows = [np.hstack(header_cells)]

    for expected_type, fname in SAMPLES:
        r = _process_one(ROOT / "imgs" / fname)
        rx = r["rx"]

        face_before = _fit_aspect(r["before_face"], face_w, row_h)
        face_after = _fit_aspect(r["after_face"], face_w, row_h)
        brow_before = _fit_aspect(r["before_brow"], brow_w, row_h)
        brow_after = _fit_aspect(r["after_brow"], brow_w, row_h)
        info = _info_tile(row_h, fname, rx, expected_type)
        # info を info_w 幅にリサイズ (_info_tile は正方形で生成)
        info = cv2.resize(info, (info_w, row_h), interpolation=cv2.INTER_AREA)

        rows.append(np.hstack([face_before, face_after,
                               brow_before, brow_after, info]))

        # 区切り線
        rows.append(np.full((2, total_w, 3), 80, dtype=np.uint8))

    grid = np.vstack(rows)

    # 全体タイトル
    title_h = 60
    title = np.full((title_h, total_w, 3), 15, dtype=np.uint8)
    draw_pil_text(title,
                  "Phase 3.1 M1: 骨格判定 → 眉タイプ処方の検証",
                  (20, 16), color=(250, 250, 200), size=24, outline=2)
    draw_pil_text(title,
                  "rule: oval→natural, round→arch, long→parallel, "
                  "inverted_triangle→straight, base→natural",
                  (20, 46), color=(180, 220, 255), size=13, outline=1)
    grid = np.vstack([title, grid])

    cv2.imwrite(str(output_path), grid)
    return output_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output",
                   default=str(ROOT / "loadmap/3/outputs/verify_grid.png"))
    args = p.parse_args()
    out = build_grid(Path(args.output))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
