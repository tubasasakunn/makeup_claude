"""
3.2 パイプラインの目視確認スクリプト。

5 顔型それぞれに 3.1 ルールで処方を生成し、3.2 パイプラインで適用、
Before | After (フル合成) の比較グリッドを作る。

Usage:
    python verify_visual.py [-o out.png]
"""
from __future__ import annotations

import argparse
import importlib
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
sys.path.insert(0, str(LOADMAP / "3" / "3.2-pipeline"))

from shared.facemesh import FaceMesh  # noqa: E402
from shared.face_metrics import draw_pil_text  # noqa: E402
from prescription import Prescription  # noqa: E402


def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rules_main = _load_mod(LOADMAP / "3" / "3.1-rules" / "main.py", "rules_m")
pipeline = _load_mod(LOADMAP / "3" / "3.2-pipeline" / "main.py", "pipeline_m")


SAMPLES = [
    ("oval",              "卵.png"),
    ("round",             "丸顔.png"),
    ("long",              "面長.png"),
    ("inverted_triangle", "逆三角.png"),
    ("base",              "ベース.png"),
]


def _fit_aspect(img: np.ndarray, w: int, h: int,
                bg=(30, 30, 30)) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    new_w = max(1, int(iw * scale)); new_h = max(1, int(ih * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)
    y_off = (h - new_h) // 2
    x_off = (w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _crop_face(image: np.ndarray, fm: FaceMesh, margin: float = 0.30) -> np.ndarray:
    h, w = image.shape[:2]
    lm = fm.landmarks_px
    x1, x2 = float(lm[:, 0].min()), float(lm[:, 0].max())
    y1, y2 = float(lm[:, 1].min()), float(lm[:, 1].max())
    bw = x2 - x1; bh = y2 - y1
    mx, my = bw * margin, bh * margin
    x1 = max(0, int(x1 - mx)); x2 = min(w, int(x2 + mx))
    y1 = max(0, int(y1 - my)); y2 = min(h, int(y2 + my))
    return image[y1:y2, x1:x2]


def _label_bar(width: int, height: int, text: str,
               bg=(50, 50, 58), fg=(255, 255, 200), size: int = 20) -> np.ndarray:
    bar = np.full((height, width, 3), bg, dtype=np.uint8)
    tw_est = int(size * 0.6 * len(text))
    x = max(8, (width - tw_est) // 2)
    y = max(4, (height - size) // 2 - 2)
    draw_pil_text(bar, text, (x, y), color=fg, size=size, outline=1)
    return bar


def _process(image_path: Path):
    image = cv2.imread(str(image_path))
    fm = FaceMesh(subdivision_level=1)
    fm.init()
    fm.detect(image)

    rx = rules_main.build_prescription(rules_main.symmetry_mod.analyze(fm))
    after = pipeline.apply_prescription(image, fm, rx, verbose=False)
    return {
        "before": _crop_face(image, fm),
        "after":  _crop_face(after, fm),
        "rx": rx,
    }


def _info_tile(width: int, height: int, fname: str,
               rx: Prescription, expected: str) -> np.ndarray:
    tile = np.full((height, width, 3), 22, dtype=np.uint8)
    ok = rx.source["skeletal_type"] == expected
    color = (140, 255, 140) if ok else (90, 90, 255)
    mark = "PASS" if ok else "MISMATCH"

    # ハイライトエリア名を短縮表示 (最後の _ 以降)
    hl_short = ", ".join(a.split("_")[-1] for a in rx.highlight.areas)
    sd_short = ", ".join(a.split("-")[-1] for a in rx.shadow.areas) if rx.shadow.enabled else "OFF"

    # enabled な eye エリア
    eye_on = [n for n in ("eyeshadow_base", "eyeshadow_crease", "eyeliner",
                          "tear_bag", "lower_outer")
              if getattr(rx.eye, n).enabled]

    lines = [
        (fname, 17, (230, 230, 235)),
        (f"骨格: {rx.source['skeletal_type']} (期待: {expected})", 14, color),
        (f"黄金比: {rx.source['golden_score']:.1f} {rx.source['golden_label']}", 13, (200, 200, 200)),
        (f"目ズレ: {rx.source.get('eye_ideal_ratio_loss', 0):.3f}  "
         f"スケール: ×{rx.source.get('intensity_scale', 1.0):.2f}", 13, (200, 200, 200)),
        ("―" * 22, 11, (80, 80, 90)),
        ("処方:", 14, (220, 220, 230)),
        (f"  base  int={rx.base.intensity:.2f}", 12, (200, 230, 255)),
        (f"  HL ({len(rx.highlight.areas)}): {hl_short}", 12, (200, 230, 255)),
        (f"  Shadow: {sd_short}", 12, (255, 220, 200) if rx.shadow.enabled else (150, 150, 160)),
        (f"  eye ON: {', '.join(eye_on)}", 11, (200, 230, 255)),
        (f"  eye_base int={rx.eye.eyeshadow_base.intensity:.2f}", 12, (200, 230, 255)),
        (f"  eyebrow: {rx.eyebrow.brow_type}", 14, (200, 255, 200)),
    ]
    y = 10
    for text, size, col in lines:
        draw_pil_text(tile, text, (12, y), color=col, size=size, outline=1)
        y += size + 6

    draw_pil_text(tile, mark, (12, height - 34), color=color,
                  size=22, outline=2)
    return tile


def build_grid(output_path: Path) -> Path:
    face_w = 380
    info_w = 360
    row_h = 380
    label_h = 42

    header = [
        _label_bar(face_w, label_h, "Before 顔"),
        _label_bar(face_w, label_h, "After (full makeup)"),
        _label_bar(info_w, label_h, "判定・処方サマリ"),
    ]
    rows = [np.hstack(header)]

    for expected_type, fname in SAMPLES:
        r = _process(ROOT / "imgs" / fname)
        rx = r["rx"]
        before = _fit_aspect(r["before"], face_w, row_h)
        after = _fit_aspect(r["after"], face_w, row_h)
        info = _info_tile(info_w, row_h, fname, rx, expected_type)
        rows.append(np.hstack([before, after, info]))
        rows.append(np.full((2, face_w * 2 + info_w, 3), 80, dtype=np.uint8))

    grid = np.vstack(rows)

    # 全体タイトル
    title_h = 72
    title = np.full((title_h, grid.shape[1], 3), 15, dtype=np.uint8)
    draw_pil_text(title,
                  "Phase 3.2 M2: 処方 → Phase 1 パイプラインでフルメイク適用",
                  (20, 16), color=(250, 250, 200), size=26, outline=2)
    draw_pil_text(title,
                  "base → highlight → shadow → eye(5 areas) → eyebrow を順次合成",
                  (20, 48), color=(180, 220, 255), size=15, outline=1)
    grid = np.vstack([title, grid])
    cv2.imwrite(str(output_path), grid)
    return output_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output",
                   default=str(ROOT / "loadmap/3/outputs/pipeline_grid.png"))
    args = p.parse_args()
    out = build_grid(Path(args.output))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
