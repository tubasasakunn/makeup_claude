"""4つの顔タイプの before/after を 1 枚の比較グリッドにまとめる"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_LOADMAP_ROOT = _HERE.parent
if str(_LOADMAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOADMAP_ROOT))

from shared.face_metrics import draw_pil_text  # noqa: E402

RESULTS = _HERE / "results"
IMGS = _LOADMAP_ROOT.parent / "imgs"

# (表示ラベル, 入力ファイル名 stem)
CASES = [
    ("卵", "卵"),
    ("丸顔", "丸顔"),
    ("面長 (base.png)", "base"),
    ("逆三角", "逆三角"),
]

SKEL_JP = {
    "oval": "卵型",
    "round": "丸型",
    "long": "面長",
    "inverted_triangle": "逆三角",
    "base": "ベース型",
}

CELL = 512

def _label(img, text, y, x: int = 12, size: int = 22):
    # PIL で日本語描画 (半透明背景付き)
    draw_pil_text(
        img, text, (x, y),
        color=(255, 255, 255), size=size,
        bg=(20, 20, 25), bg_alpha=0.72, bg_pad=5,
    )


def build_row(display_name: str, stem: str) -> np.ndarray | None:
    before_path = IMGS / f"{stem}.png"
    after_path = RESULTS / f"{stem}_after.png"
    prof_path = RESULTS / f"{stem}_profile.json"
    before = cv2.imread(str(before_path))
    after = cv2.imread(str(after_path))
    if before is None or after is None or not prof_path.exists():
        print(f"missing for {stem}")
        return None

    before = cv2.resize(before, (CELL, CELL))
    after = cv2.resize(after, (CELL, CELL))

    prof = json.loads(prof_path.read_text(encoding="utf-8"))
    plan = prof["plan"]
    summary = prof["profile"]["summary"]

    brow_type = plan["steps"][-1]["area"]
    shadow_steps = [s for s in plan["steps"] if s["stage"] == "shadow"]
    shadow_areas = ",".join(s["area"] for s in shadow_steps) or "none"
    highlight_areas = ",".join(
        s["area"] for s in plan["steps"] if s["stage"] == "highlight"
    )

    row = np.hstack([before, after])

    # Before 側にラベル
    _label(row, f"{display_name} → {SKEL_JP[plan['skeletal_type']]}", 20, size=24)
    _label(row, f"黄金比: {summary['golden_score']:.0f}", 60, size=20)

    # After 側にラベル
    ox = CELL
    _label(row, f"眉: {brow_type}", 20, x=ox + 14, size=20)
    _label(row, f"影: {shadow_areas}", 52, x=ox + 14, size=18)
    _label(row, f"HL: {highlight_areas}", 82, x=ox + 14, size=18)
    _label(row, f"全体 ×{plan['overall_modifier']:.2f}", 114, x=ox + 14, size=18)
    return row


def main():
    rows: list[np.ndarray] = []
    for display, stem in CASES:
        r = build_row(display, stem)
        if r is not None:
            rows.append(r)

    if not rows:
        raise SystemExit("no rows to build")

    # 上部ヘッダ
    header_h = 68
    header = np.zeros((header_h, rows[0].shape[1], 3), dtype=np.uint8)
    header[:] = (30, 30, 35)
    draw_pil_text(
        header, "Section 3: 顔判定 → 化粧選択 → 適用",
        (20, 18),
        color=(255, 255, 255), size=30,
        outline_color=(0, 0, 0), outline=2,
    )
    draw_pil_text(
        header, "左: Before    右: After",
        (rows[0].shape[1] - 260, 22),
        color=(200, 200, 200), size=20,
        outline_color=(0, 0, 0), outline=2,
    )

    grid = np.vstack([header] + rows)
    out_path = RESULTS / "section3_grid.png"
    cv2.imwrite(str(out_path), grid)
    print(f"saved: {out_path}  shape={grid.shape}")


if __name__ == "__main__":
    main()
