"""imgs/ 全画像に 2.2.1 顔縦横バランス判定を一括適用"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh
from shared.face_metrics import make_side_by_side
from main import RATIOS, analyze, build_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
IMGS_DIR = PROJECT_ROOT / "imgs"
OUTPUT_DIR = Path(__file__).resolve().parent / "result_images"

ROMAJI = {
    "ベース": "base",
    "丸顔": "marugao",
    "卵": "tamago",
    "逆三角": "gyakusankaku",
    "面長": "omonaga",
}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("FaceMesh 初期化中...")
    fm = FaceMesh(subdivision_level=1)
    fm.init()

    pngs = sorted(IMGS_DIR.glob("*.png"))
    results = []
    for img_path in pngs:
        if img_path.stem == "base":
            continue
        print(f"\n--- {img_path.name} ---")
        image = cv2.imread(str(img_path))
        if image is None or fm.detect(image) is None:
            print("  スキップ")
            continue

        r = analyze(fm)
        print(f"  aspect={r.aspect:.3f}  closest={r.closest_ratio} ({RATIOS[r.closest_ratio]})")
        print(f"  size: {r.face_height_cm:.1f}cm x {r.face_width_cm:.1f}cm  kogao={r.kogao_score:.1f}")

        report = build_report(image, fm, r)
        out_path = OUTPUT_DIR / f"result_{img_path.stem}.png"
        cv2.imwrite(str(out_path), report)
        print(f"  出力: {out_path.name}")
        results.append((img_path.stem, r))

    # サマリー
    if results:
        parts = []
        for stem, res in results:
            part = cv2.imread(str(OUTPUT_DIR / f"result_{stem}.png"))
            if part is None:
                continue
            h = 36
            label = np.zeros((h, part.shape[1], 3), dtype=np.uint8)
            label[:] = (40, 40, 40)
            name = ROMAJI.get(stem, stem)
            cv2.putText(
                label,
                f"{name}  aspect={res.aspect:.3f} ({res.closest_ratio})"
                f"  {res.face_height_cm:.1f}x{res.face_width_cm:.1f}cm",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
            )
            parts.append(np.vstack([label, part]))
        if parts:
            min_w = min(p.shape[1] for p in parts)
            parts = [
                cv2.resize(p, (min_w, int(p.shape[0] * min_w / p.shape[1])))
                for p in parts
            ]
            grid = np.vstack(parts)
            grid_path = OUTPUT_DIR / "_summary.png"
            cv2.imwrite(str(grid_path), grid)
            print(f"\nサマリー: {grid_path.name}")

    print("\n完了")


if __name__ == "__main__":
    main()
