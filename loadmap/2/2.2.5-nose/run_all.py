"""imgs/ 全画像に 2.2.5 鼻判定を一括適用"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh
from shared.face_metrics import make_side_by_side
from main import analyze, visualize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
IMGS_DIR = PROJECT_ROOT / "imgs"
OUTPUT_DIR = Path(__file__).resolve().parent / "result_images"
ROMAJI = {"ベース":"base","丸顔":"marugao","卵":"tamago","逆三角":"gyakusankaku","面長":"omonaga"}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    fm = FaceMesh(subdivision_level=1); fm.init()
    results = []
    for img_path in sorted(IMGS_DIR.glob("*.png")):
        if img_path.stem == "base":
            continue
        print(f"\n--- {img_path.name} ---")
        image = cv2.imread(str(img_path))
        if image is None or fm.detect(image) is None:
            continue
        r = analyze(fm)
        print(f"  wing/gap={r.wing_to_gap_ratio:.3f}  length/wing={r.length_to_wing_ratio:.3f}")
        print(f"  angle={r.nose_lip_angle_deg:.1f}  eline={r.eline_offset_px:+.1f}  [{r.overall}]")
        vis = visualize(image, fm, r)
        cv2.imwrite(str(OUTPUT_DIR / f"result_{img_path.stem}.png"),
                    make_side_by_side(image, vis))
        results.append((img_path.stem, r))

    if results:
        parts = []
        for stem, res in results:
            p = cv2.imread(str(OUTPUT_DIR / f"result_{stem}.png"))
            if p is None: continue
            lab = np.zeros((36, p.shape[1], 3), dtype=np.uint8); lab[:] = (40,40,40)
            cv2.putText(
                lab,
                f"{ROMAJI.get(stem,stem)}  wing/gap={res.wing_to_gap_ratio:.2f}  "
                f"len/wing={res.length_to_wing_ratio:.2f}  [{res.overall}]",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,255), 2, cv2.LINE_AA,
            )
            parts.append(np.vstack([lab, p]))
        min_w = min(p.shape[1] for p in parts)
        parts = [cv2.resize(p, (min_w, int(p.shape[0]*min_w/p.shape[1]))) for p in parts]
        cv2.imwrite(str(OUTPUT_DIR / "_summary.png"), np.vstack(parts))
        print("\nサマリー: _summary.png")
    print("\n完了")


if __name__ == "__main__":
    main()
