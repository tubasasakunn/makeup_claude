"""
imgs/ 配下の全画像に骨格判定を適用し、result_<名前>.png を出力する。

Usage:
    python run_all.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh
from shared.face_metrics import make_side_by_side
from main import build_report, classify, visualize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
IMGS_DIR = PROJECT_ROOT / "imgs"
OUTPUT_DIR = Path(__file__).resolve().parent / "result_images"

# cv2.putText は日本語非対応なのでラベル用のローマ字マップ
ROMAJI = {
    "ベース": "base (square)",
    "丸顔": "marugao (round)",
    "卵": "tamago (oval)",
    "逆三角": "gyakusankaku (inverted)",
    "面長": "omonaga (long)",
}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("FaceMesh 初期化中...")
    fm = FaceMesh(subdivision_level=1)
    fm.init()

    pngs = sorted(IMGS_DIR.glob("*.png"))
    if not pngs:
        print(f"Error: {IMGS_DIR} に画像がありません")
        return

    results = []
    for img_path in pngs:
        # base.png はレイアウト用で顔検出には向かないのでスキップ
        if img_path.stem == "base":
            continue

        print(f"\n--- {img_path.name} ---")
        image = cv2.imread(str(img_path))
        if image is None:
            print("  スキップ: 読み込み失敗")
            continue

        if fm.detect(image) is None:
            print("  スキップ: 顔未検出")
            continue

        result = classify(fm)
        print(f"  TYPE: {result.type}  ({result.type_label})")
        print(f"  scores: " + ", ".join(
            f"{k}={v:.3f}" for k, v in sorted(result.scores.items(), key=lambda kv: -kv[1])
        ))

        report = build_report(image, fm, result)
        out_path = OUTPUT_DIR / f"result_{img_path.stem}.png"
        cv2.imwrite(str(out_path), report)
        print(f"  出力: {out_path.name}")
        results.append((img_path.stem, result))

    # サマリーグリッド（縦に並べる）
    if results:
        grid_path = OUTPUT_DIR / "_summary.png"
        parts = []
        for stem, res in results:
            part = cv2.imread(str(OUTPUT_DIR / f"result_{stem}.png"))
            if part is None:
                continue
            # 上部に名前のラベル帯
            h = 36
            label = np.zeros((h, part.shape[1], 3), dtype=np.uint8)
            label[:] = (40, 40, 40)
            name = ROMAJI.get(stem, stem)
            cv2.putText(
                label, f"{name}  =>  {res.type}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
            )
            parts.append(np.vstack([label, part]))
        if parts:
            # 高さを揃えて横並び...ではなく縦積み
            min_w = min(p.shape[1] for p in parts)
            parts = [
                cv2.resize(p, (min_w, int(p.shape[0] * min_w / p.shape[1])))
                for p in parts
            ]
            grid = np.vstack(parts)
            cv2.imwrite(str(grid_path), grid)
            print(f"\nサマリー: {grid_path.name}")

    print("\n完了")


if __name__ == "__main__":
    main()
