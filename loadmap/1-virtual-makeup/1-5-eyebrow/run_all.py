"""全サンプル画像に眉消しを適用する"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh
from main import load_eyebrow_areas, erase_eyebrows

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
IMGS_DIR = PROJECT_ROOT / "imgs"
OUTPUT_DIR = Path(__file__).resolve().parent / "result_images"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    areas = load_eyebrow_areas()
    if "eyebrow_full" not in areas:
        print("Error: target.json に eyebrow_full がありません")
        return

    print("FaceMesh 初期化中...")
    fm = FaceMesh(subdivision_level=1)
    fm.init()

    pngs = sorted(IMGS_DIR.glob("*.png"))
    if not pngs:
        print("Error: imgs/ に画像がありません")
        return

    for img_path in pngs:
        print(f"\n処理中: {img_path.name}")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  スキップ: 読み込み失敗")
            continue

        result = fm.detect(image)
        if result is None:
            print(f"  スキップ: 顔未検出")
            continue

        output = erase_eyebrows(
            image, fm,
            eyebrow_mesh_ids=areas["eyebrow_full"],
            skin_mesh_ids=areas.get("eyebrow_skin", []),
        )

        # 左が元画像、右が眉消し
        comparison = np.hstack([image, output])
        out_path = OUTPUT_DIR / f"result_{img_path.stem}.png"
        cv2.imwrite(str(out_path), comparison)
        print(f"  出力: {out_path.name}")

    print("\n完了!")


if __name__ == "__main__":
    main()
