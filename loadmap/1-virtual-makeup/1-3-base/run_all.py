"""
imgs/ 配下の全画像にベースメイクを適用し、result_<名前>.png を出力する。

Usage:
    python run_all.py [--intensity 0.15] [--blur 2.5] [--color 235 200 170]
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh
from main import apply_base, make_side_by_side

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
IMGS_DIR = PROJECT_ROOT / "imgs"
OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="全画像にベースメイク一括適用")
    parser.add_argument("--intensity", type=float, default=0.30)
    parser.add_argument("--blur", type=float, default=2.5)
    parser.add_argument("--color", nargs=3, type=int, default=[235, 200, 170],
                        metavar=("R", "G", "B"))
    parser.add_argument("--outdir", type=str, default=None, help="出力先ディレクトリ")
    args = parser.parse_args()

    color_bgr = (args.color[2], args.color[1], args.color[0])
    out_dir = Path(args.outdir) if args.outdir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # FaceMesh 初期化 (1回だけ)
    print("FaceMesh 初期化中...")
    fm = FaceMesh(subdivision_level=1)
    fm.init()

    images = sorted(IMGS_DIR.glob("*.png"))
    if not images:
        print(f"Error: {IMGS_DIR} に画像がありません")
        return

    for img_path in images:
        name = img_path.stem

        print(f"\n--- {name} ---")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  スキップ: 読み込めません")
            continue

        result = fm.detect(image)
        if result is None:
            print(f"  スキップ: 顔未検出")
            continue

        print(f"  適用: 顔全体 ({len(fm.triangles)} meshes)")
        output = apply_base(
            image, fm,
            color_bgr=color_bgr,
            intensity=args.intensity,
            blur_scale=args.blur,
        )

        out_path = out_dir / f"result_{name}.png"
        comparison = make_side_by_side(image, output)
        cv2.imwrite(str(out_path), comparison)
        print(f"  出力: {out_path.name}")

    print("\n完了")


if __name__ == "__main__":
    main()
