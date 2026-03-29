"""
imgs/ 配下の全画像にシャドウを適用し、result_<名前>.png を出力する。

Usage:
    python run_all.py [--intensity 0.15] [--blur 2.0] [--color 139 90 43]
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh
from main import apply_shadow, load_target_areas, make_side_by_side

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
IMGS_DIR = PROJECT_ROOT / "imgs"
OUTPUT_DIR = Path(__file__).resolve().parent

# 画像ファイル名 → プリセット の対応
PRESET_MAP = {
    "卵": None,
    "丸顔": "marugao",
    "面長": "omonaga",
    "逆三角": None,
    "ベース": None,
    "base": None,
}


def main():
    parser = argparse.ArgumentParser(description="全画像にシャドウ一括適用")
    parser.add_argument("--intensity", type=float, default=0.15)
    parser.add_argument("--blur", type=float, default=1.5)
    parser.add_argument("--color", nargs=3, type=int, default=[139, 90, 43],
                        metavar=("R", "G", "B"))
    parser.add_argument("--outdir", type=str, default=None, help="出力先ディレクトリ")
    args = parser.parse_args()

    all_areas = load_target_areas("shadow")
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
        preset = PRESET_MAP.get(name)

        if preset is None:
            print(f"\n--- {name}: シャドウプリセットなし, スキップ ---")
            continue

        area_names = [n for n in all_areas if n.startswith(preset)]
        if not area_names:
            print(f"\n--- {name} (preset: {preset}): 該当エリアなし, スキップ ---")
            continue

        print(f"\n--- {name} (preset: {preset}) ---")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  スキップ: 読み込めません")
            continue

        result = fm.detect(image)
        if result is None:
            print(f"  スキップ: 顔未検出")
            continue

        output = image.copy()
        for area_name in area_names:
            mesh_ids = all_areas[area_name]
            print(f"  適用: {area_name} ({len(mesh_ids)} meshes)")
            output = apply_shadow(
                output, fm, mesh_ids,
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
