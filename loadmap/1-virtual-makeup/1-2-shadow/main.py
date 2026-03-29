"""
1.2 シャドウ - 顔のメッシュ領域にシャドウ（暗い色）を反映する

Usage:
    python main.py <input_image> [options]

Examples:
    # omonaga-upper をデフォルトで適用
    python main.py photo.jpg -a omonaga-upper

    # 複数エリアを指定
    python main.py photo.jpg -a omonaga-upper -a omonaga-lower

    # 色味と強度を指定
    python main.py photo.jpg -a marugao-side --color 139 90 43 --intensity 0.15

    # 全 omonaga エリアを一括適用
    python main.py photo.jpg --preset omonaga

    # 出力ファイル指定
    python main.py photo.jpg -a omonaga-upper -o output.png
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# 共有モジュール
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TARGET_JSON = PROJECT_ROOT / "target.json"


# ==============================================================
# Rendering
# ==============================================================
def gaussian_blur_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(mask, (ksize, ksize), ksize / 3.0)


def alpha_composite_multiply(
    src: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    intensity: float,
) -> np.ndarray:
    """シャドウ用: 乗算合成 (暗い色で影を入れる)"""
    a = (mask * intensity)[..., np.newaxis]
    color = np.array(color_bgr, dtype=np.float32) / 255.0
    src_f = src.astype(np.float32)
    # 乗算合成: 元の色 × シャドウ色 で暗くする
    multiplied = src_f * color
    # mask の強度に応じて元画像とブレンド
    result = src_f * (1.0 - a) + multiplied * a
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_shadow(
    image: np.ndarray,
    fm: FaceMesh,
    mesh_ids: list[int],
    color_bgr: tuple[int, int, int] = (90, 68, 50),
    intensity: float = 0.25,
    blur_scale: float = 2.5,
) -> np.ndarray:
    """指定メッシュ領域にシャドウを適用 (外側が濃く内側に向かって馴染む)"""
    h, w = image.shape[:2]

    # 1. メッシュ三角形でマスク作成
    mask = fm.build_mask(mesh_ids, w, h)

    # 2. 距離変換: 端からの距離 → 反転して外側ほど値が大きい
    mask_u8 = (mask * 255).astype(np.uint8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    mx = dist.max()
    if mx > 0:
        dist = dist / mx
    # 反転: 外側(輪郭側)が濃く、内側に向かってフェード
    dist = (1.0 - dist) * mask

    # 3. べき乗カーブで内側への減衰を緩やかに
    dist = np.power(dist, 1.5)

    # 4. 2段階ブラー (内側への自然なフェード)
    face_h = np.linalg.norm(
        fm.landmarks_px[10].astype(float) - fm.landmarks_px[152].astype(float)
    )
    ksize_inner = int(face_h * 0.04 * blur_scale)
    dist = gaussian_blur_mask(dist, ksize_inner)
    ksize_outer = int(face_h * 0.02 * blur_scale)
    dist = gaussian_blur_mask(dist, ksize_outer)

    # 5. 乗算合成
    return alpha_composite_multiply(image, dist, color_bgr, intensity)


def make_side_by_side(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """左: before / 右: after の比較画像"""
    return np.hstack([before, after])


# ==============================================================
# CLI
# ==============================================================
def load_target_areas(category: str = "shadow") -> dict[str, list[int]]:
    """target.json からエリア名→mesh_id辞書を返す"""
    with open(TARGET_JSON) as f:
        data = json.load(f)
    areas = {}
    for entry in data.get(category, []):
        ids = entry["mesh_id"]
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        areas[entry["name"]] = ids
    return areas


def main():
    parser = argparse.ArgumentParser(description="シャドウ適用")
    parser.add_argument("input", help="入力画像パス")
    parser.add_argument("-o", "--output", help="出力画像パス (default: shadow_result.png)")
    parser.add_argument("-a", "--area", action="append", help="適用エリア名 (複数指定可)")
    parser.add_argument("--preset", help="プリセット (omonaga/marugao で前方一致フィルタ)")
    parser.add_argument("--color", nargs=3, type=int, default=[139, 90, 43],
                        metavar=("R", "G", "B"), help="シャドウ色 (default: 139 90 43)")
    parser.add_argument("--intensity", type=float, default=0.25, help="強度 0.0-1.0 (default: 0.25)")
    parser.add_argument("--blur", type=float, default=2.5, help="ブラー倍率 (default: 2.5)")
    parser.add_argument("--list", action="store_true", help="利用可能エリア一覧を表示")
    parser.add_argument("--imgonly", action="store_true", help="結果画像のみ (比較画像なし)")
    args = parser.parse_args()

    # エリア一覧
    all_areas = load_target_areas("shadow")
    if args.list:
        print("利用可能なシャドウエリア:")
        for name, ids in all_areas.items():
            print(f"  {name:20s} ({len(ids)} meshes)")
        return

    # 適用エリア決定
    if args.area:
        area_names = args.area
    elif args.preset:
        area_names = [n for n in all_areas if n.startswith(args.preset)]
        if not area_names:
            print(f"Error: preset '{args.preset}' に一致するエリアがありません")
            return
    else:
        # デフォルト: 全エリア
        area_names = list(all_areas.keys())

    # 画像読み込み
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: 画像を読み込めません: {args.input}")
        return

    # FaceMesh 初期化・検出
    print("FaceMesh 初期化中...")
    fm = FaceMesh(subdivision_level=1)
    fm.init()

    print("顔検出中...")
    result = fm.detect(image)
    if result is None:
        print("Error: 顔が検出されませんでした")
        return

    print(f"メッシュ: {len(fm.triangles)} 三角形, {len(fm.points)} 頂点")

    # シャドウ適用
    color_bgr = (args.color[2], args.color[1], args.color[0])  # RGB→BGR
    output = image.copy()

    for name in area_names:
        if name not in all_areas:
            print(f"Warning: エリア '{name}' が見つかりません, スキップ")
            continue
        mesh_ids = all_areas[name]
        print(f"  適用: {name} ({len(mesh_ids)} meshes, intensity={args.intensity})")
        output = apply_shadow(
            output, fm, mesh_ids,
            color_bgr=color_bgr,
            intensity=args.intensity,
            blur_scale=args.blur,
        )

    # 出力
    out_path = args.output or "shadow_result.png"
    if args.imgonly:
        cv2.imwrite(out_path, output)
    else:
        comparison = make_side_by_side(image, output)
        cv2.imwrite(out_path, comparison)

    print(f"出力: {out_path}")


if __name__ == "__main__":
    main()
