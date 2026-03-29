"""
1.3 ベース - 顔全体にベースメイク（ファンデーション）を反映する

顔メッシュ全体に色を乗せて肌色を均一にする。
ハイライト/シャドウと異なり、特定エリアではなく顔全体が対象。

Usage:
    python main.py <input_image> [options]

Examples:
    # デフォルト（自然な肌色・強度0.15）で適用
    python main.py photo.jpg

    # 色味と強度を指定
    python main.py photo.jpg --color 235 200 170 --intensity 0.2

    # ブラー倍率を変更
    python main.py photo.jpg --blur 3.0

    # 出力ファイル指定
    python main.py photo.jpg -o output.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# 共有モジュール
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh


# ==============================================================
# Rendering
# ==============================================================
def gaussian_blur_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(mask, (ksize, ksize), ksize / 3.0)


def alpha_composite_normal(
    src: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    intensity: float,
) -> np.ndarray:
    """ベース用: 通常合成 (元の色とベース色を線形ブレンド)"""
    a = (mask * intensity)[..., np.newaxis]
    color = np.array(color_bgr, dtype=np.float32)
    src_f = src.astype(np.float32)
    result = src_f * (1.0 - a) + color * a
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_base(
    image: np.ndarray,
    fm: FaceMesh,
    color_bgr: tuple[int, int, int] = (170, 200, 235),
    intensity: float = 0.30,
    blur_scale: float = 2.5,
) -> np.ndarray:
    """顔全体にベースメイクを適用 (顔メッシュ全体をカバー、輪郭で自然にフェード)"""
    h, w = image.shape[:2]

    # 1. 全メッシュ三角形でマスク作成
    all_mesh_ids = list(range(len(fm.triangles)))
    mask = fm.build_mask(all_mesh_ids, w, h)

    # 2. 距離変換: 輪郭からの距離 → 中心ほど値が大きい
    mask_u8 = (mask * 255).astype(np.uint8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    mx = dist.max()
    if mx > 0:
        dist = dist / mx

    # 3. べき乗カーブで輪郭付近のフェードを自然に
    #    中心部はしっかり色が乗り、輪郭に向かって薄くなる
    dist = np.power(dist, 0.3)

    # 4. マスク範囲内に限定
    dist = dist * mask

    # 5. ブラーで輪郭を滑らかに
    face_h = np.linalg.norm(
        fm.landmarks_px[10].astype(float) - fm.landmarks_px[152].astype(float)
    )
    ksize = int(face_h * 0.05 * blur_scale)
    dist = gaussian_blur_mask(dist, ksize)

    # 6. 通常合成
    return alpha_composite_normal(image, dist, color_bgr, intensity)


def make_side_by_side(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """左: before / 右: after の比較画像"""
    return np.hstack([before, after])


# ==============================================================
# CLI
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="ベースメイク適用")
    parser.add_argument("input", help="入力画像パス")
    parser.add_argument("-o", "--output", help="出力画像パス (default: base_result.png)")
    parser.add_argument("--color", nargs=3, type=int, default=[235, 200, 170],
                        metavar=("R", "G", "B"), help="ベース色 (default: 235 200 170)")
    parser.add_argument("--intensity", type=float, default=0.30, help="強度 0.0-1.0 (default: 0.30)")
    parser.add_argument("--blur", type=float, default=2.5, help="ブラー倍率 (default: 2.5)")
    parser.add_argument("--imgonly", action="store_true", help="結果画像のみ (比較画像なし)")
    args = parser.parse_args()

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

    # ベースメイク適用
    color_bgr = (args.color[2], args.color[1], args.color[0])  # RGB→BGR
    print(f"  適用: 顔全体 ({len(fm.triangles)} meshes, intensity={args.intensity})")
    output = apply_base(
        image, fm,
        color_bgr=color_bgr,
        intensity=args.intensity,
        blur_scale=args.blur,
    )

    # 出力
    out_path = args.output or "base_result.png"
    if args.imgonly:
        cv2.imwrite(out_path, output)
    else:
        comparison = make_side_by_side(image, output)
        cv2.imwrite(out_path, comparison)

    print(f"出力: {out_path}")


if __name__ == "__main__":
    main()
