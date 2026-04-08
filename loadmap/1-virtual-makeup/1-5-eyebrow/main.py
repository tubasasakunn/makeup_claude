"""
1.5 眉メイク - 眉を消して新しい眉を描画する

Phase 1: 眉消し
  - ランドマークからスムーズなポリゴンマスクを生成（メッシュ不使用）
  - 周辺の肌色を抽出
  - cv2.inpaint + 肌色ブレンドで自然に眉を消去

Phase 2: 眉描画（今後実装）
  - 消した上に新しい眉を描画

Usage:
    python main.py <input_image> [options]

Examples:
    # 眉を消す（デフォルト）
    python main.py photo.jpg

    # inpaintの半径を変更
    python main.py photo.jpg --radius 5

    # 肌色ブレンドの強度変更
    python main.py photo.jpg --blend-strength 0.4

    # 出力ファイル指定
    python main.py photo.jpg -o output.png
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TARGET_JSON = PROJECT_ROOT / "target.json"

# =========================================================
# MediaPipe 478点 眉ランドマークID
# =========================================================
# 右眉 上端ライン（外側→内側）
RIGHT_EYEBROW_UPPER = [70, 63, 105, 66, 107]
# 右眉 下端ライン（外側→内側）
RIGHT_EYEBROW_LOWER = [46, 53, 52, 65, 55]
# 左眉 上端ライン（外側→内側）
LEFT_EYEBROW_UPPER = [300, 293, 334, 296, 336]
# 左眉 下端ライン（外側→内側）
LEFT_EYEBROW_LOWER = [276, 283, 282, 295, 285]


def gaussian_blur_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(mask, (ksize, ksize), ksize / 3.0)


def build_eyebrow_polygon_mask(fm: FaceMesh, w: int, h: int,
                               expand_px: int = 0) -> np.ndarray:
    """ランドマークからスムーズなポリゴンマスクを生成

    ランドマーク座標からポリゴンを作り cv2.fillPoly で塗りつぶす。
    expand_px でポリゴン外周をピクセル単位で拡張（眉尻の毛カバー用）。
    """
    mask = np.zeros((h, w), dtype=np.float32)

    for upper_ids, lower_ids in [
        (RIGHT_EYEBROW_UPPER, RIGHT_EYEBROW_LOWER),
        (LEFT_EYEBROW_UPPER, LEFT_EYEBROW_LOWER),
    ]:
        upper_pts = np.array([[fm.points[lid]["x"] * w, fm.points[lid]["y"] * h]
                              for lid in upper_ids], dtype=np.float64)
        lower_pts = np.array([[fm.points[lid]["x"] * w, fm.points[lid]["y"] * h]
                              for lid in lower_ids], dtype=np.float64)

        # ポリゴン（上辺→下辺逆順で閉じる）
        polygon = np.vstack([upper_pts, lower_pts[::-1]])
        pts_int = polygon.astype(np.int32)
        cv2.fillPoly(mask, [pts_int], 1.0)

    # dilate でポリゴンを均一に拡張（顔からはみ出さない程度）
    if expand_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (expand_px * 2 + 1, expand_px * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def build_skin_color_image(image: np.ndarray, brow_mask: np.ndarray,
                           sample_offset: int = 25) -> np.ndarray:
    """眉領域を周辺の肌色で塗りつぶした画像を生成

    各列ごとに眉の上(額)と下(まぶた)の肌色をサンプリングし、
    上下グラデーションで自然につなぐ。
    """
    h, w = image.shape[:2]
    brow_mask_bool = brow_mask > 0.5

    # 大きめブラーで肌テクスチャを均一化（毛穴・産毛ノイズ除去）
    skin_smooth = cv2.GaussianBlur(image, (41, 41), 12)
    result = image.copy()

    rows, cols = np.where(brow_mask_bool)
    if len(rows) == 0:
        return result

    for col in np.unique(cols):
        col_rows = rows[cols == col]
        top_row = col_rows.min()
        bottom_row = col_rows.max()

        # 眉の上下から肌色をサンプリング
        above_row = max(0, top_row - sample_offset)
        below_row = min(h - 1, bottom_row + sample_offset)
        color_above = skin_smooth[above_row, col].astype(np.float32)
        color_below = skin_smooth[below_row, col].astype(np.float32)

        for row in col_rows:
            t = (row - top_row) / max(bottom_row - top_row, 1)
            result[row, col] = (color_above * (1 - t) + color_below * t).astype(np.uint8)

    return result


def erase_eyebrows(
    image: np.ndarray,
    fm: FaceMesh,
    blur_scale: float = 1.0,
) -> np.ndarray:
    """眉を消す

    手順:
    1. ランドマークからタイトなポリゴンマスクを生成し少し拡張
    2. 周辺肌色で眉領域を直接塗りつぶし
    3. ソフトマスクで元画像と自然に合成
    """
    h, w = image.shape[:2]

    # 顔サイズ基準のパラメータ
    face_h = np.linalg.norm(
        fm.landmarks_px[10].astype(float) - fm.landmarks_px[152].astype(float)
    )
    expand_px = max(3, int(face_h * 0.015))  # 眉外周の拡張量

    # 1. タイトなポリゴンマスク + 少し拡張
    brow_mask = build_eyebrow_polygon_mask(fm, w, h, expand_px=expand_px)

    # 2. 周辺肌色で眉を塗りつぶし
    sample_offset = max(10, int(face_h * 0.04))
    skin_filled = build_skin_color_image(image, brow_mask, sample_offset=sample_offset)

    # 3. ソフトマスクでなめらかに合成
    blur_ksize = max(5, int(face_h * 0.035 * blur_scale))
    brow_mask_soft = gaussian_blur_mask(brow_mask, blur_ksize)

    alpha = brow_mask_soft[..., np.newaxis]
    result = image.astype(np.float32) * (1.0 - alpha) + skin_filled.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def crop_eyebrow_region(image: np.ndarray, fm: FaceMesh, margin_ratio: float = 0.4):
    """両眉周辺をクロップ"""
    h, w = image.shape[:2]
    # 眉関連のランドマーク
    brow_lms = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55,
                300, 293, 334, 296, 336, 276, 283, 282, 295, 285]
    xs = [fm.landmarks_px[l][0] for l in brow_lms]
    ys = [fm.landmarks_px[l][1] for l in brow_lms]
    margin = int((max(xs) - min(xs)) * margin_ratio)
    x1 = max(0, int(min(xs)) - margin)
    x2 = min(w, int(max(xs)) + margin)
    y1 = max(0, int(min(ys)) - margin)
    y2 = min(h, int(max(ys)) + margin)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def make_side_by_side(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.hstack([before, after])


def make_zoom_comparison(before: np.ndarray, after: np.ndarray, fm: FaceMesh) -> np.ndarray:
    """眉元ズームの比較画像"""
    crop_before, _ = crop_eyebrow_region(before, fm)
    crop_after, _ = crop_eyebrow_region(after, fm)
    target_w = before.shape[1]
    scale = target_w / (crop_before.shape[1] * 2)
    new_h = int(crop_before.shape[0] * scale)
    new_w = int(crop_before.shape[1] * scale)
    crop_before = cv2.resize(crop_before, (new_w, new_h))
    crop_after = cv2.resize(crop_after, (new_w, new_h))
    return np.hstack([crop_before, crop_after])


def main():
    parser = argparse.ArgumentParser(description="眉消し＆眉メイク")
    parser.add_argument("input", help="入力画像パス")
    parser.add_argument("-o", "--output", help="出力画像パス (default: eyebrow_result.png)")
    parser.add_argument("--blur", type=float, default=1.0,
                        help="マスクブラー倍率 (default: 1.0)")
    parser.add_argument("--imgonly", action="store_true",
                        help="結果画像のみ (比較画像なし)")
    parser.add_argument("--zoom", action="store_true",
                        help="眉元ズームの比較画像を出力")
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

    h, w = image.shape[:2]

    # 眉消し適用
    print("眉消し処理中...")
    output = erase_eyebrows(image, fm, blur_scale=args.blur)

    # 出力
    out_path = args.output or "eyebrow_result.png"
    if args.imgonly:
        cv2.imwrite(out_path, output)
    elif args.zoom:
        comparison = make_zoom_comparison(image, output, fm)
        cv2.imwrite(out_path, comparison)
    else:
        comparison = make_side_by_side(image, output)
        cv2.imwrite(out_path, comparison)

    print(f"出力: {out_path}")


if __name__ == "__main__":
    main()
