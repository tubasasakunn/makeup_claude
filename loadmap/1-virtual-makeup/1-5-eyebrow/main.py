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


def _mirror_tile(patch: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """パッチを反転タイリングで指定サイズに拡張

    上下反転・左右反転を交互に組み合わせることで、繰り返しの不自然さを軽減する。
    """
    ph, pw = patch.shape[:2]
    if ph == 0 or pw == 0:
        return np.zeros((target_h, target_w, patch.shape[2]), dtype=patch.dtype)

    # 縦方向: 上下反転を交互に
    rows = []
    cur_h = 0
    flip_v = False
    while cur_h < target_h:
        row_patch = patch[::-1] if flip_v else patch
        rows.append(row_patch)
        cur_h += ph
        flip_v = not flip_v
    vstacked = np.vstack(rows)[:target_h]

    # 横方向: 左右反転を交互に
    cols = []
    cur_w = 0
    flip_h = False
    while cur_w < target_w:
        col_patch = vstacked[:, ::-1] if flip_h else vstacked
        cols.append(col_patch)
        cur_w += pw
        flip_h = not flip_h
    return np.hstack(cols)[:, :target_w]


def sample_skin_color(image: np.ndarray, brow_mask: np.ndarray) -> np.ndarray:
    """眉の近くから単一の肌色をサンプリング（左右の眉ごとに別の色）

    眉の上(額)と下(まぶた)の両方からサンプリングして平均を取る。
    眉の真ん中あたりの色味になるので、置き換えても周辺となじむ。
    """
    h, w = image.shape[:2]
    brow_mask_bool = brow_mask > 0.5
    color_map = np.zeros_like(image)

    mask_u8 = brow_mask_bool.astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(mask_u8)

    for label_id in range(1, num_labels):
        comp_bool = labels == label_id
        ys, xs = np.where(comp_bool)
        if len(ys) == 0:
            continue

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        bh = y_max - y_min + 1
        bw = x_max - x_min + 1

        # 横方向: 眉の中央寄りだけサンプリング（端は影が入りやすい）
        sample_left = max(0, x_min + bw // 4)
        sample_right = min(w, x_max - bw // 4 + 1)
        if sample_right <= sample_left:
            continue

        # 上(額)サンプリング: 眉のすぐ上にマージンを設ける
        margin_above = max(5, bh)
        above_bottom = max(0, y_min - margin_above)
        above_top = max(0, above_bottom - max(5, bh // 2))
        above_area = image[above_top:above_bottom, sample_left:sample_right]

        # 下(まぶた)サンプリング: 眉のすぐ下にマージンを設ける
        margin_below = max(5, bh // 2)
        below_top = min(h, y_max + margin_below)
        below_bottom = min(h, below_top + max(5, bh // 3))
        below_area = image[below_top:below_bottom, sample_left:sample_right]

        # 上下の中央値を取得して平均
        colors = []
        if above_area.size > 0:
            colors.append(np.median(above_area.reshape(-1, 3), axis=0))
        if below_area.size > 0:
            colors.append(np.median(below_area.reshape(-1, 3), axis=0))
        if not colors:
            continue

        skin_color = np.mean(colors, axis=0).astype(np.uint8)
        color_map[comp_bool] = skin_color

    return color_map


def erase_eyebrows(
    image: np.ndarray,
    fm: FaceMesh,
    blur_scale: float = 1.0,
) -> np.ndarray:
    """眉を消す（シンプル単一肌色塗りつぶし方式）

    手順:
    1. ランドマークからタイトなポリゴンマスクを生成し少し拡張
    2. 眉のすぐ上から単一の肌色をサンプリング（中央値）
    3. その色で眉領域を塗りつぶし
    4. ソフトマスクで元画像と合成
    """
    h, w = image.shape[:2]

    # 顔サイズ基準のパラメータ
    face_h = np.linalg.norm(
        fm.landmarks_px[10].astype(float) - fm.landmarks_px[152].astype(float)
    )
    expand_px = max(3, int(face_h * 0.015))

    # 1. タイトなポリゴンマスク + 少し拡張
    brow_mask = build_eyebrow_polygon_mask(fm, w, h, expand_px=expand_px)

    # 2-3. 単一肌色で眉領域を塗りつぶし
    skin_filled = image.copy()
    color_map = sample_skin_color(image, brow_mask)
    brow_mask_bool = brow_mask > 0.5
    skin_filled[brow_mask_bool] = color_map[brow_mask_bool]

    # 4. ソフトマスクで元画像と合成（境界をなめらかに）
    # 強めのブラーで境界がフラットに見えないようにする
    blur_ksize = max(7, int(face_h * 0.05 * blur_scale))
    soft_mask = gaussian_blur_mask(brow_mask, blur_ksize)
    alpha = soft_mask[..., np.newaxis]
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
