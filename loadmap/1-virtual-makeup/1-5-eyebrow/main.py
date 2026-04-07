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


def _extend_eyebrow_tail(upper_pts: np.ndarray, lower_pts: np.ndarray,
                         extension: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """眉尻方向にポイントを延長して、ランドマーク外の毛もカバーする

    上辺・下辺それぞれの外側2点から方向ベクトルを算出し、
    眉幅の extension 倍だけ延長した点を追加する。
    """
    brow_width = np.linalg.norm(upper_pts[0] - upper_pts[-1])
    ext_len = brow_width * extension

    # 上辺の外端を延長（最初の2点から方向を推定）
    dir_upper = upper_pts[0] - upper_pts[1]
    dir_upper = dir_upper / (np.linalg.norm(dir_upper) + 1e-6)
    ext_upper = upper_pts[0] + dir_upper * ext_len

    # 下辺の外端を延長
    dir_lower = lower_pts[0] - lower_pts[1]
    dir_lower = dir_lower / (np.linalg.norm(dir_lower) + 1e-6)
    ext_lower = lower_pts[0] + dir_lower * ext_len

    upper_ext = np.vstack([[ext_upper], upper_pts])
    lower_ext = np.vstack([[ext_lower], lower_pts])
    return upper_ext, lower_ext


def build_eyebrow_polygon_mask(fm: FaceMesh, w: int, h: int,
                               expand_ratio: float = 1.4) -> np.ndarray:
    """ランドマークから直接スムーズなポリゴンマスクを生成（メッシュ不使用）

    メッシュ三角形ではなくランドマーク座標からポリゴンを作り、
    cv2.fillPoly で塗りつぶすのでモザイク状にならない。
    眉尻方向にポリゴンを延長して、はみ出す毛もカバーする。
    """
    mask = np.zeros((h, w), dtype=np.float32)

    for upper_ids, lower_ids in [
        (RIGHT_EYEBROW_UPPER, RIGHT_EYEBROW_LOWER),
        (LEFT_EYEBROW_UPPER, LEFT_EYEBROW_LOWER),
    ]:
        # ランドマークからピクセル座標を取得
        upper_pts = np.array([[fm.points[lid]["x"] * w, fm.points[lid]["y"] * h]
                              for lid in upper_ids], dtype=np.float64)
        lower_pts = np.array([[fm.points[lid]["x"] * w, fm.points[lid]["y"] * h]
                              for lid in lower_ids], dtype=np.float64)

        # 眉尻方向に延長
        upper_ext, lower_ext = _extend_eyebrow_tail(upper_pts, lower_pts, extension=0.4)

        # ポリゴン（上辺→下辺逆順で閉じる）
        polygon = np.vstack([upper_ext, lower_ext[::-1]])
        centroid = polygon.mean(axis=0)

        # 重心から均一に拡大（上下左右バランス良く）
        polygon_expanded = centroid + (polygon - centroid) * expand_ratio

        pts_int = polygon_expanded.astype(np.int32)
        cv2.fillPoly(mask, [pts_int], 1.0)

    return mask


def sample_skin_color_map(image: np.ndarray, fm: FaceMesh,
                          brow_mask: np.ndarray) -> np.ndarray:
    """眉領域の各ピクセルに対応する肌色マップを生成（上方向からサンプリング）"""
    h, w = image.shape[:2]
    brow_mask_bool = brow_mask > 0.5

    # 肌色サンプリング用のスムーズ画像
    skin_smooth = cv2.GaussianBlur(image, (31, 31), 10)
    color_map = skin_smooth.copy()

    rows, cols = np.where(brow_mask_bool)
    if len(rows) == 0:
        return color_map

    # 各列ごとに上端・下端を探し、上方の肌色からグラデーション
    for col in np.unique(cols):
        col_rows = rows[cols == col]
        top_row = col_rows.min()
        bottom_row = col_rows.max()

        # 眉の上(額)と下(まぶた付近)から肌色をサンプリング
        sample_above = max(0, top_row - 20)
        sample_below = min(h - 1, bottom_row + 15)
        color_above = skin_smooth[sample_above, col].astype(np.float32)
        color_below = skin_smooth[sample_below, col].astype(np.float32)

        # この列の眉領域を上下グラデーションで埋める
        for row in col_rows:
            t = (row - top_row) / max(bottom_row - top_row, 1)
            color_map[row, col] = (color_above * (1 - t) + color_below * t).astype(np.uint8)

    return color_map


def erase_eyebrows(
    image: np.ndarray,
    fm: FaceMesh,
    inpaint_radius: int = 5,
    blend_strength: float = 0.4,
    blur_scale: float = 1.0,
    expand_ratio: float = 1.4,
) -> np.ndarray:
    """眉を消す（ランドマークベースのスムーズマスク方式）

    手順:
    1. ランドマークからスムーズなポリゴンマスクを生成
    2. cv2.inpaint でテクスチャ付きの肌色で埋める
    3. 周辺肌色とブレンドして自然に仕上げる
    """
    h, w = image.shape[:2]

    # 1. スムーズなポリゴンマスク生成（メッシュ不使用）
    brow_mask = build_eyebrow_polygon_mask(fm, w, h, expand_ratio=expand_ratio)

    # 顔の高さに基づいたパラメータ
    face_h = np.linalg.norm(
        fm.landmarks_px[10].astype(float) - fm.landmarks_px[152].astype(float)
    )

    # 少し膨張させて毛先の端までカバー
    dilate_k = max(3, int(face_h * 0.01))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    brow_mask_dilated = cv2.dilate(brow_mask, kernel, iterations=1)

    # ソフトマスク（合成用）: 強めのブラーでなめらかに
    soft_ksize = int(face_h * 0.04 * blur_scale)
    brow_mask_soft = gaussian_blur_mask(brow_mask_dilated, soft_ksize)

    # 2. inpainting（2パスで段階的に消す）
    inpaint_mask = (brow_mask_dilated * 255).astype(np.uint8)
    inpainted = cv2.inpaint(image, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
    inpainted = cv2.inpaint(inpainted, inpaint_mask, inpaint_radius, cv2.INPAINT_NS)

    # 3. 肌色マップとブレンド（inpaintの不自然さを軽減）
    skin_color_map = sample_skin_color_map(image, fm, brow_mask_dilated)

    blended = cv2.addWeighted(
        inpainted.astype(np.float32), 1.0 - blend_strength,
        skin_color_map.astype(np.float32), blend_strength,
        0,
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # 4. ソフトマスクで元画像と合成（なめらかな境界）
    alpha = brow_mask_soft[..., np.newaxis]
    result = image.astype(np.float32) * (1.0 - alpha) + blended.astype(np.float32) * alpha
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
    parser.add_argument("--radius", type=int, default=5,
                        help="inpaint 半径 (default: 5)")
    parser.add_argument("--blend-strength", type=float, default=0.4,
                        help="肌色ブレンド強度 (default: 0.4)")
    parser.add_argument("--blur", type=float, default=1.0,
                        help="マスクブラー倍率 (default: 1.0)")
    parser.add_argument("--expand", type=float, default=1.4,
                        help="眉ポリゴン拡大率 (default: 1.4)")
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
    output = erase_eyebrows(
        image, fm,
        inpaint_radius=args.radius,
        blend_strength=args.blend_strength,
        blur_scale=args.blur,
        expand_ratio=args.expand,
    )

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
