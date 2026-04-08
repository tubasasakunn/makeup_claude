"""
1.5 眉メイク - 眉を消して新しい眉を描画する

Phase 1: 眉消し
  - ランドマークからスムーズなポリゴンマスクを生成（メッシュ不使用）
  - 眉領域を局所肌色で事前置換（暗い眉色の伝播を防ぐ）
  - cv2.inpaint TELEA でテクスチャ補完
  - cv2.seamlessClone で周辺の色味に自動マッチ
  - フェザーマスクで境界をなめらかに合成

Phase 2: 眉描画（今後実装）
  - 消した上に新しい眉を描画

Usage:
    python main.py <input_image> [options]

Examples:
    # 眉を消す（デフォルト）
    python main.py photo.jpg

    # 出力ファイル指定
    python main.py photo.jpg -o output.png

    # 眉元ズーム表示
    python main.py photo.jpg --zoom
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# =========================================================
# MediaPipe 478点 眉ランドマークID
# =========================================================
RIGHT_EYEBROW_UPPER = [70, 63, 105, 66, 107]
RIGHT_EYEBROW_LOWER = [46, 53, 52, 65, 55]
LEFT_EYEBROW_UPPER = [300, 293, 334, 296, 336]
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

    cv2.fillPoly で塗りつぶすのでモザイクにならない。
    expand_px で外周をピクセル単位で拡張（毛先カバー）。
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
        polygon = np.vstack([upper_pts, lower_pts[::-1]])
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1.0)

    if expand_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def build_skin_prefill(image: np.ndarray, brow_mask: np.ndarray) -> np.ndarray:
    """眉領域を局所肌色で事前置換した画像を生成

    cv2.inpaint は周囲のピクセル色を内側に伝播させるアルゴリズムなので、
    眉の暗い色を残したまま inpaint すると暗い色も伝播してしまう。
    あらかじめ眉領域を周辺の肌色（中央値）で塗りつぶすことで、
    後段のTELEAが綺麗な肌色テクスチャを伝播できるようになる。
    """
    h, w = image.shape[:2]
    brow_mask_bool = brow_mask > 0.5
    result = image.copy()

    rows, cols = np.where(brow_mask_bool)
    if len(rows) == 0:
        return result

    for col in np.unique(cols):
        col_rows = rows[cols == col]
        top = int(col_rows.min())
        bot = int(col_rows.max())
        bh = bot - top + 1

        # 上(額)サンプル
        above_h = max(5, bh)
        above_top = max(0, top - above_h)
        above = image[above_top:top, col].astype(np.float32)

        # 下(まぶた)サンプル
        below_h = max(5, bh // 2)
        below_bot = min(h, bot + 1 + below_h)
        below = image[bot + 1:below_bot, col].astype(np.float32)

        if len(above) == 0 and len(below) == 0:
            continue

        if len(above) > 0 and len(below) > 0:
            samples = np.vstack([above, below])
        elif len(above) > 0:
            samples = above
        else:
            samples = below

        # 中央値（暗い眉の影響を排除）
        skin_color = np.median(samples, axis=0)
        result[col_rows, col] = skin_color.astype(np.uint8)

    return result


def _seamless_clone_components(src: np.ndarray, dst: np.ndarray,
                               mask_u8: np.ndarray) -> np.ndarray:
    """連結成分（左右の眉）ごとに seamlessClone を適用"""
    num_labels, labels = cv2.connectedComponents(mask_u8)
    cloned = dst.copy()
    for label_id in range(1, num_labels):
        comp = (labels == label_id).astype(np.uint8) * 255
        ys, xs = np.where(comp > 0)
        if len(ys) == 0:
            continue
        cy = int((ys.min() + ys.max()) / 2)
        cx = int((xs.min() + xs.max()) / 2)
        try:
            cloned = cv2.seamlessClone(src, cloned, comp, (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error:
            alpha = (comp.astype(np.float32) / 255.0)[..., np.newaxis]
            cloned = (cloned.astype(np.float32) * (1 - alpha) +
                      src.astype(np.float32) * alpha).astype(np.uint8)
    return cloned


def erase_eyebrows(
    image: np.ndarray,
    fm: FaceMesh,
    blur_scale: float = 1.0,
) -> np.ndarray:
    """眉を消す（pre-fill + cv2.inpaint TELEA + seamlessClone 方式）

    手順:
    1. ランドマークからタイトなポリゴンマスク生成 + 拡張
    2. 眉領域を局所肌色で事前置換（暗い眉色の伝播を防ぐ）
    3. cv2.inpaint TELEA でテクスチャ補完
    4. cv2.seamlessClone で周辺の色味・ライティングに自動マッチ
    5. フェザーマスクで境界をなめらかに合成
    """
    h, w = image.shape[:2]

    # 顔サイズ基準のパラメータ
    face_h = np.linalg.norm(
        fm.landmarks_px[10].astype(float) - fm.landmarks_px[152].astype(float)
    )
    expand_px = max(3, int(face_h * 0.015))

    # 1. ポリゴンマスク生成
    brow_mask = build_eyebrow_polygon_mask(fm, w, h, expand_px=expand_px)
    mask_u8 = (brow_mask * 255).astype(np.uint8)

    # 2. 局所肌色で事前置換（暗い眉色の伝播を防ぐ）
    prefilled = build_skin_prefill(image, brow_mask)

    # 3. TELEA で自然なテクスチャ補完
    inpaint_radius = max(8, int(face_h * 0.025))
    inpainted = cv2.inpaint(prefilled, mask_u8, inpaint_radius, cv2.INPAINT_TELEA)

    # 4. seamlessClone で周辺の色味にマッチ
    erode_k = max(2, expand_px // 2)
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_k * 2 + 1, erode_k * 2 + 1)
    )
    clone_mask = cv2.erode(mask_u8, erode_kernel, iterations=1)
    cloned = _seamless_clone_components(inpainted, image, clone_mask)

    # 5. 中央は完全置換、外周のみフェザー
    blur_ksize = max(11, int(face_h * 0.05 * blur_scale))
    feather_dilate = blur_ksize // 2 + 4
    feather_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (feather_dilate * 2 + 1, feather_dilate * 2 + 1)
    )
    expanded_mask = cv2.dilate(brow_mask, feather_kernel, iterations=1)
    soft_mask = gaussian_blur_mask(expanded_mask, blur_ksize)
    soft_mask = gaussian_blur_mask(soft_mask, blur_ksize)

    alpha = soft_mask[..., np.newaxis]
    result = image.astype(np.float32) * (1.0 - alpha) + cloned.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def crop_eyebrow_region(image: np.ndarray, fm: FaceMesh, margin_ratio: float = 0.4):
    """両眉周辺をクロップ"""
    h, w = image.shape[:2]
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

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: 画像を読み込めません: {args.input}")
        return

    print("FaceMesh 初期化中...")
    fm = FaceMesh(subdivision_level=1)
    fm.init()

    print("顔検出中...")
    if fm.detect(image) is None:
        print("Error: 顔が検出されませんでした")
        return

    print("眉消し処理中...")
    output = erase_eyebrows(image, fm, blur_scale=args.blur)

    out_path = args.output or "eyebrow_result.png"
    if args.imgonly:
        cv2.imwrite(out_path, output)
    elif args.zoom:
        cv2.imwrite(out_path, make_zoom_comparison(image, output, fm))
    else:
        cv2.imwrite(out_path, make_side_by_side(image, output))

    print(f"出力: {out_path}")


if __name__ == "__main__":
    main()
