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


def _select_clean_patch(image: np.ndarray, x_min: int, x_max: int,
                        y_above_min: int, y_above_max: int,
                        patch_h: int, patch_w: int) -> np.ndarray | None:
    """指定範囲から複数候補パッチをサンプリングし、暗いもの（眉/影）を除外して代表を返す

    手順:
    1. 範囲内で複数の候補パッチを取得
    2. 明度でソートし、暗い下位50%を除外（眉のかけら・影を除外）
    3. 残った中で中央値明度のパッチを返す（明るすぎず暗すぎない代表）
    """
    h, w = image.shape[:2]
    candidates = []

    y_step = max(2, patch_h // 3)
    for y_top in range(y_above_min, max(y_above_min + 1, y_above_max - patch_h + 1), y_step):
        y_bot = y_top + patch_h
        if y_bot > h:
            continue
        x_step = max(3, patch_w // 3)
        for x_left in range(x_min, max(x_min + 1, x_max - patch_w + 1), x_step):
            x_right = x_left + patch_w
            if x_right > w:
                continue
            patch = image[y_top:y_bot, x_left:x_right]
            if patch.shape[0] != patch_h or patch.shape[1] != patch_w:
                continue
            brightness = float(patch.mean())
            candidates.append((brightness, patch))

    if not candidates:
        return None

    # 暗い順にソート、下位50%を除外
    candidates.sort(key=lambda c: c[0])
    clean_candidates = candidates[len(candidates) // 2:]
    # 残った中の中央値明度のパッチを返す
    return clean_candidates[len(clean_candidates) // 2][1]


def _color_match_patch(patch: np.ndarray, target_color: np.ndarray) -> np.ndarray:
    """パッチを色補正してターゲット色に平均値を合わせる

    パッチの平均色とターゲット色の差分を全ピクセルに加算する。
    テクスチャ（高周波成分）は保持しつつ、色味だけ局所平均に揃える。
    """
    patch_f = patch.astype(np.float32)
    patch_mean = patch_f.reshape(-1, 3).mean(axis=0)
    diff = target_color.astype(np.float32) - patch_mean
    return np.clip(patch_f + diff, 0, 255).astype(np.uint8)


def build_skin_texture_image(image: np.ndarray, brow_mask: np.ndarray,
                             forehead_top_y: int = 0) -> np.ndarray:
    """眉の上の額から「明るい綺麗な肌パッチ」を抽出し、反転タイリングで眉領域を埋める

    候補パッチを多数サンプリングし、明度上位群から代表パッチを選ぶことで
    影や眉のかけらが混入したパッチを避ける。
    forehead_top_y: 額の上限Y座標（髪の毛にサンプル範囲が入らないようにする）
    """
    h, w = image.shape[:2]
    brow_mask_bool = brow_mask > 0.5
    result = image.copy()

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

        # パッチサイズ: 小さめにして額の上の方からも取れるように
        patch_h = max(4, bh // 2)
        patch_w = max(8, bw // 3)

        # サンプリング範囲: 眉骨の暗い影を避けるため、額の上の方まで広く探す
        # 眉の bh*4 上 ～ bh*1 上 まで（ただし額の上限は超えない）
        margin_min = max(5, bh)
        margin_max = max(margin_min + patch_h * 2, bh * 4)
        y_above_max = y_min - margin_min
        y_above_min = max(forehead_top_y, y_min - margin_max)

        # x範囲も少し余裕を持たせる
        x_search_min = max(0, x_min - bw // 4)
        x_search_max = min(w, x_max + bw // 4 + 1)

        patch = _select_clean_patch(
            image, x_search_min, x_search_max,
            y_above_min, y_above_max,
            patch_h, patch_w,
        )
        if patch is None:
            continue

        # 局所的な肌色平均にパッチを色補正
        # 眉の上下から目標色をサンプリング（中央値）
        target_above = image[max(0, y_min - bh):y_min,
                             max(0, x_min + bw // 4):min(w, x_max - bw // 4 + 1)]
        target_below = image[y_max + 1:min(h, y_max + 1 + bh // 2),
                             max(0, x_min + bw // 4):min(w, x_max - bw // 4 + 1)]
        target_colors = []
        if target_above.size > 0:
            target_colors.append(np.median(target_above.reshape(-1, 3), axis=0))
        if target_below.size > 0:
            target_colors.append(np.median(target_below.reshape(-1, 3), axis=0))
        if target_colors:
            target_color = np.mean(target_colors, axis=0)
            patch = _color_match_patch(patch, target_color)

        # 反転タイリングで眉領域と同じサイズに拡張
        tiled = _mirror_tile(patch, bh, bw)

        # 連結成分の形で貼り付け
        comp_local = comp_bool[y_min:y_max + 1, x_min:x_max + 1]
        result_region = result[y_min:y_max + 1, x_min:x_max + 1]
        result_region[comp_local] = tiled[comp_local]
        result[y_min:y_max + 1, x_min:x_max + 1] = result_region

    return result


def erase_eyebrows(
    image: np.ndarray,
    fm: FaceMesh,
    blur_scale: float = 1.0,
) -> np.ndarray:
    """眉を消す

    手順:
    1. ランドマークからタイトなポリゴンマスクを生成し少し拡張
    2. 眉の上から複数候補パッチをサンプリング → 明るい代表パッチを選択
    3. そのパッチを反転タイリングで眉領域に貼り付け
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

    # 額の上限（髪の毛にかからないように、ランドマーク10とまゆの中間あたり）
    forehead_top_lm = fm.landmarks_px[10]  # 額の頂上
    brow_top_lm = fm.landmarks_px[9]        # 眉の中間（上）
    # 額の頂上から眉の上端の間の中央あたりを上限にする（髪に近すぎない範囲）
    forehead_top_y = int(forehead_top_lm[1] + (brow_top_lm[1] - forehead_top_lm[1]) * 0.3)

    # 2-3. 明るいパッチを選んで反転タイリングで眉領域を埋める
    skin_filled = build_skin_texture_image(image, brow_mask, forehead_top_y=forehead_top_y)

    # 4. 中央は完全置換、外周のみフェザー
    # 先に大きく膨張させてからブラーすることで、中央はα=1.0を保ち外周だけソフトに
    blur_ksize = max(7, int(face_h * 0.04 * blur_scale))
    feather_dilate = blur_ksize // 2 + 3
    feather_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (feather_dilate * 2 + 1, feather_dilate * 2 + 1)
    )
    expanded_mask = cv2.dilate(brow_mask, feather_kernel, iterations=1)
    soft_mask = gaussian_blur_mask(expanded_mask, blur_ksize)

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
