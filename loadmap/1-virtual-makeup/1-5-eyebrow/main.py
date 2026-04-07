"""
1.5 眉メイク - 眉を消して新しい眉を描画する

Phase 1: 眉消し
  - 眉領域のメッシュマスクを作成
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


def gaussian_blur_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(mask, (ksize, ksize), ksize / 3.0)


def load_eyebrow_areas() -> dict[str, list[int]]:
    """target.json から eyebrow エリアを読み込む"""
    with open(TARGET_JSON) as f:
        data = json.load(f)

    areas = {}
    for entry in data.get("eyebrow", []):
        ids = entry["mesh_id"]
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        areas[entry["name"]] = ids
    return areas


def sample_skin_color(image: np.ndarray, fm: FaceMesh, skin_mesh_ids: list[int]) -> np.ndarray:
    """肌色サンプリング領域から平均色を取得（左右別に）"""
    h, w = image.shape[:2]
    mask = fm.build_mask(skin_mesh_ids, w, h)
    mask_bool = mask > 0.5

    if mask_bool.sum() == 0:
        return np.array([200, 180, 170], dtype=np.float32)  # fallback

    # マスク領域のピクセルの平均色
    pixels = image[mask_bool]
    return pixels.mean(axis=0).astype(np.float32)


def sample_skin_color_map(image: np.ndarray, fm: FaceMesh,
                          skin_mesh_ids: list[int],
                          eyebrow_mesh_ids: list[int]) -> np.ndarray:
    """眉領域の各ピクセルに対応する肌色マップを生成（上方向からサンプリング）"""
    h, w = image.shape[:2]

    # 眉マスク
    brow_mask = fm.build_mask(eyebrow_mesh_ids, w, h)
    brow_mask_bool = brow_mask > 0.5

    # 肌色サンプリング用のスムーズ画像を作成
    # 大きめブラーで肌のテクスチャを均一化
    skin_smooth = cv2.GaussianBlur(image, (31, 31), 10)

    # 眉領域のピクセルごとに、真上の肌色を取得
    color_map = skin_smooth.copy()

    # 眉マスクの各列で上端を探して、その上方の肌色をコピー
    rows, cols = np.where(brow_mask_bool)
    if len(rows) == 0:
        return color_map

    for col in np.unique(cols):
        col_rows = rows[cols == col]
        top_row = col_rows.min()

        # 眉の上の肌色を探す（上に10-30px程度）
        sample_row = max(0, top_row - 15)
        skin_color = skin_smooth[sample_row, col]

        # この列の眉領域に肌色をセット
        for row in col_rows:
            # 上下のグラデーション: 上端は額の色、下端は元の色寄り
            t = (row - top_row) / max(col_rows.max() - top_row, 1)
            bottom_color = skin_smooth[min(h - 1, col_rows.max() + 10), col]
            color_map[row, col] = skin_color * (1 - t) + bottom_color * t

    return color_map


def erase_eyebrows(
    image: np.ndarray,
    fm: FaceMesh,
    eyebrow_mesh_ids: list[int],
    skin_mesh_ids: list[int],
    inpaint_radius: int = 4,
    blend_strength: float = 0.3,
    blur_scale: float = 1.0,
) -> np.ndarray:
    """眉を消す

    手順:
    1. 眉メッシュからマスク生成
    2. cv2.inpaint でテクスチャ付きの肌色で埋める
    3. 周辺肌色とブレンドして自然に仕上げる
    """
    h, w = image.shape[:2]

    # 1. 眉マスク生成
    brow_mask = fm.build_mask(eyebrow_mesh_ids, w, h)

    # 顔の高さに基づいたブラーカーネルサイズ
    face_h = np.linalg.norm(
        fm.landmarks_px[10].astype(float) - fm.landmarks_px[152].astype(float)
    )
    ksize = int(face_h * 0.025 * blur_scale)

    # 拡張して毛の端までカバー
    dilate_k = max(3, int(face_h * 0.012))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    brow_mask_dilated = cv2.dilate(brow_mask, kernel, iterations=2)

    # ブラーでエッジをソフトに
    brow_mask_soft = gaussian_blur_mask(brow_mask_dilated, ksize)

    # 2. inpainting（複数回で段階的に消す）
    inpaint_mask = (brow_mask_dilated * 255).astype(np.uint8)
    inpainted = cv2.inpaint(image, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
    # 2回目: 残存部分をさらに消す
    inpainted = cv2.inpaint(inpainted, inpaint_mask, inpaint_radius, cv2.INPAINT_NS)

    # 3. 肌色マップを生成して追加ブレンド（inpaintの不自然さを軽減）
    skin_color_map = sample_skin_color_map(image, fm, skin_mesh_ids, eyebrow_mesh_ids)

    # inpaint結果と肌色マップを混合
    blended = cv2.addWeighted(
        inpainted.astype(np.float32), 1.0 - blend_strength,
        skin_color_map.astype(np.float32), blend_strength,
        0,
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # 4. ソフトマスクで元画像と合成
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
    parser.add_argument("--imgonly", action="store_true",
                        help="結果画像のみ (比較画像なし)")
    parser.add_argument("--zoom", action="store_true",
                        help="眉元ズームの比較画像を出力")
    args = parser.parse_args()

    # エリア読み込み
    areas = load_eyebrow_areas()
    if "eyebrow_full" not in areas:
        print("Error: target.json に eyebrow_full がありません。先に find_eyebrow_meshes.py を実行してください")
        return

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
    print(f"メッシュ: {len(fm.triangles)} 三角形, {len(fm.points)} 頂点")

    # 眉消し適用
    print("眉消し処理中...")
    output = erase_eyebrows(
        image, fm,
        eyebrow_mesh_ids=areas["eyebrow_full"],
        skin_mesh_ids=areas.get("eyebrow_skin", []),
        inpaint_radius=args.radius,
        blend_strength=args.blend_strength,
        blur_scale=args.blur,
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
