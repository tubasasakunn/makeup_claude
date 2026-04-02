"""
1.4 アイメイク - 目の各部位にアイメイクを反映する

5つのエリアに対応:
  - eyeshadow_base   : アイホール全体（ベースカラー / 通常合成）
  - eyeshadow_crease : 二重幅（メインカラー / 乗算合成）
  - eyeliner         : 目のキワ（ポリライン / 通常合成）
  - tear_bag         : 涙袋（ハイライト / 加算合成）
  - lower_outer      : 下まぶた目尻（ポイントカラー / 通常合成）

Usage:
    python main.py <input_image> [options]

Examples:
    # 全エリアをデフォルトで適用
    python main.py photo.jpg

    # 特定エリアのみ
    python main.py photo.jpg -a eyeshadow_base -a eyeshadow_crease

    # eyeliner のみ色と太さを変えて適用
    python main.py photo.jpg -a eyeliner --color 30 20 10 --intensity 0.6

    # 全エリアの強度を一括変更
    python main.py photo.jpg --intensity 0.3

    # エリア一覧を表示
    python main.py photo.jpg --list

    # 出力ファイル指定
    python main.py photo.jpg -o output.png
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
# エリアごとのデフォルト設定
# ==============================================================
AREA_DEFAULTS = {
    "eyeshadow_base": {
        "color_rgb": (190, 145, 120),   # ウォームブラウン
        "intensity": 0.35,
        "blur_scale": 0.8,
        "blend": "normal",
    },
    "eyeshadow_crease": {
        "color_rgb": (160, 110, 85),    # ライトブラウン（薄め）
        "intensity": 0.25,
        "blur_scale": 0.5,
        "blend": "normal",
    },
    "eyeliner": {
        "color_rgb": (35, 20, 10),      # ダークブラウン/ほぼ黒
        "intensity": 0.55,
        "blur_scale": 0.3,
        "blend": "normal",
        "thickness_scale": 0.8,         # 線を細くする
    },
    "tear_bag": {
        "color_rgb": (255, 230, 215),   # ウォームハイライト
        "intensity": 0.20,
        "blur_scale": 0.5,
        "blend": "additive",
    },
    "lower_outer": {
        "color_rgb": (140, 50, 40),     # 濃いバーガンディ
        "intensity": 0.60,
        "blur_scale": 0.15,
        "blend": "normal",
    },
}


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
    """通常合成: 元の色とカラーを線形ブレンド"""
    a = (mask * intensity)[..., np.newaxis]
    color = np.array(color_bgr, dtype=np.float32)
    src_f = src.astype(np.float32)
    result = src_f * (1.0 - a) + color * a
    return np.clip(result, 0, 255).astype(np.uint8)


def alpha_composite_multiply(
    src: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    intensity: float,
) -> np.ndarray:
    """乗算合成: 暗い色で影を入れる"""
    a = (mask * intensity)[..., np.newaxis]
    color = np.array(color_bgr, dtype=np.float32) / 255.0
    src_f = src.astype(np.float32)
    multiplied = src_f * color
    result = src_f * (1.0 - a) + multiplied * a
    return np.clip(result, 0, 255).astype(np.uint8)


def alpha_composite_additive(
    src: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    intensity: float,
) -> np.ndarray:
    """加算合成: 明るい色を上乗せ"""
    a = (mask * intensity)[..., np.newaxis]
    color = np.array(color_bgr, dtype=np.float32)
    result = src.astype(np.float32) + color * a
    return np.clip(result, 0, 255).astype(np.uint8)


BLEND_FUNCS = {
    "normal": alpha_composite_normal,
    "multiply": alpha_composite_multiply,
    "additive": alpha_composite_additive,
}


def build_eyeliner_mask(fm: FaceMesh, eyeliner_data: dict, w: int, h: int, thickness_scale: float = 1.0) -> np.ndarray:
    """ランドマークに沿ったポリラインでアイラインマスクを生成（外側にオフセット）"""
    mask = np.zeros((h, w), dtype=np.float32)
    face_w = abs(fm.landmarks_px[234][0] - fm.landmarks_px[454][0])
    thickness = max(2, int(face_w * 0.012 * eyeliner_data["thickness"] * thickness_scale))
    offset_px = thickness // 2 + 1

    for side in ["right", "left"]:
        upper_ids = eyeliner_data["upper_landmarks"][side]
        lower_ids = eyeliner_data["lower_landmarks"][side]
        all_ids = upper_ids + lower_ids
        eye_center_x = np.mean([fm.points[lid]["x"] * w for lid in all_ids if lid < 478])
        eye_center_y = np.mean([fm.points[lid]["y"] * h for lid in all_ids if lid < 478])

        for landmark_ids in [upper_ids, lower_ids]:
            pts = []
            for lid in landmark_ids:
                if lid < 478:
                    p = fm.points[lid]
                    px, py = p["x"] * w, p["y"] * h
                    dx = px - eye_center_x
                    dy = py - eye_center_y
                    dist = max(np.sqrt(dx * dx + dy * dy), 1e-6)
                    px += dx / dist * offset_px
                    py += dy / dist * offset_px
                    pts.append([int(px), int(py)])
            if len(pts) >= 2:
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.polylines(mask, [pts_arr], isClosed=False, color=1.0, thickness=thickness)
    return mask


def apply_eye_area(
    image: np.ndarray,
    fm: FaceMesh,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    intensity: float,
    blur_scale: float,
    blend: str,
) -> np.ndarray:
    """マスクに対してアイメイクを適用"""
    h, w = image.shape[:2]

    # ブラーで境界を柔らかくする（マスクをそのまま使い、エッジだけソフトに）
    face_h = np.linalg.norm(
        fm.landmarks_px[10].astype(float) - fm.landmarks_px[152].astype(float)
    )
    ksize = int(face_h * 0.03 * blur_scale)
    soft_mask = gaussian_blur_mask(mask, ksize)

    # 合成
    blend_func = BLEND_FUNCS[blend]
    return blend_func(image, soft_mask, color_bgr, intensity)


def make_side_by_side(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.hstack([before, after])


def crop_eye_region(image: np.ndarray, fm: FaceMesh, margin_ratio: float = 0.3) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """両目周辺をクロップ"""
    h, w = image.shape[:2]
    eye_lms = [33, 133, 263, 362, 55, 285, 46, 276, 7, 249, 110, 339]
    xs = [fm.landmarks_px[l][0] for l in eye_lms]
    ys = [fm.landmarks_px[l][1] for l in eye_lms]
    margin = int((max(xs) - min(xs)) * margin_ratio)
    x1 = max(0, int(min(xs)) - margin)
    x2 = min(w, int(max(xs)) + margin)
    y1 = max(0, int(min(ys)) - margin)
    y2 = min(h, int(max(ys)) + margin)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def make_zoom_comparison(before: np.ndarray, after: np.ndarray, fm: FaceMesh) -> np.ndarray:
    """目元ズームの比較画像を生成"""
    crop_before, _ = crop_eye_region(before, fm)
    crop_after, _ = crop_eye_region(after, fm)
    # 横幅を揃えてリサイズ
    target_w = before.shape[1]
    scale = target_w / (crop_before.shape[1] * 2)
    new_h = int(crop_before.shape[0] * scale)
    new_w = int(crop_before.shape[1] * scale)
    crop_before = cv2.resize(crop_before, (new_w, new_h))
    crop_after = cv2.resize(crop_after, (new_w, new_h))
    return np.hstack([crop_before, crop_after])


# ==============================================================
# CLI
# ==============================================================
def load_eye_areas() -> tuple[dict[str, list[int]], dict | None]:
    """target.json から eye エリアを読み込む。メッシュ系とポリライン系を分離して返す"""
    with open(TARGET_JSON) as f:
        data = json.load(f)

    mesh_areas = {}
    eyeliner_data = None

    for entry in data.get("eye", []):
        if entry.get("type") == "polyline":
            eyeliner_data = entry
        else:
            ids = entry["mesh_id"]
            if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                ids = ids[0]
            mesh_areas[entry["name"]] = ids

    return mesh_areas, eyeliner_data


def main():
    parser = argparse.ArgumentParser(description="アイメイク適用")
    parser.add_argument("input", help="入力画像パス")
    parser.add_argument("-o", "--output", help="出力画像パス (default: eye_result.png)")
    parser.add_argument("-a", "--area", action="append", help="適用エリア名 (複数指定可)")
    parser.add_argument("--color", nargs=3, type=int, metavar=("R", "G", "B"),
                        help="色を全エリアに一括適用 (省略時は各エリアのデフォルト)")
    parser.add_argument("--intensity", type=float,
                        help="強度を全エリアに一括適用 (省略時は各エリアのデフォルト)")
    parser.add_argument("--blur", type=float, help="ブラー倍率を一括変更")
    parser.add_argument("--list", action="store_true", help="利用可能エリア一覧を表示")
    parser.add_argument("--imgonly", action="store_true", help="結果画像のみ (比較画像なし)")
    parser.add_argument("--zoom", action="store_true", help="目元ズームの比較画像を出力")
    args = parser.parse_args()

    # エリア読み込み
    mesh_areas, eyeliner_data = load_eye_areas()

    all_area_names = list(mesh_areas.keys())
    if eyeliner_data:
        all_area_names.append("eyeliner")

    if args.list:
        print("利用可能なアイメイクエリア:")
        for name, ids in mesh_areas.items():
            defaults = AREA_DEFAULTS.get(name, {})
            print(f"  {name:20s} ({len(ids):3d} meshes, "
                  f"blend={defaults.get('blend', 'normal')}, "
                  f"intensity={defaults.get('intensity', 0.15)})")
        if eyeliner_data:
            defaults = AREA_DEFAULTS.get("eyeliner", {})
            print(f"  {'eyeliner':20s} (polyline, "
                  f"blend={defaults.get('blend', 'normal')}, "
                  f"intensity={defaults.get('intensity', 0.5)})")
        return

    # 適用エリア決定
    if args.area:
        area_names = args.area
    else:
        area_names = all_area_names

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

    # アイメイク適用
    output = image.copy()

    for name in area_names:
        defaults = AREA_DEFAULTS.get(name, {
            "color_rgb": (180, 130, 100),
            "intensity": 0.15,
            "blur_scale": 1.5,
            "blend": "normal",
        })

        # CLI 引数でオーバーライド
        if args.color:
            color_rgb = tuple(args.color)
        else:
            color_rgb = defaults["color_rgb"]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        intensity = args.intensity if args.intensity is not None else defaults["intensity"]
        blur_scale = args.blur if args.blur is not None else defaults["blur_scale"]
        blend = defaults["blend"]

        if name == "eyeliner":
            if eyeliner_data is None:
                print(f"  Warning: eyeliner データが target.json にありません, スキップ")
                continue
            t_scale = defaults.get("thickness_scale", 1.0)
            mask = build_eyeliner_mask(fm, eyeliner_data, w, h, thickness_scale=t_scale)
            print(f"  適用: {name} (polyline, intensity={intensity:.2f}, blend={blend}, thickness_scale={t_scale})")
        elif name in mesh_areas:
            mesh_ids = mesh_areas[name]
            mask = fm.build_mask(mesh_ids, w, h)
            print(f"  適用: {name} ({len(mesh_ids)} meshes, intensity={intensity:.2f}, blend={blend})")
        else:
            print(f"  Warning: エリア '{name}' が見つかりません, スキップ")
            continue

        output = apply_eye_area(
            output, fm, mask,
            color_bgr=color_bgr,
            intensity=intensity,
            blur_scale=blur_scale,
            blend=blend,
        )

    # 出力
    out_path = args.output or "eye_result.png"
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
