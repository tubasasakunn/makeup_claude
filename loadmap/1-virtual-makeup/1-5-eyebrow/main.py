"""
1.5 眉メイク - 眉を消して新しい眉を描画する

Phase 1: 眉消し
  - ランドマークからスムーズなポリゴンマスクを生成
  - cv2.inpaint TELEA（大きめradius）で自然に補完

Phase 2: 眉描画
  - 3大ポイント（眉頭・眉山・眉尻）を顔ランドマークから計算
  - 眉タイプ（7種類）に応じたBezier曲線でポリゴン生成
  - 色/強度を調整してソフトに描画

Usage:
    python main.py <input_image> [options]

Examples:
    # 眉を消してナチュラルアーチを描画（デフォルト）
    python main.py photo.jpg

    # 眉タイプを指定
    python main.py photo.jpg --type angular

    # 色を変更 (R G B)
    python main.py photo.jpg --color 60 30 20

    # 眉消しのみ（描画しない）
    python main.py photo.jpg --no-draw

    # 眉元ズーム表示
    python main.py photo.jpg --zoom

    # タイプ一覧
    python main.py photo.jpg --list-types
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


def erase_eyebrows(
    image: np.ndarray,
    fm: FaceMesh,
    blur_scale: float = 1.0,
) -> np.ndarray:
    """眉を消す（シンプル: cv2.inpaint TELEA のみ）

    各種手法の比較の結果、シンプルなTELEAインペイント（大きめradius）が最も自然。
    pre-fillやseamlessCloneを足すと過度な加工感が出てしまうので、TELEAに任せる。

    手順:
    1. ランドマークからタイトなポリゴンマスク生成 + 少し拡張
    2. cv2.inpaint TELEA（顔サイズに比例した大きめradius）で補完
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

    # 2. TELEAインペイント（radius大きめで眉全体を自然に埋める）
    inpaint_radius = max(15, int(face_h * 0.04))
    result = cv2.inpaint(image, mask_u8, inpaint_radius, cv2.INPAINT_TELEA)

    return result


# =========================================================
# 眉描画 (Phase 2)
# =========================================================

# 眉タイプ定義
# EYEBROW_TYPES.md を参照
EYEBROW_TYPES = {
    "straight": {
        "peak_position": 0.5,
        "peak_height_ratio": 0.0,
        "tail_height_ratio": 0.0,
        "thickness_ratio": 1.9,
        "length_ratio": 1.0,
        "desc": "ストレート眉（若々しい・韓流）",
    },
    "parallel_thick": {
        "peak_position": 0.5,
        "peak_height_ratio": 0.15,
        "tail_height_ratio": 0.0,
        "thickness_ratio": 1.9,
        "length_ratio": 1.0,
        "desc": "並行太眉（ナチュラル・男らしい）",
    },
    "natural_arch": {
        "peak_position": 0.65,
        "peak_height_ratio": 0.35,
        "tail_height_ratio": 0.1,
        "thickness_ratio": 1.6,
        "length_ratio": 1.0,
        "desc": "ナチュラルアーチ（知的・バランス）",
    },
    "arch": {
        "peak_position": 0.65,
        "peak_height_ratio": 0.55,
        "tail_height_ratio": 0.2,
        "thickness_ratio": 1.5,
        "length_ratio": 1.0,
        "desc": "アーチ眉（上品・標準）",
    },
    "angular": {
        "peak_position": 0.6,
        "peak_height_ratio": 0.75,
        "tail_height_ratio": 0.3,
        "thickness_ratio": 1.5,
        "length_ratio": 1.0,
        "desc": "角度眉（シャープ・クール）",
    },
    "short_thick": {
        "peak_position": 0.5,
        "peak_height_ratio": 0.08,
        "tail_height_ratio": 0.0,
        "thickness_ratio": 2.2,
        "length_ratio": 0.85,
        "desc": "短め太眉（ワイルド・強い）",
    },
    "long_arch": {
        "peak_position": 0.7,
        "peak_height_ratio": 0.4,
        "tail_height_ratio": 0.25,
        "thickness_ratio": 1.3,
        "length_ratio": 1.1,
        "desc": "長めアーチ（大人・クール）",
    },
}

# デフォルト設定
DEFAULT_EYEBROW_TYPE = "natural_arch"
DEFAULT_EYEBROW_COLOR_RGB = (60, 40, 30)  # ダークブラウン
DEFAULT_EYEBROW_INTENSITY = 0.85


def compute_brow_anchors(fm: FaceMesh, side: str = "right") -> dict:
    """眉の基準点（眉頭・眉尻）と関連値を計算

    ルール:
    - 眉頭: 小鼻外縁の真上、眉のセンターライン上
    - 眉尻: 小鼻-目尻を結ぶ直線の延長上、眉頭と同じY座標
    - 眉のセンターライン Y: eye_top - eye_height * 1.86 (実測ベース)
    """
    if side == "right":
        nose_wing = fm.landmarks_px[64].astype(float)
        inner_eye = fm.landmarks_px[133].astype(float)
        outer_eye = fm.landmarks_px[33].astype(float)
        eye_top = fm.landmarks_px[159].astype(float)
        eye_bot = fm.landmarks_px[145].astype(float)
    else:
        nose_wing = fm.landmarks_px[294].astype(float)
        inner_eye = fm.landmarks_px[362].astype(float)
        outer_eye = fm.landmarks_px[263].astype(float)
        eye_top = fm.landmarks_px[386].astype(float)
        eye_bot = fm.landmarks_px[374].astype(float)

    eye_height = abs(eye_top[1] - eye_bot[1])

    # 眉頭: 小鼻外縁の真上、眉センターラインの高さ
    head_x = nose_wing[0]
    head_y = eye_top[1] - eye_height * 1.86

    # 眉尻: 小鼻-目尻の延長線上で Y == head_y となる点
    dx = outer_eye[0] - nose_wing[0]
    dy = outer_eye[1] - nose_wing[1]
    if abs(dy) < 1e-3:
        tail_x = outer_eye[0] + (outer_eye[0] - inner_eye[0]) * 0.3
        tail_y = head_y
    else:
        t = (head_y - nose_wing[1]) / dy
        tail_x = nose_wing[0] + t * dx
        tail_y = head_y

    return {
        "head": np.array([head_x, head_y], dtype=np.float64),
        "tail": np.array([tail_x, tail_y], dtype=np.float64),
        "eye_height": eye_height,
        "brow_length": abs(tail_x - head_x),
        "side": side,
    }


def _quadratic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                     n: int = 32) -> np.ndarray:
    """二次Bezier曲線を n 点でサンプリング"""
    ts = np.linspace(0, 1, n)
    points = np.zeros((n, 2))
    for i, t in enumerate(ts):
        points[i] = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    return points


def _solve_bezier_control(p0: np.ndarray, pm: np.ndarray, p2: np.ndarray,
                         t: float) -> np.ndarray:
    """パラメータ t の位置で pm を通過する二次Bezier曲線の制御点 P1 を求める

    B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2 = pm
    → P1 = (pm - (1-t)^2 * P0 - t^2 * P2) / (2(1-t)t)
    """
    if t <= 0 or t >= 1:
        return (p0 + p2) / 2
    return (pm - (1 - t) ** 2 * p0 - t ** 2 * p2) / (2 * (1 - t) * t)


def _taper(t: float) -> float:
    """眉の太さをtで調整（眉頭・眉尻が細くなる自然な形）

    t=0(頭): 0.85倍
    t=0.3: 1.0倍（最大）
    t=1(尻): 0.55倍
    """
    if t < 0.3:
        # 頭から最大へ: 0.85 → 1.0
        return 0.85 + (t / 0.3) * 0.15
    else:
        # 最大から尻へ: 1.0 → 0.55
        s = (t - 0.3) / 0.7
        return 1.0 - s * 0.45


def generate_brow_polygon(anchors: dict, brow_type: str) -> np.ndarray:
    """眉タイプに応じた上下ラインを生成してポリゴン頂点列を返す

    アプローチ:
    1. センターライン（眉の中心線）をBezier曲線で生成
       head → peak(上にズレる) → tail
    2. 各センター点の上下に thickness/2 ずらして上辺/下辺を作成
    3. taper で先端を細く

    返り値: (N*2, 2) のint32配列（上辺N点 + 下辺N点逆順）
    """
    params = EYEBROW_TYPES[brow_type]

    head = anchors["head"].copy()
    tail = anchors["tail"].copy()
    eye_h = anchors["eye_height"]

    # 長さ調整（尻を head→tail 方向に伸縮）
    length_ratio = params["length_ratio"]
    if length_ratio != 1.0:
        tail = head + (tail - head) * length_ratio

    # 眉尻の高さ調整（センターを下げる）
    tail[1] += eye_h * params["tail_height_ratio"]

    # 眉山: センターラインの上に盛り上がる
    peak_t = params["peak_position"]
    peak_x = head[0] + (tail[0] - head[0]) * peak_t
    peak_y = head[1] - eye_h * params["peak_height_ratio"]
    peak = np.array([peak_x, peak_y])

    # センターライン（Bezier曲線）
    n = 40
    p1 = _solve_bezier_control(head, peak, tail, peak_t)
    center_line = _quadratic_bezier(head, p1, tail, n=n)

    # 太さ
    base_thickness = eye_h * params["thickness_ratio"]

    # 上下辺をセンターラインから ±thickness/2 でtaper
    upper = np.zeros_like(center_line)
    lower = np.zeros_like(center_line)
    ts = np.linspace(0, 1, n)
    for i, t in enumerate(ts):
        half_thick = base_thickness * 0.5 * _taper(t)
        upper[i] = center_line[i] + np.array([0, -half_thick])  # 上(Y小)
        lower[i] = center_line[i] + np.array([0, +half_thick])  # 下(Y大)

    # ポリゴン頂点列（上辺 → 下辺逆順）
    polygon = np.vstack([upper, lower[::-1]])
    return polygon.astype(np.int32)


def build_brow_mask(fm: FaceMesh, w: int, h: int, brow_type: str) -> np.ndarray:
    """両眉のマスクを生成（float32, 0-1）"""
    mask = np.zeros((h, w), dtype=np.float32)

    for side in ["right", "left"]:
        anchors = compute_brow_anchors(fm, side=side)
        polygon = generate_brow_polygon(anchors, brow_type)
        cv2.fillPoly(mask, [polygon], 1.0)

    return mask


def draw_eyebrows(
    image: np.ndarray,
    fm: FaceMesh,
    brow_type: str = DEFAULT_EYEBROW_TYPE,
    color_rgb: tuple = DEFAULT_EYEBROW_COLOR_RGB,
    intensity: float = DEFAULT_EYEBROW_INTENSITY,
    edge_blur: float = 1.0,
) -> np.ndarray:
    """眉を描画する（眉消し済み画像が前提）

    Args:
        image: 眉消し済み画像 (BGR)
        fm: FaceMesh
        brow_type: 眉タイプ名 (EYEBROW_TYPES のキー)
        color_rgb: 眉の色 (R, G, B)
        intensity: 描画強度 0-1
        edge_blur: エッジのブラー倍率
    """
    if brow_type not in EYEBROW_TYPES:
        raise ValueError(f"Unknown brow type: {brow_type}. Available: {list(EYEBROW_TYPES.keys())}")

    h, w = image.shape[:2]
    face_h = np.linalg.norm(
        fm.landmarks_px[10].astype(float) - fm.landmarks_px[152].astype(float)
    )

    # 眉マスク生成
    mask = build_brow_mask(fm, w, h, brow_type)

    # エッジをソフトに
    blur_ksize = max(3, int(face_h * 0.008 * edge_blur))
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    soft_mask = gaussian_blur_mask(mask, blur_ksize)

    # 色を適用（BGRに変換）
    color_bgr = np.array([color_rgb[2], color_rgb[1], color_rgb[0]], dtype=np.float32)
    alpha = (soft_mask * intensity)[..., np.newaxis]

    # src * (1 - a) + color * a でブレンド
    src_f = image.astype(np.float32)
    color_layer = np.broadcast_to(color_bgr, src_f.shape)
    result = src_f * (1.0 - alpha) + color_layer * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_eyebrow_makeup(
    image: np.ndarray,
    fm: FaceMesh,
    brow_type: str = DEFAULT_EYEBROW_TYPE,
    color_rgb: tuple = DEFAULT_EYEBROW_COLOR_RGB,
    intensity: float = DEFAULT_EYEBROW_INTENSITY,
) -> np.ndarray:
    """眉消し → 新しい眉を描画 をまとめて実行"""
    erased = erase_eyebrows(image, fm)
    drawn = draw_eyebrows(erased, fm, brow_type=brow_type,
                          color_rgb=color_rgb, intensity=intensity)
    return drawn


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
    parser.add_argument("input", nargs="?", help="入力画像パス")
    parser.add_argument("-o", "--output", help="出力画像パス (default: eyebrow_result.png)")
    parser.add_argument("-t", "--type", default=DEFAULT_EYEBROW_TYPE,
                        choices=list(EYEBROW_TYPES.keys()),
                        help=f"眉タイプ (default: {DEFAULT_EYEBROW_TYPE})")
    parser.add_argument("--color", nargs=3, type=int, metavar=("R", "G", "B"),
                        default=list(DEFAULT_EYEBROW_COLOR_RGB),
                        help=f"眉の色 RGB (default: {DEFAULT_EYEBROW_COLOR_RGB})")
    parser.add_argument("--intensity", type=float, default=DEFAULT_EYEBROW_INTENSITY,
                        help=f"描画強度 0-1 (default: {DEFAULT_EYEBROW_INTENSITY})")
    parser.add_argument("--no-draw", action="store_true",
                        help="眉消しのみ（新しい眉を描画しない）")
    parser.add_argument("--blur", type=float, default=1.0,
                        help="マスクブラー倍率 (default: 1.0)")
    parser.add_argument("--imgonly", action="store_true",
                        help="結果画像のみ (比較画像なし)")
    parser.add_argument("--zoom", action="store_true",
                        help="眉元ズームの比較画像を出力")
    parser.add_argument("--list-types", action="store_true",
                        help="眉タイプ一覧を表示")
    args = parser.parse_args()

    if args.list_types:
        print("利用可能な眉タイプ:")
        for name, params in EYEBROW_TYPES.items():
            print(f"  {name:16s} - {params['desc']}")
        return

    if not args.input:
        parser.error("input image is required")

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

    if not args.no_draw:
        print(f"眉描画中... type={args.type}, color={tuple(args.color)}, intensity={args.intensity}")
        output = draw_eyebrows(
            output, fm,
            brow_type=args.type,
            color_rgb=tuple(args.color),
            intensity=args.intensity,
        )

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
