"""
眉先端の尖らせ方の比較実験
複数の手法を実装して同じ angular タイプで比較
"""
import sys
import math
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.facemesh import FaceMesh
from main import (
    compute_brow_anchors,
    _quadratic_bezier,
    _solve_bezier_control,
    gaussian_blur_mask,
    erase_eyebrows,
    crop_eyebrow_region,
    EYEBROW_TYPES,
    _apply_density_gradient,
    _make_directional_noise,
)


# ===== 各手法の generate_brow_polygon 実装 =====

def method_A_taper_to_zero(anchors, brow_type):
    """A. 既存の法線オフセット + taperを先端で 0.0 に"""
    params = EYEBROW_TYPES[brow_type]
    head = anchors["head"].copy()
    tail = anchors["tail"].copy()
    eye_h = anchors["eye_height"]
    length_ratio = params["length_ratio"]
    if length_ratio != 1.0:
        tail = head + (tail - head) * length_ratio
    tail[1] += eye_h * params["tail_height_ratio"]

    peak_t = params["peak_position"]
    peak_x = head[0] + (tail[0] - head[0]) * peak_t
    peak_y = head[1] - eye_h * params["peak_height_ratio"]
    peak = np.array([peak_x, peak_y])

    n = 100
    p1 = _solve_bezier_control(head, peak, tail, peak_t)
    center_line = _quadratic_bezier(head, p1, tail, n=n)
    base_thickness = eye_h * params["thickness_ratio"]

    # 接線→法線
    tangents = np.zeros_like(center_line)
    tangents[0] = center_line[1] - center_line[0]
    tangents[-1] = center_line[-1] - center_line[-2]
    tangents[1:-1] = (center_line[2:] - center_line[:-2]) / 2
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.maximum(lengths, 1e-6)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    # taper: head 0.7 → middle 1.0 → tail 0.0 (尖らせ)
    def taper(t):
        if t < 0.35:
            x = t / 0.35
            return 0.7 + 0.3 * (1 - math.cos(x * math.pi / 2))
        else:
            x = (t - 0.35) / 0.65
            return 1.0 * math.cos(x * math.pi / 2)  # 0 まで落ちる

    ts = np.linspace(0, 1, n)
    upper = np.zeros_like(center_line)
    lower = np.zeros_like(center_line)
    for i, t in enumerate(ts):
        ht = base_thickness * 0.5 * taper(t)
        upper[i] = center_line[i] - normals[i] * ht
        lower[i] = center_line[i] + normals[i] * ht

    return np.vstack([upper, lower[::-1]]).astype(np.int32)


def method_B_converged_tip(anchors, brow_type):
    """B. 上辺と下辺が先端で1点に収束（明確な三角の尖り）"""
    params = EYEBROW_TYPES[brow_type]
    head = anchors["head"].copy()
    tail = anchors["tail"].copy()
    eye_h = anchors["eye_height"]
    length_ratio = params["length_ratio"]
    if length_ratio != 1.0:
        tail = head + (tail - head) * length_ratio
    tail[1] += eye_h * params["tail_height_ratio"]

    peak_t = params["peak_position"]
    peak_x = head[0] + (tail[0] - head[0]) * peak_t
    peak_y = head[1] - eye_h * params["peak_height_ratio"]
    peak = np.array([peak_x, peak_y])

    n = 100
    p1 = _solve_bezier_control(head, peak, tail, peak_t)
    center_line = _quadratic_bezier(head, p1, tail, n=n)
    base_thickness = eye_h * params["thickness_ratio"]

    tangents = np.zeros_like(center_line)
    tangents[0] = center_line[1] - center_line[0]
    tangents[-1] = center_line[-1] - center_line[-2]
    tangents[1:-1] = (center_line[2:] - center_line[:-2]) / 2
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.maximum(lengths, 1e-6)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    # 先端10%は急激に0へ（鋭く尖る）
    def taper(t):
        if t < 0.3:
            x = t / 0.3
            return 0.6 + 0.4 * (1 - math.cos(x * math.pi / 2))
        elif t < 0.85:
            return 1.0
        else:
            x = (t - 0.85) / 0.15  # 0..1
            return 1.0 - x  # 線形に0へ

    ts = np.linspace(0, 1, n)
    upper = np.zeros_like(center_line)
    lower = np.zeros_like(center_line)
    for i, t in enumerate(ts):
        ht = base_thickness * 0.5 * taper(t)
        upper[i] = center_line[i] - normals[i] * ht
        lower[i] = center_line[i] + normals[i] * ht

    # 先端点を明示的に同じ点に収束
    upper[-1] = center_line[-1]
    lower[-1] = center_line[-1]

    return np.vstack([upper, lower[::-1]]).astype(np.int32)


def method_C_catmull_rom(anchors, brow_type):
    """C. Catmull-Romスプライン中心線 + 対称 taper"""
    params = EYEBROW_TYPES[brow_type]
    head = anchors["head"].copy()
    tail = anchors["tail"].copy()
    eye_h = anchors["eye_height"]
    length_ratio = params["length_ratio"]
    if length_ratio != 1.0:
        tail = head + (tail - head) * length_ratio
    tail[1] += eye_h * params["tail_height_ratio"]

    peak_t = params["peak_position"]
    peak_x = head[0] + (tail[0] - head[0]) * peak_t
    peak_y = head[1] - eye_h * params["peak_height_ratio"]

    # 5制御点: head, head_mid, peak, peak_mid, tail
    head_mid = head + (np.array([peak_x, peak_y]) - head) * 0.4
    peak_mid = np.array([peak_x, peak_y]) + (tail - np.array([peak_x, peak_y])) * 0.5
    peak_mid[1] = (peak_y + tail[1]) / 2 - eye_h * 0.05  # 眉尻側もカーブ

    control_points = [
        head - (np.array([peak_x, peak_y]) - head) * 0.3,  # 仮想前点
        head,
        np.array([peak_x, peak_y]),
        peak_mid,
        tail,
        tail + (tail - peak_mid) * 0.3,  # 仮想後点
    ]

    # Catmull-Rom補間（各区間20点）
    def catmull_rom(p0, p1, p2, p3, n_per_seg=25):
        ts = np.linspace(0, 1, n_per_seg)
        pts = []
        for t in ts:
            t2 = t * t
            t3 = t2 * t
            pt = 0.5 * (
                (2 * p1) +
                (-p0 + p2) * t +
                (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            pts.append(pt)
        return np.array(pts)

    segments = []
    for i in range(len(control_points) - 3):
        seg = catmull_rom(*control_points[i:i + 4])
        segments.append(seg if i == 0 else seg[1:])
    center_line = np.vstack(segments)
    n = len(center_line)

    base_thickness = eye_h * params["thickness_ratio"]

    tangents = np.zeros_like(center_line)
    tangents[0] = center_line[1] - center_line[0]
    tangents[-1] = center_line[-1] - center_line[-2]
    tangents[1:-1] = (center_line[2:] - center_line[:-2]) / 2
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.maximum(lengths, 1e-6)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    def taper(t):
        if t < 0.3:
            x = t / 0.3
            return 0.55 + 0.45 * (1 - math.cos(x * math.pi / 2))
        else:
            x = (t - 0.3) / 0.7
            # 先端ゆっくり 0 付近まで
            return (1 - x) ** 1.2 * 0.9 + 0.1 * (1 - x)

    ts = np.linspace(0, 1, n)
    upper = np.zeros_like(center_line)
    lower = np.zeros_like(center_line)
    for i, t in enumerate(ts):
        ht = base_thickness * 0.5 * taper(t)
        upper[i] = center_line[i] - normals[i] * ht
        lower[i] = center_line[i] + normals[i] * ht

    return np.vstack([upper, lower[::-1]]).astype(np.int32)


def method_D_triangle_cut(anchors, brow_type):
    """D. 既存のポリゴン + 先端を三角マスクでカット"""
    # method_A と同じ描画をしてから、先端15%だけを三角形で切り取り
    polygon = method_A_taper_to_zero(anchors, brow_type)
    # ここでは method_A のまま返す（カット処理はマスク生成時に適用）
    return polygon


def method_E_asymmetric_taper(anchors, brow_type):
    """E. 上辺は滑らか、下辺は直線 (上から下へ斜めに収束)"""
    params = EYEBROW_TYPES[brow_type]
    head = anchors["head"].copy()
    tail = anchors["tail"].copy()
    eye_h = anchors["eye_height"]
    length_ratio = params["length_ratio"]
    if length_ratio != 1.0:
        tail = head + (tail - head) * length_ratio
    tail[1] += eye_h * params["tail_height_ratio"]

    peak_t = params["peak_position"]
    peak_x = head[0] + (tail[0] - head[0]) * peak_t
    peak_y = head[1] - eye_h * params["peak_height_ratio"]
    peak = np.array([peak_x, peak_y])

    n = 100
    p1 = _solve_bezier_control(head, peak, tail, peak_t)
    center_line = _quadratic_bezier(head, p1, tail, n=n)
    base_thickness = eye_h * params["thickness_ratio"]

    tangents = np.zeros_like(center_line)
    tangents[0] = center_line[1] - center_line[0]
    tangents[-1] = center_line[-1] - center_line[-2]
    tangents[1:-1] = (center_line[2:] - center_line[:-2]) / 2
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.maximum(lengths, 1e-6)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    # 上側は通常テーパー（0.55→1.0→0.4）
    def upper_taper(t):
        if t < 0.35:
            x = t / 0.35
            return 0.55 + 0.45 * (1 - math.cos(x * math.pi / 2))
        else:
            x = (t - 0.35) / 0.65
            return 1.0 - x * 0.6  # 0.4 まで

    # 下側は急峻（0.55→1.0→0.0）で先端が下から上へ上がる
    def lower_taper(t):
        if t < 0.35:
            x = t / 0.35
            return 0.55 + 0.45 * (1 - math.cos(x * math.pi / 2))
        else:
            x = (t - 0.35) / 0.65
            return (1 - x) ** 1.3  # 0 まで

    ts = np.linspace(0, 1, n)
    upper = np.zeros_like(center_line)
    lower = np.zeros_like(center_line)
    for i, t in enumerate(ts):
        ut = base_thickness * 0.5 * upper_taper(t)
        lt = base_thickness * 0.5 * lower_taper(t)
        upper[i] = center_line[i] - normals[i] * ut
        lower[i] = center_line[i] + normals[i] * lt

    return np.vstack([upper, lower[::-1]]).astype(np.int32)


def method_F_type_specific(anchors, brow_type):
    """F. タイプ別テーパー + 先端収束"""
    params = EYEBROW_TYPES[brow_type]
    head = anchors["head"].copy()
    tail = anchors["tail"].copy()
    eye_h = anchors["eye_height"]
    length_ratio = params["length_ratio"]
    if length_ratio != 1.0:
        tail = head + (tail - head) * length_ratio
    tail[1] += eye_h * params["tail_height_ratio"]

    peak_t = params["peak_position"]
    peak_x = head[0] + (tail[0] - head[0]) * peak_t
    peak_y = head[1] - eye_h * params["peak_height_ratio"]
    peak = np.array([peak_x, peak_y])

    n = 100
    p1 = _solve_bezier_control(head, peak, tail, peak_t)
    center_line = _quadratic_bezier(head, p1, tail, n=n)
    base_thickness = eye_h * params["thickness_ratio"]

    tangents = np.zeros_like(center_line)
    tangents[0] = center_line[1] - center_line[0]
    tangents[-1] = center_line[-1] - center_line[-2]
    tangents[1:-1] = (center_line[2:] - center_line[:-2]) / 2
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.maximum(lengths, 1e-6)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    # タイプ別の尖り度
    SHARP_TYPES = {"angular", "arch", "long_arch", "natural_arch"}
    ROUND_TYPES = {"straight", "parallel_thick", "short_thick"}

    sharp = brow_type in SHARP_TYPES

    def taper(t):
        if t < 0.3:
            x = t / 0.3
            return 0.6 + 0.4 * (1 - math.cos(x * math.pi / 2))
        else:
            x = (t - 0.3) / 0.7
            if sharp:
                # 先端 0 へ急降下
                return (1 - x) ** 0.9
            else:
                # rounded: 0.5 で止まる
                return 1.0 - x * 0.5

    ts = np.linspace(0, 1, n)
    upper = np.zeros_like(center_line)
    lower = np.zeros_like(center_line)
    for i, t in enumerate(ts):
        ht = base_thickness * 0.5 * taper(t)
        upper[i] = center_line[i] - normals[i] * ht
        lower[i] = center_line[i] + normals[i] * ht

    # sharpタイプは先端を1点に収束
    if sharp:
        upper[-1] = center_line[-1]
        lower[-1] = center_line[-1]

    return np.vstack([upper, lower[::-1]]).astype(np.int32)


# ===== 描画関数 =====

METHODS = {
    "A_taper_zero": method_A_taper_to_zero,
    "B_converged": method_B_converged_tip,
    "C_catmull": method_C_catmull_rom,
    "D_triangle": method_D_triangle_cut,  # = A
    "E_asymmetric": method_E_asymmetric_taper,
    "F_type_specific": method_F_type_specific,
}


def build_mask_for_method(fm, w, h, brow_type, method_fn, supersample=2):
    sh, sw = h * supersample, w * supersample
    mask_hi = np.zeros((sh, sw), dtype=np.float32)
    for side in ["right", "left"]:
        anchors = compute_brow_anchors(fm, side=side)
        polygon = method_fn(anchors, brow_type)
        polygon_hi = polygon * supersample
        cv2.fillPoly(mask_hi, [polygon_hi], 1.0)
    return cv2.resize(mask_hi, (w, h), interpolation=cv2.INTER_AREA)


def draw_with_method(image, fm, brow_type, method_fn):
    h, w = image.shape[:2]
    face_h = np.linalg.norm(fm.landmarks_px[10].astype(float) - fm.landmarks_px[152].astype(float))

    mask = build_mask_for_method(fm, w, h, brow_type, method_fn)
    mask = _apply_density_gradient(mask, fm)

    # ブラー + ノイズ
    ksize1 = max(5, int(face_h * 0.018))
    density = gaussian_blur_mask(mask, ksize1)
    ksize2 = max(3, int(face_h * 0.008))
    density = gaussian_blur_mask(density, ksize2)

    noise = _make_directional_noise(density.shape, face_h)
    mask_region = density > 0.05
    density[mask_region] = density[mask_region] * (1.0 + noise[mask_region] * 0.18)
    density = np.clip(density, 0, 1)

    color_bgr = np.array([45, 60, 85], dtype=np.float32)  # BGR
    intensity = 0.5
    alpha = (density * intensity)[..., np.newaxis]
    src_f = image.astype(np.float32)
    color_layer = np.broadcast_to(color_bgr, src_f.shape)
    result = src_f * (1.0 - alpha) + color_layer * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    fm = FaceMesh(subdivision_level=1)
    fm.init()
    image = cv2.imread("imgs/base.png")
    fm.detect(image)

    # 最もわかりやすい angular で各手法を比較
    brow_type = "angular"
    erased = erase_eyebrows(image, fm)

    results = [("ORIGINAL", image)]
    for name, fn in METHODS.items():
        out = draw_with_method(erased, fm, brow_type, fn)
        results.append((name, out))

    # ズーム比較
    crops = []
    for name, img in results:
        c, _ = crop_eyebrow_region(img, fm, margin_ratio=0.3)
        cv2.putText(c, name, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        crops.append(c)

    min_h = min(c.shape[0] for c in crops)
    def rh(img):
        s = min_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), min_h))

    imgs = [rh(c) for c in crops]
    rows = []
    for i in range(0, len(imgs), 2):
        row = imgs[i:i + 2]
        while len(row) < 2:
            row.append(np.zeros_like(imgs[0]))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    cv2.imwrite("/tmp/tip_compare_angular.png", grid)
    print("saved /tmp/tip_compare_angular.png")

    # natural_arch でも比較
    results2 = [("ORIGINAL", image)]
    for name, fn in METHODS.items():
        out = draw_with_method(erased, fm, "natural_arch", fn)
        results2.append((name, out))

    crops2 = []
    for name, img in results2:
        c, _ = crop_eyebrow_region(img, fm, margin_ratio=0.3)
        cv2.putText(c, name, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        crops2.append(c)
    imgs2 = [rh(c) for c in crops2]
    rows2 = []
    for i in range(0, len(imgs2), 2):
        row = imgs2[i:i + 2]
        while len(row) < 2:
            row.append(np.zeros_like(imgs2[0]))
        rows2.append(np.hstack(row))
    grid2 = np.vstack(rows2)
    cv2.imwrite("/tmp/tip_compare_natural.png", grid2)
    print("saved /tmp/tip_compare_natural.png")


if __name__ == "__main__":
    main()
