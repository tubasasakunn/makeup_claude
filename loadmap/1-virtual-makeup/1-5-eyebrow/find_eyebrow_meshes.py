"""
眉メッシュID特定スクリプト

MediaPipeのランドマークIDを元に、眉全体に対応するメッシュ三角形IDを特定する。
src/area のFace Mesh Selectorと同じ座標系（subdivision_level=1）で動作。

右眉のみ計算し、find_mirror_meshes で左眉をミラー生成して左右対称を保証する。

眉の部位:
  - eyebrow_full    : 眉全体（消し用 + 描画ベース）
  - eyebrow_skin    : 眉の上下の肌領域（肌色サンプリング用）
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
IMGS_DIR = PROJECT_ROOT / "imgs"

# =========================================================
# MediaPipe FaceLandmarker 478点 ランドマークID（右眉のみ）
# =========================================================

# 右眉 上端ライン（外側→内側）
RIGHT_EYEBROW_UPPER = [70, 63, 105, 66, 107]
# 右眉 下端ライン（外側→内側）
RIGHT_EYEBROW_LOWER = [46, 53, 52, 65, 55]

# 左眉 上端ライン（外側→内側）
LEFT_EYEBROW_UPPER = [300, 293, 334, 296, 336]
# 左眉 下端ライン（外側→内側）
LEFT_EYEBROW_LOWER = [276, 283, 282, 295, 285]

# 肌色サンプリング用：眉の上の額領域
# 右側：眉上端よりさらに上のランドマーク
RIGHT_FOREHEAD_UPPER = [71, 68, 104, 69, 108]
# 左側
LEFT_FOREHEAD_UPPER = [301, 298, 333, 299, 337]

# 肌色サンプリング用：眉の下（眉と目の間）
RIGHT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173]


def landmarks_to_points(fm, landmark_ids, sort_by_x=True):
    """ランドマークIDリストから正規化座標リストを返す"""
    pts = []
    for lid in landmark_ids:
        if lid < len(fm.points) and lid < 478:
            p = fm.points[lid]
            pts.append((p["x"], p["y"]))
    if sort_by_x:
        pts.sort(key=lambda p: p[0])
    return pts


def make_polygon(upper_pts, lower_pts):
    """上辺と下辺の点列からポリゴンを作成"""
    return np.array(upper_pts + lower_pts[::-1], dtype=np.float64)


def point_in_polygon(px, py, polygon):
    """点がポリゴン内にあるか判定（ray casting）"""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def find_meshes_in_polygon(fm, polygon):
    """ポリゴン内にあるメッシュ三角形IDを検索"""
    mesh_ids = []
    for i in range(len(fm.triangles)):
        a, b, c = fm.triangles[i]
        pa, pb, pc = fm.points[a], fm.points[b], fm.points[c]
        cx = (pa["x"] + pb["x"] + pc["x"]) / 3
        cy = (pa["y"] + pb["y"] + pc["y"]) / 3
        if point_in_polygon(cx, cy, polygon):
            mesh_ids.append(i)
    return mesh_ids


def expand_polygon(polygon, scale=1.15):
    """ポリゴンを重心から拡大（眉の毛がはみ出る分をカバー）"""
    centroid = polygon.mean(axis=0)
    return centroid + (polygon - centroid) * scale


def symmetric_pair(fm, right_meshes):
    """右眉メッシュからミラーし、ラウンドトリップで対称ペアのみ残す"""
    r_set = set(right_meshes)
    l_set = fm.find_mirror_meshes(r_set)
    r_roundtrip = fm.find_mirror_meshes(l_set)
    r_valid = r_set & r_roundtrip
    l_valid = fm.find_mirror_meshes(r_valid)
    return sorted(r_valid), sorted(l_valid)


def identify_eyebrow_areas(fm, w, h):
    """眉領域のメッシュIDを特定（右のみ計算→ミラーで左）"""
    areas = {}

    # 右眉のランドマーク座標を取得
    r_brow_upper = landmarks_to_points(fm, RIGHT_EYEBROW_UPPER)
    r_brow_lower = landmarks_to_points(fm, RIGHT_EYEBROW_LOWER)

    # === 1. 眉全体（少し拡大してカバー） ===
    r_brow_poly = make_polygon(r_brow_upper, r_brow_lower)
    r_brow_poly_expanded = expand_polygon(r_brow_poly, scale=1.3)
    r_brow_full = find_meshes_in_polygon(fm, r_brow_poly_expanded)

    # === 2. 肌色サンプリング領域（眉の上の額） ===
    r_forehead_upper = landmarks_to_points(fm, RIGHT_FOREHEAD_UPPER)
    r_skin_poly = make_polygon(r_forehead_upper, r_brow_upper)
    r_skin = find_meshes_in_polygon(fm, r_skin_poly)
    # 眉本体のメッシュを除外
    r_brow_set = set(r_brow_full)
    r_skin = [m for m in r_skin if m not in r_brow_set]

    # === ミラーで左眉を生成 ===
    r_brow_full, l_brow_full = symmetric_pair(fm, r_brow_full)
    r_skin, l_skin = symmetric_pair(fm, r_skin)

    areas["eyebrow_full"] = sorted(set(r_brow_full) | set(l_brow_full))
    areas["eyebrow_skin"] = sorted(set(r_skin) | set(l_skin))

    return areas


def generate_guide_images(fm, image, areas):
    """各部位のガイド画像を生成"""
    h, w = image.shape[:2]
    output_dir = Path(__file__).resolve().parent / "guide_images"
    output_dir.mkdir(exist_ok=True)

    colors = {
        "eyebrow_full": (100, 180, 255),    # オレンジ
        "eyebrow_skin": (180, 255, 180),     # 薄緑
    }

    labels = {
        "eyebrow_full": "1. Eyebrow full (erase target)",
        "eyebrow_skin": "2. Skin sample (color source)",
    }

    # 各部位ごとの個別画像
    for name, mesh_ids in areas.items():
        color_bgr = colors.get(name, (200, 100, 100))

        mask = fm.build_mask(mesh_ids, w, h)
        mask_smooth = cv2.GaussianBlur(mask, (7, 7), 2)

        color_layer = np.zeros_like(image)
        color_layer[:] = color_bgr
        alpha = (mask_smooth * 0.55)[..., np.newaxis]
        overlay = (image.astype(np.float32) * (1 - alpha) + color_layer.astype(np.float32) * alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        mask_u8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_bgr, 2)

        label_en = labels.get(name, name)
        cv2.putText(overlay, label_en, (w - 420, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, f"{len(mesh_ids)} meshes", (w - 420, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        comparison = np.hstack([image, overlay])
        out_path = output_dir / f"{name}.png"
        cv2.imwrite(str(out_path), comparison)
        print(f"  ガイド画像: {out_path.name}")

    # 全部位統合画像
    overlay_all = image.copy().astype(np.float32)
    for name, mesh_ids in areas.items():
        color_bgr = colors.get(name, (200, 100, 100))
        mask = fm.build_mask(mesh_ids, w, h)
        mask_smooth = cv2.GaussianBlur(mask, (5, 5), 1.5)
        color_layer = np.zeros_like(image, dtype=np.float32)
        color_layer[:] = color_bgr
        alpha = (mask_smooth * 0.5)[..., np.newaxis]
        overlay_all = overlay_all * (1 - alpha) + color_layer * alpha

    overlay_all = np.clip(overlay_all, 0, 255).astype(np.uint8)

    legend_x = w - 420
    legend_y = 30
    for name in areas:
        color_bgr = colors.get(name, (200, 100, 100))
        label_en = labels.get(name, name)
        cv2.rectangle(overlay_all, (legend_x - 5, legend_y - 18),
                      (w - 10, legend_y + 8), (0, 0, 0), -1)
        cv2.rectangle(overlay_all, (legend_x - 5, legend_y - 18),
                      (w - 10, legend_y + 8), (100, 100, 100), 1)
        cv2.rectangle(overlay_all, (legend_x, legend_y - 12),
                      (legend_x + 20, legend_y + 2), color_bgr, -1)
        cv2.putText(overlay_all, label_en, (legend_x + 28, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 32

    comparison_all = np.hstack([image, overlay_all])
    out_path = output_dir / "all_areas.png"
    cv2.imwrite(str(out_path), comparison_all)
    print(f"  統合画像: {out_path.name}")


def main():
    img_path = IMGS_DIR / "base.png"
    if not img_path.exists():
        pngs = list(IMGS_DIR.glob("*.png"))
        if not pngs:
            print("Error: imgs/ に画像がありません")
            return
        img_path = pngs[0]

    print(f"使用画像: {img_path}")
    image = cv2.imread(str(img_path))
    if image is None:
        print("Error: 画像を読み込めません")
        return

    h, w = image.shape[:2]

    print("FaceMesh 初期化中...")
    fm = FaceMesh(subdivision_level=1)
    fm.init()

    print("顔検出中...")
    result = fm.detect(image)
    if result is None:
        print("Error: 顔が検出されませんでした")
        return

    print(f"メッシュ: {len(fm.triangles)} 三角形, {len(fm.points)} 頂点")

    areas = identify_eyebrow_areas(fm, w, h)

    print("\n=== 眉メッシュID ===")
    result_data = []
    for name, mesh_ids in areas.items():
        print(f"\n{name}: {len(mesh_ids)} meshes")
        print(f"  mesh_id: {mesh_ids[:20]}{'...' if len(mesh_ids) > 20 else ''}")
        result_data.append({
            "name": name,
            "mesh_id": mesh_ids,
        })

    # target.json に保存
    target_path = PROJECT_ROOT / "target.json"
    with open(target_path) as f:
        target = json.load(f)

    target["eyebrow"] = result_data

    with open(target_path, "w") as f:
        json.dump(target, f, indent=4, ensure_ascii=False)
    print(f"\ntarget.json 更新完了: {target_path}")

    generate_guide_images(fm, image, areas)


if __name__ == "__main__":
    main()
