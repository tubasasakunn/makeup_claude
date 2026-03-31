"""
アイメイク用メッシュID特定スクリプト

MediaPipeのランドマークIDを元に、目の各部位に対応するメッシュ三角形IDを特定する。
src/area のFace Mesh Selectorと同じ座標系（subdivision_level=1）で動作。

目の部位:
  - eye_upper_lid    : 上まぶた（アイシャドウ・アイホール）
  - eye_crease       : 二重幅ライン（メインカラー）
  - eye_lower_lid    : 下まぶた（涙袋・下アイシャドウ）
  - eye_outer_corner : 目尻（ポイントカラー・はね上げ）
  - eye_inner_corner : 目頭（インナーカラー）
  - eyebrow_area     : 眉下〜アイホール上部
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

# MediaPipe FaceLandmarker 478点のランドマークID
# 参考: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

# 右目周辺のランドマークID
RIGHT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173]  # 上まぶたライン
RIGHT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]  # 下まぶたライン
RIGHT_EYE_OUTLINE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_IRIS = [468, 469, 470, 471, 472]  # 虹彩

# 左目周辺のランドマークID
LEFT_EYE_UPPER = [466, 388, 387, 386, 385, 384, 398]
LEFT_EYE_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]
LEFT_EYE_OUTLINE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
LEFT_EYE_IRIS = [473, 474, 475, 476, 477]

# 右眉のランドマークID
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
# 左眉のランドマークID
LEFT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

# アイホール（眉下〜目）の追加ランドマーク
RIGHT_EYEHOLE_UPPER = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
                       156, 28, 27, 29, 30, 247]
LEFT_EYEHOLE_UPPER = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
                      383, 258, 257, 259, 260, 467]

# 涙袋エリアのランドマーク（目の下の細い帯に限定）
RIGHT_TEAR_BAG = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                  243, 112, 26, 22, 23, 24, 110, 25]
LEFT_TEAR_BAG = [263, 249, 390, 373, 374, 380, 381, 382, 362,
                 463, 341, 256, 252, 253, 254, 339, 255]


def get_triangle_centroid(fm, tri_id, w, h):
    """三角形の重心をピクセル座標で返す"""
    a, b, c = fm.triangles[tri_id]
    pa, pb, pc = fm.points[a], fm.points[b], fm.points[c]
    cx = (pa["x"] + pb["x"] + pc["x"]) / 3 * w
    cy = (pa["y"] + pb["y"] + pc["y"]) / 3 * h
    return cx, cy


def get_triangle_vertices_normalized(fm, tri_id):
    """三角形の3頂点を正規化座標で返す"""
    a, b, c = fm.triangles[tri_id]
    return [
        (fm.points[a]["x"], fm.points[a]["y"]),
        (fm.points[b]["x"], fm.points[b]["y"]),
        (fm.points[c]["x"], fm.points[c]["y"]),
    ]


def landmarks_to_polygon(fm, landmark_ids):
    """ランドマークIDリストからポリゴン（ピクセル座標）を作成"""
    pts = []
    for lid in landmark_ids:
        if lid < len(fm.points):
            # ランドマークは最初の478点
            p = fm.points[lid] if lid < 478 else None
            if p:
                pts.append((p["x"], p["y"]))
    return np.array(pts, dtype=np.float64)


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


def find_meshes_in_landmark_region(fm, landmark_ids, expand=0.0):
    """
    指定ランドマークで囲まれた領域内にあるメッシュ三角形IDを検索。
    expand: ポリゴンを外側に拡張する割合（0.0=そのまま、0.1=10%拡張）
    """
    polygon = landmarks_to_polygon(fm, landmark_ids)
    if len(polygon) < 3:
        return []

    # ポリゴンの拡張（必要に応じて）
    if expand > 0:
        centroid = polygon.mean(axis=0)
        polygon = centroid + (polygon - centroid) * (1.0 + expand)

    mesh_ids = []
    for i in range(len(fm.triangles)):
        verts = get_triangle_vertices_normalized(fm, i)
        # 三角形の重心がポリゴン内にあるかチェック
        cx = sum(v[0] for v in verts) / 3
        cy = sum(v[1] for v in verts) / 3
        if point_in_polygon(cx, cy, polygon):
            mesh_ids.append(i)
    return mesh_ids


def find_meshes_between_regions(fm, upper_landmarks, lower_landmarks):
    """
    上側と下側のランドマーク列に挟まれた帯状の領域にあるメッシュIDを検索。
    アイシャドウの二重幅エリアなどに使う。
    """
    upper_pts = landmarks_to_polygon(fm, upper_landmarks)
    lower_pts = landmarks_to_polygon(fm, lower_landmarks)
    if len(upper_pts) < 2 or len(lower_pts) < 2:
        return []

    # 上と下のポイントを結んでポリゴンを形成
    polygon = np.vstack([upper_pts, lower_pts[::-1]])

    mesh_ids = []
    for i in range(len(fm.triangles)):
        verts = get_triangle_vertices_normalized(fm, i)
        cx = sum(v[0] for v in verts) / 3
        cy = sum(v[1] for v in verts) / 3
        if point_in_polygon(cx, cy, polygon):
            mesh_ids.append(i)
    return mesh_ids


def find_eye_meshes_by_distance(fm, eye_center_landmarks, min_dist, max_dist, h, w,
                                 y_bias_up=0, y_bias_down=0):
    """
    目の中心からの距離でメッシュを選択。
    上下方向にバイアスをかけて非対称な選択を可能にする。
    """
    # 目の中心を算出
    center_x = np.mean([fm.points[lid]["x"] for lid in eye_center_landmarks if lid < 478])
    center_y = np.mean([fm.points[lid]["y"] for lid in eye_center_landmarks if lid < 478])

    mesh_ids = []
    for i in range(len(fm.triangles)):
        verts = get_triangle_vertices_normalized(fm, i)
        cx = sum(v[0] for v in verts) / 3
        cy = sum(v[1] for v in verts) / 3

        # y方向バイアス適用
        dy = cy - center_y
        if dy < 0:  # 上方向
            effective_cy = center_y + dy * (1.0 + y_bias_up)
        else:  # 下方向
            effective_cy = center_y + dy * (1.0 + y_bias_down)

        dist = np.sqrt((cx - center_x) ** 2 + (effective_cy - center_y) ** 2)
        if min_dist <= dist <= max_dist:
            mesh_ids.append(i)
    return mesh_ids


def identify_eye_areas(fm, w, h):
    """全てのアイメイクエリアのメッシュIDを特定"""
    areas = {}

    # --- 上まぶた（アイシャドウ: アイホール全体） ---
    # 眉下〜目の上の広いエリア
    right_eyehole = RIGHT_EYEHOLE_UPPER + list(reversed(RIGHT_EYE_UPPER))
    left_eyehole = LEFT_EYEHOLE_UPPER + list(reversed(LEFT_EYE_UPPER))
    r_eyehole = find_meshes_in_landmark_region(fm, right_eyehole, expand=0.05)
    l_eyehole = find_meshes_in_landmark_region(fm, left_eyehole, expand=0.05)
    areas["eyeshadow_base"] = sorted(set(r_eyehole + l_eyehole))

    # --- 二重幅エリア（メインカラー） ---
    # 目の上まぶたライン〜少し上の帯状エリア
    # 右目
    right_crease_upper = [156, 28, 27, 29, 30, 247]
    right_crease_lower = RIGHT_EYE_UPPER
    r_crease = find_meshes_between_regions(fm, right_crease_upper, right_crease_lower)

    # 左目
    left_crease_upper = [383, 258, 257, 259, 260, 467]
    left_crease_lower = LEFT_EYE_UPPER
    l_crease = find_meshes_between_regions(fm, left_crease_upper, left_crease_lower)
    areas["eyeshadow_crease"] = sorted(set(r_crease + l_crease))

    # --- 目のキワ（締め色・アイライン） ---
    # 上まぶたラッシュライン沿いの細い帯: 目の輪郭内だが二重幅より下
    r_eye_all = set(find_meshes_in_landmark_region(fm, RIGHT_EYE_OUTLINE, expand=0.15))
    l_eye_all = set(find_meshes_in_landmark_region(fm, LEFT_EYE_OUTLINE, expand=0.15))
    # 目の内部（白目部分）を除外
    r_eye_inner = set(find_meshes_in_landmark_region(fm, RIGHT_EYE_OUTLINE, expand=-0.2))
    l_eye_inner = set(find_meshes_in_landmark_region(fm, LEFT_EYE_OUTLINE, expand=-0.2))
    # 目の輪郭帯 = 全体 - 内部
    r_kiwa = sorted(r_eye_all - r_eye_inner)
    l_kiwa = sorted(l_eye_all - l_eye_inner)
    areas["eyeliner"] = sorted(set(r_kiwa + l_kiwa))

    # --- 涙袋（下まぶたハイライト） ---
    # 下まぶたのライン〜少し下のエリア
    r_tear = find_meshes_in_landmark_region(fm, RIGHT_TEAR_BAG, expand=0.0)
    l_tear = find_meshes_in_landmark_region(fm, LEFT_TEAR_BAG, expand=0.0)
    # 目の内部のメッシュを除外（eyelinerで算出済みのr_eye_innerを再利用）
    r_tear_filtered = [m for m in r_tear if m not in r_eye_inner]
    l_tear_filtered = [m for m in l_tear if m not in l_eye_inner]
    areas["tear_bag"] = sorted(set(r_tear_filtered + l_tear_filtered))

    # --- 下まぶた目尻1/3（ポイントカラー） ---
    # 涙袋のうち目尻側1/3のみ抽出
    # 右目: 目尻=133, 目頭=33 → 目尻側のランドマーク付近のみ
    right_outer_corner = [133, 155, 154, 153, 121, 120, 119]
    left_outer_corner = [362, 382, 381, 380, 350, 349, 348]
    r_outer = find_meshes_in_landmark_region(fm, right_outer_corner, expand=0.0)
    l_outer = find_meshes_in_landmark_region(fm, left_outer_corner, expand=0.0)
    # 目の内部を除外
    r_outer_filtered = [m for m in r_outer if m not in r_eye_inner]
    l_outer_filtered = [m for m in l_outer if m not in l_eye_inner]
    areas["lower_outer"] = sorted(set(r_outer_filtered + l_outer_filtered))

    return areas


def main():
    # base.png を使用
    img_path = IMGS_DIR / "base.png"
    if not img_path.exists():
        # 任意の画像を使用
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

    # FaceMesh 初期化・検出
    print("FaceMesh 初期化中...")
    fm = FaceMesh(subdivision_level=1)
    fm.init()

    print("顔検出中...")
    result = fm.detect(image)
    if result is None:
        print("Error: 顔が検出されませんでした")
        return

    print(f"メッシュ: {len(fm.triangles)} 三角形, {len(fm.points)} 頂点")

    # アイメイクエリア特定
    areas = identify_eye_areas(fm, w, h)

    print("\n=== アイメイク メッシュID ===")
    result_data = []
    for name, mesh_ids in areas.items():
        print(f"\n{name}: {len(mesh_ids)} meshes")
        print(f"  mesh_id: {mesh_ids[:20]}{'...' if len(mesh_ids) > 20 else ''}")
        result_data.append({
            "name": name,
            "mesh_id": mesh_ids,
        })

    # target.json を更新
    target_path = PROJECT_ROOT / "target.json"
    with open(target_path) as f:
        target = json.load(f)

    target["eye"] = result_data

    with open(target_path, "w") as f:
        json.dump(target, f, indent=4, ensure_ascii=False)
    print(f"\ntarget.json 更新完了: {target_path}")

    # 確認用画像を生成
    generate_guide_images(fm, image, areas)


def generate_guide_images(fm, image, areas):
    """各部位のガイド画像を生成"""
    h, w = image.shape[:2]
    output_dir = Path(__file__).resolve().parent / "guide_images"
    output_dir.mkdir(exist_ok=True)

    # カラーマップ（部位ごとの色）
    colors = {
        "eyeshadow_base": (180, 160, 255),    # 薄紫（ベースカラー）
        "eyeshadow_crease": (140, 100, 220),  # 紫（メインカラー）
        "eyeliner": (80, 50, 180),             # 濃い紫（アイライン）
        "tear_bag": (200, 220, 255),          # 薄いピンク白（涙袋）
        "lower_outer": (120, 80, 180),        # 濃い紫（ポイントカラー）
    }

    labels = {
        "eyeshadow_base": "アイホール（ベースカラー）",
        "eyeshadow_crease": "二重幅（メインカラー）",
        "eyeliner": "目のキワ（アイライン・締め色）",
        "tear_bag": "涙袋（ハイライト）",
        "lower_outer": "下まぶた目尻（ポイントカラー）",
    }

    # --- 1. 各部位ごとの個別画像 ---
    for name, mesh_ids in areas.items():
        overlay = image.copy()
        color_bgr = colors.get(name, (200, 100, 100))
        mask = fm.build_mask(mesh_ids, w, h)

        # 半透明でオーバーレイ
        color_layer = np.zeros_like(image)
        color_layer[:] = color_bgr
        alpha = (mask * 0.5)[..., np.newaxis]
        overlay = (image.astype(np.float32) * (1 - alpha) + color_layer.astype(np.float32) * alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # 輪郭線を描画
        mask_u8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_bgr, 2)

        # ラベル追加
        label = labels.get(name, name)
        cv2.putText(overlay, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f"mesh count: {len(mesh_ids)}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # before/after
        comparison = np.hstack([image, overlay])
        out_path = output_dir / f"{name}.png"
        cv2.imwrite(str(out_path), comparison)
        print(f"  ガイド画像: {out_path.name}")

    # --- 2. 全部位統合画像 ---
    overlay_all = image.copy().astype(np.float32)
    for name, mesh_ids in areas.items():
        color_bgr = colors.get(name, (200, 100, 100))
        mask = fm.build_mask(mesh_ids, w, h)
        color_layer = np.zeros_like(image, dtype=np.float32)
        color_layer[:] = color_bgr
        alpha = (mask * 0.4)[..., np.newaxis]
        overlay_all = overlay_all * (1 - alpha) + color_layer * alpha

    overlay_all = np.clip(overlay_all, 0, 255).astype(np.uint8)

    # 凡例
    legend_y = 30
    for name, color_bgr in colors.items():
        label = labels.get(name, name)
        cv2.rectangle(overlay_all, (15, legend_y - 15), (35, legend_y + 5), color_bgr, -1)
        cv2.rectangle(overlay_all, (15, legend_y - 15), (35, legend_y + 5), (255, 255, 255), 1)
        cv2.putText(overlay_all, label, (45, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 30

    comparison_all = np.hstack([image, overlay_all])
    out_path = output_dir / "all_areas.png"
    cv2.imwrite(str(out_path), comparison_all)
    print(f"  統合画像: {out_path.name}")


if __name__ == "__main__":
    main()
