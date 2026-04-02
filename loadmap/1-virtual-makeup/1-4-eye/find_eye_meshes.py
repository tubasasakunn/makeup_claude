"""
アイメイク用メッシュID特定スクリプト

MediaPipeのランドマークIDを元に、目の各部位に対応するメッシュ三角形IDを特定する。
src/area のFace Mesh Selectorと同じ座標系（subdivision_level=1）で動作。

右目のみ計算し、find_mirror_meshes で左目をミラー生成して左右対称を保証する。

目の部位:
  - eyeshadow_base   : アイホール全体（眉下〜上まぶた）
  - eyeshadow_crease : 二重幅（上まぶたの上半分の帯状エリア）
  - eyeliner         : 目のキワ（上下まぶたラッシュライン、眼球部分除外）
  - tear_bag         : 涙袋（下まぶた全体）
  - lower_outer      : 下まぶた目尻1/3（ポイントカラー）
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
# MediaPipe FaceLandmarker 478点 ランドマークID（右目のみ）
# =========================================================

RIGHT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173]   # 上まぶたライン
RIGHT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]  # 下まぶたライン
RIGHT_EYE_OUTLINE = [33, 246, 161, 160, 159, 158, 157, 173, 133,
                     155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_CORNER_OUTER = 33    # 目尻（顔の外側）
RIGHT_EYE_CORNER_INNER = 133   # 目頭（鼻側）

# 右眉（下端ライン = アイホール上端）
RIGHT_EYEBROW_LOWER = [55, 65, 52, 53, 46]

# アイホール中間ライン（眉下と目の間）
RIGHT_EYEHOLE_MID = [156, 28, 27, 29, 30, 247]

# --- 左目 ---
LEFT_EYE_UPPER = [466, 388, 387, 386, 385, 384, 398]
LEFT_EYE_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]
LEFT_EYE_CORNER_OUTER = 263   # 目尻（顔の外側）
LEFT_EYE_CORNER_INNER = 362   # 目頭（鼻側）

# 涙袋の下端（目の下のやや下）
RIGHT_UNDER_EYE = [243, 112, 26, 22, 23, 24, 110, 25]


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
    """上辺と下辺の点列からポリゴンを作成（両辺は左→右でソート済み前提）"""
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


def interpolate_points(pts_a, pts_b, t):
    """2つの点列を t (0-1) で線形補間した点列を返す"""
    n = max(len(pts_a), len(pts_b))
    result = []
    for i in range(n):
        ia = int(i * (len(pts_a) - 1) / max(n - 1, 1))
        ib = int(i * (len(pts_b) - 1) / max(n - 1, 1))
        x = pts_a[ia][0] * (1 - t) + pts_b[ib][0] * t
        y = pts_a[ia][1] * (1 - t) + pts_b[ib][1] * t
        result.append((x, y))
    return result


def identify_eye_areas(fm, w, h):
    """全てのアイメイクエリアのメッシュIDを特定（右目のみ計算→ミラーで左目）"""
    areas = {}

    # 右目のランドマーク座標を取得
    r_brow_lower = landmarks_to_points(fm, RIGHT_EYEBROW_LOWER)
    r_eye_upper = landmarks_to_points(fm, RIGHT_EYE_UPPER)
    r_eye_lower = landmarks_to_points(fm, RIGHT_EYE_LOWER)
    r_mid = landmarks_to_points(fm, RIGHT_EYEHOLE_MID)
    r_under = landmarks_to_points(fm, RIGHT_UNDER_EYE)

    # 眼球周辺のメッシュ（除外用）— 周回順序を維持（ソートしない）
    # 目の開口部が狭いため、ポリゴンを拡大して目のキワに接するメッシュも除外
    r_eye_outline = landmarks_to_points(fm, RIGHT_EYE_OUTLINE, sort_by_x=False)
    r_eye_outline_arr = np.array(r_eye_outline, dtype=np.float64)
    # ポリゴンを重心から20%拡大（目の周囲の皮膚ギリギリのメッシュも除外）
    centroid = r_eye_outline_arr.mean(axis=0)
    r_eye_expanded = centroid + (r_eye_outline_arr - centroid) * 1.2
    r_eyeball_meshes = set(find_meshes_in_polygon(fm, r_eye_expanded))

    # === 1. アイホール（ベースカラー） ===
    r_eyehole_poly = make_polygon(r_brow_lower, r_eye_upper)
    r_eyehole = find_meshes_in_polygon(fm, r_eyehole_poly)
    r_eyehole = [m for m in r_eyehole if m not in r_eyeball_meshes]

    # === 2. 二重幅（メインカラー） ===
    # 眼球除外なし（二重幅は目のキワに近いため除外すると削れすぎる）
    r_crease_poly = make_polygon(r_mid, r_eye_upper)
    r_crease = find_meshes_in_polygon(fm, r_crease_poly)

    # === 3. 目のキワ（アイライン） ===
    # メッシュではなくランドマークに沿ったポリラインで定義
    # 上下それぞれ目頭側・中央・目尻側に3分割
    r_upper_pts = landmarks_to_points(fm, RIGHT_EYE_UPPER)
    r_lower_pts = landmarks_to_points(fm, RIGHT_EYE_LOWER)
    l_upper_pts = landmarks_to_points(fm, LEFT_EYE_UPPER)
    l_lower_pts = landmarks_to_points(fm, LEFT_EYE_LOWER)

    def split_into_thirds(pts):
        """点列を3分割（目頭・中央・目尻）"""
        n = len(pts)
        t1 = n // 3
        t2 = 2 * n // 3
        return {
            "inner": pts[:t1 + 1],      # 目頭側（オーバーラップ1点）
            "center": pts[t1:t2 + 1],   # 中央
            "outer": pts[t2:],           # 目尻側
        }

    eyeliner_lines = {
        "upper": {
            "right": RIGHT_EYE_UPPER,
            "left": LEFT_EYE_UPPER,
            "right_parts": split_into_thirds(r_upper_pts),
            "left_parts": split_into_thirds(l_upper_pts),
        },
        "lower": {
            "right": RIGHT_EYE_LOWER,
            "left": LEFT_EYE_LOWER,
            "right_parts": split_into_thirds(r_lower_pts),
            "left_parts": split_into_thirds(l_lower_pts),
        },
        "thickness": 3,
    }

    # === 4. 涙袋（ハイライト） ===
    r_tear_poly = make_polygon(r_eye_lower, r_under)
    r_tear = find_meshes_in_polygon(fm, r_tear_poly)
    r_tear = [m for m in r_tear if m not in r_eyeball_meshes]

    # === 5. 下まぶた目尻1/3（ポイントカラー） ===
    # 右目: 目尻(outer)はx小さい側、目頭(inner)はx大きい側
    r_outer_corner_x = fm.points[RIGHT_EYE_CORNER_OUTER]["x"]  # x小
    r_inner_corner_x = fm.points[RIGHT_EYE_CORNER_INNER]["x"]  # x大
    r_eye_width = abs(r_inner_corner_x - r_outer_corner_x)
    # 目尻側40%: outer_corner_x から内側に40%分まで
    r_threshold_x = r_outer_corner_x + r_eye_width * 0.4

    r_outer = []
    for mid in r_tear:
        a, b, c = fm.triangles[mid]
        cx = (fm.points[a]["x"] + fm.points[b]["x"] + fm.points[c]["x"]) / 3
        if cx <= r_threshold_x:
            r_outer.append(mid)

    # === ミラーで左目を生成（左右完全対称） ===
    # ミラー対応がないメッシュを除外して完全対称にする
    def symmetric_pair(right_meshes):
        """右目メッシュからミラーし、ラウンドトリップで対称ペアのみ残す"""
        r_set = set(right_meshes)
        l_set = fm.find_mirror_meshes(r_set)
        # 左→右に戻して、元の右に存在するもののみ残す
        r_roundtrip = fm.find_mirror_meshes(l_set)
        r_valid = r_set & r_roundtrip
        l_valid = fm.find_mirror_meshes(r_valid)
        return sorted(r_valid), sorted(l_valid)

    r_eyehole, l_eyehole = symmetric_pair(r_eyehole)
    r_crease, l_crease = symmetric_pair(r_crease)
    r_tear, l_tear = symmetric_pair(r_tear)

    # lower_outer: symmetric_pair + ミラー後にも目尻側フィルタを適用
    r_outer, l_outer_candidates = symmetric_pair(r_outer)
    # 左目: 目尻(outer)はx大きい側、目頭(inner)はx小さい側
    l_outer_corner_x = fm.points[LEFT_EYE_CORNER_OUTER]["x"]  # x大
    l_inner_corner_x = fm.points[LEFT_EYE_CORNER_INNER]["x"]  # x小
    l_eye_width = abs(l_outer_corner_x - l_inner_corner_x)
    l_threshold_x = l_outer_corner_x - l_eye_width * 0.4
    l_outer = [mid for mid in l_outer_candidates
               if (fm.points[fm.triangles[mid][0]]["x"] + fm.points[fm.triangles[mid][1]]["x"] + fm.points[fm.triangles[mid][2]]["x"]) / 3 >= l_threshold_x]

    areas["eyeshadow_base"] = sorted(set(r_eyehole) | set(l_eyehole))
    areas["eyeshadow_crease"] = sorted(set(r_crease) | set(l_crease))
    areas["tear_bag"] = sorted(set(r_tear) | set(l_tear))
    areas["lower_outer"] = sorted(set(r_outer) | set(l_outer))

    return areas, eyeliner_lines


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

    areas, eyeliner_lines = identify_eye_areas(fm, w, h)

    print("\n=== アイメイク メッシュID ===")
    result_data = []
    for name, mesh_ids in areas.items():
        print(f"\n{name}: {len(mesh_ids)} meshes")
        print(f"  mesh_id: {mesh_ids[:20]}{'...' if len(mesh_ids) > 20 else ''}")
        result_data.append({
            "name": name,
            "mesh_id": mesh_ids,
        })

    # eyelinerはランドマークID + 太さで保存
    result_data.append({
        "name": "eyeliner",
        "type": "polyline",
        "upper_landmarks": {
            "right": eyeliner_lines["upper"]["right"],
            "left": eyeliner_lines["upper"]["left"],
        },
        "lower_landmarks": {
            "right": eyeliner_lines["lower"]["right"],
            "left": eyeliner_lines["lower"]["left"],
        },
        "thickness": eyeliner_lines["thickness"],
    })
    print(f"\neyeliner: polyline (upper: {len(eyeliner_lines['upper']['right'])} + lower: {len(eyeliner_lines['lower']['right'])} landmarks x 2 eyes)")

    target_path = PROJECT_ROOT / "target.json"
    with open(target_path) as f:
        target = json.load(f)

    target["eye"] = result_data

    with open(target_path, "w") as f:
        json.dump(target, f, indent=4, ensure_ascii=False)
    print(f"\ntarget.json 更新完了: {target_path}")

    generate_guide_images(fm, image, areas, eyeliner_lines)


def build_eyeliner_mask(fm, eyeliner_lines, w, h):
    """ランドマークに沿ったポリラインでアイラインマスクを生成（外側にオフセット）"""
    mask = np.zeros((h, w), dtype=np.float32)
    face_w = abs(fm.landmarks_px[234][0] - fm.landmarks_px[454][0])
    thickness = max(2, int(face_w * 0.012 * eyeliner_lines["thickness"]))
    # 外側オフセット量（太さの半分＋少し余裕）
    offset_px = thickness // 2 + 1

    # 目の中心を算出（オフセット方向の基準）
    for side in ["right", "left"]:
        upper_ids = eyeliner_lines["upper"][side]
        lower_ids = eyeliner_lines["lower"][side]
        all_ids = upper_ids + lower_ids
        eye_center_y = np.mean([fm.points[lid]["y"] * h for lid in all_ids if lid < 478])
        eye_center_x = np.mean([fm.points[lid]["x"] * w for lid in all_ids if lid < 478])

        for part in ["upper", "lower"]:
            landmark_ids = eyeliner_lines[part][side]
            pts = []
            for lid in landmark_ids:
                if lid < 478:
                    p = fm.points[lid]
                    px, py = p["x"] * w, p["y"] * h
                    # 目の中心から外側にオフセット
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


def build_eyeliner_part_mask(fm, eyeliner_lines, w, h, part, section):
    """eyelinerの特定パーツのマスクを生成（part=upper/lower, section=inner/center/outer）"""
    mask = np.zeros((h, w), dtype=np.float32)
    face_w = abs(fm.landmarks_px[234][0] - fm.landmarks_px[454][0])
    thickness = max(2, int(face_w * 0.012 * eyeliner_lines["thickness"]))
    offset_px = thickness // 2 + 1

    for side in ["right", "left"]:
        upper_ids = eyeliner_lines["upper"][side]
        lower_ids = eyeliner_lines["lower"][side]
        all_ids = upper_ids + lower_ids
        eye_center_y = np.mean([fm.points[lid]["y"] * h for lid in all_ids if lid < 478])
        eye_center_x = np.mean([fm.points[lid]["x"] * w for lid in all_ids if lid < 478])

        part_pts = eyeliner_lines[part][f"{side}_parts"][section]
        pts = []
        for (px_n, py_n) in part_pts:
            px, py = px_n * w, py_n * h
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


def generate_guide_images(fm, image, areas, eyeliner_lines):
    """各部位のガイド画像を生成"""
    h, w = image.shape[:2]
    output_dir = Path(__file__).resolve().parent / "guide_images"
    output_dir.mkdir(exist_ok=True)

    # カラーマップ（BGR）
    colors = {
        "eyeshadow_base": (200, 180, 255),     # 薄ピンク
        "eyeshadow_crease": (160, 100, 230),   # ピンク紫
        "eyeliner": (60, 30, 150),             # 濃いブラウン赤
        "tear_bag": (180, 130, 220),           # 紫ピンク（視認性向上）
        "lower_outer": (150, 80, 200),         # 紫ピンク
    }

    labels = {
        "eyeshadow_base": "1. Eye hole (Base color)",
        "eyeshadow_crease": "2. Crease (Main color)",
        "eyeliner": "3. Lash line (Eyeliner)",
        "tear_bag": "4. Tear bag (Highlight)",
        "lower_outer": "5. Lower outer (Point color)",
    }

    # 全エリア名（eyeliner含む描画順）
    all_area_names = list(areas.keys())
    # eyelinerをcrease の次に挿入
    eyeliner_idx = all_area_names.index("eyeshadow_crease") + 1 if "eyeshadow_crease" in all_area_names else 2
    all_area_names.insert(eyeliner_idx, "eyeliner")

    # eyeliner のマスクを事前計算
    eyeliner_mask = build_eyeliner_mask(fm, eyeliner_lines, w, h)

    # --- 1. 各部位ごとの個別画像 ---
    for name in all_area_names:
        color_bgr = colors.get(name, (200, 100, 100))

        if name == "eyeliner":
            mask = eyeliner_mask
        else:
            mask = fm.build_mask(areas[name], w, h)

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
        cv2.putText(overlay, label_en, (w - 380, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if name == "eyeliner":
            cv2.putText(overlay, "polyline", (w - 380, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(overlay, f"{len(areas[name])} meshes", (w - 380, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        comparison = np.hstack([image, overlay])
        out_path = output_dir / f"{name}.png"
        cv2.imwrite(str(out_path), comparison)
        print(f"  ガイド画像: {out_path.name}")

    # --- 1b. eyeliner サブパーツ画像 ---
    eyeliner_sub_colors = {
        "upper_inner": (80, 40, 180),
        "upper_center": (60, 30, 150),
        "upper_outer": (40, 20, 120),
        "lower_inner": (120, 60, 200),
        "lower_center": (100, 50, 180),
        "lower_outer": (80, 40, 160),
    }
    eyeliner_sub_labels = {
        "upper_inner": "3a. Upper inner",
        "upper_center": "3b. Upper center",
        "upper_outer": "3c. Upper outer",
        "lower_inner": "3d. Lower inner",
        "lower_center": "3e. Lower center",
        "lower_outer": "3f. Lower outer",
    }
    for part in ["upper", "lower"]:
        for section in ["inner", "center", "outer"]:
            sub_name = f"{part}_{section}"
            color_bgr = eyeliner_sub_colors[sub_name]
            mask = build_eyeliner_part_mask(fm, eyeliner_lines, w, h, part, section)
            mask_smooth = cv2.GaussianBlur(mask, (5, 5), 1.5)
            color_layer = np.zeros_like(image)
            color_layer[:] = color_bgr
            alpha = (mask_smooth * 0.6)[..., np.newaxis]
            overlay = (image.astype(np.float32) * (1 - alpha) + color_layer.astype(np.float32) * alpha)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            label_en = eyeliner_sub_labels[sub_name]
            cv2.putText(overlay, label_en, (w - 380, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            comparison = np.hstack([image, overlay])
            out_path = output_dir / f"eyeliner_{sub_name}.png"
            cv2.imwrite(str(out_path), comparison)
            print(f"  ガイド画像: {out_path.name}")

    # --- 2. 全部位統合画像 ---
    overlay_all = image.copy().astype(np.float32)
    for name in all_area_names:
        color_bgr = colors.get(name, (200, 100, 100))
        if name == "eyeliner":
            mask = eyeliner_mask
        else:
            mask = fm.build_mask(areas[name], w, h)
        mask_smooth = cv2.GaussianBlur(mask, (5, 5), 1.5)
        color_layer = np.zeros_like(image, dtype=np.float32)
        color_layer[:] = color_bgr
        alpha = (mask_smooth * 0.5)[..., np.newaxis]
        overlay_all = overlay_all * (1 - alpha) + color_layer * alpha

    overlay_all = np.clip(overlay_all, 0, 255).astype(np.uint8)

    legend_x = w - 400
    legend_y = 30
    for name in all_area_names:
        color_bgr = colors.get(name, (200, 100, 100))
        label_en = labels.get(name, name)
        cv2.rectangle(overlay_all, (legend_x - 5, legend_y - 18),
                      (w - 10, legend_y + 8), (0, 0, 0), -1)
        cv2.rectangle(overlay_all, (legend_x - 5, legend_y - 18),
                      (w - 10, legend_y + 8), (100, 100, 100), 1)
        cv2.rectangle(overlay_all, (legend_x, legend_y - 12),
                      (legend_x + 20, legend_y + 2), color_bgr, -1)
        cv2.rectangle(overlay_all, (legend_x, legend_y - 12),
                      (legend_x + 20, legend_y + 2), (255, 255, 255), 1)
        cv2.putText(overlay_all, label_en, (legend_x + 28, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 32

    comparison_all = np.hstack([image, overlay_all])
    out_path = output_dir / "all_areas.png"
    cv2.imwrite(str(out_path), comparison_all)
    print(f"  統合画像: {out_path.name}")


if __name__ == "__main__":
    main()
