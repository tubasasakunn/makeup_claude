"""
アイメイク用メッシュID特定スクリプト

MediaPipeのランドマークIDを元に、目の各部位に対応するメッシュ三角形IDを特定する。
src/area のFace Mesh Selectorと同じ座標系（subdivision_level=1）で動作。

目の部位:
  - eyeshadow_base   : アイホール全体（眉下〜上まぶた）
  - eyeshadow_crease : 二重幅（上まぶたの上半分の帯状エリア）
  - eyeliner         : 目のキワ（上まぶたラッシュライン）
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
# MediaPipe FaceLandmarker 478点 ランドマークID
# =========================================================

# --- 右目 ---
RIGHT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173]   # 上まぶたライン
RIGHT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]  # 下まぶたライン
RIGHT_EYE_OUTLINE = [33, 246, 161, 160, 159, 158, 157, 173, 133,
                     155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_CORNER_OUTER = 133   # 目尻
RIGHT_EYE_CORNER_INNER = 33    # 目頭

# 右眉（下端ライン = アイホール上端）
RIGHT_EYEBROW_LOWER = [55, 65, 52, 53, 46]

# --- 左目 ---
LEFT_EYE_UPPER = [466, 388, 387, 386, 385, 384, 398]
LEFT_EYE_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]
LEFT_EYE_OUTLINE = [263, 466, 388, 387, 386, 385, 384, 398, 362,
                    382, 381, 380, 374, 373, 390, 249]
LEFT_EYE_CORNER_OUTER = 362
LEFT_EYE_CORNER_INNER = 263

# 左眉（下端ライン = アイホール上端）
LEFT_EYEBROW_LOWER = [285, 295, 282, 283, 276]

# アイホール中間ライン（眉下と目の間）
RIGHT_EYEHOLE_MID = [156, 28, 27, 29, 30, 247]
LEFT_EYEHOLE_MID = [383, 258, 257, 259, 260, 467]

# 涙袋の下端（目の下のやや下）
RIGHT_UNDER_EYE = [243, 112, 26, 22, 23, 24, 110, 25]
LEFT_UNDER_EYE = [463, 341, 256, 252, 253, 254, 339, 255]


def landmarks_to_points(fm, landmark_ids):
    """ランドマークIDリストから正規化座標リストを返す（x座標昇順でソート）"""
    pts = []
    for lid in landmark_ids:
        if lid < len(fm.points) and lid < 478:
            p = fm.points[lid]
            pts.append((p["x"], p["y"]))
    # x座標昇順（左→右）にソート
    pts.sort(key=lambda p: p[0])
    return pts


def make_polygon(upper_pts, lower_pts):
    """上辺と下辺の点列からポリゴンを作成（両辺は左→右でソート済み前提）"""
    # 上辺: 左→右、下辺: 右→左 で閉じたポリゴンにする
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
    # 点列の長さが異なる場合、リサンプリング
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
    """全てのアイメイクエリアのメッシュIDを特定"""
    areas = {}

    # ランドマーク座標を取得
    r_brow_lower = landmarks_to_points(fm, RIGHT_EYEBROW_LOWER)
    r_eye_upper = landmarks_to_points(fm, RIGHT_EYE_UPPER)
    r_eye_lower = landmarks_to_points(fm, RIGHT_EYE_LOWER)
    r_mid = landmarks_to_points(fm, RIGHT_EYEHOLE_MID)
    r_under = landmarks_to_points(fm, RIGHT_UNDER_EYE)

    l_brow_lower = landmarks_to_points(fm, LEFT_EYEBROW_LOWER)
    l_eye_upper = landmarks_to_points(fm, LEFT_EYE_UPPER)
    l_eye_lower = landmarks_to_points(fm, LEFT_EYE_LOWER)
    l_mid = landmarks_to_points(fm, LEFT_EYEHOLE_MID)
    l_under = landmarks_to_points(fm, LEFT_UNDER_EYE)

    # === 1. アイホール（ベースカラー） ===
    # 眉下端〜上まぶたの広いエリア
    r_eyehole_poly = make_polygon(r_brow_lower, r_eye_upper)
    l_eyehole_poly = make_polygon(l_brow_lower, l_eye_upper)
    r_eyehole = find_meshes_in_polygon(fm, r_eyehole_poly)
    l_eyehole = find_meshes_in_polygon(fm, l_eyehole_poly)
    areas["eyeshadow_base"] = sorted(set(r_eyehole + l_eyehole))

    # === 2. 二重幅（メインカラー） ===
    # アイホール中間ライン〜上まぶたの帯状エリア（アイホール下半分）
    r_crease_poly = make_polygon(r_mid, r_eye_upper)
    l_crease_poly = make_polygon(l_mid, l_eye_upper)
    r_crease = find_meshes_in_polygon(fm, r_crease_poly)
    l_crease = find_meshes_in_polygon(fm, l_crease_poly)
    areas["eyeshadow_crease"] = sorted(set(r_crease + l_crease))

    # === 3. 目のキワ（アイライン・締め色） ===
    # 上まぶたラッシュラインの細い帯
    # 上まぶたラインと、それを少し上にオフセットした線の間
    r_kiwa_upper = interpolate_points(r_eye_upper, r_mid, 0.25)
    l_kiwa_upper = interpolate_points(l_eye_upper, l_mid, 0.25)
    r_kiwa_poly = make_polygon(r_kiwa_upper, r_eye_upper)
    l_kiwa_poly = make_polygon(l_kiwa_upper, l_eye_upper)
    r_kiwa = find_meshes_in_polygon(fm, r_kiwa_poly)
    l_kiwa = find_meshes_in_polygon(fm, l_kiwa_poly)

    # 下まぶた側のキワも追加（目の輪郭に沿った細い帯）
    r_lower_kiwa_lower = interpolate_points(r_eye_lower, r_under, 0.2)
    l_lower_kiwa_lower = interpolate_points(l_eye_lower, l_under, 0.2)
    r_lower_kiwa_poly = make_polygon(r_eye_lower, r_lower_kiwa_lower)
    l_lower_kiwa_poly = make_polygon(l_eye_lower, l_lower_kiwa_lower)
    r_lower_kiwa = find_meshes_in_polygon(fm, r_lower_kiwa_poly)
    l_lower_kiwa = find_meshes_in_polygon(fm, l_lower_kiwa_poly)

    areas["eyeliner"] = sorted(set(r_kiwa + l_kiwa + r_lower_kiwa + l_lower_kiwa))

    # === 4. 涙袋（ハイライト） ===
    # 下まぶたライン〜その下の帯状エリア
    r_tear_poly = make_polygon(r_eye_lower, r_under)
    l_tear_poly = make_polygon(l_eye_lower, l_under)
    r_tear = find_meshes_in_polygon(fm, r_tear_poly)
    l_tear = find_meshes_in_polygon(fm, l_tear_poly)
    areas["tear_bag"] = sorted(set(r_tear + l_tear))

    # === 5. 下まぶた目尻1/3（ポイントカラー） ===
    # 涙袋のうち目尻側1/3のみ
    # 右目: 目尻=133(x大), 目頭=33(x小) → x座標が目尻寄りのもの
    r_outer_corner_x = fm.points[RIGHT_EYE_CORNER_OUTER]["x"]
    r_inner_corner_x = fm.points[RIGHT_EYE_CORNER_INNER]["x"]
    r_eye_width = abs(r_outer_corner_x - r_inner_corner_x)
    # 目尻側1/3の閾値
    r_threshold_x = r_outer_corner_x - r_eye_width * 0.4  # 目尻側40%

    l_outer_corner_x = fm.points[LEFT_EYE_CORNER_OUTER]["x"]
    l_inner_corner_x = fm.points[LEFT_EYE_CORNER_INNER]["x"]
    l_eye_width = abs(l_outer_corner_x - l_inner_corner_x)
    l_threshold_x = l_outer_corner_x + l_eye_width * 0.4  # 左目は x が小さい方が目尻

    # 涙袋メッシュの中から目尻側のみフィルタ
    r_outer = []
    for mid in r_tear:
        a, b, c = fm.triangles[mid]
        cx = (fm.points[a]["x"] + fm.points[b]["x"] + fm.points[c]["x"]) / 3
        if cx >= r_threshold_x:
            r_outer.append(mid)

    l_outer = []
    for mid in l_tear:
        a, b, c = fm.triangles[mid]
        cx = (fm.points[a]["x"] + fm.points[b]["x"] + fm.points[c]["x"]) / 3
        if cx <= l_threshold_x:
            l_outer.append(mid)

    areas["lower_outer"] = sorted(set(r_outer + l_outer))

    return areas


def main():
    # base.png を使用
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

    # カラーマップ（BGR）
    colors = {
        "eyeshadow_base": (200, 180, 255),     # 薄ピンク
        "eyeshadow_crease": (160, 100, 230),   # ピンク紫
        "eyeliner": (60, 30, 150),             # 濃いブラウン赤
        "tear_bag": (220, 200, 255),           # 薄いピンク
        "lower_outer": (150, 80, 200),         # 紫ピンク
    }

    labels = {
        "eyeshadow_base": "1. Eye hole (Base color)",
        "eyeshadow_crease": "2. Crease (Main color)",
        "eyeliner": "3. Lash line (Eyeliner)",
        "tear_bag": "4. Tear bag (Highlight)",
        "lower_outer": "5. Lower outer (Point color)",
    }

    labels_ja = {
        "eyeshadow_base": "アイホール（ベースカラー）",
        "eyeshadow_crease": "二重幅（メインカラー）",
        "eyeliner": "目のキワ（アイライン・締め色）",
        "tear_bag": "涙袋（ハイライト）",
        "lower_outer": "下まぶた目尻（ポイントカラー）",
    }

    # --- 1. 各部位ごとの個別画像 ---
    for name, mesh_ids in areas.items():
        color_bgr = colors.get(name, (200, 100, 100))

        # マスク生成
        mask = fm.build_mask(mesh_ids, w, h)

        # ガウシアンブラーで境界を滑らかに
        mask_smooth = cv2.GaussianBlur(mask, (7, 7), 2)

        # 半透明オーバーレイ
        color_layer = np.zeros_like(image)
        color_layer[:] = color_bgr
        alpha = (mask_smooth * 0.5)[..., np.newaxis]
        overlay = (image.astype(np.float32) * (1 - alpha) + color_layer.astype(np.float32) * alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # 輪郭線を描画
        mask_u8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_bgr, 2)

        # ラベル追加（右上）
        label_en = labels.get(name, name)
        label_ja = labels_ja.get(name, name)
        cv2.putText(overlay, label_en, (w - 380, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, f"{len(mesh_ids)} meshes", (w - 380, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # before/after
        comparison = np.hstack([image, overlay])
        out_path = output_dir / f"{name}.png"
        cv2.imwrite(str(out_path), comparison)
        print(f"  ガイド画像: {out_path.name} ({len(mesh_ids)} meshes)")

    # --- 2. 全部位統合画像 ---
    overlay_all = image.copy().astype(np.float32)
    for name, mesh_ids in areas.items():
        color_bgr = colors.get(name, (200, 100, 100))
        mask = fm.build_mask(mesh_ids, w, h)
        mask_smooth = cv2.GaussianBlur(mask, (5, 5), 1.5)
        color_layer = np.zeros_like(image, dtype=np.float32)
        color_layer[:] = color_bgr
        alpha = (mask_smooth * 0.45)[..., np.newaxis]
        overlay_all = overlay_all * (1 - alpha) + color_layer * alpha

    overlay_all = np.clip(overlay_all, 0, 255).astype(np.uint8)

    # 凡例（右上、見やすいサイズ）
    legend_x = w - 400
    legend_y = 30
    for name in areas:
        color_bgr = colors.get(name, (200, 100, 100))
        label_en = labels.get(name, name)
        # 背景ボックス
        cv2.rectangle(overlay_all, (legend_x - 5, legend_y - 18),
                      (w - 10, legend_y + 8), (0, 0, 0), -1)
        cv2.rectangle(overlay_all, (legend_x - 5, legend_y - 18),
                      (w - 10, legend_y + 8), (100, 100, 100), 1)
        # 色見本
        cv2.rectangle(overlay_all, (legend_x, legend_y - 12),
                      (legend_x + 20, legend_y + 2), color_bgr, -1)
        cv2.rectangle(overlay_all, (legend_x, legend_y - 12),
                      (legend_x + 20, legend_y + 2), (255, 255, 255), 1)
        # テキスト
        cv2.putText(overlay_all, label_en, (legend_x + 28, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 32

    comparison_all = np.hstack([image, overlay_all])
    out_path = output_dir / "all_areas.png"
    cv2.imwrite(str(out_path), comparison_all)
    print(f"  統合画像: {out_path.name}")


if __name__ == "__main__":
    main()
