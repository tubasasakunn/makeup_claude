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
import json
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
    # 眉消しは強めに拡張（元眉を確実にカバー、二重眉を防止）
    expand_px = max(5, int(face_h * 0.025))

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
        "peak_position": 0.55,
        "peak_height_ratio": 0.15,
        "tail_height_ratio": 0.0,
        "thickness_ratio": 0.5,
        "length_ratio": 1.0,
        "desc": "ストレート眉（直線的・卵型向け）",
    },
    "arch": {
        "peak_position": 0.62,
        "peak_height_ratio": 0.55,
        "tail_height_ratio": 0.0,
        "thickness_ratio": 0.48,
        "length_ratio": 1.0,
        "desc": "アーチ眉（緩やかカーブ・丸顔向け）",
    },
    "parallel": {
        "peak_position": 0.5,
        "peak_height_ratio": 0.03,
        "tail_height_ratio": 0.0,
        "thickness_ratio": 0.58,
        "length_ratio": 1.0,
        "desc": "平行眉（水平・太め・面長向け）",
    },
    "corner": {
        "peak_position": 0.6,
        "peak_height_ratio": 0.7,
        "tail_height_ratio": 0.0,
        "thickness_ratio": 0.48,
        "length_ratio": 0.95,
        "desc": "コーナー眉（上がり眉・ベース型向け）",
    },
    "natural": {
        "peak_position": 0.6,
        "peak_height_ratio": 0.3,
        "tail_height_ratio": 0.0,
        "thickness_ratio": 0.5,
        "length_ratio": 0.9,
        "desc": "ナチュラル眉（自然・逆三角向け）",
    },
}

# デフォルト設定
DEFAULT_EYEBROW_TYPE = "straight"
DEFAULT_EYEBROW_COLOR_RGB = (85, 60, 45)   # 濃いめブラウン
DEFAULT_EYEBROW_INTENSITY = 0.75           # ハッキリ


def compute_brow_anchors(fm: FaceMesh, side: str = "right") -> dict:
    """眉の基準点を計算（左右対称化済み）

    左右のランドマークから各値を取得し、平均化して顔中心で対称にする。
    眉山位置は虹彩の外縁を基準に配置。
    """
    # 両側のランドマークを取得
    sides_data = {}
    for s, cfg in [
        ("right", {
            "nose_wing": 64, "inner_eye": 133, "outer_eye": 33,
            "eye_top": 159, "eye_bot": 145,
            "brow_head": 55, "brow_tail": 46,
            "iris": [468, 469, 470, 471, 472],
        }),
        ("left", {
            "nose_wing": 294, "inner_eye": 362, "outer_eye": 263,
            "eye_top": 386, "eye_bot": 374,
            "brow_head": 285, "brow_tail": 276,
            "iris": [473, 474, 475, 476, 477],
        }),
    ]:
        d = {k: fm.landmarks_px[v].astype(float) for k, v in cfg.items() if isinstance(v, int)}
        d["iris_center"] = np.mean([fm.landmarks_px[i].astype(float) for i in cfg["iris"]], axis=0)
        d["eye_height"] = abs(d["eye_top"][1] - d["eye_bot"][1])
        sides_data[s] = d

    # 顔中心X
    face_cx = fm.landmarks_px[1][0].astype(float)

    # 左右の値を平均化して対称にする
    r, l = sides_data["right"], sides_data["left"]

    # 眉の高さ (Y): 両目の上端から eye_height * 1.85 上
    avg_eye_height = (r["eye_height"] + l["eye_height"]) / 2
    avg_eye_top_y = (r["eye_top"][1] + l["eye_top"][1]) / 2
    head_y = avg_eye_top_y - avg_eye_height * 1.85

    # 眉頭の中心からのオフセット (平均化)
    r_head_off = abs(r["brow_head"][0] - face_cx)
    l_head_off = abs(l["brow_head"][0] - face_cx)
    avg_head_off = (r_head_off + l_head_off) / 2

    # 眉尻の中心からのオフセット
    # 黄金比ルール vs 実ランドマーク の広い方
    def tail_offset(sd):
        nose = sd["nose_wing"]
        eye = sd["outer_eye"]
        dx = eye[0] - nose[0]
        dy = eye[1] - nose[1]
        if abs(dy) < 1e-3:
            golden_x = eye[0] + (eye[0] - sd["inner_eye"][0]) * 0.3
        else:
            t = (head_y - nose[1]) / dy
            golden_x = nose[0] + t * dx
        return max(abs(golden_x - face_cx), abs(sd["brow_tail"][0] - face_cx))

    r_tail_off = tail_offset(r)
    l_tail_off = tail_offset(l)
    avg_tail_off = (r_tail_off + l_tail_off) / 2

    # 眉山の位置: 虹彩外縁の真上（中心からのオフセットで対称化）
    r_iris_off = abs(r["iris_center"][0] - face_cx)
    l_iris_off = abs(l["iris_center"][0] - face_cx)
    avg_iris_off = (r_iris_off + l_iris_off) / 2

    # 対称な座標を生成
    # 右眉: 顔中心のX小(左)側にある、head(鼻寄り)→tail(外側)
    # 左眉: 顔中心のX大(右)側にある、head(鼻寄り)→tail(外側)
    if side == "right":
        head_x = face_cx - avg_head_off   # 右=X小、鼻寄り(中心に近い)
        tail_x = face_cx - avg_tail_off   # 右=X小、外側(中心から遠い)
        iris_x = face_cx - avg_iris_off
    else:
        head_x = face_cx + avg_head_off   # 左=X大、鼻寄り
        tail_x = face_cx + avg_tail_off   # 左=X大、外側
        iris_x = face_cx + avg_iris_off

    # 眉山のt値を虹彩位置から逆算
    brow_len = abs(tail_x - head_x)
    if brow_len > 1e-3:
        iris_peak_t = abs(iris_x - head_x) / brow_len
        iris_peak_t = np.clip(iris_peak_t, 0.5, 0.75)
    else:
        iris_peak_t = 0.62

    return {
        "head": np.array([head_x, head_y], dtype=np.float64),
        "tail": np.array([tail_x, head_y], dtype=np.float64),
        "eye_height": avg_eye_height,
        "brow_length": brow_len,
        "iris_peak_t": iris_peak_t,
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


# タイプごとに先端の尖り度を制御
SHARP_BROW_TYPES = {"corner", "arch", "straight"}


def _asymmetric_centerline(head: np.ndarray, peak: np.ndarray, tail: np.ndarray,
                           n: int = 120, head_flat: float = 0.5) -> np.ndarray:
    """頭側を平らに保ち、眉山で折れ曲がり、眉尻に向かって下がる非対称な中心線

    - Head → Peak 区間: head付近を平らに保ち、peak近くで上昇（ease-in）
    - Peak → Tail 区間: peak から tail に向かって下降（ease-out）

    head_flat: 頭側の「平らな区間」の比率 (0-1)
               0.5 なら、head-peak 区間の前半50%は head_y に近い位置を維持
    """
    # head → peak と peak → tail に分割
    # 長さの比で点数配分
    head_peak_len = np.linalg.norm(peak - head)
    peak_tail_len = np.linalg.norm(tail - peak)
    total = head_peak_len + peak_tail_len
    if total < 1e-3:
        return np.tile(head, (n, 1))

    n1 = max(2, int(n * head_peak_len / total))
    n2 = n - n1 + 1

    # Section 1: Head → Peak
    # 二次Bezier: 制御点を head_y 付近・peak 寄りに置くと、頭側が平らになる
    ctrl1 = np.array([
        head[0] + (peak[0] - head[0]) * head_flat,  # peak 方向に head_flat 分進む
        head[1],                                      # Y は head のまま（平ら）
    ])
    ts1 = np.linspace(0, 1, n1)
    seg1 = np.zeros((n1, 2))
    for i, t in enumerate(ts1):
        seg1[i] = (1 - t) ** 2 * head + 2 * (1 - t) * t * ctrl1 + t ** 2 * peak

    # Section 2: Peak → Tail
    # 二次Bezier: 制御点を peak_y の少し下、tail 寄り
    # peak を通過後すぐに下降（ease-out）
    peak_y_level_ctrl = np.array([
        peak[0] + (tail[0] - peak[0]) * 0.35,  # peak から 35% 進んだ位置
        peak[1] + (tail[1] - peak[1]) * 0.7,   # Y は tail 寄り (急降下)
    ])
    ts2 = np.linspace(0, 1, n2)
    seg2 = np.zeros((n2, 2))
    for i, t in enumerate(ts2):
        seg2[i] = (1 - t) ** 2 * peak + 2 * (1 - t) * t * peak_y_level_ctrl + t ** 2 * tail

    # 結合 (seg1 の最後と seg2 の最初は peak で同じ)
    center_line = np.vstack([seg1, seg2[1:]])

    # n 点にリサンプル
    if len(center_line) != n:
        idx = np.linspace(0, len(center_line) - 1, n).astype(int)
        center_line = center_line[idx]
    return center_line


def _catmull_rom_centerline(head: np.ndarray, peak: np.ndarray, tail: np.ndarray,
                            peak_t: float, n: int = 100) -> np.ndarray:
    """5制御点 + Catmull-Romスプラインで滑らかな中心線を生成

    制御点: [仮想前, head, peak, peak_mid, tail, 仮想後]
    """
    peak_mid_x = peak[0] + (tail[0] - peak[0]) * 0.5
    peak_mid_y = (peak[1] + tail[1]) / 2
    peak_mid = np.array([peak_mid_x, peak_mid_y])

    # 仮想前点: head の手前に head→peak と対称な点
    v_pre = head - (peak - head) * 0.3
    # 仮想後点: tail の後ろ
    v_post = tail + (tail - peak_mid) * 0.3

    cps = [v_pre, head, peak, peak_mid, tail, v_post]

    def segment(p0, p1, p2, p3, n_pts):
        ts = np.linspace(0, 1, n_pts)
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

    # 3区間を均等サンプリング
    n_per_seg = n // 3 + 1
    segments = []
    for i in range(len(cps) - 3):
        seg = segment(*cps[i:i + 4], n_pts=n_per_seg)
        segments.append(seg if i == 0 else seg[1:])
    center_line = np.vstack(segments)
    # 最終的に n 点にリサンプル
    if len(center_line) != n:
        idx = np.linspace(0, len(center_line) - 1, n).astype(int)
        center_line = center_line[idx]
    return center_line


def _taper(t: float, sharp: bool = False) -> float:
    """眉の太さを t で調整

    sharp: True の場合、85%までほぼ幅を保ち、最後の15%で急降下して
          キリッと尖らせる (2段階テーパー)
    sharp: False の場合、終端 0.35 で丸く留める (rounded)
    """
    import math
    if t < 0.3:
        # 眉頭: 0.6 → 1.0
        x = t / 0.3
        return 0.6 + 0.4 * (1 - math.cos(x * math.pi / 2))

    if sharp:
        # sharp: 30-90% で 1.0 → 0.7 で緩やかに、90-100% で ease-out で 0 へ
        if t < 0.9:
            x = (t - 0.3) / 0.6  # 0..1
            return 1.0 - x * 0.3
        else:
            x = min(1.0, max(0.0, (t - 0.9) / 0.1))  # 0..1
            # ease-out: 0.7 → 0 だが、 98% まで 0.2 程度残す
            return 0.7 * (1 - x ** 0.5)
    else:
        # rounded: 30-90% で 1.0 → 0.35、90-100% で 0.35 → 0.1 （欠け感解消）
        if t < 0.9:
            x = (t - 0.3) / 0.6  # 0..1
            return 1.0 - x * 0.65
        else:
            x = min(1.0, max(0.0, (t - 0.9) / 0.1))
            return 0.35 * (1 - x) + 0.1 * x


def _peak_bump(t: float, peak_t: float, peak_height: float,
               sigma_left: float = 0.25, sigma_right: float = 0.18) -> float:
    """眉山の盛り上がり量を計算（非対称ガウシアンバンプ）

    t < peak_t: sigma_left (緩やかな上昇)
    t > peak_t: sigma_right (急峻な下降)

    返り値: t 位置でのピーク上昇量（0 〜 peak_height）
    """
    import math
    sigma = sigma_left if t < peak_t else sigma_right
    x = (t - peak_t) / max(sigma, 1e-6)
    return peak_height * math.exp(-x * x)


def _corner_peak(t: float, peak_t: float, peak_height: float,
                 rise_len: float = 0.18, fall_len: float = 0.10) -> float:
    """コーナー眉用: 区分線形で鋭い折れ曲がりを作る

    - 0 〜 (peak_t - rise_len): 0 (平坦)
    - (peak_t - rise_len) 〜 peak_t: 線形に 0 → peak_height
    - peak_t 〜 (peak_t + fall_len): 線形に peak_height → 0
    - (peak_t + fall_len) 〜 1: 0 (平坦)
    """
    rise_start = peak_t - rise_len
    fall_end = peak_t + fall_len
    if t <= rise_start:
        return 0.0
    if t <= peak_t:
        return peak_height * (t - rise_start) / rise_len
    if t <= fall_end:
        return peak_height * (1.0 - (t - peak_t) / fall_len)
    return 0.0


def _arch_peak(t: float, peak_t: float, peak_height: float) -> float:
    """アーチ眉用: 眉頭から滑らかに立ち上がる広いアーチ"""
    import math
    # 全区間で cos^2 に近い形で盛り上げる（広く、ピーク周辺がなめらか）
    sigma = 0.32
    x = (t - peak_t) / sigma
    return peak_height * math.exp(-x * x)


def _gentle_peak(t: float, peak_t: float, peak_height: float,
                 sigma: float = 0.28) -> float:
    """straight/natural 用: 控えめな対称ガウシアン"""
    import math
    x = (t - peak_t) / max(sigma, 1e-6)
    return peak_height * math.exp(-x * x)


# =========================================================
# トレース形状ファイル読み込み
# =========================================================
SHAPES_FILE = Path(__file__).parent / "eyebrow_shapes.json"

_shapes_cache = None


def _load_shapes() -> dict:
    """eyebrow_shapes.json があれば読み込む（キャッシュ付き）"""
    global _shapes_cache
    if _shapes_cache is not None:
        return _shapes_cache
    if SHAPES_FILE.exists():
        with open(SHAPES_FILE) as f:
            _shapes_cache = json.load(f)
    else:
        _shapes_cache = {}
    return _shapes_cache


def _interpolate_contour(pts_norm: list, n: int = 120) -> np.ndarray:
    """正規化された輪郭点を n 点にスムーズ補間

    pts_norm: [[t, offset], ...] のリスト
    返り値: (n, 2) 配列 (t, offset)
    """
    pts = np.array(pts_norm)
    ts = pts[:, 0]
    offsets = pts[:, 1]

    # t が単調増加するようにソート
    order = np.argsort(ts)
    ts = ts[order]
    offsets = offsets[order]

    ts_new = np.linspace(ts[0], ts[-1], n)

    # scipy が使えればスプライン補間、なければ線形補間
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(ts, offsets, bc_type='natural')
        offsets_new = cs(ts_new)
    except ImportError:
        offsets_new = np.interp(ts_new, ts, offsets)

    return np.column_stack([ts_new, offsets_new])


def _make_upward_normal(axis_unit: np.ndarray) -> np.ndarray:
    """常に上向き（画像座標で負のy方向）を指す法線を返す"""
    normal = np.array([axis_unit[1], -axis_unit[0]])
    if normal[1] > 0:
        normal = -normal
    return normal


def generate_brow_polygon_from_shape(anchors: dict, shape_data: dict,
                                     n: int = 120,
                                     thickness_scale: float = 1.0) -> np.ndarray:
    """トレースした形状データからポリゴンを生成

    shape_data: {"upper": [[t, offset], ...], "lower": [[t, offset], ...]}
    thickness_scale: 厚み倍率。上辺と下辺の中心線を保ちつつ厚みだけ拡大。
    座標系:
      t: HEAD→TAIL 軸上の位置 (0=HEAD, 1=TAIL)
      offset: 軸に垂直な距離 / 眉長さ (正=上方向)
    """
    head = anchors["head"].copy()
    tail = anchors["tail"].copy()

    axis = tail - head
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-3:
        return np.array([[0, 0]], dtype=np.int32)

    axis_unit = axis / axis_len
    normal = _make_upward_normal(axis_unit)

    upper = _interpolate_contour(shape_data["upper"], n)
    lower = _interpolate_contour(shape_data["lower"], n)

    # thickness_scale: 中心線を基準に上下に拡大
    if thickness_scale != 1.0:
        # t を揃えてから中心と半厚みを計算
        ts = upper[:, 0]
        # lower を upper と同じ t でリサンプル
        lower_offsets_resampled = np.interp(ts, lower[:, 0], lower[:, 1])
        center = (upper[:, 1] + lower_offsets_resampled) / 2
        half_thick = (upper[:, 1] - lower_offsets_resampled) / 2
        upper[:, 1] = center + half_thick * thickness_scale
        lower_scaled = np.column_stack([ts, center - half_thick * thickness_scale])
        lower = lower_scaled

    def to_world(pts_norm):
        result = []
        for t, offset in pts_norm:
            pos = head + t * axis + offset * axis_len * normal
            result.append(pos)
        return np.array(result)

    upper_pts = to_world(upper)
    lower_pts = to_world(lower)

    polygon = np.vstack([upper_pts, lower_pts[::-1]])
    return polygon.astype(np.int32)


def generate_brow_polygon(anchors: dict, brow_type: str) -> np.ndarray:
    """眉タイプに応じた上下ラインを生成してポリゴン頂点列を返す

    **下辺は直線**（眉骨ラインに沿う）、**上辺だけに眉山を乗せる**方式。
    これにより「眉山で下辺がくぼむ」現象を防ぐ。

    - 下辺: head から tail への直線
    - 上辺: 下辺 - (thickness(taper) + peak_bump) で上方向へ盛る
    - 眉山は非対称ガウシアンで、頭側を平らに保つ
    """
    params = EYEBROW_TYPES[brow_type]

    head = anchors["head"].copy()
    tail = anchors["tail"].copy()
    eye_h = anchors["eye_height"]

    length_ratio = params["length_ratio"]
    if length_ratio != 1.0:
        tail = head + (tail - head) * length_ratio

    # tail_height_ratio: 下辺の末端を少し下げる
    tail[1] += eye_h * params["tail_height_ratio"]

    base_thickness = eye_h * params["thickness_ratio"]
    peak_height = eye_h * params["peak_height_ratio"]
    peak_t = params["peak_position"]

    sharp = brow_type in SHARP_BROW_TYPES

    n = 120
    ts = np.linspace(0, 1, n)

    # 下辺: head から tail への直線
    lower = np.zeros((n, 2))
    for i, t in enumerate(ts):
        lower[i] = head + (tail - head) * t

    # 上辺: 下辺から上方向に (厚さ + 眉山バンプ) で盛る
    # タイプ別のピーク関数を使用
    if brow_type == "corner":
        peak_fn = lambda t: _corner_peak(t, peak_t, peak_height,
                                         rise_len=0.20, fall_len=0.12)
    elif brow_type == "arch":
        peak_fn = lambda t: _arch_peak(t, peak_t, peak_height)
    elif brow_type == "natural":
        peak_fn = lambda t: _gentle_peak(t, peak_t, peak_height, sigma=0.30)
    elif brow_type == "straight":
        peak_fn = lambda t: _gentle_peak(t, peak_t, peak_height, sigma=0.35)
    else:  # parallel
        peak_fn = lambda t: _gentle_peak(t, peak_t, peak_height, sigma=0.45)

    upper = np.zeros((n, 2))
    for i, t in enumerate(ts):
        thickness = base_thickness * _taper(t, sharp=sharp)
        bump = peak_fn(t)
        # 上辺 = 下辺 - (厚さ + 眉山盛り上がり)
        upper[i, 0] = lower[i, 0]
        upper[i, 1] = lower[i, 1] - (thickness + bump)

    # sharp タイプは先端を1点に完全収束
    if sharp:
        upper[-1] = lower[-1]

    polygon = np.vstack([upper, lower[::-1]])
    return polygon.astype(np.int32)


def build_brow_mask(fm: FaceMesh, w: int, h: int, brow_type: str,
                    supersample: int = 2,
                    thickness_scale: float = 1.0) -> np.ndarray:
    """両眉のマスクを生成（float32, 0-1）

    eyebrow_shapes.json にトレース済み形状があればそちらを優先。
    なければパラメータベースのフォールバック。

    supersample: 指定倍率で描画してから縮小（アンチエイリアス用）
    thickness_scale: トレース形状の厚み倍率
    """
    # トレース形状があるか確認
    shapes = _load_shapes()
    use_traced = brow_type in shapes

    # supersample 倍の解像度で描画
    sh, sw = h * supersample, w * supersample
    mask_hi = np.zeros((sh, sw), dtype=np.float32)

    for side in ["right", "left"]:
        anchors = compute_brow_anchors(fm, side=side)
        if use_traced:
            polygon = generate_brow_polygon_from_shape(
                anchors, shapes[brow_type],
                thickness_scale=thickness_scale)
        else:
            polygon = generate_brow_polygon(anchors, brow_type)
        polygon_hi = polygon * supersample
        cv2.fillPoly(mask_hi, [polygon_hi], 1.0)

    # INTER_AREA で縮小 → アンチエイリアス効果
    mask = cv2.resize(mask_hi, (w, h), interpolation=cv2.INTER_AREA)
    return mask


def _make_directional_noise(shape: tuple, face_h: float) -> np.ndarray:
    """眉の方向（水平）に沿って引き伸ばされた毛流れノイズを生成"""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1.0, shape).astype(np.float32)

    # 水平方向モーションブラー（毛流れ風）
    motion_len = max(5, int(face_h * 0.015)) | 1  # 奇数
    kernel = np.zeros((1, motion_len), dtype=np.float32)
    kernel[0, :] = 1.0 / motion_len
    noise = cv2.filter2D(noise, -1, kernel)

    # さらに縦に軽くブラーして粒状すぎないように
    noise = cv2.GaussianBlur(noise, (3, 3), 0.8)

    # 正規化
    noise = (noise - noise.mean()) / (noise.std() + 1e-6)
    return noise


def _make_soft_density(mask: np.ndarray, face_h: float, blur_scale: float,
                       noise_amount: float = 0.18) -> np.ndarray:
    """マスクから密度マップを作成

    1. 2段階ブラーで滑らかなフェード
    2. 方向性ノイズ（水平方向に引き伸ばし）で毛流れ感
    """
    # 中ブラー
    ksize1 = max(5, int(face_h * 0.018 * blur_scale))
    density = gaussian_blur_mask(mask, ksize1)
    # 小ブラー
    ksize2 = max(3, int(face_h * 0.008 * blur_scale))
    density = gaussian_blur_mask(density, ksize2)

    # 方向性ノイズで毛流れ感
    if noise_amount > 0:
        noise = _make_directional_noise(density.shape, face_h)
        mask_region = density > 0.05
        density[mask_region] = density[mask_region] * (1.0 + noise[mask_region] * noise_amount)
        density = np.clip(density, 0, 1)

    return density


def _apply_density_gradient(mask: np.ndarray, fm: FaceMesh,
                            head_fade: float = 0.3, tail_fade: float = 0.25) -> np.ndarray:
    """マスクに眉頭/眉尻のフェードを適用

    眉頭（head）は head_fade の範囲で薄く、
    眉尻（tail）は tail_fade の範囲で薄くフェード。
    プロガイドライン: 眉頭は薄くぼかし、眉尻は眉頭の1/3の太さ
    """
    h, w = mask.shape[:2]
    result = mask.copy()

    for side in ["right", "left"]:
        anchors = compute_brow_anchors(fm, side=side)
        head = anchors["head"]
        tail = anchors["tail"]

        # 眉の軸ベクトル（head → tail）
        axis = tail - head
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-3:
            continue
        axis_unit = axis / axis_len

        # マスク内の各ピクセルを head→tail 軸に投影
        ys, xs = np.where(mask > 0.05)
        if len(ys) == 0:
            continue

        # 対応する眉の連結成分だけに適用するため、眉近傍だけ処理
        pts = np.column_stack([xs - head[0], ys - head[1]]).astype(np.float32)
        proj = pts @ axis_unit  # 0 ～ axis_len の範囲
        t = proj / axis_len  # 0 ～ 1 (軸外は負や1超)

        # 範囲内のピクセルのみ
        in_range = (t >= -0.1) & (t <= 1.1)
        if not in_range.any():
            continue

        # フェード係数を計算（プロ基準: 眉頭は薄く、眉尻も薄く）
        fade = np.ones_like(t)
        # 眉頭フェード: 0% → 40%密度、head_fade で 100% に
        head_mask = t < head_fade
        if head_mask.any():
            x = t[head_mask] / head_fade  # 0..1
            fade[head_mask] = 0.35 + 0.65 * x
        # 眉尻フェード: tail_start で 100% → 末端で 25%
        tail_start = 1.0 - tail_fade
        tail_mask = t > tail_start
        if tail_mask.any():
            x = (t[tail_mask] - tail_start) / tail_fade  # 0..1
            fade[tail_mask] = 1.0 - x * 0.75  # 1.0 → 0.25

        # 軸外は変更なし
        fade[~in_range] = 1.0

        # 適用（最小値で合成: 既に弱いところはそのまま）
        apply_mask = in_range
        idx = np.where(apply_mask)[0]
        ys_apply = ys[idx]
        xs_apply = xs[idx]
        fade_apply = fade[idx]
        result[ys_apply, xs_apply] = result[ys_apply, xs_apply] * fade_apply

    return result


def draw_eyebrows(
    image: np.ndarray,
    fm: FaceMesh,
    brow_type: str = DEFAULT_EYEBROW_TYPE,
    color_rgb: tuple = DEFAULT_EYEBROW_COLOR_RGB,
    intensity: float = DEFAULT_EYEBROW_INTENSITY,
    blur_scale: float = 1.0,
    thickness_scale: float = 1.0,
) -> np.ndarray:
    """眉を描画する（眉消し済み画像が前提）

    Args:
        image: 眉消し済み画像 (BGR)
        fm: FaceMesh
        brow_type: 眉タイプ名 (EYEBROW_TYPES のキー)
        color_rgb: 眉の色 (R, G, B)
        intensity: 描画強度 0-1
        blur_scale: ブラー倍率
        thickness_scale: トレース形状の厚み倍率
    """
    # トレース形状がある場合は EYEBROW_TYPES になくてもOK
    shapes = _load_shapes()
    if brow_type not in EYEBROW_TYPES and brow_type not in shapes:
        raise ValueError(f"Unknown brow type: {brow_type}. Available: {list(EYEBROW_TYPES.keys())}")

    h, w = image.shape[:2]
    face_h = np.linalg.norm(
        fm.landmarks_px[10].astype(float) - fm.landmarks_px[152].astype(float)
    )

    # 眉マスク生成（super-samplingでアンチエイリアス）
    mask = build_brow_mask(fm, w, h, brow_type, thickness_scale=thickness_scale)

    # 眉頭/眉尻の濃度グラデーション
    mask = _apply_density_gradient(mask, fm)

    # ソフトな密度マップ（ブラー + 方向性ノイズ）
    density = _make_soft_density(mask, face_h, blur_scale)

    # 色をアルファブレンド（normal blend）
    color_bgr = np.array([color_rgb[2], color_rgb[1], color_rgb[0]], dtype=np.float32)
    alpha = (density * intensity)[..., np.newaxis]

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
    parser.add_argument("--thickness", type=float, default=1.0,
                        help="眉の厚み倍率 (default: 1.0, 2.0で2倍太く)")
    parser.add_argument("--imgonly", action="store_true",
                        help="結果画像のみ (比較画像なし)")
    parser.add_argument("--zoom", action="store_true",
                        help="眉元ズームの比較画像を出力")
    parser.add_argument("--all-types", action="store_true",
                        help="トレース済み全タイプを一括生成")
    parser.add_argument("--list-types", action="store_true",
                        help="眉タイプ一覧を表示")
    args = parser.parse_args()

    if args.list_types:
        print("利用可能な眉タイプ:")
        for name, params in EYEBROW_TYPES.items():
            traced = " [traced]" if name in _load_shapes() else ""
            print(f"  {name:16s} - {params['desc']}{traced}")
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

    # 一括生成モード
    if args.all_types:
        shapes = _load_shapes()
        types_to_run = list(shapes.keys()) if shapes else list(EYEBROW_TYPES.keys())
        for btype in types_to_run:
            print(f"\n--- {btype} ---")
            erased = erase_eyebrows(image, fm, blur_scale=args.blur)
            result = draw_eyebrows(
                erased, fm, brow_type=btype,
                color_rgb=tuple(args.color),
                intensity=args.intensity,
                thickness_scale=args.thickness,
            )
            out_path = f"eyebrow_{btype}.png"
            cv2.imwrite(out_path, make_zoom_comparison(image, result, fm))
            print(f"  出力: {out_path}")
        print("\n完了!")
        return

    print("眉消し処理中...")
    output = erase_eyebrows(image, fm, blur_scale=args.blur)

    if not args.no_draw:
        print(f"眉描画中... type={args.type}, color={tuple(args.color)}, "
              f"intensity={args.intensity}, thickness={args.thickness}")
        output = draw_eyebrows(
            output, fm,
            brow_type=args.type,
            color_rgb=tuple(args.color),
            intensity=args.intensity,
            thickness_scale=args.thickness,
        )

    out_path = args.output or f"eyebrow_{args.type}.png"
    if args.imgonly:
        cv2.imwrite(out_path, output)
    elif args.zoom:
        cv2.imwrite(out_path, make_zoom_comparison(image, output, fm))
    else:
        cv2.imwrite(out_path, make_side_by_side(image, output))

    print(f"出力: {out_path}")


if __name__ == "__main__":
    main()
