"""
FaceMetrics - 2章「顔判定」で共通利用するランドマーク計測ユーティリティ

MediaPipe FaceLandmarker の 478 点ランドマークから、
距離・角度・比率などの幾何計測を行う。

設計方針:
  - 入力は shared.facemesh.FaceMesh インスタンス（detect 済み）
  - 出力はピクセル単位 (int/float) または正規化比率 (0-1)
  - 判定ロジックは含めない。あくまで「計測値の提供」に徹する
  - 各判定サブセクションはこの値を使って独自に分類する

主要ランドマークID（MediaPipe Face Mesh 468点準拠 + iris 10点）:
  * 10   : おでこ中央（生え際の近似）
  * 152  : あご先
  * 234  : 右耳側（こめかみ外）
  * 454  : 左耳側（こめかみ外）
  * 127  : 右頬骨外
  * 356  : 左頬骨外
  * 172  : 右エラ（下顎角）
  * 397  : 左エラ（下顎角）
  * 132  : 右あご側
  * 361  : 左あご側
  * 136  : 右下あご
  * 365  : 左下あご
  * 54   : 右おでこ横
  * 284  : 左おでこ横
  * 9    : 眉間
  * 168  : 鼻根
  * 1    : 鼻先
  * 2    : 鼻下
  * 33   : 右目外角
  * 133  : 右目内角
  * 362  : 左目内角
  * 263  : 左目外角
  * 159  : 右目上
  * 145  : 右目下
  * 386  : 左目上
  * 374  : 左目下
  * 468-472: 右目虹彩
  * 473-477: 左目虹彩
  * 64   : 右小鼻
  * 294  : 左小鼻
  * 61   : 口右角
  * 291  : 口左角
  * 13   : 上唇中央内側
  * 14   : 下唇中央内側
  * 0    : 上唇中央外側
  * 17   : 下唇中央外側
  * 55/285: 右/左眉頭
  * 46/276: 右/左眉尻
  * 105/334: 右/左眉山近傍
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ----------------------------------------------------------
# ランドマーク ID 定数
# ----------------------------------------------------------
class LM:
    # 縦軸
    FOREHEAD_TOP = 10
    CHIN_BOTTOM = 152
    GLABELLA = 9      # 眉間
    NOSE_ROOT = 168   # 鼻根
    NOSE_TIP = 1      # 鼻先
    SUBNASAL = 2      # 鼻下

    # 横軸
    TEMPLE_R = 234
    TEMPLE_L = 454
    CHEEKBONE_R = 127
    CHEEKBONE_L = 356
    GONION_R = 172   # 下顎角（エラ）
    GONION_L = 397
    JAW_R = 132
    JAW_L = 361
    LOWER_JAW_R = 136
    LOWER_JAW_L = 365
    FOREHEAD_R = 54
    FOREHEAD_L = 284

    # 目
    EYE_OUTER_R = 33
    EYE_INNER_R = 133
    EYE_INNER_L = 362
    EYE_OUTER_L = 263
    EYE_TOP_R = 159
    EYE_BOT_R = 145
    EYE_TOP_L = 386
    EYE_BOT_L = 374
    IRIS_R = (468, 469, 470, 471, 472)
    IRIS_L = (473, 474, 475, 476, 477)

    # 鼻
    NOSE_WING_R = 64    # 右小鼻
    NOSE_WING_L = 294   # 左小鼻
    NOSE_WING_R_OUT = 129
    NOSE_WING_L_OUT = 358

    # 口
    MOUTH_R = 61
    MOUTH_L = 291
    UPPER_LIP_TOP = 0
    UPPER_LIP_IN = 13
    LOWER_LIP_IN = 14
    LOWER_LIP_BOT = 17

    # 眉
    BROW_HEAD_R = 55
    BROW_HEAD_L = 285
    BROW_TAIL_R = 46
    BROW_TAIL_L = 276
    BROW_PEAK_R = 105
    BROW_PEAK_L = 334


# ----------------------------------------------------------
# 小ヘルパー
# ----------------------------------------------------------
def _p(fm, idx: int) -> np.ndarray:
    """ランドマーク pixel 座標を float2 で返す"""
    return fm.landmarks_px[idx].astype(np.float64)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _mean(fm, ids) -> np.ndarray:
    return np.mean([fm.landmarks_px[i].astype(np.float64) for i in ids], axis=0)


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """2ベクトルのなす角度(0-180)"""
    c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _ratio_loss(value: float, target: float) -> float:
    """比率 value が target からどれくらい離れているかの乖離度 (0=一致)"""
    if target == 0:
        return abs(value)
    return abs(value - target) / target


# ----------------------------------------------------------
# 計測結果コンテナ
# ----------------------------------------------------------
@dataclass
class FaceMetrics:
    """顔全体の計測値まとめ"""

    # 縦方向
    face_height_px: float = 0.0       # 10 → 152
    forehead_to_brow_px: float = 0.0  # 上顔面
    brow_to_subnasal_px: float = 0.0  # 中顔面
    subnasal_to_chin_px: float = 0.0  # 下顔面

    # 横方向
    face_width_temple_px: float = 0.0     # 234 ↔ 454 こめかみ
    face_width_cheekbone_px: float = 0.0  # 127 ↔ 356 頬骨
    face_width_jaw_px: float = 0.0        # 172 ↔ 397 エラ
    forehead_width_px: float = 0.0        # 54 ↔ 284 おでこ

    # 目
    eye_width_r_px: float = 0.0
    eye_width_l_px: float = 0.0
    eye_height_r_px: float = 0.0
    eye_height_l_px: float = 0.0
    eye_inner_gap_px: float = 0.0   # 目と目の間 (133↔362)

    # 虹彩
    iris_r_diameter_px: float = 0.0
    iris_l_diameter_px: float = 0.0

    # 鼻
    nose_length_px: float = 0.0     # 鼻根 → 鼻下
    nose_wing_width_px: float = 0.0 # 64 ↔ 294

    # 口
    mouth_width_px: float = 0.0     # 61 ↔ 291
    upper_lip_thickness_px: float = 0.0  # 0 ↔ 13
    lower_lip_thickness_px: float = 0.0  # 14 ↔ 17
    philtrum_px: float = 0.0        # 鼻下 → 上唇中央外

    # あご
    chin_angle_deg: float = 0.0     # 172-152-397 のなす角（大きいほど平ら）

    # 主要比率
    face_aspect: float = 0.0        # 縦/横 (height/temple)

    # 生データ（詳細可視化用）
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "raw"}
        d["raw"] = self.raw
        return d


# ----------------------------------------------------------
# 計測関数
# ----------------------------------------------------------
def measure(fm) -> FaceMetrics:
    """FaceMesh.detect() 済みのインスタンスから計測値をまとめて算出"""
    m = FaceMetrics()

    # --- 縦 ---
    forehead = _p(fm, LM.FOREHEAD_TOP)
    chin = _p(fm, LM.CHIN_BOTTOM)
    glabella = _p(fm, LM.GLABELLA)
    subnasal = _p(fm, LM.SUBNASAL)

    m.face_height_px = _dist(forehead, chin)

    # 上/中/下顔面 (髪の生え際は10番で近似、眉頭下端は眉間9で近似)
    m.forehead_to_brow_px = abs(glabella[1] - forehead[1])
    m.brow_to_subnasal_px = abs(subnasal[1] - glabella[1])
    m.subnasal_to_chin_px = abs(chin[1] - subnasal[1])

    # --- 横 ---
    m.face_width_temple_px = _dist(_p(fm, LM.TEMPLE_R), _p(fm, LM.TEMPLE_L))
    m.face_width_cheekbone_px = _dist(_p(fm, LM.CHEEKBONE_R), _p(fm, LM.CHEEKBONE_L))
    m.face_width_jaw_px = _dist(_p(fm, LM.GONION_R), _p(fm, LM.GONION_L))
    m.forehead_width_px = _dist(_p(fm, LM.FOREHEAD_R), _p(fm, LM.FOREHEAD_L))

    # --- 目 ---
    m.eye_width_r_px = _dist(_p(fm, LM.EYE_OUTER_R), _p(fm, LM.EYE_INNER_R))
    m.eye_width_l_px = _dist(_p(fm, LM.EYE_OUTER_L), _p(fm, LM.EYE_INNER_L))
    m.eye_height_r_px = _dist(_p(fm, LM.EYE_TOP_R), _p(fm, LM.EYE_BOT_R))
    m.eye_height_l_px = _dist(_p(fm, LM.EYE_TOP_L), _p(fm, LM.EYE_BOT_L))
    m.eye_inner_gap_px = _dist(_p(fm, LM.EYE_INNER_R), _p(fm, LM.EYE_INNER_L))

    iris_r_pts = np.array([_p(fm, i) for i in LM.IRIS_R])
    iris_l_pts = np.array([_p(fm, i) for i in LM.IRIS_L])
    m.iris_r_diameter_px = float(
        np.linalg.norm(iris_r_pts[1] - iris_r_pts[3])
    )  # 左右方向の虹彩径
    m.iris_l_diameter_px = float(
        np.linalg.norm(iris_l_pts[1] - iris_l_pts[3])
    )

    # --- 鼻 ---
    m.nose_length_px = abs(subnasal[1] - _p(fm, LM.NOSE_ROOT)[1])
    m.nose_wing_width_px = _dist(_p(fm, LM.NOSE_WING_R), _p(fm, LM.NOSE_WING_L))

    # --- 口 ---
    m.mouth_width_px = _dist(_p(fm, LM.MOUTH_R), _p(fm, LM.MOUTH_L))
    m.upper_lip_thickness_px = _dist(_p(fm, LM.UPPER_LIP_TOP), _p(fm, LM.UPPER_LIP_IN))
    m.lower_lip_thickness_px = _dist(_p(fm, LM.LOWER_LIP_IN), _p(fm, LM.LOWER_LIP_BOT))
    m.philtrum_px = _dist(subnasal, _p(fm, LM.UPPER_LIP_TOP))

    # --- あご角度 (エラ-あご先-エラ の角度: 小さい=尖り、大きい=平ら) ---
    gr = _p(fm, LM.GONION_R)
    gl = _p(fm, LM.GONION_L)
    m.chin_angle_deg = _angle_deg(gr - chin, gl - chin)

    # --- 主要比率 ---
    if m.face_width_temple_px > 1e-3:
        m.face_aspect = m.face_height_px / m.face_width_temple_px

    # --- raw 座標 ---
    m.raw = {
        "forehead_top": forehead.tolist(),
        "chin": chin.tolist(),
        "glabella": glabella.tolist(),
        "subnasal": subnasal.tolist(),
        "temple_r": _p(fm, LM.TEMPLE_R).tolist(),
        "temple_l": _p(fm, LM.TEMPLE_L).tolist(),
        "cheekbone_r": _p(fm, LM.CHEEKBONE_R).tolist(),
        "cheekbone_l": _p(fm, LM.CHEEKBONE_L).tolist(),
        "gonion_r": gr.tolist(),
        "gonion_l": gl.tolist(),
        "forehead_r": _p(fm, LM.FOREHEAD_R).tolist(),
        "forehead_l": _p(fm, LM.FOREHEAD_L).tolist(),
    }

    return m


# ----------------------------------------------------------
# Ratio / 乖離度
# ----------------------------------------------------------
def ratio_loss(value: float, target: float) -> float:
    return _ratio_loss(value, target)


# ----------------------------------------------------------
# 描画ヘルパー（判定可視化用、cv2 は遅延 import）
# ----------------------------------------------------------
def draw_point(img, p, color=(0, 255, 255), r=3, label: str | None = None):
    import cv2
    p = tuple(int(x) for x in p)
    cv2.circle(img, p, r, color, -1, lineType=cv2.LINE_AA)
    if label:
        cv2.putText(
            img, label, (p[0] + 5, p[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )


def draw_line(img, a, b, color=(0, 255, 255), thickness=1):
    import cv2
    a = tuple(int(x) for x in a)
    b = tuple(int(x) for x in b)
    cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def put_label(img, text: str, org, color=(255, 255, 255),
              scale: float = 0.5, thickness: int = 1, bg=(0, 0, 0)):
    """背景付きテキスト"""
    import cv2
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + bl), bg, -1)
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
    )


def draw_line_outlined(img, a, b, color=(0, 255, 255),
                       thickness: int = 2, outline_color=(0, 0, 0),
                       outline: int = 1):
    """縁取り付きの線（ダーク縁で視認性アップ）"""
    import cv2
    a = tuple(int(x) for x in a)
    b = tuple(int(x) for x in b)
    if outline > 0:
        cv2.line(img, a, b, outline_color, thickness + outline * 2, cv2.LINE_AA)
    cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def draw_dashed_line(img, a, b, color=(0, 255, 255),
                     thickness: int = 2, dash_len: int = 10,
                     gap_len: int = 6, outline: int = 1,
                     outline_color=(0, 0, 0)):
    """破線を描画 (理想値の参照線などに使う)"""
    import cv2
    import math
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return
    ux, uy = dx / length, dy / length
    step = dash_len + gap_len
    t = 0.0
    while t < length:
        x0 = ax + ux * t
        y0 = ay + uy * t
        end = min(t + dash_len, length)
        x1 = ax + ux * end
        y1 = ay + uy * end
        p0 = (int(x0), int(y0))
        p1 = (int(x1), int(y1))
        if outline > 0:
            cv2.line(
                img, p0, p1, outline_color,
                thickness + outline * 2, cv2.LINE_AA,
            )
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
        t += step


def draw_point_outlined(img, p, color=(0, 255, 255), r: int = 5,
                        outline_color=(0, 0, 0), outline: int = 1):
    """縁取り付きの点"""
    import cv2
    p = tuple(int(x) for x in p)
    if outline > 0:
        cv2.circle(img, p, r + outline, outline_color, -1, cv2.LINE_AA)
    cv2.circle(img, p, r, color, -1, cv2.LINE_AA)


def draw_text_outlined(img, text: str, org, color=(255, 255, 255),
                       scale: float = 0.6, thickness: int = 1,
                       outline_color=(0, 0, 0), outline: int = 2):
    """黒縁取り付きテキスト (背景ボックス無しで画面に浮かせる)"""
    import cv2
    x, y = int(org[0]), int(org[1])
    if outline > 0:
        cv2.putText(
            img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
            outline_color, thickness + outline * 2, cv2.LINE_AA,
        )
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA,
    )


# ----------------------------------------------------------
# レポートパネル レンダラー (UI/UX 改善版) - PIL で日本語対応
# ----------------------------------------------------------
# テーマカラー (RGB for PIL)
_THEME = {
    "bg": (32, 32, 38),              # パネル背景
    "bg_section": (48, 48, 58),      # セクション帯
    "fg": (230, 230, 235),           # 本文
    "fg_dim": (150, 150, 160),       # サブテキスト
    "accent": (80, 210, 255),        # アクセント (水色)
    "accent2": (140, 255, 160),      # 成功 (緑)
    "accent3": (255, 200, 80),       # 注意 (黄)
    "divider": (78, 78, 90),         # 区切り線
    "bar_bg": (55, 55, 65),          # バー背景
    "bar_dim": (110, 110, 125),      # バー非ハイライト
}

_JP_FONT = "/etc/alternatives/fonts-japanese-gothic.ttf"
_MONO_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
_font_cache: dict = {}


def _get_font(size: int, mono: bool = False):
    from PIL import ImageFont
    key = (size, mono)
    if key in _font_cache:
        return _font_cache[key]
    path = _MONO_FONT if mono else _JP_FONT
    try:
        f = ImageFont.truetype(path, size)
    except Exception:
        f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def _pil_text_size(draw, text: str, font) -> tuple[int, int]:
    """テキストの (width, height) を返す (PIL バージョン差を吸収)"""
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except AttributeError:
        return draw.textsize(text, font=font)


def _bgr_to_rgb(c):
    if len(c) >= 3:
        return (int(c[2]), int(c[1]), int(c[0]))
    return c


def _rgb_to_bgr(c):
    return (int(c[2]), int(c[1]), int(c[0]))


def render_report_panel(spec: list, width: int, height: int) -> "np.ndarray":
    """レポートパネルをレンダリングする (PIL / 日本語対応)

    spec の各要素は以下の tuple のいずれか:
      ('title', text, color_bgr)                大きなタイトル
      ('subtitle', text)                        副タイトル
      ('section', text)                         セクションヘッダ (背景帯)
      ('kv', key, value, [color_bgr])           キーバリュー行 (右詰め値)
      ('bar', label, value_0_1, is_best, text)  横向きバー
      ('text', text, [color_bgr])               自由テキスト
      ('divider',)                              水平区切り線
      ('spacer', height_px)                     縦方向の余白
      ('big', text, color_bgr)                  超大見出し
      ('radar', labels, values, highlight_idx, fill_bgr)
          レーダーチャート (width-48 幅の正方形) を現在位置に描画
    """
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), _THEME["bg"])
    draw = ImageDraw.Draw(img)

    # フォント (UI/UX レビュー反映: 全体的にサイズアップ)
    f_big = _get_font(56)       # big
    f_title = _get_font(30)     # title
    f_section = _get_font(20)   # section
    f_kv_key = _get_font(18)    # kv key
    f_kv_val = _get_font(20, mono=True)  # kv value
    f_bar = _get_font(17)       # bar label
    f_bar_val = _get_font(16, mono=True)
    f_text = _get_font(16)      # text
    f_sub = _get_font(15)       # subtitle

    pad_x = 26
    y = 28

    for item in spec:
        tag = item[0]

        if tag == "title":
            text, color = item[1], _bgr_to_rgb(item[2])
            draw.text((pad_x, y), text, font=f_title, fill=color)
            y += 42

        elif tag == "big":
            text, color = item[1], _bgr_to_rgb(item[2])
            draw.text((pad_x, y), text, font=f_big, fill=color)
            y += 68

        elif tag == "subtitle":
            draw.text((pad_x, y), item[1], font=f_sub, fill=_THEME["fg_dim"])
            y += 28

        elif tag == "section":
            # 背景帯
            draw.rectangle(
                [(pad_x - 14, y - 4), (width - pad_x + 14, y + 30)],
                fill=_THEME["bg_section"],
            )
            # 左端のアクセント棒
            draw.rectangle(
                [(pad_x - 14, y - 4), (pad_x - 8, y + 30)],
                fill=_THEME["accent"],
            )
            draw.text((pad_x, y + 4), item[1], font=f_section,
                      fill=_THEME["accent"])
            y += 42

        elif tag == "kv":
            key = item[1]
            value = item[2]
            color = _bgr_to_rgb(item[3]) if len(item) > 3 else _THEME["fg"]
            draw.text((pad_x, y), key, font=f_kv_key, fill=_THEME["fg_dim"])
            vw, _vh = _pil_text_size(draw, value, f_kv_val)
            draw.text(
                (width - pad_x - vw, y - 2),
                value, font=f_kv_val, fill=color,
            )
            y += 28

        elif tag == "bar":
            label = item[1]
            value = max(0.0, min(1.0, float(item[2])))
            is_best = bool(item[3])
            numeric = item[4] if len(item) > 4 else f"{value:.2f}"
            col = _THEME["accent"] if is_best else _THEME["bar_dim"]
            label_col = _THEME["fg"] if is_best else _THEME["fg_dim"]
            num_w, _ = _pil_text_size(draw, numeric, f_bar_val)
            # ラベル (上の行)
            draw.text((pad_x, y), label, font=f_bar, fill=label_col)
            y += 22
            # バー背景
            bar_x0 = pad_x
            bar_x1 = width - pad_x - num_w - 10
            bar_y0 = y + 2
            bar_y1 = y + 16
            draw.rectangle(
                [(bar_x0, bar_y0), (bar_x1, bar_y1)],
                fill=_THEME["bar_bg"],
            )
            fill_x = bar_x0 + int((bar_x1 - bar_x0) * value)
            if fill_x > bar_x0:
                draw.rectangle([(bar_x0, bar_y0), (fill_x, bar_y1)], fill=col)
            # 数値
            draw.text(
                (bar_x1 + 8, y - 2),
                numeric, font=f_bar_val, fill=label_col,
            )
            y += 22

        elif tag == "text":
            text = item[1]
            color = _bgr_to_rgb(item[2]) if len(item) > 2 else _THEME["fg"]
            draw.text((pad_x, y), text, font=f_text, fill=color)
            y += 24

        elif tag == "divider":
            draw.line(
                [(pad_x, y), (width - pad_x, y)],
                fill=_THEME["divider"], width=1,
            )
            y += 14

        elif tag == "spacer":
            y += int(item[1])

        elif tag == "ratio_compare":
            # 比率を四角形で視覚化: 同じ基準 (width) で、height = aspect * width
            # 複数項目を横並びで表示、差分 % を下に表示
            #
            # spec: ('ratio_compare', items, highlight_label)
            #   items = [(label, aspect, value_str, color_bgr), ...]
            #   highlight_label: 最も近い項目のラベル
            items = item[1]
            highlight = item[2] if len(item) > 2 else None
            if not items:
                continue

            n = len(items)
            avail_w = width - pad_x * 2
            gap = 18
            # 各アイテムは四角 + ラベル を持つ
            item_w = (avail_w - gap * (n - 1)) // n
            # アスペクト最大値で高さを決定
            max_aspect = max(max(a for _, a, _, _ in items), 0.01)
            max_h = min(int(item_w * max_aspect), 180)
            base_w = int(max_h / max_aspect)  # 基準幅 (比率に合わせて統一)

            # 描画位置: y 開始
            row_y = y + 8
            arr = np.array(img)[:, :, ::-1].copy()
            import cv2 as _cv2
            for i, (label, aspect, value_str, col_bgr) in enumerate(items):
                is_hi = (label == highlight)
                cx = pad_x + i * (item_w + gap) + item_w // 2
                rect_h = int(base_w * aspect)
                rect_w = base_w
                x0 = int(cx - rect_w / 2)
                y0 = row_y
                x1 = x0 + rect_w
                y1 = y0 + rect_h
                col = col_bgr
                # 枠線 (ハイライトは太線、他は薄)
                thickness = 3 if is_hi else 2
                _cv2.rectangle(arr, (x0, y0), (x1, y1),
                               col if is_hi else (110, 110, 120),
                               thickness, _cv2.LINE_AA)
                # 半透明塗り (ハイライトのみ)
                if is_hi:
                    overlay = arr.copy()
                    _cv2.rectangle(overlay, (x0, y0), (x1, y1), col, -1)
                    _cv2.addWeighted(overlay, 0.22, arr, 0.78, 0, arr)
            img = Image.fromarray(arr[:, :, ::-1])
            draw = ImageDraw.Draw(img, "RGBA")

            # ラベル (四角の下)
            label_y = row_y + max_h + 6
            f_lab = _get_font(14)
            f_val = _get_font(15, mono=True)
            for i, (label, aspect, value_str, col_bgr) in enumerate(items):
                is_hi = (label == highlight)
                cx = pad_x + i * (item_w + gap) + item_w // 2
                col_rgb = (
                    _bgr_to_rgb(col_bgr) if is_hi else _THEME["fg_dim"]
                )
                tw, _ = _pil_text_size(draw, label, f_lab)
                draw.text(
                    (cx - tw / 2, label_y),
                    label, font=f_lab, fill=col_rgb,
                )
                vw, _ = _pil_text_size(draw, value_str, f_val)
                draw.text(
                    (cx - vw / 2, label_y + 18),
                    value_str, font=f_val, fill=col_rgb,
                )
            y = label_y + 44

        elif tag == "diff_bar":
            # 差分バー: 中央を 0、左右に +/- 差を表示
            #
            # spec: ('diff_bar', label, diff_pct, max_pct, color_bgr)
            label = item[1]
            diff_pct = float(item[2])
            max_pct = float(item[3]) if len(item) > 3 else 20.0
            col_bgr = item[4] if len(item) > 4 else (80, 220, 255)

            # ラベル
            draw.text((pad_x, y), label, font=f_text, fill=_THEME["fg"])
            y += 22
            # バー (中央 0、左右 ±max_pct)
            bar_x0 = pad_x
            bar_x1 = width - pad_x
            bar_cx = (bar_x0 + bar_x1) // 2
            bar_y0 = y
            bar_y1 = y + 14
            # 背景
            draw.rectangle(
                [(bar_x0, bar_y0), (bar_x1, bar_y1)],
                fill=_THEME["bar_bg"],
            )
            # 中央線
            draw.line(
                [(bar_cx, bar_y0 - 2), (bar_cx, bar_y1 + 2)],
                fill=_THEME["divider"], width=1,
            )
            # 差分バー本体
            half_w = (bar_x1 - bar_x0) // 2
            pct_ratio = max(-1.0, min(1.0, diff_pct / max_pct))
            col_rgb = _bgr_to_rgb(col_bgr)
            if pct_ratio >= 0:
                fill_x0 = bar_cx
                fill_x1 = int(bar_cx + half_w * pct_ratio)
            else:
                fill_x0 = int(bar_cx + half_w * pct_ratio)
                fill_x1 = bar_cx
            if fill_x1 > fill_x0:
                draw.rectangle(
                    [(fill_x0, bar_y0), (fill_x1, bar_y1)],
                    fill=col_rgb,
                )
            # 目盛り数値 (左端 -max、中央 0、右端 +max)
            f_small = _get_font(12, mono=True)
            draw.text(
                (bar_x0, bar_y1 + 2),
                f"-{max_pct:.0f}%", font=f_small, fill=_THEME["fg_dim"],
            )
            draw.text(
                (bar_cx - 8, bar_y1 + 2),
                "0", font=f_small, fill=_THEME["fg_dim"],
            )
            tw, _ = _pil_text_size(draw, f"+{max_pct:.0f}%", f_small)
            draw.text(
                (bar_x1 - tw, bar_y1 + 2),
                f"+{max_pct:.0f}%", font=f_small, fill=_THEME["fg_dim"],
            )
            # 差分値のテキスト
            diff_str = f"{diff_pct:+.1f} %"
            f_diff = _get_font(17, mono=True)
            dw, _ = _pil_text_size(draw, diff_str, f_diff)
            draw.text(
                (bar_cx - dw / 2, bar_y1 + 18),
                diff_str, font=f_diff, fill=col_rgb,
            )
            y = bar_y1 + 44

        elif tag == "radar":
            # 現在位置に正方形エリアでレーダーを描画
            labels = item[1]
            values = item[2]
            highlight_idx = item[3] if len(item) > 3 else None
            fill_bgr = item[4] if len(item) > 4 else (80, 210, 255)
            remaining = height - y - 10
            size = min(width - pad_x * 2, remaining)
            if size >= 80:
                cx = width // 2
                cy = y + size // 2
                radius = int(size * 0.42)
                # 一旦 PIL を numpy に書き戻して描画ヘルパー共用
                arr = np.array(img)[:, :, ::-1].copy()
                draw_radar_chart(
                    arr, (cx, cy), radius,
                    labels=labels, values=values,
                    max_value=max(values) if values else 1.0,
                    fill_color=fill_bgr, highlight_idx=highlight_idx,
                    highlight_color=fill_bgr, size_label=14,
                )
                img = Image.fromarray(arr[:, :, ::-1])
                draw = ImageDraw.Draw(img, "RGBA")
                y += size + 10

        if y >= height - 10:
            break

    # RGB → BGR (OpenCV 互換)
    arr = np.array(img)[:, :, ::-1].copy()
    return arr


def draw_pil_text(image: "np.ndarray", text: str, org, color=(255, 255, 255),
                  size: int = 20, outline_color=(0, 0, 0), outline: int = 2,
                  bg: tuple | None = None, bg_alpha: float = 0.65,
                  bg_pad: int = 4):
    """BGR numpy 画像に PIL で日本語テキストを描画 (黒縁 or 半透明背景)

    bg: (B,G,R) を指定すると、その色の半透明ボックスを背景に敷く
    """
    from PIL import Image, ImageDraw
    pil = Image.fromarray(image[:, :, ::-1])
    draw = ImageDraw.Draw(pil, "RGBA")
    font = _get_font(size)
    x, y = int(org[0]), int(org[1])
    rgb = _bgr_to_rgb(color)

    if bg is not None:
        # 背景ボックス (半透明)
        tw, th = _pil_text_size(draw, text, font)
        bg_rgb = _bgr_to_rgb(bg)
        alpha = int(255 * bg_alpha)
        draw.rectangle(
            [
                (x - bg_pad, y - bg_pad),
                (x + tw + bg_pad, y + th + bg_pad),
            ],
            fill=(bg_rgb[0], bg_rgb[1], bg_rgb[2], alpha),
        )
        draw.text((x, y), text, font=font, fill=rgb)
    else:
        outline_rgb = _bgr_to_rgb(outline_color)
        if outline > 0:
            for dx in range(-outline, outline + 1):
                for dy in range(-outline, outline + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), text, font=font,
                              fill=outline_rgb)
        draw.text((x, y), text, font=font, fill=rgb)

    image[:] = np.array(pil)[:, :, ::-1]
    return image


def draw_pil_pill(image: "np.ndarray", text: str, org,
                  text_color=(255, 255, 255), pill_color=(50, 180, 255),
                  size: int = 22, pad_x: int = 12, pad_y: int = 6,
                  radius: int = 14):
    """ピル型バッジを描画 (annotated 画像の左上バッジなど)"""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(image[:, :, ::-1])
    draw = ImageDraw.Draw(pil, "RGBA")
    font = _get_font(size)
    x, y = int(org[0]), int(org[1])
    tw, th = _pil_text_size(draw, text, font)
    # 半透明黒の外枠
    draw.rounded_rectangle(
        [(x - 2, y - 2), (x + tw + pad_x * 2 + 2, y + th + pad_y * 2 + 2)],
        radius=radius + 2, fill=(0, 0, 0, 160),
    )
    # アクセント色のピル
    pill_rgb = _bgr_to_rgb(pill_color)
    draw.rounded_rectangle(
        [(x, y), (x + tw + pad_x * 2, y + th + pad_y * 2)],
        radius=radius, fill=(pill_rgb[0], pill_rgb[1], pill_rgb[2], 230),
    )
    # テキスト (黒、濃いめ)
    text_rgb = _bgr_to_rgb(text_color)
    draw.text((x + pad_x, y + pad_y), text, font=font, fill=text_rgb)

    image[:] = np.array(pil)[:, :, ::-1]
    return image


def draw_radar_chart(image: "np.ndarray", center, radius: int,
                     labels: list[str], values: list[float],
                     max_value: float = 1.0,
                     bg_color=(40, 40, 48), axis_color=(100, 100, 115),
                     fill_color=(80, 210, 255), label_color=(220, 220, 230),
                     highlight_idx: int | None = None,
                     highlight_color=(80, 210, 255),
                     size_label: int = 13):
    """レーダーチャートを image に描画

    labels / values は同数。values は 0..max_value 推奨。
    highlight_idx が指定されると、その軸の点を強調する。
    """
    import math
    from PIL import Image, ImageDraw

    pil = Image.fromarray(image[:, :, ::-1])
    draw = ImageDraw.Draw(pil, "RGBA")
    cx, cy = int(center[0]), int(center[1])
    n = len(values)
    if n < 3:
        return image

    # 各軸の角度 (上から時計回り)
    def axis_xy(i, r):
        angle = -math.pi / 2 + 2 * math.pi * i / n
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    # 格子 (25%, 50%, 75%, 100%)
    for ring in [0.25, 0.5, 0.75, 1.0]:
        pts = [axis_xy(i, radius * ring) for i in range(n)]
        draw.polygon(pts, outline=axis_color, fill=None)

    # 軸線
    for i in range(n):
        draw.line([(cx, cy), axis_xy(i, radius)], fill=axis_color, width=1)

    # 値ポリゴン
    val_pts = [
        axis_xy(i, radius * min(1.0, values[i] / max(max_value, 1e-6)))
        for i in range(n)
    ]
    fill_rgb = _bgr_to_rgb(fill_color)
    draw.polygon(val_pts, outline=fill_rgb,
                 fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], 80))

    # 頂点
    for i, p in enumerate(val_pts):
        px, py = int(p[0]), int(p[1])
        is_h = i == highlight_idx
        col = _bgr_to_rgb(highlight_color) if is_h else fill_rgb
        r_pt = 5 if is_h else 3
        draw.ellipse(
            [(px - r_pt, py - r_pt), (px + r_pt, py + r_pt)],
            fill=col, outline=(0, 0, 0, 180), width=1,
        )

    # ラベル
    font = _get_font(size_label)
    label_rgb = _bgr_to_rgb(label_color)
    for i, name in enumerate(labels):
        lx, ly = axis_xy(i, radius + 12)
        tw, th = _pil_text_size(draw, name, font)
        # ラベル位置調整: 軸の方向に応じてアンカー
        angle = -math.pi / 2 + 2 * math.pi * i / n
        if math.cos(angle) < -0.3:
            lx -= tw
        elif -0.3 <= math.cos(angle) <= 0.3:
            lx -= tw / 2
        if math.sin(angle) < -0.3:
            ly -= th
        elif -0.3 <= math.sin(angle) <= 0.3:
            ly -= th / 2
        draw.text((lx, ly), name, font=font, fill=label_rgb)

    image[:] = np.array(pil)[:, :, ::-1]
    return image


def compose_report(before, annotated, panel):
    """[before | annotated | panel] を横連結"""
    import cv2
    h = max(before.shape[0], annotated.shape[0], panel.shape[0])
    # 高さを揃える (不足分は黒埋め)
    def _pad_h(img, target_h):
        if img.shape[0] == target_h:
            return img
        pad = np.zeros(
            (target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8
        )
        return np.vstack([img, pad])

    return np.hstack([_pad_h(before, h), _pad_h(annotated, h), _pad_h(panel, h)])



def make_side_by_side(before, after):
    return np.hstack([before, after])
