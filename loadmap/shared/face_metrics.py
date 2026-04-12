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


def make_side_by_side(before, after):
    return np.hstack([before, after])
