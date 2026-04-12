"""
2.2.1 顔全体の縦横バランス判定

計測項目:
    - 顔の縦幅（全頭高の近似）: landmark 10→152 × 1.25 補正（生え際まで）
    - 顔の横幅: 234↔454（耳付け根間, こめかみ）
    - 縦横比 ratio = height / width
    - 3つの理想比との乖離度:
        * 黄金比      1.618
        * 白銀比      1.414
        * 日本人理想比 1.460
    - 実寸換算（虹彩径 11.7mm 基準で mm/pixel を算出）
    - 産総研データ比較 (男性平均 縦23.2cm / 横14.5cm)
    - 小顔度スコア（基準: 縦20cm以下, 横14cm以下）

注意:
    MediaPipe の landmark 10 は実際の生え際ではなくおでこ上部。
    このため raw の face_height_px は真の全頭高より短く、
    aspect は 1.2 付近に偏ってしまう。
    面積補正係数 FOREHEAD_EXTEND = 1.25 を掛けて生え際までを推定する
    （平均的な日本人顔における統計値）。

Usage:
    python main.py <input_image> [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.facemesh import FaceMesh
from shared.face_metrics import (
    LM,
    compose_report,
    draw_line_outlined,
    draw_pil_pill,
    draw_pil_text,
    draw_point_outlined,
    make_side_by_side,
    measure,
    render_report_panel,
)


# ==============================================================
# 定数
# ==============================================================
# MediaPipe landmark 10 → 真の生え際までの補正係数
FOREHEAD_EXTEND = 1.25

# 理想比率
RATIOS = {
    "golden": 1.618,   # 黄金比 シャープ・クール
    "silver": 1.414,   # 白銀比 安定・親しみ
    "japanese": 1.460, # 日本人理想比 現代的
}

# 虹彩径の標準値（mm）: 成人平均 11.7mm
IRIS_DIAMETER_MM = 11.7

# 産総研男性平均（cm）
AIST_MALE_HEIGHT_CM = 23.2
AIST_MALE_WIDTH_CM = 14.5

# 小顔基準（cm）
KOGAO_HEIGHT_CM = 20.0
KOGAO_WIDTH_CM = 14.0


# ==============================================================
# 計測
# ==============================================================
@dataclass
class FaceRatioResult:
    # pixel 値
    face_height_px_raw: float = 0.0
    face_height_px: float = 0.0       # 補正済み
    face_width_px: float = 0.0
    aspect_raw: float = 0.0
    aspect: float = 0.0               # 補正済み縦/横

    # 理想比との乖離
    losses: dict[str, float] = field(default_factory=dict)  # 小さいほど近い
    closest_ratio: str = ""           # golden/silver/japanese
    impression: str = ""

    # 実寸換算
    mm_per_pixel: float = 0.0
    face_height_cm: float = 0.0
    face_width_cm: float = 0.0

    # 小顔度
    kogao_score: float = 0.0          # 0-100 の複合スコア
    kogao_label: str = ""

    # 産総研比較
    vs_aist_height: float = 0.0       # 平均に対する割合
    vs_aist_width: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _iris_diameter_px(fm) -> float:
    """虹彩径 (ピクセル) を左右平均で返す"""
    ir = np.array([fm.landmarks_px[i].astype(np.float64) for i in LM.IRIS_R])
    il = np.array([fm.landmarks_px[i].astype(np.float64) for i in LM.IRIS_L])
    # ir[1] と ir[3] が左右の端で距離 = 直径
    d_r = float(np.linalg.norm(ir[1] - ir[3]))
    d_l = float(np.linalg.norm(il[1] - il[3]))
    return (d_r + d_l) / 2.0


def _impression_for(ratio_name: str) -> str:
    return {
        "golden": "Sharp / Cool / Refined",
        "silver": "Stable / Harmonious / Friendly",
        "japanese": "Modern Balance",
    }.get(ratio_name, "")


def analyze(fm: FaceMesh) -> FaceRatioResult:
    m = measure(fm)
    r = FaceRatioResult()

    r.face_height_px_raw = m.face_height_px
    r.face_height_px = m.face_height_px * FOREHEAD_EXTEND
    r.face_width_px = m.face_width_temple_px
    r.aspect_raw = m.face_height_px / max(m.face_width_temple_px, 1.0)
    r.aspect = r.face_height_px / max(m.face_width_temple_px, 1.0)

    # 理想比との乖離 (相対%)
    for name, target in RATIOS.items():
        r.losses[name] = abs(r.aspect - target) / target

    r.closest_ratio = min(r.losses.items(), key=lambda kv: kv[1])[0]
    r.impression = _impression_for(r.closest_ratio)

    # 実寸換算
    iris_px = _iris_diameter_px(fm)
    if iris_px > 1e-3:
        r.mm_per_pixel = IRIS_DIAMETER_MM / iris_px
        r.face_height_cm = r.face_height_px * r.mm_per_pixel / 10.0
        r.face_width_cm = r.face_width_px * r.mm_per_pixel / 10.0

    # 小顔度 (100% = 基準値以下)
    if r.face_height_cm > 0 and r.face_width_cm > 0:
        h_score = max(0.0, min(100.0, (KOGAO_HEIGHT_CM / r.face_height_cm) * 100.0))
        w_score = max(0.0, min(100.0, (KOGAO_WIDTH_CM / r.face_width_cm) * 100.0))
        r.kogao_score = (h_score + w_score) / 2.0

        if r.kogao_score >= 95:
            r.kogao_label = "Very Small (+)"
        elif r.kogao_score >= 85:
            r.kogao_label = "Small"
        elif r.kogao_score >= 75:
            r.kogao_label = "Average"
        else:
            r.kogao_label = "Large"

        # 産総研平均比
        r.vs_aist_height = r.face_height_cm / AIST_MALE_HEIGHT_CM
        r.vs_aist_width = r.face_width_cm / AIST_MALE_WIDTH_CM

    return r


# ==============================================================
# 可視化 (UI/UX 改善版: レポートパネル + annotate 分離)
# ==============================================================
C_H = (80, 220, 255)     # 縦軸 アンバー
C_W = (255, 200, 80)     # 横軸 シアン
C_G = (140, 255, 160)    # 理想比参照線 グリーン
C_E = (180, 180, 255)    # 補正 (extended) ピンク淡

RATIO_JP = {
    "golden": "黄金比 1.618",
    "silver": "白銀比 1.414",
    "japanese": "日本人 1.460",
}
RATIO_COLOR = {
    "golden": (80, 220, 255),
    "silver": (180, 220, 90),
    "japanese": (140, 255, 160),
}


def annotate_face(image: np.ndarray, fm: FaceMesh, r: FaceRatioResult,
                  scale: float = 1.0) -> np.ndarray:
    """顔画像に縦横寸法と理想比率を重ねる"""
    if scale != 1.0:
        img = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
        )
    else:
        img = image.copy()
    h, w = img.shape[:2]

    def S(p):
        return (float(p[0]) * scale, float(p[1]) * scale)

    p_top = S(fm.landmarks_px[LM.FOREHEAD_TOP])
    p_chin = S(fm.landmarks_px[LM.CHIN_BOTTOM])
    p_tr = S(fm.landmarks_px[LM.TEMPLE_R])
    p_tl = S(fm.landmarks_px[LM.TEMPLE_L])

    # 補正した生え際 (scale 済み)
    ext_raw = r.face_height_px - r.face_height_px_raw  # original px
    ext = ext_raw * scale
    # 方向 (上向き)
    dy_top_chin = (p_top[1] - p_chin[1])
    ext_len = abs(dy_top_chin) if abs(dy_top_chin) > 1e-3 else 1.0
    dy_unit = dy_top_chin / ext_len
    p_top_ext = (p_top[0], p_top[1] + dy_unit * ext)

    lw = max(2, int(2 * scale))

    # 縦軸 (補正前 → 補正後は破線風で)
    draw_line_outlined(img, p_top, p_chin, C_H, thickness=lw)
    draw_line_outlined(img, p_top_ext, p_top, C_E, thickness=lw)

    # 横軸
    draw_line_outlined(img, p_tr, p_tl, C_W, thickness=lw)

    for pt, col in [(p_top, C_H), (p_top_ext, C_E), (p_chin, C_H),
                    (p_tr, C_W), (p_tl, C_W)]:
        draw_point_outlined(img, pt, col, r=max(3, int(4 * scale)))

    # ---- 寸法ラベル ----
    label_bg = (20, 20, 25)
    label_size = max(16, int(18 * scale))

    draw_pil_text(
        img, f"縦 {r.face_height_px:.0f}",
        (p_chin[0] + 14, p_chin[1] - 12),
        color=C_H, size=label_size, bg=label_bg, bg_alpha=0.78, bg_pad=5,
    )
    draw_pil_text(
        img, f"横 {r.face_width_px:.0f}",
        (max(p_tr[0], p_tl[0]) + 12, (p_tr[1] + p_tl[1]) / 2 - 12),
        color=C_W, size=label_size, bg=label_bg, bg_alpha=0.78, bg_pad=5,
    )
    # aspect の値
    draw_pil_text(
        img, f"aspect {r.aspect:.2f}",
        (p_chin[0] + 14, p_chin[1] + 16),
        color=(230, 230, 235), size=max(14, int(15 * scale)),
        bg=label_bg, bg_alpha=0.78, bg_pad=5,
    )

    # ---- 左上ピル: closest ratio ----
    label_color = RATIO_COLOR.get(r.closest_ratio, (255, 255, 255))
    pill_text = RATIO_JP.get(r.closest_ratio, r.closest_ratio).upper()
    pill_size = int(24 * max(1.0, scale * 0.9))
    draw_pil_pill(
        img, pill_text, (18, 18),
        text_color=(20, 20, 30), pill_color=label_color, size=pill_size,
        pad_x=18, pad_y=10, radius=22,
    )

    # ---- 右上凡例 ----
    legend_items = [
        ("raw top", C_H),
        ("hairline +25%", C_E),
        ("横幅", C_W),
    ]
    legend_font = max(14, int(15 * scale))
    sw_w = int(18 * scale)
    sw_h = int(14 * scale)
    row_h = int(26 * scale)
    lx = w - int(170 * scale)
    ly = int(20 * scale)
    cv2.rectangle(
        img, (lx - 8, ly - 6),
        (w - 14, ly + row_h * len(legend_items) + 4),
        (0, 0, 0), -1,
    )
    for i, (name, col) in enumerate(legend_items):
        y_item = ly + i * row_h
        cv2.rectangle(
            img, (lx, y_item + 4), (lx + sw_w, y_item + 4 + sw_h), col, -1,
        )
        draw_pil_text(
            img, name, (lx + sw_w + 8, y_item),
            color=(235, 235, 240), size=legend_font,
        )

    return img


def build_panel(r: FaceRatioResult, width: int, height: int) -> np.ndarray:
    label_color = RATIO_COLOR.get(r.closest_ratio, (255, 255, 255))

    # 理想比の正規化スコア (1 - 相対ずれ)
    ideal_scores = {k: max(0.0, 1.0 - loss * 20) for k, loss in r.losses.items()}

    spec = [
        ("title", "2.2.1  縦横バランス", (230, 230, 230)),
        ("subtitle", "Face Aspect vs Ideal Ratios"),
        ("divider",),
        ("spacer", 4),
        ("section", "判定結果"),
        ("big", r.closest_ratio.upper(), label_color),
        ("text", RATIO_JP.get(r.closest_ratio, "") + "  →  " + r.impression,
            (210, 210, 215)),
        ("spacer", 6),
        ("section", "計測値"),
        ("kv", "縦 (補正後)",  f"{r.face_height_px:.0f} px"),
        ("kv", "横 (こめかみ)", f"{r.face_width_px:.0f} px"),
        ("kv", "aspect (raw)", f"{r.aspect_raw:.3f}"),
        ("kv", "aspect (ext)", f"{r.aspect:.3f}", label_color),
    ]

    if r.face_height_cm > 0:
        kogao_color = (150, 255, 180) if r.kogao_score >= 85 else (230, 230, 235)
        spec += [
            ("spacer", 6),
            ("section", "実寸換算 (虹彩 11.7mm 基準)"),
            ("kv", "縦幅",        f"{r.face_height_cm:.1f} cm"),
            ("kv", "横幅",        f"{r.face_width_cm:.1f} cm"),
            ("kv", "小顔スコア",  f"{r.kogao_score:.0f} / 100", kogao_color),
            ("kv", "",            f"[{r.kogao_label}]", kogao_color),
        ]

    spec += [
        ("spacer", 8),
        ("section", "理想比マッチ度"),
    ]

    # レーダー: 3 つの理想比
    radar_keys = list(RATIOS.keys())
    radar_labels = [RATIO_JP[k] for k in radar_keys]
    radar_vals = [ideal_scores[k] for k in radar_keys]
    best_idx = radar_keys.index(r.closest_ratio)
    # min-max 正規化
    rv_min = min(radar_vals)
    rv_max = max(radar_vals)
    rv_range = max(rv_max - rv_min, 1e-6)
    radar_norm = [0.3 + 0.7 * (v - rv_min) / rv_range for v in radar_vals]
    spec.append(("radar", radar_labels, radar_norm, best_idx, label_color))

    return render_report_panel(spec, width, height)


def build_report(image: np.ndarray, fm: FaceMesh, r: FaceRatioResult) -> np.ndarray:
    scale = 1.5
    src_big = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
    )
    ann_big = annotate_face(image, fm, r, scale=scale)
    panel = build_panel(r, 620, src_big.shape[0])
    return compose_report(src_big, ann_big, panel)


# 後方互換
def visualize(image: np.ndarray, fm: FaceMesh, r: FaceRatioResult) -> np.ndarray:
    return annotate_face(image, fm, r, scale=1.0)


# ==============================================================
# CLI
# ==============================================================
def run_one(image_path: Path, output_path: Path | None = None,
            imgonly: bool = False, as_json: bool = False):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: 画像を読み込めません: {image_path}")
        return None

    fm = FaceMesh(subdivision_level=1)
    fm.init()
    if fm.detect(image) is None:
        print("Error: 顔が検出されませんでした")
        return None

    r = analyze(fm)

    print(f"\n=== 2.2.1 顔全体の縦横バランス ===")
    print(f"aspect(raw={r.aspect_raw:.3f}, ext={r.aspect:.3f})")
    print(f"closest: {r.closest_ratio} ({RATIOS[r.closest_ratio]}) -> {r.impression}")
    print(f"losses: " + ", ".join(f"{k}={v*100:.2f}%" for k, v in r.losses.items()))
    if r.face_height_cm > 0:
        print(f"size: {r.face_height_cm:.1f}cm x {r.face_width_cm:.1f}cm")
        print(f"kogao: {r.kogao_score:.1f} [{r.kogao_label}]")

    if as_json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))

    out_path = output_path or image_path.parent / f"face_ratio_{image_path.stem}.png"
    if imgonly:
        cv2.imwrite(str(out_path), annotate_face(image, fm, r))
    else:
        cv2.imwrite(str(out_path), build_report(image, fm, r))
    print(f"出力: {out_path}")
    return r


def main():
    parser = argparse.ArgumentParser(description="2.2.1 顔全体の縦横バランス判定")
    parser.add_argument("input", help="入力画像パス")
    parser.add_argument("-o", "--output", help="出力画像パス")
    parser.add_argument("--imgonly", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    run_one(
        Path(args.input),
        Path(args.output) if args.output else None,
        imgonly=args.imgonly,
        as_json=args.json,
    )


if __name__ == "__main__":
    main()
