"""
2.2.8 左右対称性・骨格感の総合判定

計測項目:
    - パーツごとの対称性 (目・眉・鼻・口・輪郭)
    - 総合シンメトリースコア
    - 顎ライン (ジョーライン) の鮮明度 (エラ→あご先 の直線性)
    - 頬の黄金比: 小鼻→輪郭 vs 黒目真下→輪郭 の比率 (理想 1:2)
    - 全パーツの判定結果を統合した総合スコア

2.2.1-2.2.7 の各サブセクションを実行して結果を集約する。

Usage:
    python main.py <input_image>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "loadmap"))
from shared.facemesh import FaceMesh
from shared.face_metrics import (
    LM,
    draw_line,
    draw_point,
    make_side_by_side,
    measure,
    put_label,
)

# 各サブセクションをインポート
SECTION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SECTION_ROOT / "2.1-skeletal"))
sys.path.insert(0, str(SECTION_ROOT / "2.2.1-face-ratio"))
sys.path.insert(0, str(SECTION_ROOT / "2.2.2-vertical"))
sys.path.insert(0, str(SECTION_ROOT / "2.2.3-horizontal"))
sys.path.insert(0, str(SECTION_ROOT / "2.2.4-eye"))
sys.path.insert(0, str(SECTION_ROOT / "2.2.5-nose"))
sys.path.insert(0, str(SECTION_ROOT / "2.2.6-mouth"))
sys.path.insert(0, str(SECTION_ROOT / "2.2.7-eyebrow"))

# 各サブセクションは同名の `main` モジュール。
# sys.path の先頭から順に探すので、import は "同名上書き問題" を避けるために
# importlib で直接ファイル読み込み。
import importlib.util


def _load(section: str, module_name: str):
    path = SECTION_ROOT / section / "main.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass が mod.__module__ → sys.modules lookup するので登録が必須
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


skeletal_mod = _load("2.1-skeletal", "skeletal_m")
face_ratio_mod = _load("2.2.1-face-ratio", "face_ratio_m")
vertical_mod = _load("2.2.2-vertical", "vertical_m")
horizontal_mod = _load("2.2.3-horizontal", "horizontal_m")
eye_mod = _load("2.2.4-eye", "eye_m")
nose_mod = _load("2.2.5-nose", "nose_m")
mouth_mod = _load("2.2.6-mouth", "mouth_m")
eyebrow_mod = _load("2.2.7-eyebrow", "eyebrow_m")


@dataclass
class SymmetryResult:
    # パーツごと対称性 (0-1)
    eye_sym: float = 0.0
    brow_sym: float = 0.0
    face_contour_sym: float = 0.0    # 左右の頬・顎幅対称
    overall_sym: float = 0.0

    # 輪郭
    jaw_line_sharpness: float = 0.0  # ジョーラインの直線度 (0-1, 1=完全直線)
    cheek_ratio_r: float = 0.0       # 右の 小鼻→輪郭 : 黒目下→輪郭
    cheek_ratio_l: float = 0.0

    # サブセクション総合
    sub_results: dict = field(default_factory=dict)

    # 総合黄金比スコア (0-100)
    golden_score: float = 0.0
    golden_label: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        # ネスト dataclass は to_dict 優先
        sub = {}
        for k, v in self.sub_results.items():
            if hasattr(v, "to_dict"):
                sub[k] = v.to_dict()
            else:
                sub[k] = v
        d["sub_results"] = sub
        return d


def _sym(a: float, b: float) -> float:
    m = (a + b) / 2
    if m < 1e-6:
        return 1.0
    return max(0.0, 1.0 - abs(a - b) / m)


def _jaw_line_sharpness(fm) -> float:
    """エラ(gonion) → あご先(chin) の直線性

    エラからあご先の直線と、中間の輪郭点との平均距離を評価。
    0 なら完全直線、1 に近いほど角度がある。
    """
    mids = [172, 136, 150, 149, 176, 148]  # 右顎下輪郭
    g = fm.landmarks_px[LM.GONION_R].astype(np.float64)
    c = fm.landmarks_px[LM.CHIN_BOTTOM].astype(np.float64)
    ab = c - g
    ab_len = np.linalg.norm(ab)
    if ab_len < 1e-3:
        return 0.0
    dists = []
    for idx in mids:
        if idx < len(fm.landmarks_px):
            p = fm.landmarks_px[idx].astype(np.float64)
            cross = ab[0] * (p[1] - g[1]) - ab[1] * (p[0] - g[0])
            dists.append(abs(cross) / ab_len)
    if not dists:
        return 0.0
    # 線からの平均距離を顎長さで正規化 → 小さいほど鮮明
    avg = np.mean(dists) / ab_len
    return max(0.0, 1.0 - avg * 4.0)


def analyze(fm: FaceMesh) -> SymmetryResult:
    r = SymmetryResult()

    # 各サブセクションを実行（可視化はしない）
    r.sub_results["skeletal"] = skeletal_mod.classify(fm)
    r.sub_results["face_ratio"] = face_ratio_mod.analyze(fm)
    r.sub_results["vertical"] = vertical_mod.analyze(fm)
    r.sub_results["horizontal"] = horizontal_mod.analyze(fm)
    r.sub_results["eye"] = eye_mod.analyze(fm)
    r.sub_results["nose"] = nose_mod.analyze(fm)
    r.sub_results["mouth"] = mouth_mod.analyze(fm)
    r.sub_results["eyebrow"] = eyebrow_mod.analyze(fm)

    r.eye_sym = r.sub_results["eye"].symmetry_score
    r.brow_sym = r.sub_results["eyebrow"].symmetry_score

    # 左右の頬幅対称性
    cr = fm.landmarks_px[LM.CHEEKBONE_R].astype(np.float64)
    cl = fm.landmarks_px[LM.CHEEKBONE_L].astype(np.float64)
    nose = fm.landmarks_px[LM.NOSE_TIP].astype(np.float64)
    r_half = abs(cr[0] - nose[0])
    l_half = abs(cl[0] - nose[0])
    r.face_contour_sym = _sym(r_half, l_half)

    r.overall_sym = float(np.mean([r.eye_sym, r.brow_sym, r.face_contour_sym]))

    # ジョーライン
    r.jaw_line_sharpness = _jaw_line_sharpness(fm)

    # 頬の黄金比: 小鼻→輪郭 vs 黒目真下→輪郭
    face_m = measure(fm)
    nose_r = fm.landmarks_px[LM.NOSE_WING_R].astype(np.float64)
    nose_l = fm.landmarks_px[LM.NOSE_WING_L].astype(np.float64)
    iris_r_c = np.mean([fm.landmarks_px[i] for i in LM.IRIS_R], axis=0)
    iris_l_c = np.mean([fm.landmarks_px[i] for i in LM.IRIS_L], axis=0)
    r_contour = fm.landmarks_px[LM.GONION_R].astype(np.float64)
    l_contour = fm.landmarks_px[LM.GONION_L].astype(np.float64)

    r_nose_to_contour = abs(r_contour[0] - nose_r[0])
    r_iris_to_contour = abs(r_contour[0] - iris_r_c[0])
    l_nose_to_contour = abs(l_contour[0] - nose_l[0])
    l_iris_to_contour = abs(l_contour[0] - iris_l_c[0])

    if r_nose_to_contour > 1e-3:
        r.cheek_ratio_r = r_iris_to_contour / r_nose_to_contour
    if l_nose_to_contour > 1e-3:
        r.cheek_ratio_l = l_iris_to_contour / l_nose_to_contour

    # 総合黄金比スコア (0-100)
    weights = {
        "face_ratio_loss": 15,    # 縦横比
        "vertical_loss": 15,      # 三分割
        "horizontal_loss": 10,    # 五分割
        "eye_ratio_loss": 10,     # 目縦横
        "eye_sym": 5,
        "nose_wing_loss": 5,
        "nose_len_loss": 5,
        "mouth_lip": 5,
        "mouth_phil": 5,
        "brow_peak_loss": 5,
        "brow_sym": 5,
        "contour_sym": 5,
        "jaw_sharpness": 10,
    }

    def _score_loss(loss: float, ideal: float = 0.0) -> float:
        # loss 0 → 100, loss 0.3 → 0
        return max(0.0, 100.0 - loss * 333.3)

    score = 0.0
    face_ratio = r.sub_results["face_ratio"]
    vertical = r.sub_results["vertical"]
    horizontal = r.sub_results["horizontal"]
    eye = r.sub_results["eye"]
    nose = r.sub_results["nose"]
    mouth = r.sub_results["mouth"]
    eyebrow = r.sub_results["eyebrow"]

    score += _score_loss(face_ratio.losses[face_ratio.closest_ratio]) * weights["face_ratio_loss"]
    score += _score_loss(min(vertical.traditional_loss, vertical.reiwa_loss)) * weights["vertical_loss"]
    score += _score_loss(min(horizontal.ideal_loss_1, horizontal.jp_loss)) * weights["horizontal_loss"]
    score += _score_loss(eye.ideal_ratio_loss * 3) * weights["eye_ratio_loss"]
    score += eye.symmetry_score * 100 * weights["eye_sym"]
    score += _score_loss(nose.wing_loss) * weights["nose_wing_loss"]
    score += _score_loss(nose.length_loss) * weights["nose_len_loss"]
    score += _score_loss(abs(mouth.lip_ratio - 1.75) / 1.75) * weights["mouth_lip"]
    score += _score_loss(abs(mouth.philtrum_ratio - 2.0) / 2.0) * weights["mouth_phil"]
    score += _score_loss(abs(eyebrow.right.peak_ratio - 2.0) / 2.0) * weights["brow_peak_loss"]
    score += eyebrow.symmetry_score * 100 * weights["brow_sym"]
    score += r.face_contour_sym * 100 * weights["contour_sym"]
    score += r.jaw_line_sharpness * 100 * weights["jaw_sharpness"]

    total_weight = sum(weights.values())
    r.golden_score = score / total_weight

    if r.golden_score >= 85:
        r.golden_label = "S (Excellent)"
    elif r.golden_score >= 75:
        r.golden_label = "A (Good)"
    elif r.golden_score >= 65:
        r.golden_label = "B (Average)"
    else:
        r.golden_label = "C (Needs Work)"
    return r


# ==============================================================
# 可視化
# ==============================================================
def visualize(image: np.ndarray, fm: FaceMesh, r: SymmetryResult) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]

    # 顔中央縦線
    nose = fm.landmarks_px[LM.NOSE_TIP]
    draw_line(img, (int(nose[0]), 0), (int(nose[0]), h), (255, 100, 100), 1)

    # 輪郭対称線
    cr = fm.landmarks_px[LM.CHEEKBONE_R]
    cl = fm.landmarks_px[LM.CHEEKBONE_L]
    draw_line(img, cr, cl, (100, 200, 255), 1)

    # ジョーライン
    gr = fm.landmarks_px[LM.GONION_R]
    gl = fm.landmarks_px[LM.GONION_L]
    chin = fm.landmarks_px[LM.CHIN_BOTTOM]
    draw_line(img, gr, chin, (255, 200, 0), 2)
    draw_line(img, gl, chin, (255, 200, 0), 2)

    # 頬黄金比の基準線
    nose_r = fm.landmarks_px[LM.NOSE_WING_R]
    iris_r_c = np.mean([fm.landmarks_px[i] for i in LM.IRIS_R], axis=0)
    draw_line(img, nose_r, gr, (180, 180, 255), 1)
    draw_line(img, (int(iris_r_c[0]), int(iris_r_c[1])), gr, (255, 100, 255), 1)

    # パネル
    panel_w = int(w * 0.55)
    panel_h = int(h * 0.52)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (32, 32, 32)
    put_label(panel, "2.2.8 OVERALL GOLDEN SCORE",
              (10, 22), color=(0, 255, 255), scale=0.55, thickness=2)
    put_label(panel, f"TOTAL: {r.golden_score:.1f}  [{r.golden_label}]",
              (10, 42), color=(255, 255, 0), scale=0.55, thickness=2)

    sk = r.sub_results["skeletal"]
    fr = r.sub_results["face_ratio"]
    vr = r.sub_results["vertical"]
    hr = r.sub_results["horizontal"]
    er = r.sub_results["eye"]
    nr = r.sub_results["nose"]
    mr = r.sub_results["mouth"]
    br = r.sub_results["eyebrow"]

    lines = [
        f"2.1  skeletal   : {sk.type}",
        f"2.2.1 face      : aspect={fr.aspect:.2f} closest={fr.closest_ratio}",
        f"2.2.2 vertical  : {vr.category}",
        f"2.2.3 horizontal: gap={hr.eye_gap_ratio:.2f} [{hr.category}]",
        f"2.2.4 eye       : h/w={er.mean_width_ratio:.3f} sym={er.symmetry_score*100:.0f}%",
        f"2.2.5 nose      : wing/gap={nr.wing_to_gap_ratio:.2f} [{nr.overall}]",
        f"2.2.6 mouth     : lip={mr.lip_ratio:.2f} [{mr.overall}]",
        f"2.2.7 eyebrow   : sym={br.symmetry_score*100:.0f}% [{br.category}]",
        f"",
        f"eye sym         : {r.eye_sym*100:.1f}%",
        f"brow sym        : {r.brow_sym*100:.1f}%",
        f"contour sym     : {r.face_contour_sym*100:.1f}%",
        f"OVERALL SYM     : {r.overall_sym*100:.1f}%",
        f"jaw sharpness   : {r.jaw_line_sharpness*100:.1f}%",
        f"cheek ratio R   : {r.cheek_ratio_r:.2f}  L: {r.cheek_ratio_l:.2f}",
    ]
    for i, line in enumerate(lines):
        put_label(panel, line, (10, 66 + i * 15), scale=0.38)

    y0 = h - panel_h - 10
    x0 = 10
    if y0 >= 0 and x0 + panel_w <= w:
        img[y0:y0 + panel_h, x0:x0 + panel_w] = panel
    return img


def run_one(image_path: Path, output_path=None, imgonly=False, as_json=False):
    image = cv2.imread(str(image_path))
    if image is None: return None
    fm = FaceMesh(subdivision_level=1); fm.init()
    if fm.detect(image) is None:
        print("顔未検出"); return None
    r = analyze(fm)
    print(f"\n=== 2.2.8 総合判定 ===")
    print(f"GOLDEN SCORE: {r.golden_score:.1f}  [{r.golden_label}]")
    print(f"overall sym: {r.overall_sym*100:.1f}%  jaw: {r.jaw_line_sharpness*100:.1f}%")
    if as_json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False, default=str))
    vis = visualize(image, fm, r)
    out = output_path or image_path.parent / f"symmetry_{image_path.stem}.png"
    cv2.imwrite(str(out), vis if imgonly else make_side_by_side(image, vis))
    print(f"出力: {out}")
    return r


def main():
    parser = argparse.ArgumentParser(description="2.2.8 総合判定")
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--imgonly", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    run_one(Path(args.input), Path(args.output) if args.output else None,
            imgonly=args.imgonly, as_json=args.json)


if __name__ == "__main__":
    main()
