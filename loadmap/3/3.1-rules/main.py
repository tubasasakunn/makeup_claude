"""
3.1 判定 → 処方ルール

入力画像を Phase 2.2.8 に渡して判定結果を得て、ルール表を適用して処方
(Prescription JSON) を生成する。

ルール一覧:
    R1: skeletal.type (2.1) → eyebrow_type
        oval              → natural  (バランス型、標準)
        round             → arch     (縦ラインで頬の丸みを引き締める)
        long              → parallel (水平で縦長を抑える)
        inverted_triangle → straight (シャープさを和らげる)
        base              → natural  (角張りを和らげる)

    R2: skeletal.type → highlight.areas (target.json の顔型別セット)
        round/base        → marugao 系 (縦方向ハイライト)
        long              → omonaga 系 (横方向ハイライト)
        oval/inv_triangle → base 系   (標準セット)

    R3: skeletal.type → shadow on/off & areas (小顔効果)
        round/base        → shadow [marugao-side]         (顔横を削る)
        long              → shadow [omonaga-upper/lower]  (縦を短く)
        oval/inv_triangle → shadow OFF

    R4: eye.ideal_ratio_loss (2.2.4) → eyeshadow_base intensity
        目が小さい (loss 大) ほど intensity を底上げして目力を強調

    R5: golden_score (2.2.8) → 全体 intensity スケーリング
        A (Good) 以上 → 0.85x (ナチュラル寄り)
        C 以下  (<70) → 1.15x (強調寄り)
        その他         → 1.00x

Usage:
    python main.py <input_image> [-o prescription.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2

# --- パス設定 ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOADMAP = ROOT / "loadmap"
SCHEMA_DIR = LOADMAP / "3" / "3.0-schema"
sys.path.insert(0, str(LOADMAP))
sys.path.insert(0, str(SCHEMA_DIR))

from shared.facemesh import FaceMesh  # noqa: E402
from prescription import Prescription, EyebrowRx, HighlightRx, ShadowRx  # noqa: E402


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 2.2.8 を経由すれば 2.1-2.2.7 を一括実行できる
SYMMETRY_MAIN = LOADMAP / "2" / "2.2.8-symmetry" / "main.py"
symmetry_mod = _load_module(SYMMETRY_MAIN, "symmetry_m")


# --- ルール表 ----------------------------------------------------------------
# R1: 骨格 → 眉タイプ
SKELETAL_TO_BROW = {
    "oval":              ("natural",  "卵型はバランス型。標準の緩やかカーブで自然に仕上げる"),
    "round":             ("arch",     "丸型は頬のふっくら感を引き締めるため縦ラインを作るアーチ"),
    "long":              ("parallel", "面長は縦長を抑えるため水平・太めの平行眉"),
    "inverted_triangle": ("straight", "逆三角はシャープな印象を和らげるためストレート眉"),
    "base":              ("natural",  "ベース型は角張りを和らげるため緩やかな自然カーブ"),
}

# R2: 骨格 → highlight エリアセット
BASE_HL    = ["base_t-zone", "base_c-zone", "base_under-eye",
              "base_megasira", "base_zintyuu"]
MARU_HL    = ["marugao_t-zone", "marugao_c-zone", "marugao_ago"]
OMONAGA_HL = ["omonaga_t-zone", "omonaga_c-zone"]

SKELETAL_TO_HIGHLIGHT = {
    "oval":              (BASE_HL,    "卵型は標準のハイライト配置"),
    "round":             (MARU_HL,    "丸型は縦方向ハイライト(T+C+顎)で縦長効果"),
    "long":              (OMONAGA_HL, "面長は狭めのハイライト(T+C)で縦を強調しない"),
    "inverted_triangle": (BASE_HL,    "逆三角は標準のハイライト配置(広いおでこを強調しない)"),
    "base":              (MARU_HL,    "ベース型は縦方向ハイライトで細く見せる"),
}

# R3: 骨格 → shadow ON/OFF + エリア
SKELETAL_TO_SHADOW = {
    "oval":              (False, [],                                "卵型はシャドウ不要"),
    "round":             (True,  ["marugao-side"],                  "丸型は頬サイドシャドウで小顔効果"),
    "long":              (True,  ["omonaga-upper", "omonaga-lower"], "面長は上下シャドウで縦を短く"),
    "inverted_triangle": (False, [],                                "逆三角はシャドウ不要(狭い顎を強調しない)"),
    "base":              (True,  ["marugao-side"],                  "ベース型はエラ側シャドウで横幅を削る"),
}


def _scale_from_golden(score: float) -> tuple[float, str]:
    """黄金比スコア → 全体 intensity スケール

    score >= 80 → 0.85x (ナチュラル)
    score <= 70 → 1.15x (強調)
    その他        → 1.00x
    """
    if score >= 80.0:
        return 0.85, "高スコア: ナチュラル寄り (×0.85)"
    if score <= 70.0:
        return 1.15, "低スコア: 強調寄り (×1.15)"
    return 1.00, "標準スコア: 標準の強度 (×1.00)"


def _boost_from_eye(ideal_loss: float) -> tuple[float, str]:
    """2.2.4 の ideal_ratio_loss (目縦横の理想からのズレ) → eyeshadow_base intensity 加算量

    loss=0 → +0.00、loss=0.10 → +0.20 (上限 +0.20)。
    """
    boost = min(0.20, ideal_loss * 2.0)
    return boost, f"目の小ささ補正: eyeshadow_base intensity +{boost:.2f}"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# --- 主処理 ------------------------------------------------------------------
def analyze_face(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")
    fm = FaceMesh(subdivision_level=1)
    fm.init()
    if fm.detect(image) is None:
        raise RuntimeError("顔が検出できませんでした")
    return fm, symmetry_mod.analyze(fm)


def build_prescription(sym_result) -> Prescription:
    rx = Prescription()
    rationale: list[str] = []

    skeletal = sym_result.sub_results["skeletal"]
    skel_type = skeletal.type

    # R1: 眉タイプ
    brow_type, reason_brow = SKELETAL_TO_BROW.get(
        skel_type, ("natural", f"未知の skeletal={skel_type} → natural")
    )
    rx.eyebrow = EyebrowRx(brow_type=brow_type)
    rationale.append(f"R1 skeletal={skel_type} → eyebrow={brow_type}: {reason_brow}")

    # R2: ハイライトエリア
    hl_areas, reason_hl = SKELETAL_TO_HIGHLIGHT.get(
        skel_type, (BASE_HL, "default base-*")
    )
    rx.highlight = HighlightRx(areas=list(hl_areas))
    rationale.append(f"R2 skeletal={skel_type} → highlight={hl_areas}: {reason_hl}")

    # R3: シャドウ
    sd_enabled, sd_areas, reason_sd = SKELETAL_TO_SHADOW.get(
        skel_type, (False, [], "default: OFF")
    )
    rx.shadow = ShadowRx(enabled=sd_enabled, areas=list(sd_areas))
    rationale.append(
        f"R3 skeletal={skel_type} → shadow={'ON ' + str(sd_areas) if sd_enabled else 'OFF'}: {reason_sd}"
    )

    # R4: 目の小ささに応じて eyeshadow_base intensity 加算
    eye_result = sym_result.sub_results["eye"]
    boost, reason_eye = _boost_from_eye(eye_result.ideal_ratio_loss)
    rx.eye.eyeshadow_base.intensity = _clamp(
        rx.eye.eyeshadow_base.intensity + boost, 0.10, 0.70
    )
    rationale.append(
        f"R4 eye.ideal_loss={eye_result.ideal_ratio_loss:.3f} → {reason_eye}"
    )

    # R5: 黄金比スコアで全体 intensity スケール
    scale, reason_scale = _scale_from_golden(sym_result.golden_score)
    rx.base.intensity           = _clamp(rx.base.intensity * scale,           0.05, 0.90)
    rx.highlight.intensity      = _clamp(rx.highlight.intensity * scale,      0.02, 0.50)
    rx.shadow.intensity         = _clamp(rx.shadow.intensity * scale,         0.05, 0.60)
    rx.eyebrow.intensity        = _clamp(rx.eyebrow.intensity * scale,        0.10, 1.00)
    for name in ("eyeshadow_base", "eyeshadow_crease", "lower_outer"):
        area = getattr(rx.eye, name)
        area.intensity = _clamp(area.intensity * scale, 0.05, 0.70)
    rationale.append(f"R5 golden_score={sym_result.golden_score:.1f} → {reason_scale}")

    # 処方のメタ
    rx.source = {
        "skeletal_type": skel_type,
        "skeletal_type_label": skeletal.type_label,
        "golden_score": round(sym_result.golden_score, 2),
        "golden_label": sym_result.golden_label,
        "eye_ideal_ratio_loss": round(eye_result.ideal_ratio_loss, 4),
        "intensity_scale": scale,
    }
    rx.rationale = rationale
    return rx


def run(image_path: Path, output_path: Path | None = None,
        verbose: bool = True) -> Prescription:
    _, sym_result = analyze_face(image_path)
    rx = build_prescription(sym_result)

    if output_path is None:
        output_path = image_path.parent / f"prescription_{image_path.stem}.json"
    rx.save(output_path)

    if verbose:
        print("=== 処方 (Prescription) ===")
        print(f"source: {json.dumps(rx.source, ensure_ascii=False, indent=2)}")
        print("rationale:")
        for r in rx.rationale:
            print(f"  - {r}")
        print(f"eyebrow: type={rx.eyebrow.brow_type} color={rx.eyebrow.color_rgb} "
              f"intensity={rx.eyebrow.intensity:.2f}")
        print(f"highlight: areas={rx.highlight.areas} intensity={rx.highlight.intensity:.2f}")
        print(f"shadow: enabled={rx.shadow.enabled} areas={rx.shadow.areas} "
              f"intensity={rx.shadow.intensity:.2f}")
        print(f"saved: {output_path}")
    return rx


def main():
    p = argparse.ArgumentParser(description="3.1 判定 → 処方")
    p.add_argument("input", help="入力画像パス")
    p.add_argument("-o", "--output", help="出力 JSON パス")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()
    run(
        Path(args.input),
        Path(args.output) if args.output else None,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
