"""
3.1 判定 → 処方ルール (最小版: 骨格 → 眉タイプ)

入力画像を Phase 2.2.8 に渡して判定結果を得て、ルール表を適用して処方
(Prescription JSON) を生成する。

現時点のルール:
    skeletal.type (2.1)
        oval              → eyebrow_type = natural    (バランス型、そのまま)
        round             → eyebrow_type = arch       (縦ラインを作り頬のふっくらを引き締める)
        long              → eyebrow_type = parallel   (水平・太めで縦長を抑える)
        inverted_triangle → eyebrow_type = straight   (シャープさを和らげる)
        base              → eyebrow_type = natural    (角張りを和らげる緩やかカーブ)

Usage:
    python main.py <input_image> [-o prescription.json]

Examples:
    python main.py imgs/卵.png
    python main.py imgs/逆三角.png -o out.json
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
from prescription import Prescription, EyebrowRx  # noqa: E402


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
SKELETAL_TO_BROW = {
    "oval":              ("natural",  "卵型はバランス型。標準の緩やかカーブで自然に仕上げる"),
    "round":             ("arch",     "丸型は頬のふっくら感を引き締めるため縦ラインを作るアーチ"),
    "long":              ("parallel", "面長は縦長を抑えるため水平・太めの平行眉"),
    "inverted_triangle": ("straight", "逆三角はシャープな印象を和らげるためストレート眉"),
    "base":              ("natural",  "ベース型は角張りを和らげるため緩やかな自然カーブ"),
}


# --- 主処理 ------------------------------------------------------------------
def analyze_face(image_path: Path) -> tuple[FaceMesh, "symmetry_mod.SymmetryResult"]:
    """画像を読み込み Phase 2 を一括実行"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")
    fm = FaceMesh(subdivision_level=1)
    fm.init()
    if fm.detect(image) is None:
        raise RuntimeError("顔が検出できませんでした")
    result = symmetry_mod.analyze(fm)
    return fm, result


def build_prescription(sym_result) -> Prescription:
    """SymmetryResult → Prescription へのルール適用"""
    rx = Prescription()

    skeletal = sym_result.sub_results["skeletal"]
    skel_type = skeletal.type

    brow_type, reason = SKELETAL_TO_BROW.get(
        skel_type, ("natural", "unknown skeletal type → default natural")
    )
    rx.eyebrow = EyebrowRx(brow_type=brow_type)

    rx.source = {
        "skeletal_type": skel_type,
        "skeletal_type_label": skeletal.type_label,
        "skeletal_scores": skeletal.scores,
        "golden_score": round(sym_result.golden_score, 2),
        "golden_label": sym_result.golden_label,
    }
    rx.rationale = [f"skeletal={skel_type} → eyebrow={brow_type}: {reason}"]
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
              f"intensity={rx.eyebrow.intensity}")
        print(f"saved: {output_path}")
    return rx


def main():
    p = argparse.ArgumentParser(description="3.1 判定 → 処方 (最小版)")
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
