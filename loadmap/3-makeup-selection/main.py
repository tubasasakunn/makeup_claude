"""
3. 化粧選択 (Makeup Selection) - オーケストレーター CLI

顔画像を 1 枚受け取り、次の 3 ステップを順に実行する:

  1. Evaluator  : 2.x (2.1 〜 2.2.8) の判定を全て走らせて FaceProfile を作る
  2. Selector   : 判定結果を元に MakeupPlan (どこをどう塗るか) を決定
  3. Applicator : 1.x の apply_* を連鎖呼び出しして化粧を適用

出力:
  - <stem>_after.png      : 化粧適用後の単独画像
  - <stem>_compare.png    : before | after の比較画像
  - <stem>_profile.json   : FaceProfile + MakeupPlan + applied_log

Usage:
    python main.py <input_image> [-o output_dir]
    python main.py imgs/base.png              # デフォルト: 同ディレクトリに出力
    python main.py imgs/丸顔.png -o result/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# shared のパスを通す
_HERE = Path(__file__).resolve().parent
_LOADMAP_ROOT = _HERE.parent
if str(_LOADMAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOADMAP_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from shared.facemesh import FaceMesh  # noqa: E402

from evaluator import FaceEvaluator  # noqa: E402
from selector import MakeupSelector  # noqa: E402
from applicator import MakeupApplicator  # noqa: E402


# ==============================================================
# 比較画像生成
# ==============================================================
def make_side_by_side(before: np.ndarray, after: np.ndarray,
                      label_left: str = "Before",
                      label_right: str = "After") -> np.ndarray:
    """左右比較画像 (ラベル付き)"""
    combined = np.hstack([before, after])
    h = combined.shape[0]
    _put_label(combined, label_left, (20, 50))
    _put_label(combined, label_right, (before.shape[1] + 20, 50))
    return combined


def _put_label(img: np.ndarray, text: str, org: tuple[int, int]):
    """コントラスト付きラベル"""
    x, y = org
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2, cv2.LINE_AA)


# ==============================================================
# Pipeline
# ==============================================================
def run(image_path: Path, output_dir: Path, verbose: bool = True) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(f"入力画像が見つかりません: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"画像を読み込めません: {image_path}")

    if verbose:
        print(f"\n=== 3. Makeup Selection Pipeline ===")
        print(f"入力画像: {image_path}  {image.shape}")

    # FaceMesh 初期化 & 検出 (全段階で使い回す)
    if verbose:
        print("FaceMesh 初期化中...")
    fm = FaceMesh(subdivision_level=1)
    fm.init()
    if fm.detect(image) is None:
        raise RuntimeError("顔が検出されませんでした")

    if verbose:
        print(f"  メッシュ: {len(fm.triangles)} 三角形")

    # 1. Evaluate
    if verbose:
        print("\n[1/3] Evaluating (2.1 - 2.2.8)...")
    evaluator = FaceEvaluator()
    profile = evaluator.evaluate(fm)

    if verbose:
        summary = profile["summary"]
        print(f"  骨格タイプ : {summary['skeletal_type']} ({summary['skeletal_label']})")
        print(f"  顔比率    : {summary['closest_ratio']}  小顔度={summary['kogao_label']}")
        print(f"  三分割    : {summary['vertical_category']}")
        print(f"  五分割    : {summary['horizontal_category']}")
        print(f"  目        : {summary['eye_category']}")
        print(f"  鼻        : {summary['nose_overall']}")
        print(f"  口        : {summary['mouth_overall']}")
        print(f"  眉        : {summary['eyebrow_category']}")
        print(f"  黄金比    : {summary['golden_score']:.1f} ({summary['golden_label']})")

    # 2. Select
    if verbose:
        print("\n[2/3] Selecting makeup plan...")
    selector = MakeupSelector()
    plan = selector.select(profile)

    if verbose:
        print(f"  方針: {plan.advice}")
        print(f"  全体強度倍率: {plan.overall_modifier:.2f}")
        print(f"  化粧ステップ: {len(plan.steps)} 個")
        for line in plan.rationale:
            print(f"    - {line}")

    # 3. Apply
    if verbose:
        print("\n[3/3] Applying makeup...")
    applicator = MakeupApplicator()
    after, applied_log = applicator.apply(image, fm, plan, verbose=verbose)

    # 出力
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    path_after = output_dir / f"{stem}_after.png"
    path_compare = output_dir / f"{stem}_compare.png"
    path_profile = output_dir / f"{stem}_profile.json"

    cv2.imwrite(str(path_after), after)
    comparison = make_side_by_side(image, after)
    cv2.imwrite(str(path_compare), comparison)

    result_doc = {
        "input": str(image_path),
        "profile": profile,
        "plan": plan.to_dict(),
        "applied_log": applied_log,
        "output": {
            "after": str(path_after),
            "compare": str(path_compare),
        },
    }
    with open(path_profile, "w", encoding="utf-8") as f:
        json.dump(result_doc, f, indent=2, ensure_ascii=False, default=str)

    if verbose:
        print("\n出力:")
        print(f"  {path_after}")
        print(f"  {path_compare}")
        print(f"  {path_profile}")

    return result_doc


# ==============================================================
# CLI
# ==============================================================
def main():
    parser = argparse.ArgumentParser(
        description="3. 化粧選択: 顔判定 → 化粧プラン → 適用"
    )
    parser.add_argument("input", help="入力画像パス")
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="出力ディレクトリ (default: 入力画像と同じ場所)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="ログを抑制")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    run(input_path, out_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
