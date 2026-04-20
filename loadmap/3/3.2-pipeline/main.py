"""
3.2 処方パイプライン (Prescription → Phase 1 各ステップ順次適用)

処方 JSON または Prescription オブジェクトを受け取り、
base → highlight → shadow → eye → eyebrow の順で Phase 1 の関数を
呼び出して 1 枚の完成画像を生成する。

Usage:
    # 処方 JSON を既に持っている場合
    python main.py <input_image> --rx prescription.json -o out.png

    # 判定から処方まで一気通貫 (3.1 ルールを内部で呼ぶ)
    python main.py <input_image> --auto -o out.png
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOADMAP = ROOT / "loadmap"
sys.path.insert(0, str(LOADMAP))
sys.path.insert(0, str(LOADMAP / "3" / "3.0-schema"))
sys.path.insert(0, str(LOADMAP / "3" / "3.1-rules"))

from shared.facemesh import FaceMesh  # noqa: E402
from prescription import Prescription  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Phase 1 モジュール
hl_mod = _load(LOADMAP / "1-virtual-makeup/1-1-highlight/main.py", "hl_mod")
sd_mod = _load(LOADMAP / "1-virtual-makeup/1-2-shadow/main.py",    "sd_mod")
bs_mod = _load(LOADMAP / "1-virtual-makeup/1-3-base/main.py",      "bs_mod")
ey_mod = _load(LOADMAP / "1-virtual-makeup/1-4-eye/main.py",       "ey_mod")
br_mod = _load(LOADMAP / "1-virtual-makeup/1-5-eyebrow/main.py",   "br_mod")


# ---------------------------------------------------------------
# 個別ステップ適用
# ---------------------------------------------------------------
def _merge_mesh_ids(area_names: list[str],
                    area_dict: dict[str, list[int]]) -> list[int]:
    """処方で指定されたエリア名群に対応する mesh_id を重複除去で束ねる"""
    seen: set[int] = set()
    merged: list[int] = []
    for name in area_names:
        ids = area_dict.get(name)
        if ids is None:
            print(f"  Warning: area '{name}' not found in target.json, skipping")
            continue
        for i in ids:
            if i not in seen:
                seen.add(i)
                merged.append(i)
    return merged


def _rgb_to_bgr(rgb) -> tuple:
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def apply_base(image: np.ndarray, fm: FaceMesh, rx_base) -> np.ndarray:
    if not rx_base.enabled:
        return image
    return bs_mod.apply_base(
        image, fm,
        color_bgr=tuple(rx_base.color_bgr),
        intensity=rx_base.intensity,
        blur_scale=rx_base.blur_scale,
    )


def apply_highlight(image: np.ndarray, fm: FaceMesh, rx_hl) -> np.ndarray:
    if not rx_hl.enabled or not rx_hl.areas:
        return image
    areas = hl_mod.load_target_areas("highlight")
    mesh_ids = _merge_mesh_ids(list(rx_hl.areas), areas)
    if not mesh_ids:
        return image
    return hl_mod.apply_highlight(
        image, fm, mesh_ids=mesh_ids,
        color_bgr=tuple(rx_hl.color_bgr),
        intensity=rx_hl.intensity,
        blur_scale=rx_hl.blur_scale,
    )


def apply_shadow(image: np.ndarray, fm: FaceMesh, rx_sd) -> np.ndarray:
    if not rx_sd.enabled or not rx_sd.areas:
        return image
    areas = sd_mod.load_target_areas("shadow")
    mesh_ids = _merge_mesh_ids(list(rx_sd.areas), areas)
    if not mesh_ids:
        return image
    return sd_mod.apply_shadow(
        image, fm, mesh_ids=mesh_ids,
        color_bgr=tuple(rx_sd.color_bgr),
        intensity=rx_sd.intensity,
        blur_scale=rx_sd.blur_scale,
    )


def apply_eye(image: np.ndarray, fm: FaceMesh, rx_eye) -> np.ndarray:
    mesh_areas, eyeliner_data = ey_mod.load_eye_areas()
    h, w = image.shape[:2]
    out = image

    # eyeshadow_base / eyeshadow_crease / tear_bag / lower_outer (メッシュ系)
    for field in ("eyeshadow_base", "eyeshadow_crease", "tear_bag", "lower_outer"):
        rx_area = getattr(rx_eye, field)
        if not rx_area.enabled:
            continue
        if field not in mesh_areas:
            print(f"  Warning: eye area '{field}' not in target.json, skipping")
            continue
        mask = fm.build_mask(mesh_areas[field], w, h)
        out = ey_mod.apply_eye_area(
            out, fm, mask,
            color_bgr=_rgb_to_bgr(rx_area.color_rgb),
            intensity=rx_area.intensity,
            blur_scale=rx_area.blur_scale,
            blend=rx_area.blend,
        )

    # eyeliner (polyline)
    if rx_eye.eyeliner.enabled and eyeliner_data is not None:
        mask = ey_mod.build_eyeliner_mask(
            fm, eyeliner_data, w, h,
            thickness_scale=rx_eye.eyeliner.thickness_scale,
        )
        out = ey_mod.apply_eye_area(
            out, fm, mask,
            color_bgr=_rgb_to_bgr(rx_eye.eyeliner.color_rgb),
            intensity=rx_eye.eyeliner.intensity,
            blur_scale=rx_eye.eyeliner.blur_scale,
            blend=rx_eye.eyeliner.blend,
        )

    return out


def apply_eyebrow(image: np.ndarray, fm: FaceMesh, rx_brow) -> np.ndarray:
    if not rx_brow.enabled:
        return image
    return br_mod.apply_eyebrow_makeup(
        image, fm,
        brow_type=rx_brow.brow_type,
        color_rgb=tuple(rx_brow.color_rgb),
        intensity=rx_brow.intensity,
    )


# ---------------------------------------------------------------
# 全体適用
# ---------------------------------------------------------------
def apply_prescription(image: np.ndarray, fm: FaceMesh,
                       rx: Prescription, verbose: bool = False) -> np.ndarray:
    """処方を順次適用して完成画像を返す

    順序: base → highlight → shadow → eye → eyebrow
    （ベースで肌を整えてから、光→影→色味→形 の順で重ねる）
    """
    steps = [
        ("base",      rx.base,       apply_base),
        ("highlight", rx.highlight,  apply_highlight),
        ("shadow",    rx.shadow,     apply_shadow),
        ("eye",       rx.eye,        apply_eye),
        ("eyebrow",   rx.eyebrow,    apply_eyebrow),
    ]
    out = image
    for name, rx_sec, fn in steps:
        if verbose:
            print(f"  step: {name}")
        out = fn(out, fm, rx_sec)
    return out


def run(image_path: Path, prescription: Prescription,
        output_path: Path | None = None, verbose: bool = False) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    fm = FaceMesh(subdivision_level=1)
    fm.init()
    if fm.detect(image) is None:
        raise RuntimeError("顔が検出できませんでした")
    out = apply_prescription(image, fm, prescription, verbose=verbose)
    if output_path is None:
        output_path = image_path.parent / f"makeup_{image_path.stem}.png"
    cv2.imwrite(str(output_path), out)
    if verbose:
        print(f"saved: {output_path}")
    return out


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="3.2 処方 → Phase 1 パイプライン")
    p.add_argument("input", help="入力画像")
    p.add_argument("--rx", help="処方 JSON のパス")
    p.add_argument("--auto", action="store_true",
                   help="3.1 ルールで自動生成した処方を使う")
    p.add_argument("-o", "--output", help="出力画像パス")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    if not args.rx and not args.auto:
        p.error("--rx か --auto のどちらかが必要です")

    if args.auto:
        import importlib
        rules_main = importlib.import_module("main")  # 3.1-rules/main.py
        _, sym = rules_main.analyze_face(Path(args.input))
        rx = rules_main.build_prescription(sym)
    else:
        rx = Prescription.load(Path(args.rx))

    run(
        Path(args.input), rx,
        Path(args.output) if args.output else None,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
