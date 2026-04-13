"""
3. 化粧選択 / Applicator

MakeupPlan に従って 1.x の各 apply_* 関数を連鎖呼び出しして、
BGR numpy 配列を順番に更新していく。

1.x モジュールはディレクトリ名がハイフン/ドット区切りのため、
evaluator.py と同じく importlib で直接ロードする。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# shared モジュールのパス
_HERE = Path(__file__).resolve().parent
_LOADMAP_ROOT = _HERE.parent
_PROJECT_ROOT = _LOADMAP_ROOT.parent
if str(_LOADMAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOADMAP_ROOT))

from shared.facemesh import FaceMesh  # noqa: E402

from selector import MakeupPlan, MakeupStep  # noqa: E402


# ==============================================================
# 1.x モジュールロード
# ==============================================================
_MAKEUP_ROOT = _LOADMAP_ROOT / "1-virtual-makeup"
_TARGET_JSON = _PROJECT_ROOT / "target.json"


def _load(section_dir: str, module_name: str):
    path = _MAKEUP_ROOT / section_dir / "main.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# モジュールはクラスインスタンスで遅延ロードする (import 時副作用を避ける)
class _Makeup:
    def __init__(self):
        self.highlight = _load("1-1-highlight", "sec3_apply_highlight")
        self.shadow = _load("1-2-shadow", "sec3_apply_shadow")
        self.base = _load("1-3-base", "sec3_apply_base")
        self.eye = _load("1-4-eye", "sec3_apply_eye")
        self.eyebrow = _load("1-5-eyebrow", "sec3_apply_eyebrow")


# ==============================================================
# target.json アクセス
# ==============================================================
def _load_target_areas() -> dict[str, dict[str, list[int] | dict]]:
    """target.json 全体を {category: {name: mesh_ids or polyline_dict}} で返す"""
    with open(_TARGET_JSON, encoding="utf-8") as f:
        data = json.load(f)
    result: dict[str, dict[str, Any]] = {}
    for category in ("highlight", "shadow", "eye", "eyebrow"):
        areas: dict[str, Any] = {}
        for entry in data.get(category, []):
            if entry.get("type") == "polyline":
                areas[entry["name"]] = entry
                continue
            ids = entry["mesh_id"]
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            areas[entry["name"]] = ids
        result[category] = areas
    return result


def _resolve_area_names(category_map: dict[str, Any], area_key: str) -> list[str]:
    """area_key がプリセット名 (末尾が '_' または '-') ならその接頭辞で絞り込む。

    末尾が _ / - なら前方一致のプリセット、それ以外は単一エリア名として扱う。
    """
    names = list(category_map.keys())
    if area_key.endswith("_") or area_key.endswith("-"):
        return [n for n in names if n.startswith(area_key)]
    if area_key in category_map:
        return [area_key]
    # プリセット prefix として試す
    matches = [n for n in names if n.startswith(area_key)]
    return matches


# ==============================================================
# Applicator
# ==============================================================
class MakeupApplicator:
    """MakeupPlan を連鎖適用して BGR 画像を返す"""

    def __init__(self):
        self._mod = _Makeup()
        self._areas = _load_target_areas()

    def apply(
        self,
        image: np.ndarray,
        fm: FaceMesh,
        plan: MakeupPlan,
        verbose: bool = True,
    ) -> tuple[np.ndarray, list[dict]]:
        """plan.steps の順にメイクを重ねた画像を返す

        Returns:
            (final_image, applied_log)
            applied_log は [{"stage", "area", "intensity"}, ...]
        """
        current = image.copy()
        h, w = current.shape[:2]
        log: list[dict] = []

        for step in plan.steps:
            before = current
            applied_names: list[str] = []

            if step.stage == "base":
                current = self._apply_base(current, fm, step)
                applied_names.append("full_face")

            elif step.stage == "highlight":
                resolved = _resolve_area_names(
                    self._areas["highlight"], step.area
                )
                for name in resolved:
                    mesh_ids = self._areas["highlight"][name]
                    if isinstance(mesh_ids, dict):
                        continue  # polyline は無視
                    current = self._mod.highlight.apply_highlight(
                        current, fm, mesh_ids,
                        color_bgr=_rgb_to_bgr(step.color_rgb),
                        intensity=step.intensity,
                    )
                    applied_names.append(name)

            elif step.stage == "shadow":
                resolved = _resolve_area_names(
                    self._areas["shadow"], step.area
                )
                for name in resolved:
                    mesh_ids = self._areas["shadow"][name]
                    if isinstance(mesh_ids, dict):
                        continue
                    current = self._mod.shadow.apply_shadow(
                        current, fm, mesh_ids,
                        color_bgr=_rgb_to_bgr(step.color_rgb),
                        intensity=step.intensity,
                    )
                    applied_names.append(name)

            elif step.stage == "eye":
                current = self._apply_eye(current, fm, step, w, h)
                applied_names.append(step.area)

            elif step.stage == "eyebrow":
                current = self._mod.eyebrow.apply_eyebrow_makeup(
                    current, fm,
                    brow_type=step.area,
                    color_rgb=step.color_rgb,
                    intensity=step.intensity,
                )
                applied_names.append(step.area)

            else:
                if verbose:
                    print(f"  [skip] unknown stage: {step.stage}")
                continue

            if verbose:
                diff = float(np.mean(np.abs(
                    current.astype(np.int16) - before.astype(np.int16)
                )))
                print(
                    f"  [{step.stage:9s}] "
                    f"areas={','.join(applied_names) or '-':20s} "
                    f"intensity={step.intensity:.3f} diff={diff:.2f}"
                )

            log.append({
                "stage": step.stage,
                "areas": applied_names,
                "intensity": step.intensity,
                "color_rgb": list(step.color_rgb),
            })

        return current, log

    # ------------------------------------------------------
    # 個別ステージ
    # ------------------------------------------------------
    def _apply_base(self, image: np.ndarray, fm: FaceMesh, step: MakeupStep) -> np.ndarray:
        return self._mod.base.apply_base(
            image, fm,
            color_bgr=_rgb_to_bgr(step.color_rgb),
            intensity=step.intensity,
        )

    def _apply_eye(
        self,
        image: np.ndarray,
        fm: FaceMesh,
        step: MakeupStep,
        w: int,
        h: int,
    ) -> np.ndarray:
        name = step.area
        eye_mod = self._mod.eye
        areas = self._areas["eye"]
        if name not in areas:
            return image
        entry = areas[name]

        if isinstance(entry, dict):
            # polyline (eyeliner)
            t_scale = 0.2  # 1-4-eye の AREA_DEFAULTS と同じ
            mask = eye_mod.build_eyeliner_mask(fm, entry, w, h, thickness_scale=t_scale)
        else:
            mask = fm.build_mask(entry, w, h)

        return eye_mod.apply_eye_area(
            image, fm, mask,
            color_bgr=_rgb_to_bgr(step.color_rgb),
            intensity=step.intensity,
            blur_scale=0.5,  # 共通のソフト境界
            blend=step.blend,
        )


# ==============================================================
# 補助
# ==============================================================
def _rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


__all__ = ["MakeupApplicator"]
