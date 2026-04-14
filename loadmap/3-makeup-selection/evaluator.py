"""
3. 化粧選択 / Evaluator

2 章の各判定モジュール (2.1 - 2.2.8) を一気に呼び出し、
単一の FaceProfile dict にまとめるオーケストレーター。

2.x の各ディレクトリは "2.1-skeletal" のようにハイフン/ドットを含むため、
通常の import は使えない。2.2.8-symmetry と同じく importlib で直接 main.py を
ロードする。FaceMesh.detect() 済みインスタンスを渡すだけで済むように、
検出とモデル初期化は呼び出し側で行う。

Usage:
    from shared.facemesh import FaceMesh
    from evaluator import FaceEvaluator

    fm = FaceMesh(subdivision_level=1)
    fm.init()
    fm.detect(image_bgr)

    ev = FaceEvaluator()
    profile = ev.evaluate(fm)
    print(profile["skeletal"]["type"])  # "oval" / "round" / "long" / ...
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any

# shared モジュールへのパスを通す (2.x 側と同じ仕組み)
_HERE = Path(__file__).resolve().parent
_LOADMAP_ROOT = _HERE.parent
if str(_LOADMAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOADMAP_ROOT))

from shared.facemesh import FaceMesh  # noqa: E402


# ==============================================================
# 2.x モジュールの動的ロード
# ==============================================================
_SECTION_ROOT = _LOADMAP_ROOT / "2"

# section_dir, attr name (analyze|classify), profile key の 3 つ組
_SECTIONS: list[tuple[str, str, str]] = [
    ("2.1-skeletal",   "classify", "skeletal"),
    ("2.2.1-face-ratio", "analyze", "face_ratio"),
    ("2.2.2-vertical", "analyze", "vertical"),
    ("2.2.3-horizontal", "analyze", "horizontal"),
    ("2.2.4-eye",      "analyze", "eye"),
    ("2.2.5-nose",     "analyze", "nose"),
    ("2.2.6-mouth",    "analyze", "mouth"),
    ("2.2.7-eyebrow",  "analyze", "eyebrow"),
    ("2.2.8-symmetry", "analyze", "symmetry"),
]


def _load_module(section_dir: str, module_name: str):
    path = _SECTION_ROOT / section_dir / "main.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass が sys.modules を参照するので登録が必要
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _to_dict(obj: Any) -> Any:
    """dataclass / 独自 to_dict / dict / list を JSON 化しやすい形へ再帰変換"""
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        try:
            return _to_dict(obj.to_dict())
        except Exception:
            pass
    if is_dataclass(obj):
        return _to_dict(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)):
        return obj
    # numpy 等
    try:
        return float(obj)
    except Exception:
        return str(obj)


class FaceEvaluator:
    """2.x 全モジュールをまとめて実行して FaceProfile を返す"""

    def __init__(self):
        self._modules: dict[str, Any] = {}
        for section_dir, _, key in _SECTIONS:
            # モジュール名は一意に (selector/applicator の import と衝突しないよう prefix)
            self._modules[key] = _load_module(section_dir, f"sec3_eval_{key}")

    def evaluate(self, fm: FaceMesh) -> dict[str, Any]:
        """FaceMesh.detect() 済みインスタンスを受け取り、全判定結果を dict で返す

        Returns:
            {
              "skeletal":   {...},
              "face_ratio": {...},
              "vertical":   {...},
              ...
              "summary":    { "skeletal_type": "oval", "golden_score": 78.2, ... }
            }
        """
        if fm.landmarks_px is None:
            raise RuntimeError(
                "FaceMesh is not detected. Call fm.detect(image) first."
            )

        profile: dict[str, Any] = {}
        for section_dir, attr, key in _SECTIONS:
            mod = self._modules[key]
            fn = getattr(mod, attr)
            result = fn(fm)
            profile[key] = _to_dict(result)

        profile["summary"] = self._make_summary(profile)
        return profile

    # ----------------------------------------------------------
    # 判定値の要約
    # ----------------------------------------------------------
    @staticmethod
    def _make_summary(profile: dict[str, Any]) -> dict[str, Any]:
        s: dict[str, Any] = {}
        skel = profile.get("skeletal", {})
        s["skeletal_type"] = skel.get("type", "unknown")
        s["skeletal_label"] = skel.get("type_label", "")

        fr = profile.get("face_ratio", {})
        s["closest_ratio"] = fr.get("closest_ratio", "")
        s["kogao_label"] = fr.get("kogao_label", "")
        s["face_aspect"] = fr.get("aspect", 0.0)

        vt = profile.get("vertical", {})
        s["vertical_category"] = vt.get("category", "")
        s["vertical_closest"] = vt.get("closest", "")

        hz = profile.get("horizontal", {})
        s["horizontal_category"] = hz.get("category", "")
        s["eye_gap_ratio"] = hz.get("eye_gap_ratio", 0.0)

        ey = profile.get("eye", {})
        s["eye_category"] = ey.get("category", "")
        s["eye_to_face_ratio"] = ey.get("eye_to_face_ratio", 0.0)

        no = profile.get("nose", {})
        s["nose_overall"] = no.get("overall", "")
        s["nose_angle"] = no.get("nose_lip_angle_deg", 0.0)

        mo = profile.get("mouth", {})
        s["mouth_overall"] = mo.get("overall", "")
        s["lip_ratio"] = mo.get("lip_ratio", 0.0)

        eb = profile.get("eyebrow", {})
        s["eyebrow_category"] = eb.get("category", "")
        s["eyebrow_symmetry"] = eb.get("symmetry_score", 0.0)

        sy = profile.get("symmetry", {})
        s["overall_sym"] = sy.get("overall_sym", 0.0)
        s["golden_score"] = sy.get("golden_score", 0.0)
        s["golden_label"] = sy.get("golden_label", "")
        return s


__all__ = ["FaceEvaluator"]
