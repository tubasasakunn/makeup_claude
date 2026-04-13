"""
3. 化粧選択 / Selector

evaluator.py が出力した FaceProfile を入力に取り、
どの化粧工程をどのパラメータで適用するかを決定する。

判定と化粧工程の対応ルール:

1. 骨格タイプ (2.1) → シャドウ/ハイライトのプリセット + 眉タイプ
    - oval (卵型)              : base highlight のみ, straight brow
    - round (丸型)             : marugao highlight + marugao-side shadow, arch brow
    - long (面長)              : omonaga highlight + omonaga shadow, parallel brow
    - inverted_triangle (逆三角): base highlight + ジョーライン緩和, natural brow
    - base (ベース型)          : marugao highlight + marugao-side shadow, corner brow

2. 目 (2.2.4) → アイメイク強度
    - Big-Round Eyes  : eyeshadow 強度を控えめ (line を細く)
    - Narrow Eyes     : eyeshadow + eyeliner を強めにして目を大きく見せる
    - Balanced        : デフォルト

3. 黄金比スコア (2.2.8) → 全体補正
    - 80 以上: そもそも整っているので弱め (-15%)
    - 70-80  : 標準
    - 70 未満: 強め (+15%) で引き締める

4. 眉 (2.2.7) → 現在の眉カテゴリ
    - 骨格タイプで決まった形状を採用する (現状眉のカテゴリは参考情報)

MakeupPlan は dict 形式で applicator.py に渡される。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ==============================================================
# 骨格タイプ別プリセット
# ==============================================================
# target.json のエリア名と対応させる
# 卵型は「理想型」なので控えめのベースハイライトのみ
SKELETAL_PRESETS: dict[str, dict[str, Any]] = {
    "oval": {
        "label": "卵型 - 理想的なバランス",
        "highlight_presets": ["base_"],
        "shadow_presets": [],
        "eyebrow_type": "straight",
        "advice": "すでに整っているので余計な輪郭補正はせず、ベースのみ。",
    },
    "round": {
        "label": "丸型 - 縦を強調してシャープに",
        "highlight_presets": ["base_", "marugao_"],
        "shadow_presets": ["marugao-side"],
        "eyebrow_type": "arch",
        "advice": "サイドにシャドウで面を絞り、中央にハイライトで縦を強調。",
    },
    "long": {
        "label": "面長 - 横幅を広げて縦を抑える",
        "highlight_presets": ["base_", "omonaga_"],
        "shadow_presets": ["omonaga-upper", "omonaga-lower"],
        "eyebrow_type": "parallel",
        "advice": "おでこと顎先にシャドウを入れて縦幅を圧縮。平行眉で横強調。",
    },
    "inverted_triangle": {
        "label": "逆三角 - 顎まわりを柔らかく",
        "highlight_presets": ["base_"],
        "shadow_presets": [],
        "eyebrow_type": "natural",
        "advice": "シャープすぎる輪郭を眉で和らげ、シャドウは控えめ。",
    },
    "base": {
        "label": "ベース型 - エラを削いで縦ラインを作る",
        "highlight_presets": ["base_", "marugao_"],
        "shadow_presets": ["marugao-side"],
        "eyebrow_type": "corner",
        "advice": "エラ周辺のサイドシャドウ + 上がり眉で強さを調整。",
    },
}

# 標準のカラー設定 (RGB)
DEFAULT_BASE_COLOR_RGB = (235, 200, 170)       # 肌色ファンデーション
DEFAULT_BASE_INTENSITY = 0.25

DEFAULT_HIGHLIGHT_COLOR_RGB = (255, 248, 235)  # ウォームアイボリー
DEFAULT_HIGHLIGHT_INTENSITY = 0.10

DEFAULT_SHADOW_COLOR_RGB = (120, 85, 60)       # ウォームブラウン
DEFAULT_SHADOW_INTENSITY = 0.22

DEFAULT_EYEBROW_COLOR_RGB = (85, 60, 45)       # 濃いめブラウン
DEFAULT_EYEBROW_INTENSITY = 0.70

# アイメイクのカテゴリ別倍率 (1 = デフォルト)
EYE_CATEGORY_MODIFIERS: dict[str, dict[str, float]] = {
    "Big-Round Eyes": {
        "eyeshadow_intensity": 0.85,
        "eyeliner_intensity": 0.60,
    },
    "Narrow Eyes": {
        "eyeshadow_intensity": 1.15,
        "eyeliner_intensity": 1.25,
    },
    "Balanced": {
        "eyeshadow_intensity": 1.00,
        "eyeliner_intensity": 1.00,
    },
}


# ==============================================================
# MakeupPlan
# ==============================================================
@dataclass
class MakeupStep:
    """1 ステップ分の化粧設定"""

    stage: str           # "base" / "shadow" / "highlight" / "eye" / "eyebrow"
    area: str            # target.json のエリア名 / "full_face" / brow_type
    color_rgb: tuple[int, int, int]
    intensity: float
    blend: str = "normal"  # 1.x での合成方式 (一部 stage で参照)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "area": self.area,
            "color_rgb": list(self.color_rgb),
            "intensity": float(self.intensity),
            "blend": self.blend,
            "meta": self.meta,
        }


@dataclass
class MakeupPlan:
    skeletal_type: str
    skeletal_label: str
    advice: str
    overall_modifier: float       # 0.85-1.15 全体強度倍率
    steps: list[MakeupStep] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "skeletal_type": self.skeletal_type,
            "skeletal_label": self.skeletal_label,
            "advice": self.advice,
            "overall_modifier": self.overall_modifier,
            "steps": [s.to_dict() for s in self.steps],
            "rationale": self.rationale,
        }


# ==============================================================
# Selector 本体
# ==============================================================
class MakeupSelector:
    """FaceProfile から MakeupPlan を組み立てる"""

    def select(self, profile: dict[str, Any]) -> MakeupPlan:
        summary = profile.get("summary", {})
        skel_type = summary.get("skeletal_type", "oval")
        if skel_type not in SKELETAL_PRESETS:
            skel_type = "oval"
        preset = SKELETAL_PRESETS[skel_type]

        # 全体強度倍率 (golden_score ベース)
        golden = float(summary.get("golden_score", 70.0))
        if golden >= 80:
            overall_mod = 0.85
            mod_reason = f"golden={golden:.1f} (≥80) なので全体 -15%"
        elif golden >= 70:
            overall_mod = 1.0
            mod_reason = f"golden={golden:.1f} (70-80) なので標準"
        else:
            overall_mod = 1.15
            mod_reason = f"golden={golden:.1f} (<70) なので全体 +15%"

        plan = MakeupPlan(
            skeletal_type=skel_type,
            skeletal_label=preset["label"],
            advice=preset["advice"],
            overall_modifier=overall_mod,
        )
        plan.rationale.append(f"骨格判定: {skel_type} ({preset['label']})")
        plan.rationale.append(mod_reason)

        # ---- 1. ベース ----
        plan.steps.append(MakeupStep(
            stage="base",
            area="full_face",
            color_rgb=DEFAULT_BASE_COLOR_RGB,
            intensity=DEFAULT_BASE_INTENSITY * overall_mod,
            blend="normal",
        ))

        # ---- 2. シャドウ ----
        for preset_prefix in preset["shadow_presets"]:
            plan.steps.append(MakeupStep(
                stage="shadow",
                area=preset_prefix,
                color_rgb=DEFAULT_SHADOW_COLOR_RGB,
                intensity=DEFAULT_SHADOW_INTENSITY * overall_mod,
                blend="multiply",
            ))
        if preset["shadow_presets"]:
            plan.rationale.append(
                f"シャドウ: {preset['shadow_presets']} で輪郭を補正"
            )
        else:
            plan.rationale.append("シャドウは不要 (輪郭補正しない)")

        # ---- 3. ハイライト ----
        for preset_prefix in preset["highlight_presets"]:
            plan.steps.append(MakeupStep(
                stage="highlight",
                area=preset_prefix,
                color_rgb=DEFAULT_HIGHLIGHT_COLOR_RGB,
                intensity=DEFAULT_HIGHLIGHT_INTENSITY * overall_mod,
                blend="additive",
            ))
        plan.rationale.append(
            f"ハイライト: {preset['highlight_presets']} を塗布"
        )

        # ---- 4. アイメイク ----
        eye_category = summary.get("eye_category", "Balanced")
        eye_mods = EYE_CATEGORY_MODIFIERS.get(
            eye_category, EYE_CATEGORY_MODIFIERS["Balanced"]
        )
        plan.rationale.append(
            f"目判定: {eye_category} → shadow x{eye_mods['eyeshadow_intensity']:.2f}, "
            f"liner x{eye_mods['eyeliner_intensity']:.2f}"
        )

        eye_steps = [
            ("eyeshadow_base",   (190, 145, 120), 0.35, "normal"),
            ("eyeshadow_crease", (160, 110,  85), 0.25, "normal"),
            ("tear_bag",         (255, 230, 215), 0.12, "additive"),
            ("lower_outer",      (180, 100,  85), 0.18, "normal"),
            ("eyeliner",         ( 35,  20,  10), 0.55, "normal"),
        ]
        for name, color, base_intensity, blend in eye_steps:
            if "eyeliner" in name:
                mul = eye_mods["eyeliner_intensity"]
            else:
                mul = eye_mods["eyeshadow_intensity"]
            plan.steps.append(MakeupStep(
                stage="eye",
                area=name,
                color_rgb=color,
                intensity=base_intensity * mul * overall_mod,
                blend=blend,
                meta={"eye_category": eye_category},
            ))

        # ---- 5. 眉 ----
        eyebrow_category = summary.get("eyebrow_category", "")
        plan.rationale.append(
            f"眉判定: 現在={eyebrow_category} → 目標={preset['eyebrow_type']}"
        )
        plan.steps.append(MakeupStep(
            stage="eyebrow",
            area=preset["eyebrow_type"],
            color_rgb=DEFAULT_EYEBROW_COLOR_RGB,
            intensity=min(1.0, DEFAULT_EYEBROW_INTENSITY * overall_mod),
            blend="normal",
            meta={"current": eyebrow_category},
        ))

        return plan


__all__ = ["MakeupSelector", "MakeupPlan", "MakeupStep", "SKELETAL_PRESETS"]
