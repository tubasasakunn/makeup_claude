"""
3.0 化粧処方スキーマ (Makeup Prescription Schema)

Phase 2 の判定結果 → このスキーマ (Prescription) → Phase 1 各ステップを呼び出す。

各セクションは Phase 1 の apply_xxx 関数の引数とほぼ1対1で対応する。
`enabled=False` の場合はそのステップをスキップする。色・強度はデフォルトを
Phase 1 側の値と合わせているので、何も指定しなければ現状と同じ仕上がりになる。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ==========================================================
# 各ステップの処方
# ==========================================================
@dataclass
class BaseRx:
    """1.3 ベース: 顔全体にファンデーション相当を乗せる"""
    enabled: bool = True
    color_bgr: tuple = (170, 200, 235)   # Phase 1 のデフォルトに合わせる
    intensity: float = 0.30
    blur_scale: float = 2.5


@dataclass
class HighlightRx:
    """1.1 ハイライト: target.json の highlight エリア名を列挙"""
    enabled: bool = True
    areas: list = field(default_factory=lambda: ["base_t-zone", "base_c-zone",
                                                 "base_under-eye", "base_megasira",
                                                 "base_zintyuu"])
    color_bgr: tuple = (255, 255, 255)
    intensity: float = 0.12
    blur_scale: float = 2.0


@dataclass
class ShadowRx:
    """1.2 シャドウ: target.json の shadow エリア名を列挙"""
    enabled: bool = False                 # 顔型で必要な人のみ ON
    areas: list = field(default_factory=list)
    color_bgr: tuple = (90, 68, 50)
    intensity: float = 0.25
    blur_scale: float = 2.5


@dataclass
class EyeAreaRx:
    enabled: bool = True
    color_rgb: tuple = (190, 145, 120)
    intensity: float = 0.35
    blur_scale: float = 0.8
    blend: str = "normal"                 # normal | multiply | additive
    thickness_scale: float = 1.0          # eyeliner のみ


@dataclass
class EyeRx:
    """1.4 目: 5エリアそれぞれを個別に指定

    eyeliner と tear_bag は現状不要のため既定でオフ。
    """
    eyeshadow_base: EyeAreaRx = field(default_factory=lambda: EyeAreaRx(
        color_rgb=(190, 145, 120), intensity=0.35, blur_scale=0.8, blend="normal"))
    eyeshadow_crease: EyeAreaRx = field(default_factory=lambda: EyeAreaRx(
        color_rgb=(160, 110, 85), intensity=0.25, blur_scale=0.5, blend="normal"))
    eyeliner: EyeAreaRx = field(default_factory=lambda: EyeAreaRx(
        enabled=False,
        color_rgb=(35, 20, 10), intensity=0.55, blur_scale=0.3, blend="normal",
        thickness_scale=0.2))
    tear_bag: EyeAreaRx = field(default_factory=lambda: EyeAreaRx(
        enabled=False,
        color_rgb=(255, 230, 215), intensity=0.12, blur_scale=0.5, blend="additive"))
    lower_outer: EyeAreaRx = field(default_factory=lambda: EyeAreaRx(
        color_rgb=(180, 100, 85), intensity=0.18, blur_scale=0.3, blend="normal"))


@dataclass
class EyebrowRx:
    """1.5 眉: 形状タイプ + 色 + 強度"""
    enabled: bool = True
    brow_type: str = "natural"            # natural|straight|arch|parallel|corner
    color_rgb: tuple = (85, 60, 45)
    intensity: float = 0.75


@dataclass
class Prescription:
    """化粧処方: Phase 1 全ステップのパラメータを束ねる"""
    base: BaseRx = field(default_factory=BaseRx)
    highlight: HighlightRx = field(default_factory=HighlightRx)
    shadow: ShadowRx = field(default_factory=ShadowRx)
    eye: EyeRx = field(default_factory=EyeRx)
    eyebrow: EyebrowRx = field(default_factory=EyebrowRx)

    # 処方の根拠 (どの判定から決定されたか)
    source: dict = field(default_factory=dict)
    # 人間可読な説明 (["oval→natural: バランス型のため標準形状", ...])
    rationale: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "Prescription":
        def _mk_eye_area(x): return EyeAreaRx(**x)
        return cls(
            base=BaseRx(**d.get("base", {})),
            highlight=HighlightRx(**d.get("highlight", {})),
            shadow=ShadowRx(**d.get("shadow", {})),
            eye=EyeRx(**{k: _mk_eye_area(v) for k, v in d.get("eye", {}).items()}),
            eyebrow=EyebrowRx(**d.get("eyebrow", {})),
            source=d.get("source", {}),
            rationale=d.get("rationale", []),
        )

    @classmethod
    def load(cls, path: Path) -> "Prescription":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


if __name__ == "__main__":
    # デフォルト処方を吐いてスキーマを確認
    import sys
    rx = Prescription()
    rx.rationale.append("default prescription (no rules applied)")
    out = Path(__file__).parent / "sample_prescription.json"
    rx.save(out)
    print(rx.to_json())
    print(f"\nsaved: {out}", file=sys.stderr)
