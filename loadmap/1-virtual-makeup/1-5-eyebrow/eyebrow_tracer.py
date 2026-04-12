"""
眉形状トレーサー

リファレンス画像上で眉の輪郭をクリックでトレースし、
正規化した形状データを eyebrow_shapes.json に保存する。

使い方:
    python eyebrow_tracer.py [画像パス]

操作:
    1. 眉頭(HEAD)をクリック
    2. 眉尻(TAIL)をクリック
    3. 眉の輪郭に沿ってクリックで1周トレース
    4. [Save] ボタンで保存 → 即座にJSONに書き出し
    5. [Next]/[Prev] で別のタイプへ
    右クリックで直前のポイント削除
"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import rcParams

# 日本語フォント設定 (macOS)
for font in ['Hiragino Sans', 'Hiragino Kaku Gothic Pro', 'AppleGothic']:
    try:
        rcParams['font.sans-serif'] = [font] + rcParams['font.sans-serif']
        break
    except Exception:
        pass

BROW_TYPES = ["natural", "straight", "arch", "parallel", "corner"]
BROW_TYPE_JA = {
    "natural": "ナチュラル",
    "straight": "ストレート",
    "arch": "アーチ",
    "parallel": "平行",
    "corner": "コーナー",
}


class EyebrowTracer:
    def __init__(self, image_path: str, json_path: str = None):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"画像を読み込めません: {image_path}")
        self.img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self.type_idx = 0
        self.phase = "head"  # head → tail → trace
        self.head = None
        self.tail = None
        self.outline = []
        self.shapes = {}

        # 既存のJSONがあれば読み込む
        self.json_path = Path(json_path) if json_path else Path(__file__).parent / "eyebrow_shapes.json"
        if self.json_path.exists():
            with open(self.json_path) as f:
                self.shapes = json.load(f)
            if self.shapes:
                print(f"既存データ: {', '.join(BROW_TYPE_JA.get(t, t) for t in self.shapes)}")

        # --- UI構築 ---
        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        self.fig.subplots_adjust(bottom=0.13)

        # ボタン配置
        btn_w, btn_h = 0.10, 0.045
        btn_y = 0.03
        buttons_spec = [
            (0.15, "Save",  self._btn_save),
            (0.27, "Next",  self._btn_next),
            (0.39, "Prev",  self._btn_prev),
            (0.51, "Clear", self._btn_clear),
            (0.63, "Quit",  self._btn_quit),
        ]
        self._buttons = []
        for bx, label, callback in buttons_spec:
            ax_btn = self.fig.add_axes([bx, btn_y, btn_w, btn_h])
            btn = Button(ax_btn, label)
            btn.on_clicked(callback)
            self._buttons.append(btn)  # prevent GC

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._redraw()
        plt.show()

    # ---- properties ----
    @property
    def current_type(self):
        return BROW_TYPES[self.type_idx]

    # ---- button callbacks ----
    def _btn_save(self, event=None):
        self._save_current()
        self._redraw()

    def _btn_next(self, event=None):
        self._clear()
        self.type_idx = (self.type_idx + 1) % len(BROW_TYPES)
        print(f"\n--- {BROW_TYPE_JA[self.current_type]} ---")
        self._redraw()

    def _btn_prev(self, event=None):
        self._clear()
        self.type_idx = (self.type_idx - 1) % len(BROW_TYPES)
        print(f"\n--- {BROW_TYPE_JA[self.current_type]} ---")
        self._redraw()

    def _btn_clear(self, event=None):
        self._clear()
        print("  クリア")
        self._redraw()

    def _btn_quit(self, event=None):
        plt.close()

    # ---- mouse / key ----
    def _on_click(self, event):
        # ボタン領域のクリックは無視
        if event.inaxes != self.ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata

        if event.button == 1:  # 左クリック
            if self.phase == "head":
                self.head = [x, y]
                self.phase = "tail"
                print(f"  HEAD: ({x:.0f}, {y:.0f})")
            elif self.phase == "tail":
                self.tail = [x, y]
                self.phase = "trace"
                print(f"  TAIL: ({x:.0f}, {y:.0f})")
                print("  → 輪郭をクリックでトレースしてください")
            elif self.phase == "trace":
                self.outline.append([x, y])
                print(f"  point {len(self.outline)}: ({x:.0f}, {y:.0f})")

        elif event.button == 3:  # 右クリック → 取り消し
            if self.phase == "trace" and self.outline:
                self.outline.pop()
                print(f"  undo → {len(self.outline)} pts")
            elif self.phase == "trace" and not self.outline:
                self.phase = "tail"
                self.tail = None
                print("  undo TAIL")
            elif self.phase == "tail":
                self.phase = "head"
                self.head = None
                print("  undo HEAD")

        self._redraw()

    def _on_key(self, event):
        k = event.key
        if k == 's':
            self._btn_save()
        elif k == 'n':
            self._btn_next()
        elif k == 'p':
            self._btn_prev()
        elif k == 'c':
            self._btn_clear()
        elif k in ('q', 'w'):
            self._btn_quit()

    # ---- core logic ----
    def _clear(self):
        self.head = None
        self.tail = None
        self.outline.clear()
        self.phase = "head"

    def _make_normal(self, axis_unit):
        """常に上向き（画像座標で負のy方向）を指す法線"""
        normal = np.array([axis_unit[1], -axis_unit[0]])
        if normal[1] > 0:
            normal = -normal
        return normal

    def _auto_split(self):
        """アウトライン点群を上辺と下辺に自動分割"""
        if not self.head or not self.tail or len(self.outline) < 4:
            return None

        head = np.array(self.head)
        tail = np.array(self.tail)
        axis = tail - head
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-3:
            return None

        axis_unit = axis / axis_len
        normal = self._make_normal(axis_unit)

        # ローカル座標に変換
        local = []
        for p in self.outline:
            v = np.array(p) - head
            t = float(np.dot(v, axis_unit) / axis_len)
            offset = float(np.dot(v, normal) / axis_len)
            local.append((t, offset))

        # t 最小/最大点で分割
        ts = [l[0] for l in local]
        head_idx = int(np.argmin(ts))
        tail_idx = int(np.argmax(ts))

        n = len(self.outline)

        def walk(start, end):
            idx = []
            i = start
            while True:
                idx.append(i)
                if i == end:
                    break
                i = (i + 1) % n
            return idx

        path1_idx = walk(head_idx, tail_idx)
        path2_idx = walk(tail_idx, head_idx)

        path1 = [local[i] for i in path1_idx]
        path2 = [local[i] for i in path2_idx]

        # 平均 offset が大きい方が上辺
        avg1 = np.mean([p[1] for p in path1])
        avg2 = np.mean([p[1] for p in path2])

        if avg1 >= avg2:
            upper_local, lower_local = path1, path2
        else:
            upper_local, lower_local = path2, path1

        # t 昇順にソート
        upper_local.sort(key=lambda p: p[0])
        lower_local.sort(key=lambda p: p[0])

        return {
            "upper": [[round(t, 4), round(o, 4)] for t, o in upper_local],
            "lower": [[round(t, 4), round(o, 4)] for t, o in lower_local],
        }

    def _save_current(self):
        data = self._auto_split()
        if data:
            self.shapes[self.current_type] = data
            # 即座にJSONに書き出し
            self._write_json()
            print(f"  OK: {BROW_TYPE_JA[self.current_type]} "
                  f"(upper {len(data['upper'])}pts, lower {len(data['lower'])}pts) "
                  f"→ {self.json_path.name}")
        else:
            missing = []
            if not self.head:
                missing.append("HEAD")
            if not self.tail:
                missing.append("TAIL")
            if len(self.outline) < 4:
                missing.append(f"輪郭点(現在{len(self.outline)}点, 4点以上必要)")
            print(f"  NG: {', '.join(missing)}")

    def _write_json(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.shapes, f, indent=2, ensure_ascii=False)

    # ---- drawing ----
    def _redraw(self):
        self.ax.clear()
        self.ax.imshow(self.img)

        type_ja = BROW_TYPE_JA[self.current_type]
        already_saved = self.current_type in self.shapes
        phase_msg = {
            "head": "1. 眉頭(HEAD)をクリック",
            "tail": "2. 眉尻(TAIL)をクリック",
            "trace": f"3. 輪郭をトレース ({len(self.outline)}pts)",
        }

        saved = [
            f"{'*' if t == self.current_type else ''}{BROW_TYPE_JA[t]}"
            for t in self.shapes
        ]
        saved_str = ', '.join(saved) if saved else ''
        status = f"  |  saved: {saved_str}" if saved_str else ""

        mark = " [saved]" if already_saved else ""
        self.ax.set_title(
            f"[{type_ja}{mark}]  {phase_msg[self.phase]}{status}\n"
            f"左Click:追加  右Click:戻る  |  ボタンまたは S/N/P/C/Q キー",
            fontsize=11, fontweight='bold',
        )

        # HEAD
        if self.head:
            self.ax.plot(*self.head, 'g^', markersize=14, zorder=10)
            self.ax.annotate('HEAD', self.head, fontsize=9, color='green',
                             fontweight='bold', ha='center', va='bottom',
                             xytext=(0, 12), textcoords='offset points')

        # TAIL
        if self.tail:
            self.ax.plot(*self.tail, 'gv', markersize=14, zorder=10)
            self.ax.annotate('TAIL', self.tail, fontsize=9, color='green',
                             fontweight='bold', ha='center', va='bottom',
                             xytext=(0, 12), textcoords='offset points')

        # ベースライン
        if self.head and self.tail:
            self.ax.plot([self.head[0], self.tail[0]],
                         [self.head[1], self.tail[1]],
                         'g--', linewidth=1, alpha=0.4)

        # アウトライン
        if self.outline:
            pts = np.array(self.outline)
            self.ax.plot(pts[:, 0], pts[:, 1], '.-', color='red',
                         linewidth=2.5, markersize=8)
            if len(self.outline) >= 3:
                self.ax.fill(pts[:, 0], pts[:, 1], alpha=0.2, color='orange')

        self.fig.canvas.draw_idle()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="眉形状トレーサー")
    parser.add_argument("image", nargs="?",
                        default=str(Path(__file__).parent / "reference_medicalbrows.jpg"),
                        help="リファレンス画像パス")
    parser.add_argument("-o", "--output",
                        default=str(Path(__file__).parent / "eyebrow_shapes.json"),
                        help="保存先JSONファイル名 (default: eyebrow_shapes.json)")
    args = parser.parse_args()

    print("=" * 50)
    print("  眉形状トレーサー")
    print("=" * 50)
    print(f"画像: {args.image}")
    print(f"保存先: {args.output}")
    print()
    print("  1. HEAD → 2. TAIL → 3. 輪郭トレース → [Save]")
    print("  ツールバーの虫眼鏡で拡大可")
    print()

    EyebrowTracer(args.image, json_path=args.output)


if __name__ == "__main__":
    main()
