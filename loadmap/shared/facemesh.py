"""
FaceMesh - src/area/facemesh.js の Python ポート

MediaPipe FaceLandmarker で顔検出 → メッシュ三角形抽出 → 中点分割
target.json のメッシュIDと同じ座標系を提供する。

Usage:
    fm = FaceMesh(subdivision_level=1)
    fm.init()
    result = fm.detect(image_bgr)
    # result["points"]    : [(x,y,z), ...] 正規化座標 0-1
    # result["triangles"] : [(i,j,k), ...]
    # result["landmarks_px"] : (478, 2) ピクセル座標
"""

import mediapipe as mp
import numpy as np
from pathlib import Path

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


class FaceMesh:
    def __init__(self, subdivision_level: int = 1):
        self.subdivision_level = subdivision_level
        self._landmarker = None

        self.points: list[dict] = []          # [{x,y,z}, ...]
        self.triangles: list[tuple] = []      # [(i,j,k), ...]
        self.landmarks_px: np.ndarray | None = None  # (478,2) pixel coords

        self._midpoint_cache: dict[tuple, int] = {}
        self._reverse_cache: dict[int, tuple] = {}

        self._raw_points: list[dict] = []
        self._raw_triangles: list[tuple] = []

    # ----------------------------------------------------------
    # Init
    # ----------------------------------------------------------
    def init(self, model_path: str | Path | None = None):
        """MediaPipe FaceLandmarker を初期化"""
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python import BaseOptions

        if model_path is None:
            import urllib.request, tempfile
            cache = Path(tempfile.gettempdir()) / "face_landmarker.task"
            if not cache.exists():
                print("Downloading face_landmarker model...")
                urllib.request.urlretrieve(_MODEL_URL, cache)
            model_path = str(cache)

        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(opts)

    # ----------------------------------------------------------
    # Detect
    # ----------------------------------------------------------
    def detect(self, image_bgr: np.ndarray) -> dict | None:
        """顔検出 → ベースメッシュ構築。顔未検出時 None"""
        import mediapipe as mp

        rgb = image_bgr[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None

        lm = result.face_landmarks[0]
        h, w = image_bgr.shape[:2]

        self._raw_points = [{"x": p.x, "y": p.y, "z": p.z or 0} for p in lm]
        self.landmarks_px = np.array(
            [[int(p.x * w), int(p.y * h)] for p in lm], dtype=np.int32
        )

        # Tesselation → triangles
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
        connections = [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]
        self._raw_triangles = self._extract_triangles(connections)

        # Working copy
        self.points = [dict(p) for p in self._raw_points]
        self.triangles = [tuple(t) for t in self._raw_triangles]
        self._midpoint_cache.clear()
        self._reverse_cache.clear()

        # Auto subdivide
        for _ in range(self.subdivision_level):
            self._subdivide_all()

        return {
            "points": self.points,
            "triangles": self.triangles,
            "landmarks_px": self.landmarks_px,
        }

    # ----------------------------------------------------------
    # Mesh Rendering Helpers
    # ----------------------------------------------------------
    def get_triangle_pixels(self, tri_id: int, img_w: int, img_h: int) -> np.ndarray:
        """三角形の3頂点をピクセル座標 (3,2) で返す"""
        a, b, c = self.triangles[tri_id]
        pts = []
        for idx in (a, b, c):
            p = self.points[idx]
            pts.append([int(p["x"] * img_w), int(p["y"] * img_h)])
        return np.array(pts, dtype=np.int32)

    def build_mask(self, mesh_ids: list[int], img_w: int, img_h: int) -> np.ndarray:
        """指定メッシュIDの三角形を塗りつぶした float32 マスク (0-1)"""
        mask = np.zeros((img_h, img_w), dtype=np.float32)
        import cv2
        for mid in mesh_ids:
            if 0 <= mid < len(self.triangles):
                pts = self.get_triangle_pixels(mid, img_w, img_h)
                cv2.fillPoly(mask, [pts], 1.0)
        return mask

    # ----------------------------------------------------------
    # Mirror
    # ----------------------------------------------------------
    def find_mirror_meshes(self, mesh_ids: set[int]) -> set[int]:
        """左右反転メッシュIDを返す"""
        mirror = self._get_mirror_map()
        tri_map = {}
        for i, t in enumerate(self.triangles):
            key = tuple(sorted(t))
            tri_map[key] = i

        result = set()
        for mid in mesh_ids:
            a, b, c = self.triangles[mid]
            key = tuple(sorted([mirror[a], mirror[b], mirror[c]]))
            if key in tri_map:
                result.add(tri_map[key])
        return result

    # ----------------------------------------------------------
    # Private
    # ----------------------------------------------------------
    def _get_midpoint(self, i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in self._midpoint_cache:
            return self._midpoint_cache[key]

        a, b = self.points[i], self.points[j]
        mid = {
            "x": (a["x"] + b["x"]) / 2,
            "y": (a["y"] + b["y"]) / 2,
            "z": (a.get("z", 0) + b.get("z", 0)) / 2,
        }
        idx = len(self.points)
        self.points.append(mid)
        self._midpoint_cache[key] = idx
        self._reverse_cache[idx] = (i, j)
        return idx

    def _subdivide_all(self):
        new_tris = []
        for a, b, c in self.triangles:
            mab = self._get_midpoint(a, b)
            mbc = self._get_midpoint(b, c)
            mac = self._get_midpoint(a, c)
            new_tris.extend([
                (a, mab, mac),
                (mab, b, mbc),
                (mac, mbc, c),
                (mab, mbc, mac),
            ])
        self.triangles = new_tris

    def _get_mirror_map(self) -> list[int]:
        raw_n = len(self._raw_points)
        n = len(self.points)
        pts = self.points
        mirror = list(range(n))

        # Step 1: raw landmarks - nearest neighbor with face center
        sum_x = sum(pts[i]["x"] for i in range(raw_n))
        cx = sum_x / raw_n

        for i in range(raw_n):
            mx = 2 * cx - pts[i]["x"]
            my = pts[i]["y"]
            best, best_d = i, float("inf")
            for j in range(raw_n):
                dx = pts[j]["x"] - mx
                dy = pts[j]["y"] - my
                d = dx * dx + dy * dy
                if d < best_d:
                    best_d = d
                    best = j
            mirror[i] = best

        # Step 2: midpoints - structural
        for i in range(raw_n, n):
            parents = self._reverse_cache.get(i)
            if not parents:
                continue
            a, b = parents
            ma, mb = mirror[a], mirror[b]
            key = (min(ma, mb), max(ma, mb))
            mid_idx = self._midpoint_cache.get(key)
            if mid_idx is not None:
                mirror[i] = mid_idx
        return mirror

    @staticmethod
    def _extract_triangles(connections) -> list[tuple]:
        adj: dict[int, set[int]] = {}
        for u, v in connections:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        result = []
        seen = set()
        for u, v in connections:
            for w in adj.get(u, set()):
                if w != v and w in adj.get(v, set()):
                    tri = tuple(sorted([u, v, w]))
                    if tri not in seen:
                        seen.add(tri)
                        result.append(tri)
        return result
