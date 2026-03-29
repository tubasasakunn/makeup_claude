/**
 * FaceMesh - 再利用可能な顔メッシュモジュール
 *
 * MediaPipe FaceLandmarker で検出した478点のランドマークから
 * 三角形メッシュを構築し、自動で中点分割を適用する。
 *
 * Usage:
 *   import { FaceMesh } from './facemesh.js';
 *
 *   const fm = new FaceMesh({ subdivisionLevel: 1 });
 *   await fm.init();
 *
 *   const result = await fm.detect(imageElement);
 *   // result.points    : [{x,y,z}, ...]  正規化座標 (0-1)
 *   // result.triangles : [[i,j,k], ...]  points へのインデックス
 *
 *   fm.subdivide(new Set([0, 3, 7]));  // 選択メッシュをさらに分割
 *   fm.subdivideAll();                  // 全メッシュをさらに1段階分割
 *   fm.reset();                         // ベースメッシュに戻す
 *   fm.findMirrorMeshes(ids);           // 左右反転メッシュを検索
 */

import { FaceLandmarker, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const DEFAULT_SUBDIVISION_LEVEL = 1;

export class FaceMesh {
  /**
   * @param {Object} opts
   * @param {number} opts.subdivisionLevel - 初期分割レベル (default: 1, 各レベルで4倍)
   */
  constructor({ subdivisionLevel = DEFAULT_SUBDIVISION_LEVEL } = {}) {
    this.subdivisionLevel = subdivisionLevel;
    this._landmarker = null;

    // Raw MediaPipe data (分割前)
    this._rawPoints = [];
    this._rawTriangles = [];

    // Base mesh (自動分割後 = 標準状態)
    this._basePoints = [];
    this._baseTriangles = [];
    this._baseMidpointCache = new Map();
    this._baseReverseCache = new Map();

    // Working mesh
    this.points = [];
    this.triangles = [];
    this._midpointCache = new Map();
    this._reverseCache = new Map();   // pointIndex → [parentA, parentB]

    // Mirror cache (invalidated on mesh change)
    this._mirrorMap = null;
    this._mirrorMapLen = 0;
  }

  /** MediaPipe を初期化 */
  async init(onStatus) {
    if (onStatus) onStatus("loading");
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    this._landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "IMAGE",
      numFaces: 1,
      outputFacialTransformationMatrixes: false,
      outputFaceBlendshapes: false,
    });
    if (onStatus) onStatus("ready");
  }

  /**
   * 画像から顔を検出しベースメッシュを構築
   * @param {HTMLImageElement|HTMLCanvasElement} imageElement
   * @returns {{ points, triangles } | null} 顔未検出時は null
   */
  detect(imageElement) {
    if (!this._landmarker) {
      throw new Error("FaceMesh not initialized. Call init() first.");
    }

    const result = this._landmarker.detect(imageElement);
    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
      return null;
    }

    const landmarks = result.faceLandmarks[0];

    // Raw mesh
    this._rawPoints = landmarks.map((l) => ({ x: l.x, y: l.y, z: l.z || 0 }));
    this._rawTriangles = FaceMesh._extractTriangles(
      FaceLandmarker.FACE_LANDMARKS_TESSELATION
    );

    // Working copy → auto subdivide
    this.points = this._rawPoints.map((p) => ({ ...p }));
    this.triangles = this._rawTriangles.map((t) => [...t]);
    this._midpointCache.clear();
    this._reverseCache.clear();
    this._mirrorMap = null;

    for (let i = 0; i < this.subdivisionLevel; i++) {
      this._subdivideAllInternal();
    }

    // Save as base
    this._basePoints = this.points.map((p) => ({ ...p }));
    this._baseTriangles = this.triangles.map((t) => [...t]);
    this._baseMidpointCache = new Map(this._midpointCache);
    this._baseReverseCache = new Map(this._reverseCache);

    return { points: this.points, triangles: this.triangles };
  }

  /**
   * 指定メッシュを中点分割 (1三角形 → 4三角形)
   * @param {Set<number>} meshIds - 分割対象のメッシュID
   */
  subdivide(meshIds) {
    if (!meshIds || meshIds.size === 0) {
      return { points: this.points, triangles: this.triangles };
    }

    const newTriangles = [];
    for (let i = 0; i < this.triangles.length; i++) {
      if (meshIds.has(i)) {
        const [a, b, c] = this.triangles[i];
        const mab = this._getMidpoint(a, b);
        const mbc = this._getMidpoint(b, c);
        const mac = this._getMidpoint(a, c);
        newTriangles.push(
          [a, mab, mac],
          [mab, b, mbc],
          [mac, mbc, c],
          [mab, mbc, mac]
        );
      } else {
        newTriangles.push(this.triangles[i]);
      }
    }
    this.triangles = newTriangles;
    this._mirrorMap = null; // invalidate
    return { points: this.points, triangles: this.triangles };
  }

  /**
   * 全メッシュを指定レベル分割
   * @param {number} levels - 分割回数 (default: 1)
   */
  subdivideAll(levels = 1) {
    for (let i = 0; i < levels; i++) {
      this._subdivideAllInternal();
    }
    return { points: this.points, triangles: this.triangles };
  }

  /** ベースメッシュ (自動分割後の標準状態) に戻す */
  reset() {
    this.points = this._basePoints.map((p) => ({ ...p }));
    this.triangles = this._baseTriangles.map((t) => [...t]);
    this._midpointCache = new Map(this._baseMidpointCache);
    this._reverseCache = new Map(this._baseReverseCache);
    this._mirrorMap = null;
    return { points: this.points, triangles: this.triangles };
  }

  /** MediaPipe 生メッシュ (分割前) を取得 */
  getRawMesh() {
    return {
      points: this._rawPoints.map((p) => ({ ...p })),
      triangles: this._rawTriangles.map((t) => [...t]),
    };
  }

  /** ベースメッシュのコピーを取得 */
  getBaseMesh() {
    return {
      points: this._basePoints.map((p) => ({ ...p })),
      triangles: this._baseTriangles.map((t) => [...t]),
    };
  }

  /**
   * 選択メッシュの左右反転（ミラー）メッシュIDを返す
   * 生ランドマークは顔中心基準の最近傍マッチ、
   * 分割中点は親頂点から構造的に導出 (1:1対応)
   * @param {Set<number>|Array<number>} meshIds
   * @returns {Set<number>}
   */
  findMirrorMeshes(meshIds) {
    const mirrorPt = this._getMirrorMap();

    // sorted-vertex-key → triangle index
    const triMap = new Map();
    for (let i = 0; i < this.triangles.length; i++) {
      const key = [...this.triangles[i]].sort((a, b) => a - b).join(",");
      triMap.set(key, i);
    }

    const result = new Set();
    for (const id of meshIds) {
      const [a, b, c] = this.triangles[id];
      const key = [mirrorPt[a], mirrorPt[b], mirrorPt[c]]
        .sort((a, b) => a - b)
        .join(",");
      const mirrorId = triMap.get(key);
      if (mirrorId !== undefined) result.add(mirrorId);
    }
    return result;
  }

  /** メッシュ統計情報 */
  get stats() {
    return {
      rawTriangles: this._rawTriangles.length,
      rawPoints: this._rawPoints.length,
      baseTriangles: this._baseTriangles.length,
      basePoints: this._basePoints.length,
      currentTriangles: this.triangles.length,
      currentPoints: this.points.length,
    };
  }

  // -----------------------------------------------------------
  // Private
  // -----------------------------------------------------------

  /**
   * 頂点ごとの左右ミラー先インデックスを構築 (lazy cached)
   *
   * - 生ランドマーク (0..rawN-1): 顔の中心X基準で x反転→最近傍
   * - 分割中点 (rawN..): 親頂点のミラーから構造的に導出
   *   midpoint(A,B) → midpoint(mirror(A), mirror(B))
   */
  _getMirrorMap() {
    if (this._mirrorMap && this._mirrorMapLen === this.points.length) {
      return this._mirrorMap;
    }

    const rawN = this._rawPoints.length;
    const n = this.points.length;
    const pts = this.points;
    const map = new Array(n);

    // --- Step 1: 生ランドマーク同士で最近傍マッチ ---
    // 顔の中心Xを生ランドマークから算出
    let sumX = 0;
    for (let i = 0; i < rawN; i++) sumX += pts[i].x;
    const centerX = sumX / rawN;

    for (let i = 0; i < rawN; i++) {
      const mx = 2 * centerX - pts[i].x;
      const my = pts[i].y;
      let bestIdx = i;
      let bestDist = Infinity;
      for (let j = 0; j < rawN; j++) {
        const dx = pts[j].x - mx;
        const dy = pts[j].y - my;
        const d = dx * dx + dy * dy;
        if (d < bestDist) {
          bestDist = d;
          bestIdx = j;
        }
      }
      map[i] = bestIdx;
    }

    // --- Step 2: 分割中点は構造的に導出 ---
    // midpoint(A,B) の mirror = midpoint(mirror(A), mirror(B))
    // インデックス昇順なので、親は必ず先に解決済み
    for (let i = rawN; i < n; i++) {
      const parents = this._reverseCache.get(i);
      if (!parents) {
        map[i] = i;
        continue;
      }
      const [a, b] = parents;
      const ma = map[a], mb = map[b];
      const cacheKey = ma < mb ? `${ma},${mb}` : `${mb},${ma}`;
      const mirrorIdx = this._midpointCache.get(cacheKey);
      map[i] = mirrorIdx !== undefined ? mirrorIdx : i;
    }

    this._mirrorMap = map;
    this._mirrorMapLen = n;
    return map;
  }

  _getMidpoint(i, j) {
    const key = i < j ? `${i},${j}` : `${j},${i}`;
    if (this._midpointCache.has(key)) return this._midpointCache.get(key);

    const a = this.points[i], b = this.points[j];
    const mid = {
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2,
      z: ((a.z || 0) + (b.z || 0)) / 2,
    };
    const idx = this.points.length;
    this.points.push(mid);
    this._midpointCache.set(key, idx);
    this._reverseCache.set(idx, [i, j]);
    return idx;
  }

  _subdivideAllInternal() {
    const all = new Set();
    for (let i = 0; i < this.triangles.length; i++) all.add(i);
    this.subdivide(all);
  }

  /**
   * tesselation の辺リストから三角形を抽出
   * @param {Array<{start:number, end:number}>} connections
   */
  static _extractTriangles(connections) {
    const adj = new Map();
    for (const { start: u, end: v } of connections) {
      if (!adj.has(u)) adj.set(u, new Set());
      if (!adj.has(v)) adj.set(v, new Set());
      adj.get(u).add(v);
      adj.get(v).add(u);
    }

    const result = [];
    const seen = new Set();

    for (const { start: u, end: v } of connections) {
      const nu = adj.get(u);
      const nv = adj.get(v);
      for (const w of nu) {
        if (w !== v && nv.has(w)) {
          const tri = [u, v, w].sort((a, b) => a - b);
          const key = `${tri[0]},${tri[1]},${tri[2]}`;
          if (!seen.has(key)) {
            seen.add(key);
            result.push(tri);
          }
        }
      }
    }
    return result;
  }
}
