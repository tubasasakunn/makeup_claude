import { FaceMesh } from "./facemesh.js";

// ============================================================
// Core
// ============================================================
const faceMesh = new FaceMesh({ subdivisionLevel: 1 });

// ============================================================
// State (UI only)
// ============================================================
let img = null;
let selectedMeshes = new Set();
let showIds = false;

// Zoom & Pan
let zoom = 1;
let panX = 0;
let panY = 0;
let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let panStartPanX = 0;
let panStartPanY = 0;
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 30;

// ============================================================
// DOM
// ============================================================
const canvas     = document.getElementById("mainCanvas");
const ctx        = canvas.getContext("2d");
const imageInput = document.getElementById("imageInput");
const statusEl   = document.getElementById("status");
const dropZone   = document.getElementById("dropZone");
const meshListEl = document.getElementById("meshList");
const totalCountEl   = document.getElementById("totalCount");
const selectedCountEl = document.getElementById("selectedCount");
const idInput    = document.getElementById("idInput");
const loadIdsBtn = document.getElementById("loadIdsBtn");
const showIdsCb  = document.getElementById("showIds");
const clearBtn   = document.getElementById("clearBtn");
const copyBtn    = document.getElementById("copyBtn");
const mirrorBtn  = document.getElementById("mirrorBtn");
const toastEl    = document.getElementById("toast");
const zoomLevelEl    = document.getElementById("zoomLevel");
const resetZoomBtn   = document.getElementById("resetZoomBtn");
const zoomInBtn      = document.getElementById("zoomInBtn");
const zoomOutBtn     = document.getElementById("zoomOutBtn");
const canvasContainer = document.getElementById("canvas-container");
const canvasArea     = document.getElementById("canvas-area");

// ============================================================
// Helpers
// ============================================================
function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = cls || "";
}

function showToast(text) {
  toastEl.textContent = text;
  toastEl.classList.add("show");
  setTimeout(() => toastEl.classList.remove("show"), 1500);
}

// ============================================================
// Zoom & Pan
// ============================================================
function updateTransform() {
  canvasContainer.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
  zoomLevelEl.textContent = `${Math.round(zoom * 100)}%`;
}

function fitToView() {
  const area = canvasArea.getBoundingClientRect();
  const scaleX = area.width / canvas.width;
  const scaleY = area.height / canvas.height;
  zoom = Math.min(scaleX, scaleY, 1);
  panX = (area.width - canvas.width * zoom) / 2;
  panY = (area.height - canvas.height * zoom) / 2;
  updateTransform();
}

function zoomAtPoint(factor, centerX, centerY) {
  const newZoom = Math.min(Math.max(zoom * factor, MIN_ZOOM), MAX_ZOOM);
  panX = centerX - (centerX - panX) * (newZoom / zoom);
  panY = centerY - (centerY - panY) * (newZoom / zoom);
  zoom = newZoom;
  updateTransform();
}

// ============================================================
// Image Loading & Processing
// ============================================================
function loadImage(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const image = new Image();
    image.onload = () => {
      img = image;
      processImage();
    };
    image.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

async function processImage() {
  if (!img) return;

  dropZone.classList.add("hidden");
  setStatus("顔メッシュ検出中...", "loading");

  canvas.width = img.width;
  canvas.height = img.height;

  const result = faceMesh.detect(img);

  if (!result) {
    setStatus("顔が検出されませんでした", "");
    ctx.drawImage(img, 0, 0);
    return;
  }

  const s = faceMesh.stats;
  setStatus(`${s.currentTriangles} メッシュ / ${s.currentPoints} 頂点`, "ready");
  totalCountEl.textContent = s.currentTriangles;

  selectedMeshes.clear();
  updateSelectedCount();
  buildMeshList();
  fitToView();
  render();
}

// ============================================================
// Rendering
// ============================================================
function render() {
  if (!img) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);

  const { points, triangles } = faceMesh;
  if (triangles.length === 0) return;

  const w = canvas.width;
  const h = canvas.height;

  for (let i = 0; i < triangles.length; i++) {
    const [a, b, c] = triangles[i];
    const pa = points[a], pb = points[b], pc = points[c];
    const ax = pa.x * w, ay = pa.y * h;
    const bx = pb.x * w, by = pb.y * h;
    const cx = pc.x * w, cy = pc.y * h;

    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.lineTo(cx, cy);
    ctx.closePath();

    if (selectedMeshes.has(i)) {
      ctx.fillStyle = "rgba(236, 72, 153, 0.35)";
      ctx.fill();
      ctx.strokeStyle = "rgba(236, 72, 153, 0.9)";
      ctx.lineWidth = 1.5;
    } else {
      ctx.strokeStyle = "rgba(0, 255, 100, 0.25)";
      ctx.lineWidth = 0.5;
    }
    ctx.stroke();
  }

  // Draw IDs on canvas (toggle)
  if (showIds) {
    const fontSize = Math.max(7, Math.min(12, w / 80));
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let i = 0; i < triangles.length; i++) {
      const [a, b, c] = triangles[i];
      const pa = points[a], pb = points[b], pc = points[c];
      const cx = ((pa.x + pb.x + pc.x) / 3) * w;
      const cy = ((pa.y + pb.y + pc.y) / 3) * h;

      if (selectedMeshes.has(i)) {
        ctx.font = `bold ${fontSize + 2}px monospace`;
        ctx.fillStyle = "#fff";
        ctx.strokeStyle = "rgba(0,0,0,0.8)";
        ctx.lineWidth = 2.5;
        ctx.strokeText(String(i), cx, cy);
        ctx.fillText(String(i), cx, cy);
      } else {
        ctx.font = `${fontSize}px monospace`;
        ctx.fillStyle = "rgba(255,255,255,0.55)";
        ctx.fillText(String(i), cx, cy);
      }
    }
  }
}

// ============================================================
// Hit Testing
// ============================================================
function triSign(px, py, ax, ay, bx, by) {
  return (px - bx) * (ay - by) - (ax - bx) * (py - by);
}

function pointInTriangle(px, py, ax, ay, bx, by, cx, cy) {
  const d1 = triSign(px, py, ax, ay, bx, by);
  const d2 = triSign(px, py, bx, by, cx, cy);
  const d3 = triSign(px, py, cx, cy, ax, ay);
  const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
  const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(hasNeg && hasPos);
}

function hitTest(clientX, clientY) {
  const { points, triangles } = faceMesh;
  if (triangles.length === 0) return -1;

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (clientX - rect.left) * scaleX;
  const py = (clientY - rect.top) * scaleY;
  const w = canvas.width;
  const h = canvas.height;

  let bestId = -1;
  let bestArea = Infinity;

  for (let i = 0; i < triangles.length; i++) {
    const [a, b, c] = triangles[i];
    const pa = points[a], pb = points[b], pc = points[c];
    const ax = pa.x * w, ay = pa.y * h;
    const bx = pb.x * w, by = pb.y * h;
    const cx = pc.x * w, cy = pc.y * h;

    if (pointInTriangle(px, py, ax, ay, bx, by, cx, cy)) {
      const area = Math.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay));
      if (area < bestArea) {
        bestArea = area;
        bestId = i;
      }
    }
  }
  return bestId;
}

// ============================================================
// Sidebar Mesh List
// ============================================================
function buildMeshList() {
  meshListEl.innerHTML = "";
  const { triangles } = faceMesh;

  for (let i = 0; i < triangles.length; i++) {
    const [a, b, c] = triangles[i];
    const item = document.createElement("div");
    item.className = "mesh-item";
    item.dataset.id = i;
    item.innerHTML = `
      <span class="dot"></span>
      <span>Mesh #${i}</span>
      <span class="vertices">[${a}, ${b}, ${c}]</span>
    `;
    item.addEventListener("click", () => toggleMesh(i));
    meshListEl.appendChild(item);
  }
}

function updateMeshListHighlight() {
  const items = meshListEl.querySelectorAll(".mesh-item");
  items.forEach((item) => {
    const id = parseInt(item.dataset.id);
    item.classList.toggle("selected", selectedMeshes.has(id));
  });
}

function scrollToMesh(id) {
  const item = meshListEl.querySelector(`[data-id="${id}"]`);
  if (item) item.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function toggleMesh(id) {
  if (selectedMeshes.has(id)) {
    selectedMeshes.delete(id);
  } else {
    selectedMeshes.add(id);
  }
  updateSelectedCount();
  updateMeshListHighlight();
  render();
}

function updateSelectedCount() {
  selectedCountEl.textContent = selectedMeshes.size;
}

// ============================================================
// Copy
// ============================================================
function copySelection() {
  if (selectedMeshes.size === 0) {
    showToast("メッシュが選択されていません");
    return;
  }
  const ids = [...selectedMeshes].sort((a, b) => a - b);
  const text = JSON.stringify(ids);
  navigator.clipboard.writeText(text).then(() => {
    showToast(`${ids.length} メッシュIDをコピーしました`);
  });
}

// ============================================================
// Event Handlers
// ============================================================

imageInput.addEventListener("change", (e) => {
  if (e.target.files[0]) loadImage(e.target.files[0]);
});

dropZone.addEventListener("click", () => imageInput.click());

canvasArea.addEventListener("dragover", (e) => e.preventDefault());
canvasArea.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) loadImage(e.dataTransfer.files[0]);
});

canvas.addEventListener("click", (e) => {
  if (e.altKey) return;
  const id = hitTest(e.clientX, e.clientY);
  if (id >= 0) {
    toggleMesh(id);
    scrollToMesh(id);
  }
});

showIdsCb.addEventListener("change", (e) => {
  showIds = e.target.checked;
  render();
});

clearBtn.addEventListener("click", () => {
  selectedMeshes.clear();
  updateSelectedCount();
  updateMeshListHighlight();
  render();
});

copyBtn.addEventListener("click", () => copySelection());

mirrorBtn.addEventListener("click", () => {
  if (selectedMeshes.size === 0) {
    showToast("メッシュが選択されていません");
    return;
  }
  const mirrored = faceMesh.findMirrorMeshes(selectedMeshes);
  for (const id of mirrored) selectedMeshes.add(id);
  updateSelectedCount();
  updateMeshListHighlight();
  render();
  showToast(`${mirrored.size} ミラーメッシュを追加選択`);
});

// ID読込
function loadIds() {
  const raw = idInput.value.trim();
  if (!raw) return;
  // "0,3,7" / "[0,3,7]" / "0 3 7" いずれも受け付ける
  const nums = raw.replace(/[\[\]]/g, "").split(/[,\s]+/).map(Number).filter(Number.isFinite);
  const max = faceMesh.triangles.length;
  const valid = nums.filter((n) => n >= 0 && n < max);
  if (valid.length === 0) {
    showToast("有効なIDがありません");
    return;
  }
  selectedMeshes.clear();
  for (const id of valid) selectedMeshes.add(id);
  updateSelectedCount();
  updateMeshListHighlight();
  render();
  scrollToMesh(valid[0]);
  showToast(`${valid.length} メッシュを選択`);
}

loadIdsBtn.addEventListener("click", loadIds);
idInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") loadIds();
});

// ============================================================
// Zoom & Pan Events
// ============================================================

canvasArea.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = canvasArea.getBoundingClientRect();
  zoomAtPoint(
    e.deltaY < 0 ? 1.15 : 1 / 1.15,
    e.clientX - rect.left,
    e.clientY - rect.top
  );
}, { passive: false });

canvasArea.addEventListener("mousedown", (e) => {
  if (e.button === 1 || (e.button === 0 && e.altKey)) {
    isPanning = true;
    panStartX = e.clientX;
    panStartY = e.clientY;
    panStartPanX = panX;
    panStartPanY = panY;
    canvasArea.classList.add("panning");
    e.preventDefault();
  }
});

window.addEventListener("mousemove", (e) => {
  if (!isPanning) return;
  panX = panStartPanX + (e.clientX - panStartX);
  panY = panStartPanY + (e.clientY - panStartY);
  updateTransform();
});

window.addEventListener("mouseup", () => {
  if (isPanning) {
    isPanning = false;
    canvasArea.classList.remove("panning");
  }
});

zoomInBtn.addEventListener("click", () => {
  const rect = canvasArea.getBoundingClientRect();
  zoomAtPoint(1.3, rect.width / 2, rect.height / 2);
});

zoomOutBtn.addEventListener("click", () => {
  const rect = canvasArea.getBoundingClientRect();
  zoomAtPoint(1 / 1.3, rect.width / 2, rect.height / 2);
});

resetZoomBtn.addEventListener("click", () => { if (img) fitToView(); });
window.addEventListener("resize", () => { if (img) fitToView(); });

// ============================================================
// Start
// ============================================================
setStatus("MediaPipe 読み込み中...", "loading");
faceMesh.init((s) => {
  if (s === "ready") setStatus("画像を選択してください", "ready");
}).catch((err) => {
  console.error("MediaPipe init error:", err);
  setStatus("MediaPipe初期化エラー", "");
});
