// COVID-19 CT Classifier, running entirely in the browser with ONNX Runtime Web.
// The ONNX model (made by export_onnx.py) takes a 224x224 image with values in [0, 1] and returns
// the probability of each class and one Grad-CAM heatmap per class.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.bundle.min.mjs";

const ORT_WASM_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.wasm";
const ORT_WASM_BYTES = 14239897;          // size of that file, used for the progress before it is known
const SIZE = 224;                         // model input size
const VIEW = 672;                         // resolution of the image shown on screen

// Same order as the model outputs (alphabetical folder names: NiCT, nCT, pCT)
const CLASSES = [
  { name: "Non-informative", short: "Non-informative", evidence: "a non-informative slice", color: "--gray" },
  { name: "Normal", short: "Normal", evidence: "normal lungs", color: "--green" },
  { name: "COVID-19 positive", short: "COVID-19", evidence: "COVID-19", color: "--red" },
];
const ORDER = [2, 1, 0];                  // display order: COVID-19 first

const EXAMPLES = [
  { src: "examples/pCT1227.jpg", label: "COVID-19" },
  { src: "examples/nCT2413.jpg", label: "Healthy lungs" },
  { src: "examples/NiCT5567.jpg", label: "Neck" },
];

const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (id) => document.getElementById(id);

let session = null;
let result = null;                        // { probs, cams, predicted, explain, ms }
let heatmapOn = true;

const baseCanvas = makeCanvas(VIEW);      // the CT slice, cropped like the model sees it
const heatCanvas = makeCanvas(SIZE);      // the coloured heatmap of the selected class
const inputCanvas = makeCanvas(SIZE);     // the pixels sent to the model

function makeCanvas(size) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  return canvas;
}

/* ---------------- Loading ---------------- */

const progress = { shown: 0, target: 0, loaded: 0, total: 0, running: false, downloading: false };

function setLoaderText(title, detail) {
  $("loader-title").textContent = title;
  $("loader-detail").textContent = detail;
}

function setProgress(loaded, total) {
  progress.loaded = loaded;
  progress.total = Math.max(total, loaded);
  progress.target = Math.min(1, loaded / progress.total);
  if (!progress.running) {
    progress.running = true;
    requestAnimationFrame(stepProgress);
  }
}

// Height of the liquid for a progress between 0 and 1. The lungs (y 190 to 58) fill during the first
// 88%, the narrow airway above them (y 58 to -4) during the rest, so the level matches the percentage.
function liquidLevel(p) {
  const split = 0.88;
  return p <= split ? 190 - (p / split) * 132 : 58 - ((p - split) / (1 - split)) * 62;
}

// Ease the displayed value towards the real one so the liquid rises smoothly
function stepProgress() {
  progress.shown += (progress.target - progress.shown) * (reducedMotion ? 1 : 0.14);
  if (Math.abs(progress.target - progress.shown) < 0.0005) progress.shown = progress.target;
  $("liquid-level").setAttribute("transform", `translate(0 ${liquidLevel(progress.shown).toFixed(2)})`);
  $("loader-percent").textContent = Math.floor(progress.shown * 100);
  if (progress.downloading && progress.total) {
    $("loader-detail").textContent = `${megabytes(progress.loaded)} of ${megabytes(progress.total)} MB`;
  }
  if (progress.shown !== progress.target || progress.shown < 1) requestAnimationFrame(stepProgress);
  else progress.running = false;
}

// Resolves once the displayed percentage has reached 100%
function progressFilled() {
  return new Promise((resolve) => {
    const check = () => (progress.shown >= 1 ? resolve() : requestAnimationFrame(check));
    check();
  });
}

const megabytes = (bytes) => (bytes / 1e6).toFixed(1);

async function download(url, onBytes) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} answered ${response.status}`);
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    onBytes(bytes.length);
    return bytes;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onBytes(received);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

// The model is served next to the page. If the host returns something else (for example a Git LFS
// pointer instead of the file), it is downloaded from the fallback URL written by deploy_space.py.
async function downloadModel(meta, onBytes) {
  try {
    const bytes = await download(meta.file, onBytes);
    if (bytes.length === meta.bytes) return bytes;
  } catch (error) {
    if (!meta.fallback) throw error;
  }
  if (!meta.fallback) throw new Error("The model file is incomplete");
  onBytes(0);
  return download(meta.fallback, onBytes);
}

async function load() {
  document.body.classList.add("is-loading");
  $("retry").hidden = true;
  setLoaderText("Loading the model", "Connecting");
  try {
    const meta = await (await fetch("model.json", { cache: "no-cache" })).json();
    let model = 0;
    let engine = 0;
    const total = meta.bytes + ORT_WASM_BYTES;
    const report = () => setProgress(model + engine, total);
    progress.downloading = true;
    const [modelBytes, wasmBytes] = await Promise.all([
      downloadModel(meta, (n) => { model = n; report(); }),
      download(ORT_WASM_URL, (n) => { engine = n; report(); }),
    ]);
    progress.downloading = false;
    setProgress(total, total);
    setLoaderText("Starting the engine", "Almost ready");

    ort.env.wasm.wasmBinary = wasmBytes;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    // A first run on an empty image, so the first real prediction is instant
    await session.run({ image: new ort.Tensor("float32", new Float32Array(3 * SIZE * SIZE), [1, 3, SIZE, SIZE]) });
    await progressFilled();
    finishLoading();
  } catch (error) {
    console.error(error);
    progress.downloading = false;
    document.body.classList.remove("is-loading");
    setLoaderText("The model couldn't load", "Check your internet connection, then try again.");
    $("retry").hidden = false;
  }
}

function finishLoading() {
  setLoaderText("Ready", "");
  document.body.classList.add("is-complete");
  setTimeout(() => {
    document.body.classList.remove("is-loading");
    document.body.classList.add("is-ready");
    document.querySelector(".nav").inert = false;
    $("top").inert = false;
  }, reducedMotion ? 0 : 650);
}

/* ---------------- Prediction ---------------- */

// Same crop as the training pipeline: shorter side resized to 256, then the central 224 x 224
function cropBox(width, height) {
  const side = (SIZE / 256) * Math.min(width, height);
  return [(width - side) / 2, (height - side) / 2, side];
}

function drawCrop(canvas, image, size) {
  const [sx, sy, side] = cropBox(image.width, image.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: canvas === inputCanvas });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(image, sx, sy, side, side, 0, 0, size, size);
  return ctx;
}

function toTensor(image) {
  const { data } = drawCrop(inputCanvas, image, SIZE).getImageData(0, 0, SIZE, SIZE);
  const plane = SIZE * SIZE;
  const input = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i += 1, p += 4) {
    input[i] = data[p] / 255;
    input[i + plane] = data[p + 1] / 255;
    input[i + 2 * plane] = data[p + 2] / 255;
  }
  return new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]);
}

async function analyse(image) {
  if (!session) return;
  const stage = $("stage");
  stage.classList.add("has-image", "is-analysing");
  document.body.classList.add("is-scanning");
  $("dropzone").hidden = true;
  $("clear").hidden = false;
  $("viewer-controls").hidden = false;

  drawCrop(baseCanvas, image, VIEW);
  render(0);

  const started = performance.now();
  const output = await session.run({ image: toTensor(image) });
  const ms = performance.now() - started;

  const probs = Array.from(output.probs.data);
  const predicted = probs.indexOf(Math.max(...probs));
  result = { probs, cams: output.cams.data, predicted, explain: predicted, ms };

  showResult();
  showHeatmap();
  setTimeout(() => {
    stage.classList.remove("is-analysing");
    document.body.classList.remove("is-scanning");
  }, reducedMotion ? 0 : 900);
}

const percent = (p) => `${(p * 100).toFixed(p >= 0.1 || p === 0 ? 0 : 1)}%`;

function showResult() {
  const { probs, predicted } = result;
  const cls = CLASSES[predicted];
  $("results-empty").hidden = true;
  $("results-content").hidden = false;
  $("prediction").style.setProperty("--cls", `var(${cls.color})`);
  $("prediction-name").textContent = cls.name;
  $("confidence").textContent = `${percent(probs[predicted])} confidence`;

  for (const index of ORDER) {
    const row = document.querySelector(`.class-row[data-class="${index}"]`);
    row.querySelector(".class-pct").textContent = percent(probs[index]);
    row.querySelector(".bar span").style.setProperty("--p", probs[index].toFixed(4));
  }
  setExplain(predicted, false);
}

function setExplain(index, redraw = true) {
  result.explain = index;
  const position = ORDER.indexOf(index);
  $("segmented").style.setProperty("--i", position);
  for (const button of document.querySelectorAll(".segment")) {
    const selected = Number(button.dataset.class) === index;
    button.setAttribute("aria-checked", String(selected));
    button.tabIndex = selected ? 0 : -1;
  }
  const evidence = index === result.predicted
    ? "Warm areas weighed most in the prediction."
    : `Warm areas show the evidence for ${CLASSES[index].evidence}.`;
  $("footnote").textContent = `${evidence} Analysed in ${Math.max(1, Math.round(result.ms))} ms on this device.`;
  if (redraw) showHeatmap();
}

/* ---------------- Heatmap ---------------- */

// "Jet" colour map, as in the Grad-CAM figures of the project
function jet(v) {
  const c = (x) => Math.max(0, Math.min(1, x));
  return [c(1.5 - Math.abs(4 * v - 3)), c(1.5 - Math.abs(4 * v - 2)), c(1.5 - Math.abs(4 * v - 1))];
}

const smoothstep = (a, b, x) => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

function paintHeatmap(index) {
  const ctx = heatCanvas.getContext("2d");
  const image = ctx.createImageData(SIZE, SIZE);
  const offset = index * SIZE * SIZE;
  for (let i = 0; i < SIZE * SIZE; i += 1) {
    const v = result.cams[offset + i];
    const [r, g, b] = jet(v);
    image.data[i * 4] = r * 255;
    image.data[i * 4 + 1] = g * 255;
    image.data[i * 4 + 2] = b * 255;
    image.data[i * 4 + 3] = smoothstep(0.15, 0.85, v) * 255;   // cold areas stay transparent
  }
  ctx.putImageData(image, 0, 0);
}

function render(alpha) {
  const ctx = $("canvas").getContext("2d");
  ctx.clearRect(0, 0, VIEW, VIEW);
  ctx.drawImage(baseCanvas, 0, 0);
  if (heatmapOn && result && alpha > 0) {
    ctx.globalAlpha = alpha;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(heatCanvas, 0, 0, VIEW, VIEW);
    ctx.globalAlpha = 1;
  }
}

const opacity = () => Number($("opacity").value);

function showHeatmap() {
  paintHeatmap(result.explain);
  if (reducedMotion) return render(opacity());
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / 360);
    render(opacity() * (1 - (1 - t) ** 3));
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ---------------- Inputs ---------------- */

function showHint(message, isError = false) {
  const hint = $("dropzone-hint");
  hint.textContent = message;
  hint.classList.toggle("is-error", isError);
}

async function openFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    clearImage();
    showHint("That file isn't an image. Choose a JPG or PNG file.", true);
    return;
  }
  try {
    markExample(null);
    await analyse(await createImageBitmap(file));
  } catch (error) {
    console.error(error);
    clearImage();
    showHint("This image couldn't be read. Try another JPG or PNG file.", true);
  }
}

async function openExample(example, button) {
  markExample(button);
  const blob = await (await fetch(example.src)).blob();
  await analyse(await createImageBitmap(blob));
}

function markExample(button) {
  for (const b of document.querySelectorAll(".example")) b.setAttribute("aria-pressed", String(b === button));
}

function clearImage() {
  result = null;
  markExample(null);
  const stage = $("stage");
  stage.classList.remove("has-image", "is-analysing");
  $("dropzone").hidden = false;
  $("clear").hidden = true;
  $("viewer-controls").hidden = true;
  $("results-empty").hidden = false;
  $("results-content").hidden = true;
  showHint("One axial chest slice, JPG or PNG");
}

function buildInterface() {
  const list = $("class-list");
  const segmented = $("segmented");
  for (const index of ORDER) {
    const cls = CLASSES[index];
    const row = document.createElement("li");
    row.className = "class-row";
    row.dataset.class = index;
    row.style.setProperty("--cls", `var(${cls.color})`);
    row.innerHTML = `<span class="class-name"></span><span class="class-pct">0%</span><span class="bar"><span></span></span>`;
    row.querySelector(".class-name").textContent = cls.name;
    list.append(row);

    const segment = document.createElement("button");
    segment.type = "button";
    segment.className = "segment";
    segment.dataset.class = index;
    segment.setAttribute("role", "radio");
    segment.setAttribute("aria-checked", "false");
    segment.textContent = cls.short;
    segment.addEventListener("click", () => result && setExplain(index));
    segmented.append(segment);
  }

  // Arrow keys move the selection, like a native segmented control
  segmented.addEventListener("keydown", (event) => {
    if (!result || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    const position = ORDER.indexOf(result.explain) + (event.key === "ArrowRight" ? 1 : -1);
    const next = ORDER[(position + ORDER.length) % ORDER.length];
    setExplain(next);
    segmented.querySelector(`.segment[data-class="${next}"]`).focus();
    event.preventDefault();
  });

  for (const example of EXAMPLES) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "example";
    button.setAttribute("aria-pressed", "false");
    button.innerHTML = `<img alt="" width="36" height="36"><span></span>`;
    button.querySelector("img").src = example.src;
    button.querySelector("span").textContent = example.label;
    button.addEventListener("click", () => openExample(example, button));
    $("examples").append(button);
  }

  const fileInput = $("file");
  $("choose").addEventListener("click", (event) => { event.stopPropagation(); fileInput.click(); });
  $("dropzone").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => { openFile(fileInput.files[0]); fileInput.value = ""; });
  $("clear").addEventListener("click", clearImage);
  $("retry").addEventListener("click", load);

  const heatmapSwitch = $("heatmap-switch");
  heatmapSwitch.addEventListener("click", () => {
    heatmapOn = !heatmapOn;
    heatmapSwitch.setAttribute("aria-checked", String(heatmapOn));
    $("opacity").disabled = !heatmapOn;
    render(opacity());
  });
  $("opacity").addEventListener("input", () => render(opacity()));

  // Drag and drop on the image area; dropping elsewhere must not open the file in the tab
  const stage = $("stage");
  for (const type of ["dragover", "drop"]) window.addEventListener(type, (event) => event.preventDefault());
  stage.addEventListener("dragenter", () => stage.classList.add("is-dragging"));
  stage.addEventListener("dragover", () => stage.classList.add("is-dragging"));
  stage.addEventListener("dragleave", (event) => {
    if (!stage.contains(event.relatedTarget)) stage.classList.remove("is-dragging");
  });
  stage.addEventListener("drop", (event) => {
    stage.classList.remove("is-dragging");
    openFile(event.dataTransfer.files[0]);
  });

  // Paste an image with Cmd + V
  window.addEventListener("paste", (event) => {
    const item = [...event.clipboardData.items].find((i) => i.type.startsWith("image/"));
    if (item) openFile(item.getAsFile());
  });

  // The glass catches the light where the pointer is
  if (!reducedMotion && matchMedia("(pointer: fine)").matches) {
    document.addEventListener("pointermove", (event) => {
      const glass = event.target.closest?.(".glass");
      if (!glass) return;
      const box = glass.getBoundingClientRect();
      glass.style.setProperty("--mx", `${event.clientX - box.left}px`);
      glass.style.setProperty("--my", `${event.clientY - box.top}px`);
    }, { passive: true });
  }

  // The top edge fades content only once something has scrolled under the bar
  const markScrolled = () => document.body.classList.toggle("is-scrolled", window.scrollY > 8);
  window.addEventListener("scroll", markScrolled, { passive: true });
  markScrolled();
}

buildInterface();
load();
