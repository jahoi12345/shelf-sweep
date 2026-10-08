// Read-only frame inspection, independent of recognition credentials.
const runSelect = document.querySelector("#runSelect");
const filterSelect = document.querySelector("#frameFilter");
const gallery = document.querySelector("#frameGallery");
const image = document.querySelector("#frameImage");
const imageStage = document.querySelector("#imageStage");
const positionInput = document.querySelector("#framePosition");
const errorEl = document.querySelector("#error");
const zoomButton = document.querySelector("#zoomButton");
let frames = [], visibleFrames = [], position = 0, loadRevision = 0;
const seconds = (value) => typeof value === "number" ? `${value.toFixed(2)}s` : "time unknown";

async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(response.status === 404 ? "This sweep has expired. Choose a saved run, or return to the sweep and start again." : `Could not load frames (${response.status}).`);
  return response.json();
}

function showError(message) {
  errorEl.textContent = message || "";
  errorEl.hidden = !message;
}

function setZoom(zoomed) {
  imageStage.classList.toggle("zoomed", zoomed);
  zoomButton.textContent = zoomed ? "Fit image" : "100% size";
  zoomButton.setAttribute("aria-pressed", String(zoomed));
}

function renderGallery() {
  const fragment = document.createDocumentFragment();
  visibleFrames.forEach((frame, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "frame-thumb";
    button.setAttribute("aria-label", `${frame.frame_id} at ${seconds(frame.timestamp_s)}${frame.selected ? ", keyframe" : ""}`);
    const thumb = document.createElement("img");
    thumb.src = frame.image_url;
    thumb.alt = "";
    thumb.loading = "lazy";
    const label = document.createElement("span");
    label.className = "thumb-label";
    const name = document.createElement("span");
    name.textContent = frame.frame_id;
    const time = document.createElement("span");
    time.textContent = seconds(frame.timestamp_s);
    label.append(name, time);
    const kind = document.createElement("span");
    kind.className = "thumb-kind";
    kind.textContent = frame.selected ? "Keyframe" : "Sampled frame";
    button.append(thumb, label, kind);
    button.addEventListener("click", () => selectFrame(index));
    fragment.append(button);
  });
  gallery.replaceChildren(fragment);
}

function selectFrame(index) {
  position = Math.max(0, Math.min(index, visibleFrames.length - 1));
  const frame = visibleFrames[position];
  document.querySelector("#previousFrame").disabled = !frame || position === 0;
  document.querySelector("#nextFrame").disabled = !frame || position === visibleFrames.length - 1;
  positionInput.disabled = !frame;
  zoomButton.disabled = !frame;
  positionInput.max = Math.max(1, visibleFrames.length);
  positionInput.value = frame ? position + 1 : 1;
  document.querySelector("#positionTotal").textContent = `/ ${visibleFrames.length}`;
  document.querySelector("#emptyMessage").hidden = Boolean(frame);
  image.hidden = !frame;
  document.querySelector("#fullResolution").hidden = !frame;
  document.querySelector("#downloadFrame").hidden = !frame;
  setZoom(false);
  imageStage.scrollTop = imageStage.scrollLeft = 0;
  document.querySelector("#keyframeBadge").hidden = !frame?.selected;
  Array.from(gallery.children).forEach((button, i) => button.setAttribute("aria-current", String(i === position && Boolean(frame))));
  if (!frame) {
    image.removeAttribute("src");
    document.querySelector("#emptyMessage").textContent = filterSelect.value === "keyframes" && frames.length ? "No keyframes selected yet. Try All frames or refresh after the sweep ends." : "No frames saved in this sweep yet. Replay a video or choose a saved run.";
    document.querySelector("#frameTitle").textContent = "No frames to display";
    ["frameMeta", "qualityMeta", "selectionMeta"].forEach(id => document.getElementById(id).textContent = "");
    return;
  }
  image.alt = `${frame.frame_id}, ${seconds(frame.timestamp_s)}${frame.selected ? ", selected keyframe" : ""}`;
  if (image.getAttribute("src") !== frame.image_url) image.src = frame.image_url;
  document.querySelector("#frameTitle").textContent = frame.frame_id;
  document.querySelector("#frameMeta").textContent = `${seconds(frame.timestamp_s)} · ${frame.width_px ?? "?"} × ${frame.height_px ?? "?"} px · ${position + 1} of ${visibleFrames.length}`;
  const quality = frame.quality || {};
  document.querySelector("#qualityMeta").textContent = `Quality flags: ${(quality.reasons || []).join(", ") || "none"}${typeof quality.sharpness === "number" ? ` · Sharpness ${quality.sharpness.toFixed(1)}` : ""}${typeof quality.glare_fraction === "number" ? ` · Glare ${(quality.glare_fraction * 100).toFixed(1)}%` : ""}`;
  document.querySelector("#selectionMeta").textContent = frame.selected ? `Selected: ${frame.selection_reason || "keyframe chosen by the pipeline"}.` : "Sampled frame; not selected as a keyframe.";
  document.querySelector("#fullResolution").href = frame.image_url;
  const download = document.querySelector("#downloadFrame");
  download.href = frame.image_url;
  download.download = `${frame.frame_id}.jpg`;
}

function applyFilter(keepId) {
  visibleFrames = filterSelect.value === "keyframes" ? frames.filter(frame => frame.selected) : frames;
  renderGallery();
  const keptPosition = visibleFrames.findIndex(frame => frame.frame_id === keepId);
  selectFrame(keptPosition >= 0 ? keptPosition : 0);
  document.querySelector("#counts").textContent = `${frames.length} saved frames · ${frames.filter(f => f.selected).length} keyframes · showing ${visibleFrames.length}`;
}

let pendingFrame = null;   // deep link: frame to show once the chosen source has loaded
async function loadFrames(preserve = false) {
  const revision = ++loadRevision;
  const keepId = pendingFrame || (preserve ? visibleFrames[position]?.frame_id : null);
  pendingFrame = null;
  showError("");
  frames = [];
  applyFilter();
  if (!runSelect.value) return;
  const [kind, id] = runSelect.value.split(":");
  try {
    const data = await getJson(kind === "session" ? `/api/sessions/${encodeURIComponent(id)}/frame-index` : `/api/frame-runs/${encodeURIComponent(id)}`);
    if (revision !== loadRevision) return;
    frames = data.frames;
    document.querySelector("#sourceNote").textContent = `${data.fixture ? "FIXTURE run — synthetic inference; images alone do not prove recognition accuracy. " : ""}${data.video_file || data.run_id} · ${kind === "session" ? "Current sweep; refresh to load newly saved frames." : "Saved run; frames and keyframe selection come from its manifest."} Quality scores are capture heuristics.`;
    applyFilter(keepId);
  } catch (error) {
    if (revision === loadRevision) showError(error.message);
  }
}

document.querySelector("#previousFrame").addEventListener("click", () => selectFrame(position - 1));
document.querySelector("#nextFrame").addEventListener("click", () => selectFrame(position + 1));
positionInput.addEventListener("change", () => selectFrame((Number(positionInput.value) || 1) - 1));
filterSelect.addEventListener("change", () => applyFilter(visibleFrames[position]?.frame_id));
runSelect.addEventListener("change", () => loadFrames());
document.querySelector("#refreshButton").addEventListener("click", () => loadFrames(true));
image.addEventListener("click", () => setZoom(!imageStage.classList.contains("zoomed")));
zoomButton.addEventListener("click", () => setZoom(!imageStage.classList.contains("zoomed")));
image.addEventListener("error", () => showError("This frame image could not be loaded. Refresh or choose another saved run."));
image.addEventListener("load", () => showError(""));
document.addEventListener("keydown", event => {
  if (event.target.closest("input, select") || event.target.isContentEditable || event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    selectFrame(position + (event.key === "ArrowRight" ? 1 : -1));
  }
});

// Same-origin deep links from the unified site: ?source=session:<id>|run:<name>&frame=f00012 (embed=1 hides the back link)
window.openFrameSource = async (source, frameId) => {
  if (source && ![...runSelect.options].some(o => o.value === source)) {
    const option = document.createElement("option");
    option.value = source;
    option.textContent = source.startsWith("session:") ? `Active sweep ${source.slice(8)} (refresh for new frames)` : source.slice(4);
    runSelect.prepend(option);
  }
  if (source) runSelect.value = source;
  pendingFrame = frameId || null;
  filterSelect.value = "all";
  await loadFrames(false);
};

async function init() {
  try {
    const showAll = document.querySelector("#showAllRuns")?.checked;
    const { runs } = await getJson(`/api/frame-runs${showAll ? "?all=1" : ""}`);
    runSelect.replaceChildren();
    for (const run of runs) {
      const option = document.createElement("option");
      option.value = `run:${run.run_id}`;
      const label = run.label ? `${run.label} (${run.created || run.run_id})` : run.run_id.startsWith("sweep_") ? `Sweep ${run.created || ""} · ${run.video_file || "live camera"}` : run.run_id;
      option.textContent = `${label} · ${run.frame_count} frames${run.created && !run.run_id.startsWith("sweep_") ? " · " + run.created : ""}${run.fixture ? " · FIXTURE" : ""}`;
      option.title = run.run_id;
      runSelect.append(option);
    }
    const current = sessionStorage.getItem("sweepSession");
    if (current) {
      const option = document.createElement("option");
      option.value = `session:${current}`;
      option.textContent = "Current sweep (refresh for new frames)";
      runSelect.prepend(option);
    }
    // The original offline sample is ready to inspect even before a live replay.
    runSelect.value = current ? `session:${current}` : (runs[0] ? `run:${runs[0].run_id}` : "");   // newest featured run first
    const q = new URLSearchParams(location.search);
    if (q.get("embed") === "1") document.body.classList.add("embedded");
    if (q.get("source")) await window.openFrameSource(q.get("source"), q.get("frame"));
    else await loadFrames();
  } catch (error) { showError(error.message); }
}
document.querySelector("#showAllRuns")?.addEventListener("change", () => init());
init();
