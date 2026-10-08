// Adapted from insurance_claim_live_agent_team/live_demo/app.js (awesome-llm-apps @ 4bf51ab, Apache-2.0; see app/NOTICE.md).
// KEPT (functionally unchanged): session/connect flow, generation counter for reset races, WebSocket handling,
//   resampleToPcm16/arrayBufferToBase64/playPcm24/stopPlayback, startLiveVoice/stopLiveVoice, mergeTranscript,
//   upsertStreamingTurn, tool-activity merge, camera start/stop and teardown.
// MODIFIED: notebook/pinboard/policy/avatar rendering replaced by the inventory view; sendFrame now sends a 512px
//   voice-context frame AND a high-resolution evidence frame; added recorded-video replay through the same
//   frame path; added End sweep.
const transcriptEl = document.querySelector("#transcript");
const teamFeedEl = document.querySelector("#teamFeed");
const callStatus = document.querySelector("#callStatus");
const modelLabel = document.querySelector("#modelLabel");
const micButton = document.querySelector("#micButton");
const cameraButton = document.querySelector("#cameraButton");
const endButton = document.querySelector("#endButton");
const replayButton = document.querySelector("#replayButton");
const replaySelect = document.querySelector("#replaySelect");
const modeLabel = document.querySelector("#modeLabel");
const cameraStage = document.querySelector("#cameraStage");
const cameraPreview = document.querySelector("#cameraPreview");
const frameCanvas = document.querySelector("#frameCanvas");
const evidenceCanvas = document.querySelector("#evidenceCanvas");
const newIntakeButton = document.querySelector("#newIntakeButton");
const textForm = document.querySelector("#textForm");
const textInput = document.querySelector("#textInput");
const downloadEl = document.querySelector("#downloadPacket");

const DEFAULT_API_ORIGIN = "http://127.0.0.1:4177";
const API_ORIGIN = window.location.protocol === "file:" ? DEFAULT_API_ORIGIN : window.location.origin;
const WS_ORIGIN = API_ORIGIN.replace(/^http/, "ws");
const VOICE_FRAME_INTERVAL_MS = 1000;   // compressed frame for the voice model's context only
const VOICE_FRAME_WIDTH = 512;
const EVIDENCE_INTERVAL_MS = 200;       // 5 fps high-resolution evidence frames: the track association was tuned at 5 fps (replay default); 2 fps fragments handheld sweeps (IESTest2: 611 groups for 155 books)
const EVIDENCE_MAX_WIDTH = 1920;        // frames wider than this are scaled down (the pipeline's tuning size) unless ?res=4k asks for native pixels up to 3840 on the long side
const EVIDENCE_MAX_LONG_4K = 3840;
const CAPTURE_FPS_DEFAULT = 60;  // ask the camera for 60 fps: it halves the longest shutter the camera may use (1/60 s vs 1/30 s), i.e. half the motion blur of YugoTest7
                                 // (median sharpness 66 at 4K30), and the best-of-window picker gets ~12 candidates per 5 fps evidence frame instead of ~6. ?fps=30 opts out.
const CAPTURE_FPS = (() => { const v = Number(new URLSearchParams(window.location.search).get("fps")); return Number.isFinite(v) && v >= 5 && v <= 120 ? v : CAPTURE_FPS_DEFAULT; })();
const SHARP_W = 320;             // candidate ranking runs on a 320-px grey copy (cheap); it only orders frames inside one 200 ms window
const MAX_SOCKET_BACKLOG = 3000000;   // bytes queued on the WebSocket; above this an evidence frame is skipped (and counted) instead of arriving late in a burst
const EVIDENCE_QUALITY = 0.95;   // match the offline extractor (JPEG q95): the track association is sensitive to frame quantisation

if (window.location.protocol === "file:") {
  window.location.replace(`${API_ORIGIN}/index.html`);
}

let liveSocket = null;
let connectionPromise = null;
let generation = 0;
let processing = false;
let micPending = false;
let cameraPending = false;
let resetting = false;
const playbackSources = new Set();
let audioContext = null;
let inputProcessor = null;
let inputSource = null;
let audioStream = null;
let cameraStream = null;
let replayActive = false;
let frameTimer = null;
let evidenceTimer = null;
let isRecording = false;
let nextPlaybackTime = 0;
let sessionId = null;
let state = null;
let writing = false;
let health = {};

const teamLabels = {
  get_inventory_status: "Inventory",
  record_correction: "Corrections",
  confirm_market: "Market",
  end_sweep: "Packet",
};

const emptyState = {
  transcript: [], tool_activity: [], totals: {}, books: [], items: [], excluded_by_user: [], review_queue: [],
  coverage_gaps: [], provisional_unmerged: [], progress: {}, nudges: [], provenance: {}, stage_status: {}, status: "capturing",
  final_paths: {}, finalize: {},
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function setStatus(text, tone = "") {
  callStatus.textContent = text;
  callStatus.className = `pill ${tone}`.trim();
}

function setWriting(next) {
  writing = next;
}

function setState(nextState) {
  const previous = state || emptyState;
  state = { ...emptyState, ...nextState, tool_activity: nextState.tool_activity ?? previous.tool_activity };
  render();
}

function render() {
  renderTranscript();
  renderInventory();
  renderTeam();
  renderMeasurement();
  renderRecStrip();
  renderMode();
  renderSweepStatus();
  renderSummary();
  const done = state.status === "finished" && sessionId;
  if (done) setStatus("Packet ready", "");
  downloadEl.hidden = !done;
  downloadEl.href = done ? `${API_ORIGIN}/api/sessions/${sessionId}/packet` : "#";
}

function renderTranscript() {
  transcriptEl.innerHTML = (state.transcript || [])
    .map((turn) => {
      const cls = turn.speaker === "Agent" ? "agent" : turn.speaker === "System" ? "system" : "claimant";
      const who = turn.speaker === "Agent" ? "AI" : turn.speaker === "System" ? "!" : "You";
      return `<article class="turn ${cls} ${turn.streaming ? "streaming" : ""}"><span class="who">${who}</span><p>${escapeHtml(turn.text)}</p></article>`;
    })
    .join("");
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// ---- inventory view (unknown values print as "unknown"; nothing is invented client-side)
const unk = '<span class="unk">unknown</span>';
const val = (x) => (x === null || x === undefined || x === "" ? unk : escapeHtml(x));

function thumb(line) {
  const ev = line.evidence || {};
  if (!sessionId || !ev.frame_id) return unk;
  const bb = ev.bbox_xyxy, w = ev.frame_width_px, h = ev.frame_height_px;
  const box = bb && w && h
    ? `<span class="box" style="left:${(100 * bb[0] / w).toFixed(2)}%;top:${(100 * bb[1] / h).toFixed(2)}%;width:${(100 * (bb[2] - bb[0]) / w).toFixed(2)}%;height:${(100 * (bb[3] - bb[1]) / h).toFixed(2)}%"></span>`
    : "";
  const src = `${API_ORIGIN}/api/sessions/${sessionId}/frames/${escapeHtml(ev.frame_id)}.jpg`;
  return `<a class="thumb" href="${src}" target="_blank" rel="noopener"><img loading="lazy" src="${src}" alt="evidence ${escapeHtml(ev.frame_id)}">${box}</a><div class="small">${escapeHtml(ev.frame_id)} @ ${val(ev.timestamp_s)}s</div>${evidenceLinks(line)}`;
}

// Books section: the evidence frame zoomed onto the detected book (padded for context) with the detector's rectangle drawn on it and the
// rest of the picture dimmed, so it is obvious which physical book the row is. A toggle shows the whole frame with the same rectangle.
const wholeFrameBooks = new Set();   // row ids flipped to the whole-frame view; survives the re-render on every state update
function highlightView(src, bb, poly, w, h, region, width) {
  // Like the debugger: the exact detected outline (rotated polygon) filled + stroked over the frame, everything else dimmed; bbox only if no polygon.
  const [rx0, ry0, rx1, ry1] = region, rw = rx1 - rx0, rh = ry1 - ry0;
  const pts = (Array.isArray(poly) && poly.length >= 3 ? poly : [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]])
    .map(([x, y]) => `${(+x).toFixed(1)},${(+y).toFixed(1)}`).join(" ");
  const sw = Math.max(1.5, rw / 110);
  return `<svg class="hl" viewBox="${rx0} ${ry0} ${rw} ${rh}" style="width:${width}px;aspect-ratio:${rw} / ${rh}" role="img" aria-label="detected book in context">` +
    `<image href="${src}" x="0" y="0" width="${w}" height="${h}" preserveAspectRatio="none"/>` +
    `<path fill="rgba(0,0,0,.5)" fill-rule="evenodd" d="M${rx0} ${ry0}h${rw}v${rh}h${-rw}z M${pts.replace(/ /g, " L")}z"/>` +
    `<polygon points="${pts}" fill="rgba(37,235,188,.22)" stroke="#25ebbc" stroke-width="${sw}" stroke-linejoin="round"/></svg>`;
}

function bookEvidence(line) {
  const ev = line.evidence || {}, bb = ev.bbox_xyxy, w = ev.frame_width_px, h = ev.frame_height_px;
  if (!sessionId || !ev.frame_id || !bb || !w || !h) return thumb(line);
  const src = `${API_ORIGIN}/api/sessions/${sessionId}/frames/${escapeHtml(ev.frame_id)}.jpg`;
  const whole = wholeFrameBooks.has(line.id);
  let view;
  if (whole) {
    view = highlightView(src, bb, ev.polygon_xy, w, h, [0, 0, w, h], 170);
  } else {
    const pad = Math.max(24, 0.25 * Math.max(bb[2] - bb[0], bb[3] - bb[1]));
    const region = [Math.max(0, bb[0] - pad), Math.max(0, bb[1] - pad), Math.min(w, bb[2] + pad), Math.min(h, bb[3] + pad)];
    const aspect = (region[2] - region[0]) / (region[3] - region[1]);
    view = highlightView(src, bb, ev.polygon_xy, w, h, region, Math.round(Math.max(72, Math.min(150, 210 * aspect))));
  }
  const toggle = `<a href="#" class="evid hl-toggle" data-book-view="${escapeHtml(line.id)}">${whole ? "zoom to book" : "whole frame"}</a><a class="evid" href="${src}" target="_blank" rel="noopener">open ↗</a>`;
  return `${view}<div class="small">${escapeHtml(ev.frame_id)} @ ${val(ev.timestamp_s)}s</div><div>${toggle}</div>${evidenceLinks(line)}`;
}

// Physical-instance lines (pipeline/instances.py): a line merged from several track groups shows its members and the evidence of every merge, with a
// "split" control per member (undo that merge: the member becomes its own line again; a `split` correction on a finished sweep).
function mergeEvidenceText(ev) {
  if (ev.kind === "panorama_overlap") return `panorama spine ${ev.panorama_spine || "?"}${ev.iou ? ` (IoU ${Object.values(ev.iou).map((v) => (v == null ? "?" : Number(v).toFixed(2))).join("/")})` : ""}`;
  if (ev.kind === "cross_segment_registration") return `frames ${(ev.frame_ids || []).join(" → ")} registered (${ev.inliers ?? "?"} inliers, appearance ${ev.appearance_score != null ? Number(ev.appearance_score).toFixed(2) : "?"})`;
  if (ev.kind === "short_gap_registration") return `frames ${(ev.frame_ids || []).join(" → ")} ${ev.gap_s != null ? Number(ev.gap_s).toFixed(1) + " s apart " : ""}registered (${ev.inliers ?? "?"} inliers, appearance ${ev.appearance_score != null ? Number(ev.appearance_score).toFixed(2) : "?"})`;
  if (ev.kind === "revisit_registration") return `seen again ${ev.gap_s != null ? Number(ev.gap_s).toFixed(1) + " s later" : "later"}: frames ${(ev.frame_ids || []).join(" → ")} registered (${ev.inliers ?? "?"} inliers, appearance ${ev.appearance_score != null ? Number(ev.appearance_score).toFixed(2) : "?"}${ev.layout ? `, ${ev.layout[0]} of ${ev.layout[1]} aligned spines alike` : ""})`;
  if (ev.kind === "shelf_plane_position") return `same spine position on the shelf plane of the 3D map (${ev.model || "?"}): ${ev.across_spine_widths != null ? Number(ev.across_spine_widths).toFixed(2) : "?"} spine widths apart, segments ${(ev.segments || []).join(" / ")}`;
  if (ev.kind === "hosted_pair_adjudication") return `merged by hosted pair adjudication (model-read): frames ${(ev.frame_ids || []).join(" + ")}${ev.reason ? ` — ${ev.reason}` : ""}`;
  return ev.kind || "evidence";
}
function membersDisplay(line) {
  const members = line.member_group_ids || [];
  if (members.length < 2) return "";
  const ev = (line.merge_evidence || []).map((x) => `<li>${escapeHtml((x.groups || []).join(" + "))}: ${escapeHtml(mergeEvidenceText(x))}</li>`).join("");
  const rows = members.map((m) => `<li>${escapeHtml(m)} <a href="#" class="evid split-member" data-split-member="${escapeHtml(m)}" title="undo the merge: this track group becomes its own line again">split</a></li>`).join("");
  return `<details class="members small"><summary>merged from ${members.length} track groups (${escapeHtml((line.merge_evidence_kinds || []).join(", ").replace(/_/g, " "))})</summary><ul>${rows}</ul><div>Evidence:</div><ul>${ev}</ul></details>`;
}
async function splitMember(member) {
  if (!sessionId) return;
  try {
    const res = await api(`/api/sessions/${sessionId}/instances/split`, { method: "POST", body: JSON.stringify({ member_group_id: member }) });
    if (res.state) applyServerState(res.state);
    appendSystem(`Split ${member} into its own line (correction ${res.correction_id})${res.attach && res.attach.attached ? "; the saved packet was rebuilt" : ""}.`);
  } catch (err) {
    appendSystem(`Split failed: ${err.message}`);
  }
}

// Source-evidence links: same frame in the Frames view, same observation in the Debugger (only for this session's own saved run).
function evidenceLinks(line) {
  const ev = line.evidence || {};
  const id = state.identity || {};
  const obs = (ev.observation_ids || [])[0];
  const frame = `<a class="evid" href="#frames" data-open-frame="${escapeHtml(ev.frame_id)}">Frame ↗</a>`;
  const ready = id.scan && ((state.progress || {}).merges > 0 || state.status === "finished");
  const dbg = obs ? (ready ? `<a class="evid" href="#debug" data-open-debug="${escapeHtml(obs)}">Debugger ↗</a>` :
    `<span class="evid small" title="The debugger lists a run once it has a merged manifest bound to a registered scan">debugger pending</span>`) : "";
  return `<div>${frame}${dbg}</div>`;
}

function priceDisplay(offer) {
  if (!offer || offer.amount == null) return "unknown";
  // a converted line (no US seller offered the equivalent edition): always labelled, with the native amount and the cited ECB rate
  const fx = offer.fx || {};
  const conv = offer.converted === true ? `<br><small><b>converted</b> from ${val(offer.original_currency)} ${val(offer.original_amount)} at ${val(fx.rate)} (ECB ${val(fx.rate_date)})</small>` : "";
  return `${val(offer.amount)} ${val(offer.currency)}${conv}<br><a href="${escapeHtml(offer.url || "")}" target="_blank" rel="noopener">${val(offer.source)}</a><br><small>${val(offer.retrieved_at)} · ${val(offer.condition_assumed)}</small>`;
}

function cropDisplay(book) {
  const reading = (book.reading_evidence || []).find(r => r.crop_path);
  if (!reading) return "";
  const name = reading.crop_path.split("/").at(-1);
  return `<br><a href="${API_ORIGIN}/api/sessions/${sessionId}/crops/${encodeURIComponent(name)}" target="_blank" rel="noopener">Native reading crop</a>`;
}

// ---- sweep state indicator: one icon + word from the session state (status, finalize, progress); no new pipeline work
let endFailure = null;          // set when the End request itself fails; cleared on a new session / retry
let roomPhase = false;
let sweepEnding = false;        // End was pressed (or the voice agent ended the sweep): no frame is sent after this          // the shelf pass is over and the room pan is being filmed; End now ends the sweep
let finalizeStartedAt = null;
let finalizePoll = null;
function sweepPhase(st, live, failure) {
  const s = st || emptyState, p = s.progress || {}, f = s.finalize || {};
  if (s.status === "failed" || failure || f.error) {
    return { phase: "failed", icon: "✕", word: "Failed", detail: `${failure || f.error || "the sweep could not be finalized"}. Press End sweep to retry.` };
  }
  if (s.status === "finished") {
    const ps = (s.finalization || {}).packet_status;
    return { phase: "done", icon: "✓", word: "Done", detail: `packet ready${ps ? ` (${ps})` : ""}${(s.finalization || {}).finalization_s != null ? ` · ${Number(s.finalization.finalization_s).toFixed(0)} s` : ""}` };
  }
  if (s.status === "finalizing") {
    let step = f.stage || "";
    if (f.read && f.read.total) step = `reading titles ${f.read.attempted}/${f.read.total}${f.read.budget_exhausted ? " (budget reached)" : ""}`;
    if (!step) {
      const running = Object.entries(s.stage_status || {}).find(([, v]) => v && v.status === "running");
      step = running ? `${running[0]} running` : "finishing";
    }
    return { phase: "processing", icon: "◐", word: "Sweep finished — processing", detail: step };
  }
  if (live) {
    return { phase: "recording", icon: "●", word: "Recording", detail: `${p.frames_saved ?? 0} frames saved · ${p.keyframes_selected ?? 0} keyframes · ${p.stages_in_flight ?? 0} in processing` };
  }
  if ((p.frames_saved ?? 0) > 0) {
    return { phase: "idle", icon: "○", word: "Not recording", detail: `${p.frames_saved} frames saved. Press End sweep to build the packet.` };
  }
  return { phase: "idle", icon: "○", word: "Idle", detail: "No sweep yet. Show the camera or replay a video." };
}

function renderSweepStatus() {
  const box = document.querySelector("#sweepStatus");
  if (!box) return;
  const v = sweepPhase(state, Boolean(cameraStream || replayActive), endFailure);
  if (v.phase === "processing" && finalizeStartedAt == null) finalizeStartedAt = Date.now();
  if (v.phase !== "processing") finalizeStartedAt = null;
  const elapsed = v.phase === "processing" && finalizeStartedAt ? ` · ${Math.round((Date.now() - finalizeStartedAt) / 1000)} s` : "";
  box.className = `sweep-status ${v.phase}`;
  box.querySelector(".ss-icon").textContent = v.icon;
  box.querySelector(".ss-word").textContent = v.word;
  box.querySelector(".ss-detail").textContent = ` ${v.detail}${elapsed}`;
  const done = v.phase === "done" && sessionId;
  const report = document.querySelector("#ssReport"), packet = document.querySelector("#ssPacket");
  report.hidden = !done; packet.hidden = !done;
  report.href = done ? `${API_ORIGIN}/api/sessions/${sessionId}/report` : "#";
  packet.href = done ? `${API_ORIGIN}/api/sessions/${sessionId}/packet` : "#";
  const banner = document.querySelector("#provBanner");
  if (banner) {
    banner.textContent = v.phase === "done"
      ? "SWEEP FINISHED: the values below are final for this recording. Anything marked unknown could not be established from the evidence (unknown, not zero)."
      : v.phase === "failed" ? "SWEEP FAILED: values below are whatever was captured before the failure; pending items were not computed."
      : "PROVISIONAL, INCOMPLETE: unpriced and unmeasured. Prices, spine sizes, shelf run and room measurements are unknown (not zero).";
  }
  // While finalizing without a live socket (closed tab/reconnect) the server state is polled; with a socket, finalize events carry progress.
  if (v.phase === "processing" && !finalizePoll && sessionId) {
    finalizePoll = setInterval(async () => {
      if (!state || state.status !== "finalizing") { clearInterval(finalizePoll); finalizePoll = null; return; }
      if (liveSocket && liveSocket.readyState === WebSocket.OPEN) { renderSweepStatus(); return; }   // elapsed-time tick only
      try { applyServerState((await api(`/api/sessions/${sessionId}`)).state); } catch { /* next tick */ }
    }, 3000);
  } else if (v.phase !== "processing" && finalizePoll) { clearInterval(finalizePoll); finalizePoll = null; }
  renderSweepBuilds();
}

// post-sweep states that are still working (server: app/post_sweep.py ACTIVE); joining_3d = the 3D merge pass after the first map
const POST_SWEEP_ACTIVE = ["queued", "waiting", "building_2d", "building_3d", "counting", "merging", "measuring", "joining_3d"];

// "building 3D map: registering frames (SfM) 40/61 · 63% · ~30 s left"; queued steps show their queue position
function mapsDetail(ps) {
  const names = { panorama: "2D panorama", recon3d: "3D map" }, parts = [];
  for (const kind of ["panorama", "recon3d"]) {
    const st = (ps.steps || {})[kind];
    if (!st || st.state === "none") continue;
    if (st.state === "running") {
      const pr = st.progress || {};
      parts.push(`building ${names[kind]}${pr.label ? ": " + pr.label : ""}${pr.fraction != null ? ` (${Math.round(100 * pr.fraction)}%)` : ""}`);
    } else if (st.state === "queued") parts.push(`${names[kind]} queued${st.position ? " #" + st.position : ""}`);
    else if (st.state === "waiting") parts.push(`${names[kind]}: ${st.note || "waiting"}`);
    else if (st.state === "done") parts.push(`${names[kind]} ready${st.markers != null ? ` (${st.markers} markers = book_count ${st.book_count})` : ""}`);
    else if (st.state === "failed") parts.push(`${names[kind]} failed: ${st.error || "unknown reason"}`);
  }
  const m = ps.merge3d;                               // the 3D merge pass after the first map (first map already usable)
  if (m && m.state === "running") { const pr = m.progress || {}; parts.push(`joining 3D fragments${pr.label ? ": " + pr.label : ""}${pr.fraction != null ? ` (${Math.round(100 * pr.fraction)}%)` : ""}`); }
  else if (m && m.state === "queued") parts.push("3D fragment join queued (first map ready)");
  else if (m && m.state === "done") parts.push(`joined 3D map ready${typeof m.registered_ratio === "number" ? ` (${Math.round(100 * m.registered_ratio)}% of frames)` : ""}`);
  else if (m && m.state === "failed") parts.push(`3D fragment join failed (first map stands): ${m.error || "unknown reason"}`);
  const o = ps.overall, active = POST_SWEEP_ACTIVE.includes(ps.state);
  let text = parts.join(" · ") || ps.summary || ps.note || "";
  if (ps.measure && ps.measure.state && ps.measure.state !== "skipped") text += ` · measurement: ${ps.measure.state === "done" ? (ps.measure.summary || "done") : (ps.measure.error || ps.measure.reason || ps.measure.state)}`;
  if (o && active) text += ` · ${Math.round(100 * o.fraction)}% overall${o.eta_s ? ` · ~${o.eta_s >= 90 ? Math.round(o.eta_s / 60) + " min" : Math.max(1, Math.round(o.eta_s)) + " s"} left` : ""}`;
  return text;
}

// ---- automatic 2D panorama + 3D map after End sweep (server: app/post_sweep.py; shows what the background builds are doing)
let buildPoll = null, buildPollSince = 0;
function renderSweepBuilds() {
  const box = document.querySelector("#sweepBuilds");
  if (!box) return;
  const finished = Boolean(state && state.status === "finished" && sessionId);
  const ps = finished ? state.post_sweep : null;
  box.hidden = !finished;
  if (!finished) { if (buildPoll) { clearInterval(buildPoll); buildPoll = null; } return; }
  const active = !ps || POST_SWEEP_ACTIVE.includes(ps.state);
  const phase = !ps ? "processing" : ps.state === "done" ? "done" : ps.state === "failed" ? "failed" : "processing";
  box.className = `sweep-status ${phase}`;
  box.querySelector(".ss-icon").textContent = phase === "done" ? "✓" : phase === "failed" ? "✕" : "◐";
  const label = !ps ? "queued" : ps.state === "queued" ? "queued" : ps.state === "waiting" ? "waiting for the sweep files to settle"
    : ps.state === "building_2d" ? "building 2D panorama" : ps.state === "building_3d" ? "building 3D map" : ps.state === "counting" ? "counting spines on the panorama" : ps.state === "merging" ? "merging track groups into books" : ps.state === "measuring" ? "scaling the 3D map" : ps.state === "joining_3d" ? "joining the 3D map fragments" : ps.state === "done" ? "done" : "failed";
  box.querySelector(".ss-word").textContent = `Maps: ${label}`;
  box.querySelector(".ss-detail").textContent = ps ? ` ${mapsDetail(ps)}` : " the 2D panorama and 3D map build in the background after the packet";
  const bar = document.querySelector("#sbBar"), fill = document.querySelector("#sbFill");
  const frac = ps && ps.overall ? ps.overall.fraction : (ps && ps.state === "done" ? 1 : 0);
  if (bar && fill) {                                  // overall progress of the 2D + 3D builds of this sweep (panorama ~1/4 of the work, 3D ~3/4)
    bar.hidden = !(active || (ps && ps.state === "done"));
    const p = Math.max(0, Math.min(100, Math.round(100 * (Number(frac) || 0))));
    fill.style.width = `${p}%`; bar.setAttribute("aria-valuenow", String(p));
  }
  const steps = (ps && ps.steps) || {};
  const pano = document.querySelector("#sbPano"), recon = document.querySelector("#sbRecon");
  pano.hidden = !(steps.panorama && steps.panorama.state === "done"); recon.hidden = !(steps.recon3d && steps.recon3d.state === "done");
  if (active && !buildPoll) {
    buildPollSince = Date.now();
    buildPoll = setInterval(async () => {
      if (!sessionId || !state || state.status !== "finished") { clearInterval(buildPoll); buildPoll = null; return; }
      try { applyServerState((await api(`/api/sessions/${sessionId}`)).state); } catch { /* next tick */ }
    }, 3000);
  } else if (!active && buildPoll) { clearInterval(buildPoll); buildPoll = null; }
}

// ---- Sweep page chrome: Details toggle, full-screen camera overlay while sweeping, summary card after the sweep. Display only: every
// number here is read from the server state (packet totals, post_sweep); nothing is estimated client-side.
const DETAILS_KEY = "sweepDetails";   // per browser; technical panels carry class "tech" and show only with body.show-tech
function detailsStored() { try { return window.localStorage.getItem(DETAILS_KEY) === "1"; } catch (_) { return false; } }
function applyDetails(on) {
  document.body?.classList.toggle("show-tech", on);
  const btn = document.querySelector("#detailsToggle");
  if (btn) { btn.setAttribute("aria-pressed", String(on)); btn.textContent = on ? "Hide details" : "Details"; }
}
function toggleDetails() {
  const on = !document.body?.classList.contains("show-tech");
  try { window.localStorage.setItem(DETAILS_KEY, on ? "1" : "0"); } catch (_) { /* storage blocked: the toggle still works for this page */ }
  applyDetails(on);
}

let captureStartedAt = null;          // wall clock of the camera/replay start (recording time on the overlay)
let ovFadeTimer = null;
const isSweeping = () => Boolean(document.body?.classList.contains("sweeping"));
const clockText = (ms) => { const s = Math.max(0, Math.floor(ms / 1000)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
// capture-quality band (provenance.capture_quality.now) -> dot colour + one word
// distance overrides sharpness (user decision 2026-10-08): while the spines are under the target the indicator is red, never good/ok
const QUALITY_BAND = { strict: ["q-good", "Good"], medium: ["q-good", "OK"], poor: ["q-warn", "Slow down"], bad: ["q-bad", "Re-capture"], too_far: ["q-bad", "Too far — move closer"] };
const qualityState = (cq) => cq.indicator || (cq.distance_now === "too_far" ? "too_far" : cq.now);

function renderSweepChrome() {
  const live = Boolean(cameraStream || replayActive);
  const sweeping = live && currentView === "sweep";
  const body = document.body;
  if (body) {
    const was = body.classList.contains("sweeping");
    body.classList.toggle("sweeping", sweeping);
    if (sweeping && !was) { try { window.scrollTo(0, 0); } catch (_) { /* no scroll API */ } }
  }
  const cq = ((state || emptyState).provenance || {}).capture_quality || {};
  const [cls, word] = QUALITY_BAND[qualityState(cq)] || ["q-unknown", "Measuring"];
  const q = document.querySelector("#ovQuality");
  if (q) { q.className = `ov-chip ov-quality ${cls}`; q.title = `Capture quality: ${word}`; }
  const qw = document.querySelector("#ovQualityWord");
  if (qw) qw.textContent = word;
  const talk = document.querySelector("#ovTalk");
  if (talk) {
    talk.classList.toggle("active", isRecording);
    const label = talk.querySelector(".ov-label");
    if (label) label.textContent = isRecording ? "Listening" : "Talk";
  }
  tickSweepClock();
}

function tickSweepClock() {
  const el = document.querySelector("#ovTime");
  if (el) el.textContent = clockText(captureStartedAt ? Date.now() - captureStartedAt : 0);
}

// The newest agent/system line (or capture notice) on the overlay: one line, the last sentence, fades after a few seconds.
function showOverlayLine(text) {
  const t = String(text || "").trim();
  if (!t || !isSweeping()) return;
  const el = document.querySelector("#ovLine");
  if (!el) return;
  const sentences = t.split(/(?<=[.!?])\s+/).filter(Boolean);
  el.textContent = sentences.at(-1) || t;
  el.title = t;
  el.classList.add("show");
  if (ovFadeTimer) clearTimeout(ovFadeTimer);
  ovFadeTimer = setTimeout(() => el.classList.remove("show"), 7000);
}

const RELIABILITY_WORDS = { high: "reliable", medium: "fairly reliable", low: "rough estimate", unknown: "reliability unknown" };
const usd = (v) => (v == null ? null : `$${Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);

// 2D / 3D / measurement chips from post_sweep (the long technical line and the progress bar stay in #sweepBuilds)
function mapChips(ps) {
  const pct = (pr) => (pr && pr.fraction != null ? ` ${Math.round(100 * pr.fraction)}%` : "");
  const step = (name, st, merge) => {
    if (!st || st.state === "none") return ["", `${name}: not started`];
    if (st.state === "running") return ["busy", `${name}: building${pct(st.progress)}`];
    if (st.state === "queued") return ["busy", `${name}: queued${st.position ? " #" + st.position : ""}`];
    if (st.state === "waiting") return ["busy", `${name}: waiting`];
    if (st.state === "failed") return ["bad", `${name}: failed`];
    if (st.state === "done" && merge && merge.state === "running") return ["busy", `${name}: ready, joining fragments${pct(merge.progress)}`];
    if (st.state === "done") return ["ok", `${name}: ready`];
    return ["", `${name}: ${st.state}`];
  };
  if (!ps) return [["busy", "2D: queued"], ["busy", "3D: queued"], ["", "Measurement: after the 3D map"]];
  const steps = ps.steps || {}, m = ps.measure || {};
  const meas = !m.state ? ["", "Measurement: after the 3D map"] : m.state === "done" ? ["ok", `Measurement: ${m.summary || "done"}`]
    : m.state === "failed" ? ["bad", "Measurement: failed"] : m.state === "skipped" ? ["", "Measurement: skipped"] : ["busy", `Measurement: ${m.state}`];
  return [step("2D", steps.panorama), step("3D", steps.recon3d, ps.merge3d), meas];
}

function renderSummary() {
  const st = state || emptyState, t = st.totals || {};
  const books = st.books || [], items = st.items || [];
  const set = (id, text) => { const el = document.querySelector(id); if (el) el.textContent = text; return el; };
  const nBooks = st.list_caps?.books?.total ?? books.length, nItems = st.list_caps?.items?.total ?? items.length;
  set("#booksHint", nBooks ? `${nBooks} line${nBooks === 1 ? "" : "s"}` : "");
  set("#itemsHint", nItems ? `${nItems} item${nItems === 1 ? "" : "s"}` : "");
  const floor = (st.room || {}).floor_area_m2, measure = (st.post_sweep || {}).measure || {};
  set("#measureHint", t.shelf_run_m != null ? `shelf run ${t.shelf_run_m} m` : "");
  const sm = document.querySelector("#sumMeasure");
  if (sm) {
    sm.innerHTML = `Shelf run: ${t.shelf_run_m != null ? `<b>${escapeHtml(t.shelf_run_m)} m</b>` : unk} · Room floor area: ${floor != null ? `<b>${escapeHtml(floor)} m²</b>` : unk}` +
      (measure.state ? ` · Automatic measurement: ${escapeHtml(measure.state === "done" ? (measure.summary || "done") : (measure.error || measure.reason || measure.state))}` : "") +
      ` · <a href="#measure">Open Measurements</a>`;
  }
  const card = document.querySelector("#summaryCard");
  const finished = st.status === "finished" && Boolean(sessionId);
  if (card) card.hidden = !finished;
  document.body?.classList.toggle("sweep-finished", finished);   // phone: the summary moves above the controls
  if (!finished) return;
  // Books: best guess, plausible range, reliability in plain words
  const range = t.book_count_range || (t.book_count_lower != null && t.book_count_upper != null ? [t.book_count_lower, t.book_count_upper] : null);
  const booksEl = set("#sumBooks", t.book_count == null ? "unknown" : `${range ? "~" : ""}${t.book_count}`);
  booksEl?.classList.toggle("unknown", t.book_count == null);
  const rel = t.count_reliability ? (RELIABILITY_WORDS[t.count_reliability] || t.count_reliability) : null;
  // the headline is the 2D panorama count when the panorama is usable (its range may not contain the true count); the line count is always shown second
  const headPan = t.headline_estimator === "panorama" || t.count_estimator === "panorama";
  const nLines = t.book_lines ?? (t.count_estimator === "panorama" ? t.book_count_track : t.book_count);
  set("#sumBooksRange", [range ? (headPan ? `panorama range ${range[0]}–${range[1]} (may not contain the true count)` : `likely ${range[0]}–${range[1]}`)
    : t.book_count != null ? "best guess" : "could not be established",
    rel, headPan ? "counted on the 2D panorama" : null,
    nLines != null ? `${nLines} book lines listed (some may be duplicates)` : null].filter(Boolean).join(" · "));
  // Titles identified
  const titlesEl = set("#sumTitles", t.books_identified == null ? "unknown" : String(t.books_identified));
  titlesEl?.classList.toggle("unknown", t.books_identified == null);
  set("#sumTitlesSub", [nLines != null ? `of ${nLines} book lines` : null, t.books_unidentified ? `${t.books_unidentified} not identified (still counted)` : null,
    t.books_needs_appraisal ? `${t.books_needs_appraisal} need appraisal` : null].filter(Boolean).join(" · "));
  // Replacement / used totals in USD; converted lines are called out
  const rep = usd(t.books_replacement_cost), used = usd(t.books_used_value);
  const moneyEl = set("#sumMoney", rep == null && used == null ? "unknown" : `${rep ?? "unknown"} / ${used ?? "unknown"}`);
  moneyEl?.classList.toggle("unknown", rep == null && used == null);
  const priced = books.filter((b) => (b.replacement_cost && b.replacement_cost.amount != null) || (b.used_value && b.used_value.amount != null)).length;
  const converted = books.filter((b) => (b.replacement_cost && b.replacement_cost.converted === true) || (b.used_value && b.used_value.converted === true)).length;
  set("#sumMoneySub", rep == null && used == null ? "no prices found (unknown, not zero)"
    : [`${priced} book${priced === 1 ? "" : "s"} priced`, converted ? `${converted} converted to USD from another currency (ECB rate)` : "all prices in USD"].join(" · "));
  // Maps
  const chips = document.querySelector("#sumMaps");
  if (chips) chips.innerHTML = mapChips(st.post_sweep).map(([c, txt]) => `<li class="${c}">${escapeHtml(txt)}</li>`).join("");
  const packet = document.querySelector("#sumPacket"), report = document.querySelector("#sumReport");
  if (packet) packet.href = `${API_ORIGIN}/api/sessions/${sessionId}/packet`;
  if (report) report.href = `${API_ORIGIN}/api/sessions/${sessionId}/report`;
}

// Frame rate and timing line(s) under the capture-quality banner: both clocks (server arrival and the phone's capture clock), the rate over time, orientation,
// and what the phone reported about its own main thread. Numbers only; the 5 fps target is the pipeline's tuning rate.
function captureTimingLines(cq) {
  if (!cq || !cq.frames) return [];
  const n = (v, d = 2) => (v == null ? "n/a" : Number(v).toFixed(d)), g = (o) => `${n((o || {}).p50)}/${n((o || {}).p90)}/${n((o || {}).max)}`;
  const ft = cq.frame_timing || {}, cc = cq.client_counters || {};
  const lines = [`Frame rate: ${n(cq.fps_effective)} fps (target 5)${ft.fps_first_third != null ? `, first third ${n(ft.fps_first_third, 1)} -> last third ${n(ft.fps_last_third, 1)} fps` : ""}` +
    ` · gap p50/p90/max ${g(ft.arrival_clock)} s (server arrival) · ${ft.capture_clock ? `${g(ft.capture_clock)} s (phone clock)` : "phone clock n/a"}` +
    `${ft.server_ingest_ms ? ` · server ingest p50/p90 ${n(ft.server_ingest_ms.p50, 0)}/${n(ft.server_ingest_ms.p90, 0)} ms` : ""}` +
    ` · ${cq.orientation || "?"} ${cq.frame_size || ""}` +
    `${cq.device?.requested_resolution ? ` · camera asked ${cq.device.requested_resolution}, granted ${cq.device.width ?? "?"}x${cq.device.height ?? "?"}${cq.device.zoom != null ? `, zoom ${cq.device.zoom}x` : ""}` : ""}`];
  if (cc.camera_callback_fps != null || cc.main_thread_lag_ms_p90 != null) {
    lines.push(`Phone: camera ${n(cc.camera_callback_fps, 1)} callbacks/s · encode p50/p90 ${n(cc.encode_ms_p50, 0)}/${n(cc.encode_ms_p90, 0)} ms · ranking p50 ${n(cc.rank_ms_p50, 0)} ms (every ${cc.rank_every_final ?? "?"} frame)` +
      ` · page main-thread lag p90/max ${n(cc.main_thread_lag_ms_p90, 0)}/${n(cc.main_thread_lag_ms_max, 0)} ms · state msgs ${cc.state_msgs ?? "?"} (p90 ${n(cc.state_kb_p90, 0)} KB, ${n(cc.state_apply_ms_p90, 0)} ms)` +
      ` · skipped: encoding ${cc.skipped_while_encoding ?? 0}, backlog ${cc.skipped_socket_backlog ?? 0} · JPEG q${cc.jpeg_quality_final ?? "?"}`);
  }
  if (cq.orientation === "portrait") lines.push("Portrait frames: hold the phone sideways (landscape) for a wider view of the shelf; the pipeline was tuned on landscape 1920x1080.");
  return lines;
}

// While the camera is capturing only the newest rows are drawn: nobody reads a 700-row table on a phone mid-sweep, and each re-render (plus the parse of the
// state message) costs main-thread time the camera loop needs (YugoTest5/6: capture fell from 4.3 to 1.5 fps as the book list grew). The full list returns after End sweep.
const LIVE_ROWS = 40;
function liveTail(rows, total) {
  if (state.status !== "capturing" || !(cameraStream || replayActive)) return { rows, note: "" };
  const shown = rows.length > LIVE_ROWS ? rows.slice(-LIVE_ROWS) : rows, all = Math.max(total ?? 0, rows.length);
  return { rows: shown, note: all > shown.length ? liveNote(shown.length, all) : "" };
}
const liveNote = (shown, total) => `<p class="small">Showing the latest ${shown} of ${total} while the camera is on; the full list appears after End sweep.</p>`;

function renderInventory() {
  const t = state.totals || {};
  const phase = sweepPhase(state, Boolean(cameraStream || replayActive), endFailure).phase;
  const final = phase === "done", failed = phase === "failed";
  // pending: a value that only exists after End sweep finishes. final+null => "unknown" (could not be established); failed => unknown.
  const stat = (label, v, unknown, pendingUntilFinal = false) => {
    const pending = pendingUntilFinal && !final && !failed;
    const cls = pending ? "pending" : unknown ? "unk" : "";
    const mark = final ? "✓" : failed ? "✕" : "◐";
    const title = final ? "final" : failed ? "sweep failed: not computed" : pending ? "pending until End sweep finishes" : "provisional: still changing";
    return `<div class="stat ${cls} ${final ? "final" : failed ? "failed" : ""}" title="${title}"><i class="mark" aria-label="${title}">${mark}</i><b>${pending ? "pending" : unknown ? "unknown" : escapeHtml(v ?? 0)}</b><span>${escapeHtml(label)}</span></div>`;
  };
  const range = t.book_count_range ? null : t.book_count_lower != null ? `${t.book_count_lower} - ${t.book_count_upper}` : null;
  const money = t.books_replacement_cost == null && t.books_used_value == null;
  const floor = (state.room || {}).floor_area_m2;
  document.querySelector("#totals").innerHTML = [
    stat(t.headline_estimator === "panorama" ? "Books: 2D panorama count (panorama range; book lines)" : t.book_count_range ? "Books: best guess (range, reliability)" : "Books (deduplicated, provisional)", countText(t) ?? t.book_count), range ? stat("Plausible range", range) : "",
    stat("Identified", t.books_identified, false, true), stat("Unidentified (still counted)", t.books_unidentified, false, true),
    stat("Needs appraisal", t.books_needs_appraisal, false, true), stat("Other items", t.item_count),
    stat("Excluded by you", t.user_excluded_count), stat("Replacement / used value", `${t.books_replacement_cost ?? "unknown"} / ${t.books_used_value ?? "unknown"} USD`, money, true),
    stat("Shelf run (m)", t.shelf_run_m, t.shelf_run_m == null, true), stat("Room floor area (m2)", floor, floor == null, true),
  ].join("");
  const p = state.progress || {};
  const unmerged = state.list_caps?.provisional_unmerged?.total ?? (state.provisional_unmerged || []).length;
  document.querySelector("#progress").textContent =
    `frames saved ${p.frames_saved ?? 0} · keyframes ${p.keyframes_selected ?? 0} (detected ${p.keyframes_detected ?? 0}) · processing ${p.stages_in_flight ?? 0}` +
    ` · detections awaiting merge ${unmerged}` + (state.ocr_coverage ? ` · OCR attempted ${state.ocr_coverage.ocr_attempted} / book observations ${state.ocr_coverage.book_observations} (budget-skipped ${state.ocr_coverage.skipped_by_budget}, queued ${state.ocr_coverage.queued_or_pending}; attempted is not identified)` : "") + (p.merge_error ? ` · merge error: ${p.merge_error}` : "");
  const pv = state.provenance || {};
  const cq = pv.capture_quality, cqEl = document.querySelector("#captureQuality");
  if (cq && cqEl) {
    const pct = (v) => (v == null ? "n/a" : `${Math.round(v * 100)}%`);
    const qs = qualityState(cq);
    const now = { strict: "good", medium: "OK", poor: "SLOW DOWN", bad: "RE-CAPTURE", too_far: "TOO FAR — MOVE CLOSER", unknown: "measuring" }[qs] || qs;
    cqEl.className = `banner ${["bad", "poor", "too_far"].includes(qs) ? "warn" : "info"} tech`;   // the raw numbers sit under Details; the overlay dot carries the band
    const head = `Capture quality: ${now} · ${pct(cq.share_medium_or_better)} of ${cq.seconds}s within limits (${pct(cq.share_strict)} strict) · count reliability: ${cq.count_reliability}${cq.count_reliability_capped ? " (spines too small)" : ""} · spine width ${cq.median_spine_px ?? "?"} px (min ${cq.thresholds?.min_spine_px}) · scene speed ${cq.motion_scene_units?.now ?? "?"} sw/s (info; ref ${cq.motion_scene_units?.reference_sw_s ?? cq.thresholds?.scene_speed_ref_sw_s}) · nudges: slow-down ${cq.nudges_issued?.slow_down ?? 0}, re-capture ${cq.nudges_issued?.recapture ?? 0}, closer ${cq.nudges_issued?.too_far ?? 0}`;
    cqEl.innerHTML = [escapeHtml(head), ...captureTimingLines(cq).map(escapeHtml)].join("<br>");
  }
  document.querySelector("#provenance").textContent =
    `Detector: ${pv.stages?.detect?.selection || "unknown"} (${pv.stages?.detect?.available ? "configured" : "unavailable"}) · OCR: ${pv.stages?.read?.selection || "unknown"} (${pv.stages?.read?.available ? "configured" : "unavailable"}) · Results: ${pv.inference_mode || "none yet"} · fresh model requests ${pv.fresh_requests ?? 0} · cached ${pv.cache_hits ?? 0} · cost ${pv.cost_usd ?? "unknown"} · sources ${(pv.capture_sources || []).join("+") || "none"}`;
  document.querySelector("#nudges").innerHTML = (state.nudges || []).length
    ? `<div class="banner warn">Capture: ${escapeHtml((state.nudges.at(-1) || {}).message || "")}</div>` : "";
  const pendingBooks = (state.provisional_unmerged || []).filter(o => o.kind === "book").map(o => ({id: o.observation_id, status: "detection awaiting association", evidence: o.evidence}));
  const bl = liveTail([...(state.books || []), ...pendingBooks], state.list_caps?.books?.total);
  const books = bl.rows;
  document.querySelector("#books").innerHTML = books.length
    ? bl.note + `<table class="lines"><tr><th>ID</th><th>Status</th><th>Title / author</th><th>Visible text</th><th>Lang</th><th>Replacement / used</th><th>Evidence</th></tr>` +
      books.map((b) => `<tr><td>${escapeHtml(b.id)}${b.display_index != null ? `<br><small>#${escapeHtml(b.display_index)}</small>` : ""}</td><td>${escapeHtml(b.status)}${b.uncertain_association ? " <b>(uncertain)</b>" : ""}${b.sightings != null ? `<br><small>${escapeHtml(b.sightings)} sighting${b.sightings === 1 ? "" : "s"}</small>` : ""}${membersDisplay(b)}</td><td>${val(b.title)}<br><small>${val(b.author)}${b.identity_source === "user_statement" ? " · from your statement" : ""}${b.status === "identified" && b.catalogue ? ` · ${b.catalogue === "google_books" ? "Google Books" : "Open Library"}${b.identity_source_url ? ` <a href="${escapeHtml(b.identity_source_url)}" target="_blank" rel="noopener">↗</a>` : ""}` : ""}</small></td><td>${val(b.visible_text)}</td><td>${val(b.language)}</td><td>${priceDisplay(b.replacement_cost)} / ${priceDisplay(b.used_value)}</td><td>${bookEvidence(b)}${cropDisplay(b)}</td></tr>`).join("") + "</table>"
    : `<p class="small">${unmerged ? `${unmerged} detections are queued for merging.` : "No books yet."}</p>`;
  const pendingItems = (state.provisional_unmerged || []).filter(o => o.kind !== "book").map(o => ({id: o.observation_id, category: o.category, status: "detection awaiting association", evidence: o.evidence}));
  const il = liveTail([...(state.items || []), ...pendingItems], state.list_caps?.items?.total);
  const items = il.rows;
  document.querySelector("#items").innerHTML = items.length
    ? il.note + `<table class="lines"><tr><th>ID</th><th>Category</th><th>Description</th><th>Status</th><th>Evidence</th></tr>` +
      items.map((i) => `<tr><td>${escapeHtml(i.id)}${i.display_index != null ? `<br><small>#${escapeHtml(i.display_index)}</small>` : ""}</td><td>${val(i.category)}</td><td>${val(i.description)}</td><td>${val(i.status)}${i.art_is_print ? " (print)" : ""}</td><td>${thumb(i)}</td></tr>`).join("") + "</table>"
    : `<p class="small">None yet.</p>`;
  const ex = state.excluded_by_user || [];
  document.querySelector("#excluded").innerHTML = ex.length
    ? `<table class="lines"><tr><th>ID</th><th>Reason</th><th>Correction</th></tr>${ex.map((x) => `<tr><td>${escapeHtml(x.instance_id)}</td><td>${val(x.reason)}</td><td>${escapeHtml(x.correction_id)}</td></tr>`).join("")}</table>`
    : `<p class="small">Nothing excluded.</p>`;
  document.querySelector("#review").innerHTML = (state.review_queue || []).slice(0, 40)
    .map((r) => `<li class="sev-${escapeHtml(r.severity)}"><b>${escapeHtml(r.ref_id)}</b> ${escapeHtml(r.code)}: ${escapeHtml(r.reason)}</li>`).join("") || "<li class='small'>Empty.</li>";
  document.querySelector("#gaps").innerHTML = (state.coverage_gaps || [])
    .map((g) => `<li><b>${escapeHtml(g.type)}</b>${g.start_s != null ? ` (${g.start_s.toFixed?.(1) ?? g.start_s}s-${g.end_s?.toFixed?.(1) ?? g.end_s}s)` : ""}: ${escapeHtml(g.detail)}</li>`).join("");
}

// ---- measurement panel: human-supplied calibration inputs (nothing is estimated client-side)
let calPoints = [];
let replayLabel = null;
let voiceAvailable = null;   // null until the server says; false => hosted voice cannot run
let voiceReason = null;
let replayTimer = null;
let replayT = 0;
let replayBusy = false;
const REPLAY_STEP_S = 0.2;   // one saved evidence frame per 0.2 s of video (5 fps), same cadence as the camera evidence timer and app.replay
const calNeed = () => ({ plane_quad: 4, segment: 2, wall: 4, truth_spine: 1 }[document.querySelector("#calMode").value]);

// "~N (range A-B, reliability X)": best guess + range from the packet totals; plain N when the packet has no range.
function countText(t) {
  if (!t || t.book_count == null) return null;
  const headPan = t.headline_estimator === "panorama" || t.count_estimator === "panorama";
  const lines = t.book_lines != null && t.book_lines !== t.book_count ? `; ${t.book_lines} book lines listed, some may be duplicates` : "";
  if (!t.book_count_range) return String(t.book_count) + lines;
  return headPan
    ? `~${t.book_count} (2D panorama count; panorama range ${t.book_count_range[0]}–${t.book_count_range[1]}, may not contain the true count${lines})`
    : `~${t.book_count} (range ${t.book_count_range[0]}–${t.book_count_range[1]}, reliability ${t.count_reliability || "unknown"}${lines})`;
}

function renderMeasurement() {
  const ms = document.querySelector("#measureSummary");
  if (ms) {
    const ct = countText(state.totals);
    ms.textContent = ct ? `Books: ${ct}` : "";
  }
  const m = state.measurement;
  const box = document.querySelector("#measureStatus");
  if (!m || m.error) { box.textContent = m && m.error ? m.error : "No data yet."; return; }
  const items = (m.items || []).map((i) => `<li class="${i.present ? "meas-ok" : i.optional ? "" : "meas-missing"}">${i.present ? "✓" : i.optional ? "○" : "✗"} ${escapeHtml(i.item)}${i.present ? "" : " - " + escapeHtml(i.how)}</li>`).join("");
  const cals = (m.calibrations || []).map((c) => `<li>${escapeHtml(c.calibration_id)} ${escapeHtml(c.kind)} on ${escapeHtml((c.plane || {}).id)}: <b>${escapeHtml(c.status)}</b>${c.problems.length ? " (" + escapeHtml(c.problems.join("; ")) + ")" : ""} <small>user-supplied</small></li>`).join("");
  const pend = (m.pending_dimensions || []).map((p) => `<li>stated for ${escapeHtml(p.plane_id)}: width ${p.width_m ?? "?"} m, height ${p.height_m ?? "?"} m - tap the corners below</li>`).join("");
  const sc = (m.scale || {}).summary;
  box.innerHTML = `<ul class="review-list">${items}</ul>` + (cals || pend ? `<ul class="review-list">${cals}${pend}</ul>` : "") +
    (sc ? `<div>Spines measured ${sc.measured}, abstained ${sc.abstained}. ${m.scale.shelf_run_m == null ? "Shelf run withheld." : "Shelf run " + m.scale.shelf_run_m + " m."}</div>` : "") +
    `<div>Room: ${escapeHtml((m.room || {}).status || "")}${((m.room || {}).abstention_reasons || []).length ? " (" + escapeHtml(m.room.abstention_reasons.join(", ")) + ")" : ""}</div>` +
    `<div class="small">${escapeHtml(m.validation || "")}</div>`;
  const sel = document.querySelector("#calFrame");
  const frames = (state.frames_index || []).map((f) => f.frame_id);
  if (sel && frames.join("|") !== [...(sel.options || [])].map((o) => o.value).join("|")) {
    const keep = sel.value;
    sel.innerHTML = frames.map((f) => `<option>${escapeHtml(f)}</option>`).join("");
    if (keep && frames.includes(keep)) sel.value = keep;
    loadCalFrame();
  }
}

// ---- automatic measurement of a saved run (app.measurement.auto_view): numbers with their +-, scale source, null + reason; nothing computed here
const fmtPm = (v, tol, unit) => (v == null ? null : `${v} ${unit}` + (tol != null ? ` ± ${tol}` : ""));

let autoMeasureSeq = 0;   // only the newest request renders (a slow earlier load must not overwrite the selected run)
async function loadAutoMeasure(runId) {
  const seq = ++autoMeasureSeq;
  const sel = document.querySelector("#autoRun");
  const box = document.querySelector("#autoMeasure");
  try {
    const list = (await (await fetch(`${API_ORIGIN}/api/auto-measurements`)).json()).runs || [];
    if (seq !== autoMeasureSeq) return;
    const own = (state && state.post_sweep && state.post_sweep.run_id) || (state && state.identity && state.identity.run_id) || null;
    const ids = [...new Set([own, ...list.map((r) => r.run_id)].filter(Boolean))];
    const want = runId || sel.value || own || ids[0];
    sel.innerHTML = ids.map((r) => `<option value="${escapeHtml(r)}">${escapeHtml(r)}${r === own ? " (this sweep)" : ""}</option>`).join("");
    if (!want) { box.textContent = "No measured run yet: the measurement runs automatically after the 3D map of a finished sweep."; return; }
    sel.value = want;
    const r = await fetch(`${API_ORIGIN}/api/runs/${encodeURIComponent(want)}/auto-measurement`);
    if (!r.ok) throw new Error(await r.text());
    const view = await r.json();
    if (seq !== autoMeasureSeq) return;
    sel.value = want;
    box.innerHTML = renderAutoMeasure(view);
  } catch (error) {
    if (seq === autoMeasureSeq) box.textContent = `Could not load the measurement: ${error.message}`;
  }
}

function renderAutoMeasure(v) {
  if (!v.available) return `<p>${escapeHtml(v.reason || "Not measured.")}</p>`;
  const sc = v.scale || {};
  const scaleLine = sc.source
    ? `<b>Scale source: ${escapeHtml(sc.label)}</b>${sc.relative_uncertainty_pct ? ` · ±${sc.relative_uncertainty_pct}%` : ""}${sc.not_a_measurement ? " · <span class='sev-warning'>not a measurement</span>" : ""}` +
      (sc.why_not_catalogue ? `<div>Why not the catalogue: ${escapeHtml(String(sc.why_not_catalogue).replace(/_/g, " "))}${sc.cross_check ? ` (cross-check: ${sc.cross_check.n_books} catalogued book(s), prior ${sc.cross_check.prior_vs_catalogue_pct}% vs catalogue)` : ""}</div>` : "")
    : `<b>No metric scale</b>: ${escapeHtml(String(sc.reason || "").replace(/_/g, " "))}`;
  const sm = v.summary || {};
  const sh = v.shelf || {};
  const qty = (label, q) => `<li>${escapeHtml(label)}: ${q.value != null ? `<b>${q.value} ${escapeHtml(q.unit)}</b>${q.pct ? ` ±${q.pct}%` : ""}` : `<i>unknown</i> - ${escapeHtml(q.reason || "not established")}`}</li>`;
  const room = v.room || {};
  const d = room.dimensions || {}, a = room.areas || {};
  const roomHtml = `<h3>Room: ${escapeHtml(room.status)}</h3><ul class="review-list">` +
    qty("Width", d.width_m || {}) + qty("Depth", d.depth_m || {}) + qty("Ceiling height", d.height_m || {}) +
    qty("Floor area", a.floor_area_m2 || {}) + qty("Wall area (gross)", a.wall_area_m2 || {}) + qty("Shelved wall (books' bounding rectangle)", a.shelved_wall_area_m2 || {}) + `</ul>` +
    ((room.reasons || []).length ? `<div>Why: ${room.reasons.map(escapeHtml).join("; ")}.</div>` : "") +
    ((room.film || []).length ? `<div>To get it next time, film: ${room.film.map(escapeHtml).join(" ")}</div>` : "");
  const thickWhy = [...new Set((v.books || []).filter((b) => b.status === "measured").map((b) => b.thickness_reason))];
  const runWideThick = thickWhy.length === 1 && thickWhy[0] ? thickWhy[0] : null;   // withheld for the whole run: said once, not on every row
  const rows = [...(v.books || [])].sort((x, y) => (x.status !== "measured") - (y.status !== "measured")).map((b) => {
    const name = escapeHtml(b.title || (b.visible_text && b.visible_text !== "[?]" ? `"${b.visible_text}"` : "unread spine"));
    if (b.status !== "measured") return `<tr class="meas-abst"><td>${escapeHtml(b.group_id)}</td><td>${name}</td><td colspan="2"><i>unknown</i> - ${escapeHtml(b.reason || "abstained")}</td></tr>`;
    return `<tr><td>${escapeHtml(b.group_id)}</td><td>${name}</td><td>${escapeHtml(fmtPm(b.spine_height_cm, b.height_pm_cm, "cm"))}</td>` +
      `<td>${b.spine_thickness_cm != null ? escapeHtml(fmtPm(b.spine_thickness_cm, b.thickness_pm_cm, "cm")) : `<i>unknown</i>${runWideThick ? "" : " - " + escapeHtml(b.thickness_reason || "withheld")}`}</td></tr>`;
  }).join("");
  return `<div>${scaleLine}</div>` +
    `<div>Spines: <b>${sm.measured}</b> of ${sm.books} books measured (${sm.with_thickness} with a thickness); ${sm.abstained} unknown, with the reason.` +
    (runWideThick ? ` <b>Thickness unknown for every book</b>: ${escapeHtml(runWideThick.replace(/^thickness withheld: /, ""))}.` : "") +
    ` Shelf run (measured part only): ${sh.shelf_run_m != null ? `<b>${sh.shelf_run_m} m</b> over ${sh.rows} row(s)` : "<i>unknown</i>"}.</div>` +
    `<div class="small">${escapeHtml(v.measure_dir)} · ${v.attached_to_packet ? "attached to the claim packet" : "not attached to the packet"} · ${escapeHtml(v.validation)}</div>` +
    roomHtml +
    `<h3>Spines (height × thickness, ± stated uncertainty)</h3><table class="meas-table"><thead><tr><th>group</th><th>book</th><th>height</th><th>thickness</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function loadCalFrame() {
  const sel = document.querySelector("#calFrame");
  const img = document.querySelector("#calImg");
  if (!sessionId || !sel.value) { img.removeAttribute("src"); return; }
  const src = `${API_ORIGIN}/api/sessions/${sessionId}/frames/${encodeURIComponent(sel.value)}.jpg`;
  if (img.getAttribute("src") !== src) { img.src = src; calPoints = []; renderCalDots(); }
}

function renderCalDots() {
  const img = document.querySelector("#calImg");
  document.querySelector("#calDots").innerHTML = calPoints.map((p, i) => {
    const w = img.naturalWidth || 1, h = img.naturalHeight || 1;
    return `<span class="cal-dot" style="left:${(100 * p[0] / w).toFixed(2)}%;top:${(100 * p[1] / h).toFixed(2)}%">${i + 1}</span>`;
  }).join("");
}

function calClick(event) {
  const img = document.querySelector("#calImg");
  if (!img.naturalWidth || calPoints.length >= calNeed()) return;
  const r = img.getBoundingClientRect();
  calPoints.push([Math.round((event.clientX - r.left) * img.naturalWidth / r.width), Math.round((event.clientY - r.top) * img.naturalHeight / r.height)]);
  renderCalDots();
}

async function postJson(path, body) {
  const r = await fetch(`${API_ORIGIN}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || `HTTP ${r.status}`);
  return j;
}

function calSubmitBody() {
  const val = (id) => document.querySelector(id).value;
  const mode = val("#calMode");
  if (mode === "truth_spine") throw new Error('This mode records held-out truth: use "Save spine reading" below, not "Save marks".');
  if (calPoints.length !== calNeed()) throw new Error(`Tap exactly ${calNeed()} points on the frame first.`);
  const base = { frame_id: val("#calFrame") };
  if (mode === "wall") return { path: "room-evidence/walls", body: { ...base, quad_px: calPoints, wall_id: val("#calWall") || undefined, spans_floor_to_ceiling: document.querySelector("#calCeiling").checked } };
  const common = { ...base, image_points_px: calPoints, unit: val("#calUnit"), tolerance_pct: Number(val("#calTol")), plane_id: val("#calPlane") || undefined,
    utterance: val("#calUtterance"), role: "shelf_face", assume_fronto_parallel: document.querySelector("#calFronto").checked };
  if (mode === "plane_quad") return { path: "calibrations", body: { ...common, kind: "plane_quad", width: Number(val("#calWidth")), height: Number(val("#calHeight")) } };
  return { path: "calibrations", body: { ...common, kind: "segment", length: Number(val("#calWidth")) } };
}

async function submitCalibration() {
  const msg = document.querySelector("#calMsg");
  try {
    const { path, body } = calSubmitBody();
    await postJson(`/api/sessions/${sessionId}/${path}`, body);
    msg.textContent = "Saved. Values update as processing allows.";
    calPoints = []; renderCalDots();
  } catch (e) { msg.textContent = `Not saved: ${e.message}`; }
}

// ---- held-out tape truth: sent to /truth/* routes, stored outside the run, never fed back into any stage
async function loadTruth() {
  if (!sessionId) return;
  try {
    const r = await fetch(`${API_ORIGIN}/api/sessions/${sessionId}/truth`);
    if (!r.ok) return;
    const t = await r.json();
    document.querySelector("#truthCount").textContent = `${t.spine_samples.length} / ${t.spine_target}`;
    document.querySelector("#truthList").innerHTML = t.spine_samples.map((s) => `<li>${escapeHtml(s.sample_id)} · ${escapeHtml(s.frame_id)} @ [${s.tap_px}] · ${s.height_cm} x ${s.thickness_cm} cm ${s.language ? "· " + escapeHtml(s.language) : ""} ${escapeHtml(s.description || "")}</li>`).join("") +
      (t.room ? `<li>room: ${t.room.length_m} x ${t.room.width_m} x ${t.room.height_m} m (saved)</li>` : "");
    if (t.annotator && !document.querySelector("#truthName").value) document.querySelector("#truthName").value = t.annotator;
  } catch (_) { /* advisory */ }
}

async function saveTruthSpine() {
  const msg = document.querySelector("#truthMsg");
  try {
    if (document.querySelector("#calMode").value !== "truth_spine") throw new Error('Set the mode to "HELD-OUT truth: tap ONE book" first.');
    if (calPoints.length !== 1) throw new Error("Tap exactly one book on the frame.");
    const val = (id) => document.querySelector(id).value;
    await postJson(`/api/sessions/${sessionId}/truth/spines`, { frame_id: val("#calFrame"), tap_px: calPoints[0], height: Number(val("#truthHeight")),
      thickness: Number(val("#truthThickness")), unit: val("#truthUnit"), annotator: val("#truthName"), description: val("#truthDesc"), language: val("#truthLang") || undefined });
    msg.textContent = "Saved (held out: not used by the software).";
    calPoints = []; renderCalDots();
    document.querySelector("#truthHeight").value = ""; document.querySelector("#truthThickness").value = "";
    await loadTruth();
  } catch (e) { msg.textContent = `Not saved: ${e.message}`; }
}

async function saveTruthRoom() {
  const msg = document.querySelector("#truthMsg");
  try {
    const val = (id) => document.querySelector(id).value;
    await postJson(`/api/sessions/${sessionId}/truth/room`, { length: Number(val("#truthRoomL")), width: Number(val("#truthRoomW")), height: Number(val("#truthRoomH")),
      unit: val("#truthRoomUnit"), annotator: val("#truthName"), notes: val("#truthRoomNotes") });
    msg.textContent = "Room reading saved (held out).";
    await loadTruth();
  } catch (e) { msg.textContent = `Not saved: ${e.message}`; }
}

async function submitCeiling() {
  const msg = document.querySelector("#calMsg");
  try {
    await postJson(`/api/sessions/${sessionId}/room-evidence/height`, { value: Number(document.querySelector("#ceilValue").value), unit: document.querySelector("#calUnit").value,
      approximate: true, utterance: document.querySelector("#calUtterance").value || "ceiling height typed by the person" });
    msg.textContent = "Ceiling height saved as user-supplied.";
  } catch (e) { msg.textContent = `Not saved: ${e.message}`; }
}

function renderMode() {
  const currentState = state || emptyState;
  const sources = [];
  if (replayActive) sources.push("REPLAY of a recorded video");
  if (cameraStream) sources.push("LIVE camera");
  modeLabel.textContent = sources.length ? `Capture source: ${sources.join(", ")}. Both use the same ingestion path.` : "";
  endButton.disabled = currentState.status === "finalizing" || currentState.status === "finished";
  const ovEnd = document.querySelector("#ovEnd");
  if (ovEnd) ovEnd.disabled = endButton.disabled;
  renderSweepChrome();
}

function renderTeam() {
  const activity = [...(state.tool_activity || [])].slice(-6).reverse();
  const stages = Object.entries(state.stage_status || {}).map(([k, v]) => `<li><span class="team-dot ${v.status === "ok" ? "done" : v.status === "failed" ? "error" : "running"}"></span><span><span class="team-name">${escapeHtml(k)}</span> · ${escapeHtml(v.status)} ${v.detail ? "· " + escapeHtml(String(v.detail).slice(0, 110)) : ""}</span></li>`);
  const tools = activity.map((item) => {
    const phase = item.phase || "running";
    const meta = phase === "running" ? "working" : item.scheduling === "INTERRUPT" ? "interrupted the agent" : item.duration_ms != null ? `${item.duration_ms} ms` : "";
    return `<li><span class="team-dot ${phase}"></span><span><span class="team-name">${escapeHtml(teamLabels[item.name] || item.name)}</span> · ${escapeHtml(item.headline || "")}</span><span class="team-meta">${escapeHtml(meta)}</span></li>`;
  });
  teamFeedEl.innerHTML = [...tools, ...stages].join("") || `<li class="empty">Detection, reading and merging show up here as they run.</li>`;
  setWriting(processing || (state.tool_activity || []).some((item) => item.phase === "running"));
}

function appendSystem(text) {
  showOverlayLine(text);
  setState({ ...state, transcript: [...(state.transcript || []), { id: crypto.randomUUID(), speaker: "System", text }] });
}

function mergeTranscript(authoritative, local) {
  const byId = new Map((authoritative || []).map(turn => [turn.id, turn]));
  for (const turn of local || []) {
    if (!byId.has(turn.id)) byId.set(turn.id, turn);
  }
  return [...byId.values()];
}

function applyServerState(nextState) {
  if (nextState.session_id !== sessionId) return;
  setState({
    ...nextState,
    transcript: mergeTranscript(nextState.transcript, state?.transcript),
    tool_activity: mergeToolActivity(nextState.tool_activity, state?.tool_activity),
  });
}

function mergeToolActivity(authoritative, local) {
  const finished = new Set(["done", "error", "cancelled"]);
  const byId = new Map();
  for (const item of authoritative || []) byId.set(item.id, item);
  for (const item of local || []) {
    const current = byId.get(item.id);
    if (!current || (finished.has(item.phase) && !finished.has(current.phase))) byId.set(item.id, item);
  }
  return [...byId.values()];
}

function applyToolEvent(message) {
  const { type, ...entry } = message;
  const others = (state.tool_activity || []).filter((item) => item.id !== entry.id);
  setState({ ...state, tool_activity: [...others, entry] });
}

function upsertStreamingTurn(speaker, text, final = false, id = crypto.randomUUID()) {
  if (!String(text || "").trim()) return;
  const transcript = [...(state.transcript || [])];
  const index = transcript.findIndex(turn => turn.id === id);
  const turn = { id, speaker, text, streaming: !final };
  if (speaker === "Agent" || speaker === "System") showOverlayLine(text);
  if (index >= 0) transcript[index] = turn;
  else transcript.push(turn);
  setState({ ...state, transcript });
}

async function api(path, options = {}) {
  const response = await fetch(`${API_ORIGIN}${path}`, { headers: { "Content-Type": "application/json" }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed with status ${response.status}`);
  return payload;
}

function stopPlayback() {
  for (const source of playbackSources) { try { source.stop(); } catch (_) {} }
  playbackSources.clear();
  nextPlaybackTime = audioContext?.currentTime || 0;
}

function clearBusy() {
  processing = false;
  state.tool_activity = (state.tool_activity || []).map(item => item.phase === "running" ? { ...item, phase: "cancelled", headline: "Connection ended" } : item);
  render();
}

function disconnectLive() {
  generation += 1;
  const socket = liveSocket;
  liveSocket = null;
  connectionPromise = null;
  stopLiveVoice(false);
  stopCamera();
  stopPlayback();
  if (socket) socket.close();
  if (state) clearBusy();
}

async function createSession(resume = false) {
  if (resetting) return;
  resetting = true;
  newIntakeButton.disabled = true;
  const previous = sessionId || sessionStorage.getItem("sweepSession");
  disconnectLive();
  const epoch = generation;
  setStatus("Connecting", "neutral");
  sessionId = null;
  endFailure = null;
  sweepEnding = false;
  setRoomPhase(false);
  setState(emptyState);
  try {
    let payload;
    if (resume && previous) {
      try { payload = await api(`/api/sessions/${previous}`); } catch (_) { sessionStorage.removeItem("sweepSession"); }
    } else if (previous) {
      await api(`/api/sessions/${previous}`, { method: "DELETE" }).catch(() => {});
    }
    payload = payload || await api("/api/sessions", { method: "POST" });
    if (epoch !== generation) return;
    sessionId = payload.session_id;
    sessionStorage.setItem("sweepSession", sessionId);
    setState(payload.state);
    health = await api("/api/health");
    modelLabel.textContent = health.has_api_key ? `${health.live_model} · detection ${health.detect_model_default}` : `Voice off · detector ${health.detector_selection} · OCR ${health.reader_selection}`;
    await loadScans();
    refreshBuild();
    if (state.status === "finished") setStatus("Packet ready", "");
    else setStatus(health.has_api_key ? "Ready" : "No API key: voice off", health.has_api_key ? "" : "warning");
    if (!(window.matchMedia && window.matchMedia("(pointer: coarse)").matches)) textInput.focus();
  } catch (error) {
    setStatus("Backend unavailable", "danger");
    appendSystem(error.message);
  } finally {
    resetting = false;
    newIntakeButton.disabled = false;
  }
}

// New sweep never loses a sweep: when the current one is still recording, writing its packet or building its maps, ask first; on OK the server lets that sweep finish in the
// background (every sweep stays under runs/). If the status cannot be read, the click goes through: the server still detaches a processing sweep on delete.
async function requestNewSweep() {
  if (resetting) return;
  let info = null;
  if (sessionId) {
    try { info = await api(`/api/sessions/${sessionId}/processing`); } catch (_) { info = null; }
  }
  if (info && info.processing) {
    const ok = typeof window.confirm === "function" ? window.confirm(info.message || "This sweep is still processing. It will finish in the background. Start a new sweep?") : true;
    if (!ok) return;
  }
  await createSession(false);
  if (info && info.processing && sessionId) appendSystem(`The previous sweep (${info.run_id}) keeps ${info.phase === "maps" ? "building its maps" : "finishing its packet and maps"} in the background; it stays under runs/${info.run_id}.`);
}

function connectLive() {
  if (connectionPromise) return connectionPromise;
  if (!sessionId || resetting) return Promise.reject(new Error("Wait for the sweep to be ready."));
  const epoch = generation;
  const socket = new WebSocket(`${WS_ORIGIN}/ws/live?session_id=${encodeURIComponent(sessionId)}`);
  liveSocket = socket;
  connectionPromise = new Promise((resolve, reject) => {
    let ready = false;
    const timeout = setTimeout(() => { reject(new Error("Live connection timed out.")); socket.close(); }, 20000);
    const current = () => epoch === generation && liveSocket === socket;
    socket.onmessage = (event) => {
      if (!current()) return;
      const t0 = nowMs();
      const message = JSON.parse(event.data);
      if (message.session_id !== sessionId) return;
      if (message.type === "ready") {
        ready = true;
        voiceAvailable = message.voice !== false;
        voiceReason = message.voice_reason || null;
        clearTimeout(timeout);
        setStatus(message.voice === false ? "Connected (voice off)" : "Live", message.voice === false ? "warning" : "");
        resolve();
      } else if (message.type === "session") {
        if (message.notice) appendSystem(message.notice);
        if (message.model) modelLabel.textContent = `${message.model} · detection ${health.detect_model_default || ""}`;
      } else if (message.type === "processing") {
        processing = message.active;
        render();
      } else if (message.type === "transcript") {
        upsertStreamingTurn(message.speaker, message.text, message.final, message.id);
      } else if (message.type === "tool") {
        applyToolEvent(message);
      } else if (message.type === "audio") {
        playPcm24(message.data);
      } else if (message.type === "state") {
        applyServerState(message.state);
        noteStateMessage(String(event.data).length, nowMs() - t0);
      } else if (message.type === "finalize") {
        const { type, session_id, ...info } = message;
        setState({ ...state, finalize: info });
      } else if (message.type === "status") {
        if (message.status === "finalizing") {           // ended by the voice agent or another tab: stop sending frames now
          sweepEnding = true;
          setRoomPhase(false);
          if (cameraStream || replayActive) stopCamera();
        }
      } else if (message.type === "end_ack") {
        if (message.action === "end") finishSweep();          // no room pan due (already done, replay, or no camera): end now
      } else if (message.type === "nudge") {
        if (message.code === "room_sweep") setRoomPhase(true);
        showOverlayLine(message.message);
        setState({ ...state, nudges: [...(state.nudges || []), message].slice(-5) });
      } else if (message.type === "interrupted") {
        stopPlayback();
      } else if (message.type === "error") {
        processing = false;
        if (state && state.status === "failed") endFailure = message.message;     // only a real failure; a late-frame refusal while finalizing is not one
        appendSystem(message.message);
        if (!ready) { reject(new Error(message.message)); socket.close(); }
      }
    };
    socket.onerror = () => { clearTimeout(timeout); reject(new Error("Live connection failed.")); socket.close(); };
    socket.onclose = () => {
      clearTimeout(timeout);
      if (!ready) reject(new Error("Live connection ended before it was ready."));
      if (!current()) return;
      liveSocket = null;
      connectionPromise = null;
      stopLiveVoice(false);
      stopCamera();
      stopPlayback();
      clearBusy();
      setStatus("Disconnected — reconnect to continue", "warning");
    };
  });
  return connectionPromise;
}

async function unlockAudio() {
  if (window.AudioContext) {
    audioContext = audioContext || new AudioContext();
    await audioContext.resume();
  }
}

async function sendClaimantTurn(text) {
  const epoch = generation;
  try {
    stopPlayback();
    await unlockAudio();
    await connectLive();
    if (epoch !== generation) return;
    const id = crypto.randomUUID();
    liveSocket.send(JSON.stringify({ type: "text", text, id }));
    upsertStreamingTurn("Claimant", text, true, id);
    processing = true;
    render();
  } catch (error) {
    if (epoch !== generation) return;
    textInput.value = text;
    appendSystem(`${error.message} Your message is back in the input; send it again to retry.`);
  }
}

async function startLiveVoice() {
  if (micPending || isRecording) return;
  const epoch = generation;
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) {
    appendSystem("This browser cannot capture microphone audio. Type instead.");
    return;
  }
  try {
    micPending = true;
    await unlockAudio();
    await connectLive();
    if (voiceAvailable === false) {   // fail clearly BEFORE opening the microphone
      appendSystem(`Voice is unavailable: ${voiceReason || "hosted voice is not configured"}. The camera sweep, frames and inventory still work; type instead of talking.`);
      setStatus("Voice unavailable", "warning");
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    if (epoch !== generation || !liveSocket || liveSocket.readyState !== WebSocket.OPEN) { stream.getTracks().forEach(track => track.stop()); return; }
    audioStream = stream;
    audioContext = audioContext || new AudioContext();
    await audioContext.resume();
    inputSource = audioContext.createMediaStreamSource(audioStream);
    inputProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    inputProcessor.onaudioprocess = (event) => {
      event.outputBuffer.getChannelData(0).fill(0);
      if (!liveSocket || liveSocket.readyState !== WebSocket.OPEN) return;
      const pcm16 = resampleToPcm16(event.inputBuffer.getChannelData(0), audioContext.sampleRate, 16000);
      liveSocket.send(JSON.stringify({ type: "audio", data: arrayBufferToBase64(pcm16.buffer) }));
    };
    inputSource.connect(inputProcessor);
    inputProcessor.connect(audioContext.destination);
    isRecording = true;
    micButton.classList.add("active");
    micButton.querySelector(".round-label").textContent = "Listening";
    setStatus("Live, listening", "");
    renderSweepChrome();
  } catch (error) {
    const denied = error.name === "NotAllowedError" || /denied|permission/i.test(error.message);
    appendSystem(denied ? "Microphone access was denied. Allow it for this site or type instead." : `Voice failed: ${error.message}`);
    stopLiveVoice(false);
  } finally { micPending = false; }
}

function stopLiveVoice(closeSocket = true) {
  isRecording = false;
  micButton.classList.remove("active");
  micButton.querySelector(".round-label").textContent = "Talk";
  renderSweepChrome();
  inputProcessor?.disconnect();
  inputSource?.disconnect();
  audioStream?.getTracks().forEach((track) => track.stop());
  inputProcessor = null;
  inputSource = null;
  audioStream = null;
  if (closeSocket) {
    disconnectLive();
    setStatus("Call ended — sweep saved for reconnect", "neutral");
  }
}

// ---- Camera capture (iPhone Safari over HTTPS, Mac webcam, Continuity Camera) ---------------------------------------------------------------
// Constraint sets, best first. All "ideal" (never "exact"/"min"): iOS Safari swaps width/height with the device orientation and a min would make
// portrait fail. Hold the phone in landscape: the pipeline was tuned on 1920x1080 landscape frames.
// ?res=4k asks for 3840x2160 first (spine crops get 2x the pixels; the browser may grant less, the granted size is reported) and sends native pixels up to 3840
// on the long side; the default stays 1920x1080-class. ?zoom=2 applies the track's zoom (where the browser exposes it) after the camera opens.
function captureRequest() {
  const q = new URLSearchParams((window.location && window.location.search) || "");
  const zoom = Number(q.get("zoom"));
  return { name: q.get("res") === "4k" ? "4k" : "1080", zoom: Number.isFinite(zoom) && zoom > 1 ? zoom : null };
}
let captureReq = { name: "1080", zoom: null };
const evidenceScale = (vw, vh) => (captureReq.name === "4k" ? Math.min(1, EVIDENCE_MAX_LONG_4K / Math.max(vw, vh)) : Math.min(1, EVIDENCE_MAX_WIDTH / vw));

function cameraConstraintSets() {
  const hd = [
    { video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: CAPTURE_FPS } } },
    { video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } } },
    { video: true },
  ];
  if (captureReq.name !== "4k") return hd;
  return [{ video: { facingMode: { ideal: "environment" }, width: { ideal: 3840 }, height: { ideal: 2160 }, frameRate: { ideal: CAPTURE_FPS } } }, ...hd];
}

async function openCameraStream() {
  let last = null;
  const sets = cameraConstraintSets();
  for (let i = 0; i < sets.length; i += 1) {
    try { return { stream: await navigator.mediaDevices.getUserMedia(sets[i]), constraintSet: i }; }
    catch (error) {
      last = error;
      if (error.name === "NotAllowedError" || error.name === "SecurityError" || /denied|permission/i.test(error.message || "")) throw error;   // asking again cannot help
    }
  }
  throw last;
}

async function tuneVideoTrack(track) {
  try {   // continuous autofocus where the browser exposes it (Chrome/Android; iOS Safari does not, and keeps its own continuous AF)
    const caps = track?.getCapabilities?.() || {};
    if (Array.isArray(caps.focusMode) && caps.focusMode.includes("continuous")) await track.applyConstraints({ advanced: [{ focusMode: "continuous" }] });
  } catch (_) { /* advisory */ }
  if (captureReq.zoom && track?.getCapabilities) {   // zoom halves the field of view: each book is seen in fewer frames at the same hand speed, so it is opt-in
    try {
      const z = track.getCapabilities().zoom;
      if (z && typeof z.max === "number") await track.applyConstraints({ advanced: [{ zoom: Math.min(Math.max(captureReq.zoom, z.min || 1), z.max) }] });
    } catch (_) { /* unsupported: reported as zoom null */ }
  }
}

let wakeLock = null;
async function holdScreenAwake() {   // iPhone auto-lock (30 s default) would end a 60-90 s sweep; the lock is released with the camera
  try { if (navigator.wakeLock?.request) wakeLock = await navigator.wakeLock.request("screen"); } catch (_) { wakeLock = null; }
}
function releaseScreen() { try { wakeLock?.release?.(); } catch (_) { /* already released */ } wakeLock = null; }

// Best-of-window capture. The camera runs at 30 fps and each 200 ms window sends its SHARPEST decoded frame (Laplacian variance on a small grey copy).
// Live iPhone sweeps (YugoTest5/6) started at ~4.3 fps and sank to ~1.5 fps by the end: the page's main thread was saturated (its 200 ms timer ran at 2.5 Hz,
// and every server state message re-rendered hundreds of book rows). So here: windows are clocked by the camera callback (a busy thread cannot stretch them), the
// ranking is budgeted (every Nth camera frame when it costs too much), up to MAX_INFLIGHT encodes overlap, JPEG quality steps down only when encoding cannot keep
// up, a frame is skipped (not queued) when the socket is backed up, and every stage is timed and reported (capture_info client_stats) so a live run shows
// where the time went. The old direct path stays for replays and browsers without requestVideoFrameCallback.
const MAX_INFLIGHT = 2;                  // JPEG encodes (toBlob + base64 + send) allowed to overlap
const RANK_BUDGET = 0.25;                // share of main-thread time the sharpness ranking may use before it samples fewer camera frames
const RANK_EVERY_MAX = 6;
const STATS_REPORT_MS = 10000;
const LAG_PROBE_MS = 100;
const STAT_RING = 400;
const JPEG_QUALITY_STEPS = [0.95, 0.9, 0.85];   // only ever stepped down when >=5 of the last 20 windows were skipped because encodes were still in flight
let evidenceQuality = EVIDENCE_QUALITY;
const nowMs = () => (typeof performance !== "undefined" && performance.now ? performance.now() : Date.now());
const r1 = (v) => (v == null || !Number.isFinite(v) ? null : Math.round(v * 10) / 10);
const bow = { active: false, handle: null, t0: null, tStart: null, small: null, sctx: null, gray: null, best: null, bctx: null, win: null, inflight: 0, lastMsgBytes: 0,
              rankEvery: 1, rankCost: null, cps: null, vw: 0, vh: 0, skipRing: [], buckets: [], samples: {}, lagTimer: null, reportTimer: null, lagLast: 0,
              stats: null };
const freshStats = () => ({ windows: 0, sent: 0, empty_windows: 0, skipped_encoding: 0, skipped_backlog: 0, candidates: 0, seen: 0, callbacks: 0, gain_sum: 0, gain_n: 0,
                            inflight_max: 0, quality_steps: 0, state_msgs: 0 });
bow.stats = freshStats();

function pushSample(name, v) {
  if (!Number.isFinite(v)) return;
  const a = bow.samples[name] || (bow.samples[name] = []);
  a.push(v);
  if (a.length > STAT_RING) a.shift();
}
function summarize(name) {
  const a = bow.samples[name];
  if (!a || !a.length) return {};
  const s = a.slice().sort((x, y) => x - y), at = (f) => s[Math.min(s.length - 1, Math.floor(f * s.length))];
  return { p50: r1(at(0.5)), p90: r1(at(0.9)), max: r1(s[s.length - 1]) };
}
const flat = (prefix, s) => ({ [`${prefix}_p50`]: s.p50 ?? null, [`${prefix}_p90`]: s.p90 ?? null, [`${prefix}_max`]: s.max ?? null });

function noteStateMessage(bytes, ms) {   // every server state message is parsed and rendered on the same thread that captures frames
  if (!bow.active) return;
  bow.stats.state_msgs += 1;
  pushSample("state_kb", bytes / 1024);
  pushSample("state_ms", ms);
}

const newWindow = (start = null) => ({ n: 0, seen: 0, best: -1, first: -1, t: 0, start });

function sizeSharpCanvas(vw, vh) {   // the ranking copy keeps the camera's aspect (portrait on a phone held upright); long side SHARP_W
  const k = SHARP_W / Math.max(vw, vh), w = Math.max(16, Math.round(vw * k)), h = Math.max(16, Math.round(vh * k));
  bow.small.width = w;
  bow.small.height = h;
  bow.sctx = bow.small.getContext("2d", { willReadFrequently: true });
  bow.gray = new Float32Array(w * h);
  bow.vw = vw;
  bow.vh = vh;
}

function startBestOfWindow() {
  const params = new URLSearchParams((window.location && window.location.search) || "");
  if (params.get("capture") === "timer") return false;   // ?capture=timer: A/B against the plain timer path
  if (typeof cameraPreview.requestVideoFrameCallback !== "function" || typeof evidenceCanvas.toBlob !== "function") return false;
  try {
    bow.small = document.createElement("canvas");
    bow.best = document.createElement("canvas");
    bow.bctx = bow.best.getContext("2d");
  } catch (_) { return false; }
  bow.stats = freshStats();
  Object.assign(bow, { t0: null, tStart: null, inflight: 0, lastMsgBytes: 0, rankEvery: 1, rankCost: null, cps: null, vw: 0, vh: 0, gray: null, sctx: null, skipRing: [], buckets: [], samples: {} });
  evidenceQuality = EVIDENCE_QUALITY;
  bow.win = newWindow();
  bow.active = true;
  bow.lagLast = nowMs();
  bow.lagTimer = window.setInterval(() => { const n = nowMs(); pushSample("lag_ms", n - bow.lagLast - LAG_PROBE_MS); bow.lagLast = n; }, LAG_PROBE_MS);
  bow.reportTimer = window.setInterval(reportCaptureStats, STATS_REPORT_MS);
  bow.handle = cameraPreview.requestVideoFrameCallback(onVideoFrame);
  return true;
}

function stopBestOfWindow() {
  bow.active = false;
  try { if (bow.handle != null) cameraPreview.cancelVideoFrameCallback?.(bow.handle); } catch (_) { /* element already detached */ }
  bow.handle = null;
  if (bow.lagTimer) window.clearInterval(bow.lagTimer);
  if (bow.reportTimer) window.clearInterval(bow.reportTimer);
  bow.lagTimer = bow.reportTimer = null;
}

function lumaLaplacianVariance(video) {
  const w = bow.small.width, h = bow.small.height, g = bow.gray;
  bow.sctx.drawImage(video, 0, 0, w, h);
  const px = bow.sctx.getImageData(0, 0, w, h).data;
  for (let i = 0, j = 0; i < g.length; i += 1, j += 4) g[i] = 0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2];
  let sum = 0, sum2 = 0, n = 0;
  for (let y = 1; y < h - 1; y += 1) {
    for (let x = 1, i = y * w + 1; x < w - 1; x += 1, i += 1) {
      const l = g[i - 1] + g[i + 1] + g[i - w] + g[i + w] - 4 * g[i];
      sum += l; sum2 += l * l; n += 1;
    }
  }
  const mean = sum / n;
  return sum2 / n - mean * mean;
}

function onVideoFrame(now, meta) {
  if (!bow.active) return;
  bow.handle = cameraPreview.requestVideoFrameCallback(onVideoFrame);
  const vw = cameraPreview.videoWidth, vh = cameraPreview.videoHeight;
  if (!vw) return;
  const st = bow.stats, win = bow.win;
  st.callbacks += 1;
  if (win.start === null) win.start = now;
  win.seen += 1;
  const t = typeof meta?.captureTime === "number" ? meta.captureTime : now;
  if (bow.t0 === null) { bow.t0 = t; bow.tStart = now; }
  if (win.n === 0 || st.callbacks % bow.rankEvery === 0) {   // the first frame of a window is always ranked: it is the fallback if no other is
    const c0 = nowMs();
    if (!bow.gray || bow.vw !== vw || bow.vh !== vh) sizeSharpCanvas(vw, vh);
    const sharp = lumaLaplacianVariance(cameraPreview);
    win.n += 1;
    if (win.first < 0) win.first = sharp;
    if (sharp > win.best) {
      const scale = evidenceScale(vw, vh);
      const w = Math.round(vw * scale), h = Math.round(vh * scale);
      if (bow.best.width !== w || bow.best.height !== h) { bow.best.width = w; bow.best.height = h; }
      const c1 = nowMs();
      bow.bctx.drawImage(cameraPreview, 0, 0, w, h);
      pushSample("copy_ms", nowMs() - c1);
      win.best = sharp;
      win.t = t;
    }
    const cost = nowMs() - c0;
    pushSample("rank_ms", cost);
    bow.rankCost = bow.rankCost == null ? cost : 0.8 * bow.rankCost + 0.2 * cost;
  }
  if (now - win.start >= EVIDENCE_INTERVAL_MS) closeWindow(now);
}

function adaptRanking() {   // keep ranking under RANK_BUDGET of the main thread: sample every Nth camera frame instead of every one
  if (bow.rankCost == null || bow.cps == null) return;
  const load = bow.rankCost * (bow.cps / bow.rankEvery) / 1000;
  if (load > RANK_BUDGET && bow.rankEvery < RANK_EVERY_MAX) bow.rankEvery += 1;
  else if (load < RANK_BUDGET * 0.4 && bow.rankEvery > 1) bow.rankEvery -= 1;
}

function noteEncodeSkip(skipped) {   // encoding cannot keep up at this quality: step JPEG quality down (never below 0.85)
  const ring = bow.skipRing;
  ring.push(skipped ? 1 : 0);
  if (ring.length > 20) ring.shift();
  if (ring.length >= 20 && ring.reduce((a, b) => a + b, 0) >= 5) {
    const i = JPEG_QUALITY_STEPS.indexOf(evidenceQuality);
    if (i >= 0 && i < JPEG_QUALITY_STEPS.length - 1) { evidenceQuality = JPEG_QUALITY_STEPS[i + 1]; bow.stats.quality_steps += 1; }
    ring.length = 0;
  }
}

function closeWindow(now) {
  const win = bow.win, st = bow.stats;
  const next = win.start + EVIDENCE_INTERVAL_MS;
  bow.win = newWindow(now - next > EVIDENCE_INTERVAL_MS ? now : next);   // stay on the 200 ms grid (5.0 fps) but never try to catch up after a stall
  st.windows += 1;
  const dur = now - win.start;
  pushSample("window_ms", dur);
  const cps = win.seen / Math.max(0.001, dur / 1000);
  bow.cps = bow.cps == null ? cps : 0.7 * bow.cps + 0.3 * cps;
  adaptRanking();
  if (win.n === 0) { st.empty_windows += 1; return; }
  if (!frameActive()) return;
  if (bow.inflight >= MAX_INFLIGHT) { st.skipped_encoding += 1; noteEncodeSkip(true); return; }
  noteEncodeSkip(false);
  if (liveSocket.bufferedAmount > Math.max(MAX_SOCKET_BACKLOG, 3 * (bow.lastMsgBytes || 0))) { st.skipped_backlog += 1; return; }   // 3 frames' worth: a 4K frame alone is >3 MB
  st.candidates += win.n;
  st.seen += win.seen;
  if (win.first > 0) { st.gain_sum += win.best / win.first; st.gain_n += 1; }   // sharpness of the picked frame vs the first candidate of the window
  const message = { type: "evidence_frame", source: "camera", client_t_s: Math.round(win.t - bow.t0) / 1000, cand: win.n, seen: win.seen, win_ms: Math.round(dur),
                    sharp_small: Math.round(win.best * 10) / 10 };
  bow.inflight += 1;
  st.inflight_max = Math.max(st.inflight_max, bow.inflight);
  const t0 = nowMs();
  bow.best.toBlob(async (blob) => {   // toBlob snapshots the canvas when called, so the next window may redraw it while this one encodes
    try {
      if (!blob || !frameActive()) return;
      const t1 = nowMs();
      pushSample("encode_ms", t1 - t0);
      pushSample("jpeg_kb", blob.size / 1024);
      message.enc_ms = Math.round(t1 - t0);
      message.q = evidenceQuality;
      message.data = await blobToBase64(blob);
      const t2 = nowMs();
      pushSample("b64_ms", t2 - t1);
      const payload = JSON.stringify(message);
      bow.lastMsgBytes = payload.length;
      liveSocket.send(payload);
      pushSample("send_ms", nowMs() - t2);
      st.sent += 1;
      const b = Math.floor((now - bow.tStart) / 10000);
      bow.buckets[b] = (bow.buckets[b] || 0) + 1;
    } catch (_) { /* socket closed while encoding */ }
    finally { bow.inflight -= 1; }
  }, "image/jpeg", evidenceQuality);
}

function watchdogWindow() {   // the camera callback closes windows; this only fires if it stalled (frozen preview, hidden tab)
  const win = bow.win, n = nowMs();
  if (win && win.start !== null && n - win.start > 2 * EVIDENCE_INTERVAL_MS) closeWindow(n);
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function reportCaptureInfo(track, constraintSet, bestOfWindow) {
  if (typeof track?.getSettings !== "function" || !liveSocket || liveSocket.readyState !== WebSocket.OPEN) return;
  const s = track.getSettings() || {};
  liveSocket.send(JSON.stringify({ type: "capture_info", label: track.label || "", width: s.width, height: s.height, frame_rate: s.frameRate, facing_mode: s.facingMode,
    focus_mode: s.focusMode, orientation: window.screen?.orientation?.type || "", secure_context: window.isSecureContext, constraint_set: constraintSet,
    capture_mode: bestOfWindow ? "best_of_window" : "timer", jpeg_quality: EVIDENCE_QUALITY, requested_resolution: captureReq.name === "4k" ? "3840x2160" : "1920x1080",
    zoom: typeof s.zoom === "number" ? s.zoom : null, zoom_requested: captureReq.zoom, zoom_max: track.getCapabilities?.().zoom?.max ?? null, user_agent: String(navigator.userAgent || "").slice(0, 160) }));
}

// Flat scalars only (the server keeps scalar values): per-stage timings in ms (p50/p90/max), camera callback rate, frames sent per 10 s, main-thread lag.
function reportCaptureStats() {
  if (!bow.active || !liveSocket || liveSocket.readyState !== WebSocket.OPEN) return;
  const st = bow.stats, elapsed = bow.tStart == null ? 0 : (nowMs() - bow.tStart) / 1000;
  liveSocket.send(JSON.stringify({ type: "capture_info", client_stats: {
    windows: st.windows, frames_sent: st.sent, empty_windows: st.empty_windows, skipped_while_encoding: st.skipped_encoding, skipped_socket_backlog: st.skipped_backlog,
    candidates_per_window: st.windows ? Math.round(10 * st.candidates / st.windows) / 10 : null,
    mean_sharpness_gain_vs_first_candidate: st.gain_n ? Math.round(100 * st.gain_sum / st.gain_n) / 100 : null,
    camera_frames_per_window: st.windows ? r1(st.seen / st.windows) : null, camera_callback_fps: elapsed > 0 ? r1(st.callbacks / elapsed) : null,
    fps_sent: elapsed > 0 ? r1(st.sent / elapsed) : null, elapsed_s: r1(elapsed), inflight_max: st.inflight_max, rank_every_final: bow.rankEvery,
    jpeg_quality_final: evidenceQuality, quality_steps: st.quality_steps,
    ...flat("window_ms", summarize("window_ms")), ...flat("rank_ms", summarize("rank_ms")), ...flat("copy_ms", summarize("copy_ms")),
    ...flat("encode_ms", summarize("encode_ms")), ...flat("jpeg_kb", summarize("jpeg_kb")), ...flat("base64_ms", summarize("b64_ms")), ...flat("send_ms", summarize("send_ms")),
    ...flat("main_thread_lag_ms", summarize("lag_ms")), state_msgs: st.state_msgs, ...flat("state_kb", summarize("state_kb")), ...flat("state_apply_ms", summarize("state_ms")),
    sent_per_10s: bow.buckets.slice(-40).map((v) => v || 0).join(",") } }));
}

function startFrameTimers() {
  frameTimer = window.setInterval(sendVoiceFrame, VOICE_FRAME_INTERVAL_MS);
  evidenceTimer = window.setInterval(sendEvidenceFrame, EVIDENCE_INTERVAL_MS);
}

async function startCamera() {
  if (cameraPending || cameraStream || replayActive) return;
  const epoch = generation;
  if (!navigator.mediaDevices?.getUserMedia) {
    appendSystem("This browser cannot access a camera.");
    return;
  }
  try {
    cameraPending = true;
    await unlockAudio();
    await connectLive();
    captureReq = captureRequest();
    const { stream, constraintSet } = await openCameraStream();
    if (epoch !== generation || !liveSocket || liveSocket.readyState !== WebSocket.OPEN) { stream.getTracks().forEach(track => track.stop()); return; }
    cameraStream = stream;
    stream.getVideoTracks().forEach(track => track.addEventListener("ended", stopCamera, { once: true }));
    liveSocket.send(JSON.stringify({ type: "camera_state", enabled: true }));
    cameraPreview.srcObject = cameraStream;
    cameraStage.hidden = false;
    cameraButton.classList.add("active");
    cameraButton.querySelector(".round-label").textContent = "Stop camera";
    captureStartedAt = Date.now();
    const videoTrack = stream.getVideoTracks()[0];
    await tuneVideoTrack(videoTrack);
    holdScreenAwake();
    const bestOfWindow = startBestOfWindow();
    reportCaptureInfo(videoTrack, constraintSet, bestOfWindow);
    startFrameTimers();
    if (!isRecording) setStatus("Camera on: sweep in progress", "");
    renderMode();
  } catch (error) {
    const denied = error.name === "NotAllowedError" || /denied|permission/i.test(error.message);
    appendSystem(denied ? "Camera access was denied. Allow it for this site to sweep the room." : `Camera failed: ${error.message}`);
    stopCamera();
  } finally { cameraPending = false; }
}

// Recorded-video preview: the SAME frame path as the camera (same canvases, same messages, same server ingestion).
let scans = [];
async function loadScans() {
  try {
    scans = (await api("/api/scans")).scans || [];
  } catch (error) {
    scans = [];
    appendSystem(`Scan registry unavailable: ${error.message}`);
  }
  replaySelect.innerHTML = scans.length
    ? scans.map((sc) => `<option value="${escapeHtml(sc.label)}"${sc.preview_available ? "" : " disabled"}>${escapeHtml(sc.label)} · ${escapeHtml(String(sc.sha256).slice(0, 8))}${sc.preview_available ? "" : " (no browser preview)"}</option>`).join("")
    : "<option value=''>no registered scans</option>";
}

async function refreshBuild() {
  try {
    const b = await api("/api/build");
    const pill = document.querySelector("#buildPill");
    pill.textContent = `backend pid ${b.pid} · ${b.backend_fingerprint_loaded}${b.backend_source_changed_since_start ? " · SOURCE CHANGED ON DISK: restart to load it" : ""} · static ${b.static_fingerprint_current}`;
    pill.classList.toggle("stale", b.backend_source_changed_since_start);
  } catch (_) { /* build info is advisory */ }
}

// Replay is driven by SEEKING the video to exact timestamps rather than playing it: decoding on seek works even when the
// browser pauses background media, frame times are deterministic, and capture never depends on the tab being visible.
function waitLoaded(video) {
  return new Promise((resolve, reject) => {
    if (video.readyState >= 2) return resolve();
    const t = setTimeout(() => reject(new Error("The video did not load.")), 15000);
    video.addEventListener("loadeddata", () => { clearTimeout(t); resolve(); }, { once: true });
    video.addEventListener("error", () => { clearTimeout(t); reject(new Error("The video could not be decoded by this browser.")); }, { once: true });
  });
}

function seekTo(video, t) {
  return new Promise((resolve) => {
    const done = () => { video.removeEventListener("seeked", done); resolve(); };
    video.addEventListener("seeked", done);
    video.currentTime = t;
    setTimeout(done, 2000);
  });
}

async function stepReplay() {
  if (!replayActive || replayBusy) return;
  replayBusy = true;
  try {
    const d = cameraPreview.duration;
    if (!Number.isFinite(d) || replayT > d) { stopCamera(); return; }
    await seekTo(cameraPreview, replayT);
    if (!replayActive) return;
    sendEvidenceFrame();
    if (Math.round(replayT / REPLAY_STEP_S) % 2 === 0) sendVoiceFrame();
    replayT += REPLAY_STEP_S;
  } finally { replayBusy = false; }
}

function startReplayStepping() {
  replayT = 0;
  replayTimer = window.setInterval(stepReplay, EVIDENCE_INTERVAL_MS);
}

async function startReplay(name) {
  if (cameraPending || cameraStream || replayActive || !name) return;
  const epoch = generation;
  try {
    cameraPending = true;
    await unlockAudio();
    await connectLive();
    if (epoch !== generation || !liveSocket || liveSocket.readyState !== WebSocket.OPEN) return;
    await postJson(`/api/sessions/${sessionId}/scan`, { label: name });   // bind the active session to this registered scan first
    replayActive = true;
    replayLabel = name;
    cameraPreview.srcObject = null;
    cameraPreview.muted = true;
    cameraPreview.src = `${API_ORIGIN}/replay/scan/${encodeURIComponent(name)}`;
    cameraPreview.onended = () => stopCamera();
    cameraStage.hidden = false;
    await waitLoaded(cameraPreview);
    liveSocket.send(JSON.stringify({ type: "camera_state", enabled: true }));
    cameraStage.hidden = false;
    replayButton.textContent = "Stop replay";
    captureStartedAt = Date.now();
    startReplayStepping();
    setStatus("Replaying recorded video", "");
    renderMode();
  } catch (error) {
    appendSystem(`Replay failed: ${error.message}`);
    stopCamera();
  } finally { cameraPending = false; }
}

function stopCamera() {
  const wasOn = Boolean(cameraStream) || replayActive;
  reportCaptureStats();
  stopBestOfWindow();
  releaseScreen();
  if (frameTimer) window.clearInterval(frameTimer);
  if (evidenceTimer) window.clearInterval(evidenceTimer);
  if (replayTimer) window.clearInterval(replayTimer);
  replayTimer = null;
  frameTimer = null;
  evidenceTimer = null;
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  if (replayActive) { try { cameraPreview.pause(); } catch (_) {} cameraPreview.removeAttribute("src"); }
  replayActive = false;
  captureStartedAt = null;
  cameraPreview.srcObject = null;
  cameraStage.hidden = true;
  cameraButton.classList.remove("active");
  cameraButton.querySelector(".round-label").textContent = "Show camera";
  replayButton.textContent = "Replay video";
  if (wasOn && liveSocket?.readyState === WebSocket.OPEN) {
    liveSocket.send(JSON.stringify({ type: "camera_state", enabled: false }));
    if (!isRecording) setStatus("Camera off", "");
  }
  renderMode();
}

function frameActive() {
  return !sweepEnding && (cameraStream || replayActive) && liveSocket && liveSocket.readyState === WebSocket.OPEN && cameraPreview.videoWidth;
}

// Compressed frame for the voice model's context only. Never used as OCR evidence.
function sendVoiceFrame() {
  if (!frameActive() || voiceAvailable === false) return;   // with no voice model the server discards the frame: do not pay for the canvas copy and JPEG encode
  const scale = VOICE_FRAME_WIDTH / cameraPreview.videoWidth;
  frameCanvas.width = VOICE_FRAME_WIDTH;
  frameCanvas.height = Math.round(cameraPreview.videoHeight * scale);
  frameCanvas.getContext("2d").drawImage(cameraPreview, 0, 0, frameCanvas.width, frameCanvas.height);
  liveSocket.send(JSON.stringify({ type: "video", data: frameCanvas.toDataURL("image/jpeg", 0.6).split(",")[1] }));
}

// High-resolution evidence frame: saved unchanged by the server and fed to the incremental pipeline.
function sendEvidenceFrame() {
  if (!frameActive()) return;
  if (bow.active && !replayActive) { watchdogWindow(); return; }
  const scale = evidenceScale(cameraPreview.videoWidth, cameraPreview.videoHeight);
  evidenceCanvas.width = Math.round(cameraPreview.videoWidth * scale);
  evidenceCanvas.height = Math.round(cameraPreview.videoHeight * scale);
  evidenceCanvas.getContext("2d").drawImage(cameraPreview, 0, 0, evidenceCanvas.width, evidenceCanvas.height);
  const message = { type: "evidence_frame", source: replayActive ? "replay" : "camera", data: evidenceCanvas.toDataURL("image/jpeg", evidenceQuality).split(",")[1] };
  if (replayActive) message.t_s = cameraPreview.currentTime;
  liveSocket.send(JSON.stringify(message));
}

// during a camera sweep the first End asks the server; it answers with the room-pan instruction (a "room_sweep" nudge, the sweep goes
// on) or end_ack "end". A second press (room phase) ends right away.
function endSweep() {
  if (!liveSocket || liveSocket.readyState !== WebSocket.OPEN) { appendSystem("Connect first (Talk or Show camera) before ending the sweep."); return; }
  if (cameraStream && !replayActive && !roomPhase) { liveSocket.send(JSON.stringify({ type: "end_request" })); return; }
  finishSweep();
}

function setRoomPhase(on) {
  roomPhase = on;
  const label = on ? "Finish" : "End sweep";
  endButton.querySelector(".round-label").textContent = label;
  const ovEnd = document.querySelector("#ovEnd");
  if (ovEnd) ovEnd.textContent = label;
}

function finishSweep() {
  if (!liveSocket || liveSocket.readyState !== WebSocket.OPEN) return;
  sweepEnding = true;                                  // the frame timers stop sending before the camera is even released
  setRoomPhase(false);
  endFailure = null;
  stopCamera();
  liveSocket.send(JSON.stringify({ type: "end_sweep" }));
  setStatus("Finishing: building the provisional packet", "neutral");
}

function resampleToPcm16(input, inputRate, outputRate) {
  const ratio = inputRate / outputRate;
  const outputLength = Math.floor(input.length / ratio);
  const pcm = new Int16Array(outputLength);
  for (let i = 0; i < outputLength; i += 1) {
    const index = i * ratio;
    const before = Math.floor(index);
    const after = Math.min(before + 1, input.length - 1);
    const weight = index - before;
    const sample = Math.max(-1, Math.min(1, input[before] * (1 - weight) + input[after] * weight));
    pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return pcm;
}

function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.byteLength; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function playPcm24(base64) {
  audioContext = audioContext || new AudioContext();
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  const pcm = new Int16Array(bytes.buffer);
  const audioBuffer = audioContext.createBuffer(1, pcm.length, 24000);
  const channel = audioBuffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i += 1) channel[i] = pcm[i] / 32768;
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);
  const startAt = Math.max(audioContext.currentTime, nextPlaybackTime);
  playbackSources.add(source);
  source.onended = () => playbackSources.delete(source);
  source.start(startAt);
  nextPlaybackTime = startAt + audioBuffer.duration;
}

micButton.addEventListener("click", () => (isRecording ? stopLiveVoice() : startLiveVoice()));
cameraButton.addEventListener("click", () => (cameraStream ? stopCamera() : startCamera()));
replayButton.addEventListener("click", () => (replayActive ? stopCamera() : startReplay(replaySelect.value)));
endButton.addEventListener("click", endSweep);
newIntakeButton.addEventListener("click", requestNewSweep);
document.querySelector("#ovTalk").addEventListener("click", () => (isRecording ? stopLiveVoice() : startLiveVoice()));
document.querySelector("#ovEnd").addEventListener("click", endSweep);
document.querySelector("#ovStop").addEventListener("click", stopCamera);
document.querySelector("#detailsToggle").addEventListener("click", toggleDetails);
applyDetails(detailsStored());
setInterval(() => { if (isSweeping()) tickSweepClock(); }, 1000);
window.addEventListener("pagehide", disconnectLive);
textForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const value = textInput.value.trim();
  if (!value) return;
  textInput.value = "";
  sendClaimantTurn(value);
});


// ---- site shell: Sweep / Measurements / Frames / Debugger on ONE page. Views are hidden, never unloaded, so the active
// session, socket, camera/replay and timers keep running while you inspect frames or the debugger.
const VIEWS = ["sweep", "measure", "frames", "debug", "label"];
let currentView = "sweep";

function parseHash() {
  const raw = (location.hash || "#sweep").slice(1);
  const [name, query = ""] = raw.split("?");
  return { view: VIEWS.includes(name) ? name : "sweep", params: Object.fromEntries(new URLSearchParams(query)) };
}

function iframeReady(frameEl) {
  return new Promise((resolve) => {
    const loaded = frameEl.getAttribute("src");
    if (!loaded) frameEl.src = frameEl.dataset.src;
    const done = () => resolve(frameEl.contentWindow);
    if (loaded && frameEl.contentDocument && frameEl.contentDocument.readyState === "complete") setTimeout(done, 50);
    else frameEl.addEventListener("load", () => setTimeout(done, 300), { once: true });
  });
}

async function showView(name, params = {}) {
  if (!VIEWS.includes(name)) name = "sweep";
  currentView = name;
  for (const v of VIEWS) document.querySelector(`#view-${v}`).hidden = v !== name;
  document.querySelectorAll("#siteNav a").forEach((a) => a.classList.toggle("active", a.dataset.view === name));
  const archive = document.querySelector("#navArchive");   // Measurements / Frames / Label titles live under "Archive"
  archive.classList.toggle("has-active", Boolean(archive.querySelector("a.active")));
  archive.open = false;
  renderRecStrip();
  renderSweepChrome();
  if (name === "measure") { loadTruth(); loadAutoMeasure(); }
  if (name === "label" && window.labelOnShow) window.labelOnShow();
  if (name === "frames") {
    const win = await iframeReady(document.querySelector("#framesFrame"));
    const source = params.source || (sessionId ? `session:${sessionId}` : null);   // default to the ACTIVE session, never a saved run by accident
    if (source && win.openFrameSource) await win.openFrameSource(source, params.frame);
  }
  if (name === "debug") {
    const win = await iframeReady(document.querySelector("#debugFrame"));
    if (params.run && params.scan) await debugSelect(win, params.scan, params.run, params.obs, params.section);
  }
}

// Thin adapter over the debugger's globals (no edits to its files; it has no native deep-link API).
// Its top-level `let` state is not reachable from here, so run lookup uses the same read-only /debug/api/scans endpoint.
async function debugSelect(win, scanLabel, runId, obs, section) {
  const note = document.querySelector("#debugNote");
  try {
    await win.catalogue();   // refresh the viewer's own catalogue (a finished sweep appears without restarting the server)
    const catalogue = await (await fetch(`${API_ORIGIN}/debug/api/scans`)).json();
    const run = (catalogue.find((s) => s.label === scanLabel) || { runs: [] }).runs.find((r) => r.run_id === runId);
    if (!run) { note.textContent = `Run ${runId} is not listed for ${scanLabel} yet (needs a merged manifest bound to the scan hash). Refresh after the sweep ends.`; return; }
    win.document.getElementById("scan").value = scanLabel; win.chooseScan();
    win.document.getElementById("profile").value = run.profile; win.chooseProfile();
    win.document.getElementById("run").value = runId; await win.loadView();
    let selected = false;
    if (obs) { try { win.select(obs); selected = true; } catch (_) { /* observation not in this run view */ } }
    if (section) {   // open the newest 2D panorama / 3D map of this run (the Debugger shows the newest build by default)
      const el = win.document.getElementById(section === "3d" ? "recon" : "panSection");
      if (el) { if (section === "3d") el.open = true; el.scrollIntoView({ block: "start" }); }
    }
    note.textContent = `Showing saved run ${runId} (scan ${scanLabel})${selected ? `, observation ${obs}` : ""}${section ? ` · ${section === "3d" ? "3D map" : "2D panorama"}` : ""}.`;
  } catch (error) { note.textContent = `Debugger could not open that record: ${error.message}`; }
}

function renderRecStrip() {
  const st = document.querySelector("#recState");
  const id = (state && state.identity) || {};
  const live = Boolean(cameraStream || replayActive);
  st.textContent = live ? (replayActive ? "● REPLAY RECORDING" : "● CAMERA RECORDING") : (state && state.status === "finished" ? "Sweep finished" : state && state.status === "finalizing" ? "Finalizing…" : "Not recording");
  st.classList.toggle("live", live);
  const scan = id.scan ? `scan ${id.scan.label} (sha ${String(id.scan.sha256 || "").slice(0, 8)})` : "no registered scan (live camera or unbound)";
  document.querySelector("#recIdentity").textContent = sessionId
    ? `session ${sessionId} · run ${id.run_dir_name || "?"} · ${scan} · ${id.frames_saved ?? 0} frames saved${live && currentView !== "sweep" ? " · recording continues while you inspect this view" : ""}`
    : "no active session";
}

document.addEventListener("click", (event) => {
  const f = event.target.closest("a[data-open-frame]");
  const d = event.target.closest("a[data-open-debug]");
  const m = event.target.closest("a[data-open-map]");
  if (m && state && state.post_sweep) {
    event.preventDefault();
    const q = { scan: state.post_sweep.scan_label, run: state.post_sweep.run_id, section: m.dataset.openMap };
    history.replaceState(null, "", `#debug?${new URLSearchParams(q)}`); showView("debug", q);
    return;
  }
  if (f) { event.preventDefault(); const q = { source: `session:${sessionId}`, frame: f.dataset.openFrame }; history.replaceState(null, "", `#frames?${new URLSearchParams(q)}`); showView("frames", q); }
  if (d) {
    event.preventDefault();
    const id = state.identity || {};
    const q = { scan: id.scan && id.scan.label, run: id.run_id, obs: d.dataset.openDebug };
    history.replaceState(null, "", `#debug?${new URLSearchParams(q)}`); showView("debug", q);
  }
});
document.querySelectorAll("#siteNav a").forEach((a) => a.addEventListener("click", (event) => {
  event.preventDefault();
  history.replaceState(null, "", a.getAttribute("href"));
  showView(a.dataset.view);
}));
window.addEventListener("hashchange", () => { const { view, params } = parseHash(); showView(view, params); });
document.querySelector("#debugRefresh").addEventListener("click", async () => {
  const win = await iframeReady(document.querySelector("#debugFrame"));
  try { await win.catalogue(); document.querySelector("#debugNote").textContent = "Debugger catalogue refreshed."; } catch (e) { document.querySelector("#debugNote").textContent = `Refresh failed: ${e.message}`; }
});
setInterval(renderRecStrip, 2000);
setInterval(refreshBuild, 30000);

document.querySelector("#calImg").addEventListener("click", calClick);
document.querySelector("#calImg").addEventListener("load", renderCalDots);
document.querySelector("#calFrame").addEventListener("change", loadCalFrame);
document.querySelector("#calClear").addEventListener("click", () => { calPoints = []; renderCalDots(); });
document.querySelector("#calSubmit").addEventListener("click", submitCalibration);
document.querySelector("#autoRun").addEventListener("change", (e) => loadAutoMeasure(e.target.value));
document.querySelector("#autoRefresh").addEventListener("click", () => loadAutoMeasure(document.querySelector("#autoRun").value));
document.querySelector("#truthSaveSpine").addEventListener("click", saveTruthSpine);
document.querySelector("#truthSaveRoom").addEventListener("click", saveTruthRoom);
document.querySelector("#ceilSubmit").addEventListener("click", submitCeiling);
document.querySelector("#calMode").addEventListener("change", () => { calPoints = []; renderCalDots(); });
document.querySelector("#books").addEventListener("click", (e) => {
  const split = e.target.closest && e.target.closest(".split-member");
  if (split) { e.preventDefault(); splitMember(split.dataset.splitMember); return; }
  const link = e.target.closest && e.target.closest(".hl-toggle");
  if (!link) return;
  e.preventDefault();
  const id = link.dataset.bookView;
  if (!wholeFrameBooks.delete(id)) wholeFrameBooks.add(id);
  renderInventory();
});

{ const { view, params } = parseHash(); showView(view, params); }

// ---------------------------------------------------------------- hosted kill-switch (Disconnect Gemini / Reconnect): server-side, persisted
(() => {
  const ind = document.querySelector("#hostedIndicator"), btn = document.querySelector("#hostedToggle"), ctr = document.querySelector("#hostedCounters");
  if (!ind || !btn) return;
  let state = null;
  const render = (st) => {
    state = st;
    const off = !!st.disabled;
    ind.textContent = off ? "Gemini: DISCONNECTED (hosted calls blocked)" : "Gemini: connected";
    ind.className = `hosted-ind ${off ? "off" : "on"}`;
    btn.textContent = off ? "Reconnect Gemini" : "Disconnect Gemini";
    btn.className = `ghost-button hosted-btn ${off ? "off" : ""}`;
    btn.title = off ? `blocked since ${st.changed_at || "?"}; ${st.blocked} call(s) refused` : "closes any live voice session and blocks every hosted call until Reconnect";
    const t = st.tokens || {};
    ctr.textContent = `hosted calls ${st.calls ?? 0} (cache hits ${st.cache_hits ?? 0}) · tokens ${t.total_tokens ?? 0} · $${(st.cost_usd ?? 0).toFixed ? (st.cost_usd ?? 0).toFixed(4) : st.cost_usd}${off ? ` · refused ${st.blocked ?? 0}` : ""}${(st.live_sessions_open || []).length ? ` · live sessions ${st.live_sessions_open.length}` : ""}`;
  };
  const refresh = () => fetch("/api/hosted").then((r) => r.json()).then(render).catch(() => {});
  btn.addEventListener("click", async () => {
    const off = state && state.disabled;
    if (!off && !window.confirm("Disconnect Gemini now? Any live voice session closes immediately and every hosted call is blocked until you press Reconnect.")) return;
    btn.disabled = true;
    try { render(await fetch(off ? "/api/hosted/enable" : "/api/hosted/disable", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }).then((r) => r.json())); }
    finally { btn.disabled = false; }
  });
  refresh(); setInterval(refresh, 5000);
  window.__hostedState = () => state;
})();

createSession(true);
