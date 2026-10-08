// Laptop watch page: read-only. Polls /api/watch/* (loopback server only) and renders the phone's sweep end to end:
// live frame + counters + capture indicator + notices, then the end-sweep steps, the post-sweep queue and the final count. No control changes a sweep.
// the book count banner (live count, then the headline + book lines + range + reliability), the voice conversation, and a claim packet viewer.
"use strict";
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const enc = encodeURIComponent;

// same words/colours as the phone's indicator (app.js QUALITY_BAND): distance overrides sharpness
const QUALITY_BAND = { strict: ["q-good", "Good"], medium: ["q-good", "OK"], poor: ["q-warn", "Slow down"], bad: ["q-bad", "Re-capture"], too_far: ["q-bad", "Too far — move closer"] };
const RELIABILITY_WORDS = { high: "reliable", medium: "fairly reliable", low: "rough estimate", unknown: "reliability unknown" };
const PHASE = { waiting: "○ Sweep page open — waiting for the camera", capturing: "● Recording", processing: "◐ Sweep ended — processing", maps: "◐ Building maps", done: "✓ Done", stopped: "■ Stopped", failed: "✕ Failed", incomplete: "○ Incomplete", unknown: "○ —" };
// the engine's end-sweep steps (app/engine.py _end_sweep, in order); "cross-checking the count" only runs with LIBRARY_ALT_AFTER_PACKET=0
const FIN = [
  ["finishing detection and OCR", "Detection catch-up"],
  ["merging duplicates", "Merging duplicates"],
  ["cross-checking the count", "Cross-checking the count", true],
  ["reading titles", "Title reading"],
  ["pricing", "Pricing"],
  ["room and scale", "Room and scale"],
  ["building the packet", "Building the packet"],
  ["rendering the report", "Rendering the report"],
];
const ICON = { pending: "○", active: "◐", done: "✓", failed: "✕", skipped: "–" };

let follow = true, current = null, lastFrame = null, seenNotices = new Set(), stateTimer = null, lastDetect = null, lastRead = null, lastOk = 0;
let packetRev = null, packetView = null, rawOpen = false, rawRev = null, talkRev = null;

const pct = (f) => Math.max(0, Math.min(100, Math.round(100 * (Number(f) || 0))));
const clock = (s) => { s = Math.max(0, Math.floor(Number(s) || 0)); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}` : `${m}:${String(r).padStart(2, "0")}`; };
const eta = (sec) => { sec = Math.round(Number(sec) || 0); return sec >= 90 ? `~${Math.round(sec / 60)} min left` : `~${Math.max(sec, 1)} s left`; };
const when = (epoch) => epoch ? new Date(epoch * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "";

async function get(path) {
  const r = await fetch(path, { cache: "no-store", credentials: "omit" });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

function progressBar(fraction, label) {
  const bar = el("div", "bar"); bar.setAttribute("role", "progressbar"); bar.setAttribute("aria-label", label);
  bar.setAttribute("aria-valuemin", "0"); bar.setAttribute("aria-valuemax", "100"); bar.setAttribute("aria-valuenow", String(pct(fraction)));
  const fill = el("div", "fill"); fill.style.width = pct(fraction) + "%"; bar.append(fill); return bar;
}

function stepRow(list, state, name, detail, fraction) {
  const li = el("li", state);
  li.append(el("span", "ic", ICON[state] || "○"), el("span", "name", name));
  const d = el("div", "detail", detail || "");
  if (fraction != null) d.append(progressBar(fraction, name));
  li.append(d);
  list.append(li);
}

// ---------------------------------------------------------------- picker / auto-follow
async function pollList() {
  let data;
  try { data = await get("/api/watch/sweeps"); } catch { return; }
  const pick = $("sweepPick"), keep = pick.value;
  pick.replaceChildren(el("option", null, data.newest_active ? `Follow the newest active sweep (now ${data.newest_active})` : "Follow the newest active sweep"));
  pick.options[0].value = "";
  for (const s of data.sweeps) {
    const o = el("option", null, `${when(s.started_epoch)} · ${s.run_id} · ${s.phase}${s.frames != null ? ` · ${s.frames} frames` : ""}${s.book_count != null ? ` · ${s.book_count} books` : ""}`);
    o.value = s.run_id; pick.append(o);
  }
  pick.value = follow ? "" : (current || keep);
  $("empty").hidden = data.sweeps.length > 0;
  if (follow) {
    const target = data.newest_active || current || data.newest;
    if (target && target !== current) show(target);
  }
}

function show(runId) {
  current = runId; lastFrame = null; seenNotices = new Set(); lastDetect = null; lastRead = null;
  packetRev = undefined; packetView = null; rawRev = null; talkRev = undefined;     // undefined: the first state always (re)loads both
  $("frame").hidden = true; $("noFrame").hidden = false; $("frameTag").hidden = true;
  $("talk").replaceChildren(); $("talkCount").textContent = ""; $("noTalk").hidden = false; $("noTalk").textContent = "No conversation yet.";
  renderPacket(null); $("pkRawSections").replaceChildren();
  if (stateTimer) clearTimeout(stateTimer);
  pollState();
}

$("sweepPick").addEventListener("change", (e) => {
  follow = !e.target.value;
  if (follow) { current = null; pollList(); } else show(e.target.value);
});

// ---------------------------------------------------------------- state
async function pollState() {
  const runId = current;
  if (!runId) { stateTimer = setTimeout(pollState, 2000); return; }
  let st = null;
  try { st = await get(`/api/watch/sweeps/${enc(runId)}`); lastOk = Date.now(); } catch { /* shown in the connection line */ }
  if (runId !== current) return;
  if (st) render(st);
  $("conn").textContent = lastOk ? (Date.now() - lastOk < 5000 ? `updated ${new Date(lastOk).toLocaleTimeString()}` : "connection lost — retrying") : "connecting";
  const phase = st && st.phase;
  stateTimer = setTimeout(pollState, phase === "waiting" || phase === "capturing" || phase === "processing" ? 1000 : phase === "maps" ? 2000 : 5000);
}

function render(st) {
  const eng = st.engine || {};
  $("runId").textContent = st.run_id;
  const chip = $("phaseChip"); chip.className = `w-chip ${st.phase}`; chip.textContent = PHASE[st.phase] || st.phase;
  if (eng.stopped_reason) chip.title = eng.stopped_reason;
  renderHero(st, eng);
  renderLive(st, eng);
  renderFinalize(st, eng);
  renderPost(st);
  renderFinal(st, eng);
  if (st.transcript_rev !== talkRev) loadTalk(st.run_id, st.transcript_rev);
  if (st.packet_rev !== packetRev) loadPacket(st.run_id, st.packet_rev);
}

function renderLive(st, eng) {
  const now = Date.now() / 1000;
  let elapsed = "—";
  if (eng.started_epoch) {
    const end = eng.ended_epoch || (eng.status === "capturing" ? now : eng.updated_epoch || now);
    elapsed = clock(end - eng.started_epoch) + (eng.ended_epoch ? " (ended)" : "");
    if (eng.status === "finalizing" && eng.ended_epoch) elapsed += ` · processing ${clock(now - eng.ended_epoch)}`;
  } else if (eng.duration_s != null) elapsed = `${clock(eng.duration_s)} (capture)`;
  $("elapsed").textContent = elapsed;
  $("frames").textContent = eng.frames != null ? String(eng.frames) : "—";
  const lc = eng.live_count || {};
  $("liveCount").textContent = lc.book_count != null ? String(lc.book_count) : (st.phase === "capturing" ? "counting…" : "—");
  const [cls, word] = eng.indicator ? (QUALITY_BAND[eng.indicator] || ["q-unknown", "Measuring"]) : ["q-unknown", eng.status === "capturing" ? "Measuring" : "—"];
  $("quality").className = `w-quality ${cls}`; $("qualityWord").textContent = eng.status === "capturing" || !eng.indicator ? word : `${word} (last)`;
  // latest evidence frame: fetched only when it changes, swapped in after it has loaded (no flicker)
  if (eng.latest_frame && eng.latest_frame !== lastFrame) {
    const fid = eng.latest_frame, url = `/api/watch/sweeps/${enc(st.run_id)}/frames/${enc(fid)}.jpg`, img = new Image();
    lastFrame = fid;
    img.onload = () => { if (lastFrame !== fid || current !== st.run_id) return; $("frame").src = url; $("frame").hidden = false; $("noFrame").hidden = true;
      const tag = $("frameTag"); tag.hidden = false; tag.textContent = `${fid}${eng.latest_frame_t_s != null ? ` · ${clock(eng.latest_frame_t_s)}` : ""}${eng.frame_size ? ` · ${eng.frame_size[0]}×${eng.frame_size[1]}` : ""}`; };
    img.src = url;
  }
  // notices, newest first
  const notes = eng.notices || [], list = $("notices");
  list.replaceChildren();
  for (const n of [...notes].reverse()) {
    const key = `${n.code}|${n.at_s}`;
    const li = el("li", [n.kind === "guidance" ? "guidance" : "", n.code === "too_far" || n.code === "closer" ? "distance" : "", seenNotices.size && !seenNotices.has(key) ? "fresh" : ""].filter(Boolean).join(" "));
    li.append(el("time", null, n.at_s != null ? clock(n.at_s) : ""), el("span", null, n.message || n.code || ""));
    list.append(li);
  }
  seenNotices = new Set(notes.map((n) => `${n.code}|${n.at_s}`).concat(["_"]));
  $("noNotices").hidden = notes.length > 0;
  $("noticeCount").textContent = notes.length ? `(${notes.length})` : "";
}

function renderFinalize(st, eng) {
  const list = $("finSteps"); list.replaceChildren();
  const status = eng.status, fin = eng.finalize || {}, reached = (eng.finalize_steps || []).map((s) => s.stage), cur = fin.stage;
  if (fin.detect && fin.detect.total) lastDetect = fin.detect;
  if (fin.read && fin.read.total) lastRead = fin.read;
  else if (fin.read_done && fin.read_done.total) lastRead = fin.read_done;
  if (status === "capturing") { stepRow(list, "pending", "After End", "starts when the sweep is ended on the phone"); $("firstCount").hidden = true; return; }
  if (!reached.length && !cur) {
    stepRow(list, status === "finished" ? "done" : "pending", "End-sweep steps", status === "finished" ? "finished (step timings not recorded for this sweep)" : status === "stopped" ? eng.stopped_reason || "stopped" : "—");
  } else {
    const curIdx = cur ? FIN.findIndex(([k]) => k === cur) : -1;
    const lastIdx = Math.max(curIdx, ...reached.map((r) => FIN.findIndex(([k]) => k === r)));
    FIN.forEach(([key, name, optional], i) => {
      let state = "pending";
      if (key === cur && status === "finalizing") state = "active";
      else if (reached.includes(key)) state = (status === "failed" || status === "stopped") && i === lastIdx ? "failed" : "done";
      else if (i < lastIdx || status === "finished") state = "skipped";
      if (optional && (state === "skipped" || state === "pending")) return;
      let detail = state === "skipped" ? "not needed for this sweep" : state === "pending" ? "" : state === "failed" ? (fin.error || eng.stopped_reason || "failed") : "";
      let fraction = null;
      if (key === "finishing detection and OCR") {
        const d = state === "active" ? fin.detect : lastDetect;
        if (d && d.total) { detail = `${d.done} of ${d.total} detection frames`; if (state === "active") fraction = d.done / d.total; }
      }
      if (key === "reading titles" && (state === "active" || state === "done")) {
        const r = state === "active" ? (fin.read || fin.read_done) : lastRead;
        if (r && r.total) {   // attempted also counts second reads of the same spines after the first pass
          const first = Math.min(r.attempted, r.total), extra = r.attempted - first;
          detail = `${first} of ${r.total} spines read${extra > 0 ? ` (+${extra} second reads)` : ""}${r.budget_exhausted ? " (time budget reached)" : ""}`;
          if (state === "active") fraction = first / r.total;
        }
      }
      const at = (eng.finalize_steps || []).find((s) => s.stage === key);
      if (at && state !== "pending") detail = [detail, `started at +${clock(at.at_s)}`].filter(Boolean).join(" · ");
      stepRow(list, state, name, detail, fraction);
    });
  }
  const fc = eng.first_count, box = $("firstCount");
  box.hidden = !fc;
  if (fc) countBox(box, "First count (packet written)", fc, eng.finalization_s != null ? `packet ready ${Math.round(eng.finalization_s)} s after End` : "");
}

function countBox(box, label, t, note) {
  box.replaceChildren();
  const range = t.book_count_range || (t.book_count_lower != null && t.book_count_upper != null ? [t.book_count_lower, t.book_count_upper] : null);
  box.append(el("span", "lbl", label), el("span", "big", t.book_count == null ? "unknown" : `${range ? "~" : ""}${t.book_count} books`));
  if (range) box.append(el("span", null, `likely ${range[0]}–${range[1]}`));
  if (t.count_reliability) box.append(el("span", null, RELIABILITY_WORDS[t.count_reliability] || t.count_reliability));
  if (t.item_count != null) box.append(el("span", "small", `${t.item_count} other items`));
  if (note) box.append(el("span", "small", note));
}

function subState(s) {
  if (!s || !s.state || s.state === "none") return "pending";
  if (["queued", "waiting", "pending"].includes(s.state)) return "pending";
  if (s.state === "running") return "active";
  if (s.state === "done") return "done";
  if (s.state === "failed") return "failed";
  return "skipped";
}

function buildDetail(s) {
  if (!s || !s.state || s.state === "none") return ["", null];
  if (s.state === "queued") return [`queued${s.position ? " #" + s.position : ""}${s.ahead && s.ahead.length ? ` (after ${s.ahead.map((a) => `${a.kind} of ${a.run_id}`).join(", ")})` : ""}`, null];
  if (s.state === "waiting") return [s.note || "waiting for the sweep files to settle", null];
  if (s.state === "running") { const p = s.progress || {}; return [`${p.label || s.stage || "starting"} · ${pct(p.fraction)}%${p.eta_s != null ? " · " + eta(p.eta_s) : ""}`, p.fraction || 0]; }
  if (s.state === "done") return [s.book_count != null && s.markers != null ? `ready (${s.markers} markers)` : "ready", null];
  if (s.state === "failed") return [`failed: ${s.error || "unknown reason"}`, null];
  return [s.state, null];
}

function hookDetail(h, running) {
  if (!h || !h.state) return ["", null];
  if (h.state === "running") return [running, null];
  if (h.state === "done") return [String(h.summary || "done"), null];
  return [`${h.state}${h.error || h.reason ? ": " + (h.error || h.reason) : ""}`, null];
}

function renderPost(st) {
  const list = $("postSteps"); list.replaceChildren();
  const ps = st.post_sweep, eng = st.engine || {};
  if (!ps) {
    const why = eng.status === "finished" ? "not queued for this sweep (post-sweep builds off, or the sweep predates the queue)" : "starts when the packet is written";
    for (const n of ["2D panorama", "Panorama count", "Duplicate merging", "3D map", "3D join", "Measurement"]) stepRow(list, "pending", n, n === "2D panorama" ? why : "");
    $("postNote").textContent = "";
    return;
  }
  const steps = ps.steps || {};
  let [d, f] = buildDetail(steps.panorama); stepRow(list, subState(steps.panorama), "2D panorama", d, f);
  [d, f] = hookDetail(ps.count, "counting spines on the panorama"); stepRow(list, subState(ps.count), "Panorama count", d || (steps.panorama && steps.panorama.state === "done" ? "" : "after the panorama"), f);
  [d, f] = hookDetail(ps.instances, "merging track groups into physical books"); stepRow(list, subState(ps.instances), "Duplicate merging", d || "after the count", f);
  [d, f] = buildDetail(steps.recon3d); stepRow(list, subState(steps.recon3d), "3D map", d, f);
  const m = ps.merge3d;
  if (!m || !m.state) stepRow(list, "pending", "3D join", steps.recon3d && steps.recon3d.state === "done" ? "not needed (single map)" : "after the first 3D map");
  else if (m.state === "running") { const p = m.progress || {}; stepRow(list, "active", "3D join", `${p.label || "joining fragments"} · ${pct(p.fraction)}%${p.eta_s != null ? " · " + eta(p.eta_s) : ""}`, p.fraction || 0); }
  else if (m.state === "queued") stepRow(list, "pending", "3D join", "queued (first map already usable)");
  else if (m.state === "done") stepRow(list, "done", "3D join", `joined map ready${typeof m.registered_ratio === "number" ? ` (${Math.round(100 * m.registered_ratio)}% of frames registered)` : ""}`);
  else stepRow(list, m.state === "failed" ? "failed" : "skipped", "3D join", `${m.state}${m.error || m.reason ? ": " + (m.error || m.reason) : ""}`);
  [d, f] = hookDetail(ps.measure, "scaling the 3D map (spines, shelf run, room)"); stepRow(list, subState(ps.measure), "Measurement", d || "after the 3D map", f);
  const fr = ps.finish_reading;
  if (fr && fr.state) {
    const frac = fr.total ? Math.min(1, (fr.done || 0) / fr.total) : null;
    if (fr.state === "running") stepRow(list, "active", "Finish reading", `${fr.phase || "starting"}${fr.total ? ` ${fr.done || 0} of ${fr.total}` : ""}${fr.note ? " · " + fr.note : ""}`, frac);
    else if (fr.state === "queued") stepRow(list, "pending", "Finish reading", "queued");
    else if (fr.state === "done") stepRow(list, "done", "Finish reading", String(fr.summary || fr.reason || "done"));
    else stepRow(list, fr.state === "failed" ? "failed" : "skipped", "Finish reading", `${fr.state}${fr.error || fr.reason ? ": " + (fr.error || fr.reason) : ""}`);
  }
  const o = ps.overall;
  $("postNote").textContent = `${ps.note || ps.state}${o && st.phase === "maps" ? ` · ${pct(o.fraction)}% overall${o.eta_s ? " · " + eta(o.eta_s) : ""}` : ""}`;
}

function renderFinal(st) {
  const box = $("finalCount");
  if (!st.packet || st.engine.status !== "finished") { box.hidden = true; return; }
  box.hidden = false;
  box.classList.toggle("final", Boolean(st.final));
  countBox(box, st.final ? "Final count" : "Count so far (post-sweep steps may still change it)", st.packet, st.packet.packet_status ? `packet ${st.packet.packet_status}` : "");
  if (st.report) {
    const a = el("a", null, "Open the report"); a.href = `/api/watch/sweeps/${enc(st.run_id)}/report`; a.target = "_blank"; a.rel = "noopener";
    box.append(a);
  }
}

// ---------------------------------------------------------------- the book count, always on top
const ESTIMATOR = { panorama: "2D panorama count", lines: "book lines", instances: "merged physical books", track: "tracked groups" };
const POST_RUNNING = new Set(["queued", "waiting", "building_2d", "building_3d", "counting", "merging", "measuring", "joining_3d", "reading"]);
const fmtMoney = (amount, cur) => amount == null ? "—" : `${cur === "USD" || !cur ? "$" : ""}${Number(amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}${cur && cur !== "USD" ? " " + cur : ""}`;
const rangeOf = (t) => t.book_count_range || (t.book_count_lower != null && t.book_count_upper != null ? [t.book_count_lower, t.book_count_upper] : null);
const reliability = (r) => r ? RELIABILITY_WORDS[r] || r : null;

function facts(dl, rows) {
  dl.replaceChildren();
  for (const [k, v, note] of rows) {
    if (v == null || v === "") continue;
    const d = el("div"); d.append(el("dt", null, k));
    const dd = el("dd", null, String(v)); if (note) dd.append(el("small", null, note));
    d.append(dd); dl.append(d);
  }
}

function runningPost(ps) {
  if (!ps) return null;
  const fr = ps.finish_reading;
  if (fr && fr.state === "running") return `finish reading${fr.total ? ` ${fr.done || 0} of ${fr.total}` : ""}`;
  if (ps.state && POST_RUNNING.has(ps.state)) return ps.note || ps.state.replace(/_/g, " ");
  return null;
}

function renderHero(st, eng) {
  const hero = $("hero"), lc = eng.live_count || {}, p = st.packet;
  hero.classList.toggle("live", st.phase === "capturing" || st.phase === "waiting");
  hero.classList.toggle("final", Boolean(st.final));
  let label, num, sub = "", rows = [];
  if (p && eng.status === "finished") {
    const range = rangeOf(p), est = p.headline_estimator;
    label = st.final ? "Book count (final)" : "Book count so far";
    num = p.book_count == null ? "unknown" : `${p.book_count}`;
    sub = est ? `headline from the ${ESTIMATOR[est] || est}` : "headline count";
    const still = runningPost(st.post_sweep);
    if (!st.final) sub += still ? ` · still running: ${still} — the count may change` : " · post-sweep steps may still change it";
    rows = [
      ["Book lines", p.book_lines != null ? `${p.book_lines} book lines listed` : null, "some may be duplicates"],
      ["Range", range ? `${range[0]}–${range[1]}` : null, p.book_count_range_label || "likely range"],
      ["Reliability", reliability(p.count_reliability)],
      ["Identified", p.books_identified != null ? `${p.books_identified} titles` : null, p.book_lines ? `of ${p.book_lines} lines` : null],
      ["Other items", p.item_count],
    ];
  } else if (st.phase === "processing") {
    label = "Live book count (last)"; num = lc.book_count != null ? `${lc.book_count}` : "—"; sub = "sweep ended — the headline count comes with the packet";
    rows = [["Frames", eng.frames], ["Reliability", reliability(lc.count_reliability)]];
  } else if (st.phase === "capturing" || st.phase === "waiting") {
    label = "Live book count"; num = lc.book_count != null ? `${lc.book_count}` : (st.phase === "capturing" ? "counting…" : "—");
    sub = st.phase === "waiting" ? "waiting for the camera" : "counted during the sweep; the headline count comes after End";
    const r = lc.book_count_range;
    rows = [["Frames", eng.frames], ["Range so far", r ? `${r[0]}–${r[1]}` : null], ["Reliability", reliability(lc.count_reliability)], ["Other items", lc.item_count]];
  } else {
    label = "Book count"; num = lc.book_count != null ? `${lc.book_count}` : "—";
    sub = eng.stopped_reason || (st.phase === "incomplete" ? "no packet was written for this sweep" : "");
  }
  $("heroLabel").textContent = label; $("heroCount").textContent = num; $("heroSub").textContent = sub;
  facts($("heroFacts"), rows);
}

// ---------------------------------------------------------------- the voice conversation
const WHO = { Claimant: ["person", "Person"], Agent: ["agent", "Voice agent"] };

async function loadTalk(runId, rev) {
  talkRev = rev;
  if (rev == null) { $("noTalk").textContent = "No conversation saved for this sweep (older sweeps did not save it)."; return; }
  let data;
  try { data = await get(`/api/watch/sweeps/${enc(runId)}/transcript`); } catch { talkRev = undefined; return; }
  if (runId !== current) return;
  const list = $("talk"), atEnd = list.scrollHeight - list.scrollTop - list.clientHeight < 40;
  list.replaceChildren();
  for (const t of data.turns || []) {
    const [cls, who] = WHO[t.speaker] || ["", t.speaker || "?"];
    const li = el("li", cls), tm = el("time", null, t.at_s != null ? `+${clock(t.at_s)}` : "");
    if (t.at_epoch) tm.title = new Date(t.at_epoch * 1000).toLocaleString();
    li.append(tm, el("span", "who", who), el("span", "say", t.text));
    list.append(li);
  }
  const n = (data.turns || []).length;
  $("noTalk").hidden = n > 0; $("talkCount").textContent = n ? `(${n} turns)` : "";
  if (atEnd) list.scrollTop = list.scrollHeight;
}

// ---------------------------------------------------------------- claim packet viewer
const bk = { q: "", filter: "", sort: "i", dir: 1 };
const BOOK_COLS = [["thumb", "", null], ["i", "#", "num"], ["title", "Title", "str"], ["author", "Author", "str"], ["status", "Status", "str"],
  ["confidence", "Confidence", "num"], ["rc", "Replacement", "num"], ["uv", "Used", "num"], ["tier", "Count tier", "str"], ["t_s", "Seen at", "num"]];
const bookKey = (b, k) => k === "rc" ? (b.replacement_cost && b.replacement_cost.amount) : k === "uv" ? (b.used_value && b.used_value.amount) : b[k];

async function loadPacket(runId, rev) {
  packetRev = rev;
  if (rev == null) { packetView = null; renderPacket(null); return; }
  let view;
  try { view = await get(`/api/watch/sweeps/${enc(runId)}/packet`); } catch { packetRev = undefined; return; }
  if (runId !== current) return;
  packetView = view; renderPacket(view);
  if (rawOpen) loadRaw(runId, rev);
}

function money(m) {
  const td = el("td", "num");
  if (!m) { td.textContent = "—"; td.classList.add("muted"); return td; }
  td.textContent = fmtMoney(m.amount, m.currency);
  if (m.converted) td.append(el("div", "conv", `converted from ${m.original_currency} ${m.original_amount}`));
  td.title = [m.source, m.condition_assumed].filter(Boolean).join(" · ");
  return td;
}

function renderPacket(v) {
  const has = Boolean(v);
  $("pkEmpty").hidden = has; $("pkBody").hidden = !has || rawOpen; $("pkRaw").hidden = !has || !rawOpen;
  const dl = $("pkDownload");
  dl.setAttribute("aria-disabled", String(!has)); $("pkRawToggle").disabled = !has;
  dl.href = has && current ? `/api/watch/sweeps/${enc(current)}/packet.json?download=1` : "#";
  const chip = $("pkStatus"); chip.className = `w-chip ${has ? v.status || "" : ""}`; chip.textContent = has ? (v.status || "") : "";
  $("pkUpdated").textContent = has && v.packet_mtime ? `updated ${new Date(v.packet_mtime * 1000).toLocaleTimeString()}` : "";
  if (!has) return;
  $("pkNotice").textContent = [v.completion_status && `Completion: ${v.completion_status}.`, v.provisional_notice].filter(Boolean).join(" ");
  const iss = $("pkIssues"); iss.replaceChildren();
  for (const i of v.issues || []) iss.append(el("li", null, `${i.severity ? i.severity + ": " : ""}${i.detail || i.code}`));
  const t = v.totals || {}, range = rangeOf(t), cur = (v.pricing || {}).currency || (v.sweep || {}).currency || "USD", lr = t.book_lines_range;
  facts($("pkTotals"), [
    ["Headline count", t.book_count, t.headline_estimator ? `from the ${ESTIMATOR[t.headline_estimator] || t.headline_estimator}` : null],
    ["Book lines", t.book_lines != null ? `${t.book_lines} book lines listed` : null, `some may be duplicates${lr ? ` · lines range ${lr[0]}–${lr[1]}` : ""}`],
    ["Range", range ? `${range[0]}–${range[1]}` : null, t.book_count_range_label || t.count_range_basis],
    ["Reliability", reliability(t.count_reliability),
      [t.count_reliability_capture && `capture ${t.count_reliability_capture}`, t.count_reliability_association && `association ${t.count_reliability_association}`].filter(Boolean).join(" · ")],
    ["Identified", t.books_identified, t.books_unidentified != null ? `${t.books_unidentified} unidentified` : null],
    ["Replacement cost", fmtMoney(t.books_replacement_cost, cur), "sum of sourced, priced lines"],
    ["Used value", fmtMoney(t.books_used_value, cur), "sum of sourced, priced lines"],
    ["Other items", t.item_count, t.items_needs_appraisal ? `${t.items_needs_appraisal} need appraisal` : null],
    ["Count groups", t.confident_groups != null ? `${t.confident_groups} confident` : null,
      [t.singleton_groups != null && `${t.singleton_groups} single sightings`, t.unconfirmed_groups != null && `${t.unconfirmed_groups} unconfirmed`].filter(Boolean).join(" · ")],
    ["Panorama count", t.panorama_count], ["Track count", t.book_count_track],
    ["Excluded from money totals", t.excluded_from_totals, "lines without a price"],
    ["Shelf run", t.shelf_run_m != null ? `${t.shelf_run_m} m` : "not measured"],
  ]);
  renderBooks(); renderItems(v);
  const pr = v.pricing || {}, comp = (v.completeness || {}).books || {}, rcC = comp.replacement_cost || {}, uvC = comp.used_value || {}, wle = pr.work_level_equivalent || {};
  facts($("pkPricing"), [
    ["Market", [pr.market, pr.currency].filter(Boolean).join(" · ")],
    ["Priced lines", rcC.known != null ? `${rcC.known} replacement · ${uvC.known || 0} used` : null, comp.total != null ? `of ${comp.total} book lines` : null],
    ["Converted lines", (rcC.converted || 0) + (uvC.converted || 0), "non-US offer converted to USD (ECB rate)"],
    ["Basis", wle.price_basis ? wle.price_basis.replace(/_/g, " ") : null, wle.online === false ? "offline" : wle.budget_s ? `budget ${wle.budget_s} s` : null],
    ["Second country", v.second_country ? [v.second_country.country, v.second_country.currency].filter(Boolean).join(" · ") : null,
      v.second_country ? `${v.second_country.status}${(v.second_country.totals || {}).books != null ? ` · ${v.second_country.totals.books} books` : ""}` : null],
    ["Device", (v.sweep || {}).device],
  ]);
  renderSecondCountry(v.second_country); renderReview(v);
  const r = v.room || {}, m = v.measurement || {};
  facts($("pkRoom"), [
    ["Room", r.status, (r.abstention_reasons || []).join(", ").replace(/_/g, " ")],
    ["Size", r.length_m != null ? `${r.length_m} × ${r.width_m} × ${r.height_m} m` : null],
    ["Floor area", r.floor_area_m2 != null ? `${r.floor_area_m2} m²` : null],
    ["Measurement", m.status, m.scale_method ? m.scale_method.replace(/_/g, " ") : null],
    ["Scale", m.scale && m.scale.cm_per_unit != null ? `${Number(m.scale.cm_per_unit).toFixed(3)} cm/unit` : (m.scale && m.scale.reason || null)],
    ["Shelf run", m.shelf && m.shelf.shelf_run_m != null ? `${m.shelf.shelf_run_m} m` : (m.shelf && m.shelf.reason || null)],
    ["Validation", m.validation],
  ]);
  const sb = $("pkStages").tBodies[0]; sb.replaceChildren();
  for (const [name, s] of Object.entries(v.stage_status || {})) {
    const x = s || {}, tr = el("tr");
    tr.append(el("td", null, name), el("td", null, x.status || ""), el("td", "num", x.elapsed_s != null ? `${Math.round(x.elapsed_s)} s` : ""), el("td", null, x.detail || ""));
    sb.append(tr);
  }
}

function renderBooks() {
  const v = packetView; if (!v) return;
  const table = $("bkTable"), head = table.tHead, body = table.tBodies[0];
  table.classList.toggle("nothumbs", !$("bkThumbs").checked);
  head.replaceChildren(); const hr = el("tr");
  for (const [k, name, type] of BOOK_COLS) {
    const th = el("th", k === "thumb" ? "thumb" : null, name);
    if (type) { th.dataset.sort = k; th.setAttribute("aria-sort", bk.sort === k ? (bk.dir > 0 ? "ascending" : "descending") : "none"); th.tabIndex = 0; }
    hr.append(th);
  }
  head.append(hr);
  const q = bk.q.trim().toLowerCase();
  let rows = (v.books || []).filter((b) => {
    if (bk.filter === "identified" && b.status !== "identified") return false;
    if (bk.filter === "unidentified" && b.status === "identified") return false;
    if (bk.filter === "priced" && !b.replacement_cost && !b.used_value) return false;
    return !q || [b.title, b.author, b.publisher, b.visible_text, b.id, b.isbn].some((x) => x && String(x).toLowerCase().includes(q));
  });
  const type = (BOOK_COLS.find((c) => c[0] === bk.sort) || [])[2];
  rows = rows.slice().sort((a, b) => {
    const x = bookKey(a, bk.sort), y = bookKey(b, bk.sort);
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return (type === "num" ? x - y : String(x).localeCompare(String(y))) * bk.dir;
  });
  const LIMIT = 1500, thumbs = $("bkThumbs").checked;
  body.replaceChildren();
  for (const b of rows.slice(0, LIMIT)) {
    const tr = el("tr"), th = el("td", "thumb");
    if (b.crop && thumbs) { const img = el("img"); img.loading = "lazy"; img.alt = ""; img.src = `/api/watch/sweeps/${enc(current)}/crops/${enc(b.crop)}`; th.append(img); }
    const title = el("td");
    if (b.title) title.append(el("strong", null, b.title));
    else title.append(el("span", "muted", b.visible_text ? `reads: “${b.visible_text}”` : "not read"));
    if (b.publisher && b.title) title.append(el("div", "muted", b.publisher));
    const st = el("td"); st.append(el("span", `tag ${b.status || ""}`, b.status || "—"));
    tr.append(th, el("td", "num", b.i != null ? String(b.i) : ""), title, el("td", null, b.author || ""), st,
      el("td", "num", b.confidence != null ? `${Math.round(100 * b.confidence)}%` : "—"), money(b.replacement_cost), money(b.used_value),
      el("td", null, b.tier || ""), el("td", "num", b.t_s != null ? clock(b.t_s) : ""));
    if (b.price_note && !b.replacement_cost && !b.used_value) tr.title = `no price: ${b.price_note.replace(/_/g, " ")}`;
    body.append(tr);
  }
  const total = (v.books || []).length;
  $("bkCount").textContent = `(${rows.length === total ? total : `${rows.length} of ${total}`} lines)`;
  $("bkMore").textContent = rows.length > LIMIT ? `Showing the first ${LIMIT}; search or filter to narrow.` : "";
}

// the same 10 books priced for the second country (native currency) next to their US lines; the review queue, one entry per line/topic.
function renderSecondCountry(sc) {
  const body = $("scTable").tBodies[0]; body.replaceChildren();
  const t = (sc && sc.totals) || {};
  $("scHead").textContent = !sc ? "" : sc.status === "ok" || sc.status === "partial"
    ? `(${sc.country} · ${sc.currency}, ${sc.status}: ${t.priced_replacement || 0} new / ${t.priced_used || 0} used priced of ${t.books || 0})` : `(${sc.status}${sc.country ? ` · ${sc.country}` : ""})`;
  for (const b of (sc && sc.books) || []) {
    const tr = el("tr"), name = el("td", null, b.title || "");
    name.append(el("div", "muted", `ISBN ${b.isbn || "—"}${b.stale ? ` · stale: ${b.stale}` : ""}`));
    tr.append(el("td", "num", b.i != null ? String(b.i) : ""), name, money(b.us_new), money(b.us_used), money(b.new), money(b.used));
    if (b.issues && b.issues.length) tr.title = b.issues.join("; ").replace(/_/g, " ");
    body.append(tr);
  }
}

function renderReview(v) {
  const body = $("rqTable").tBodies[0]; body.replaceChildren();
  const rows = v.review || [], l = v.lists || {};
  $("rqCount").textContent = `(${l.review_queue != null ? l.review_queue : rows.length} entries, ${l.review_details != null ? l.review_details : rows.length} details${rows.length < (l.review_queue || 0) ? `; first ${rows.length} shown` : ""})`;
  for (const r of rows) {
    const tr = el("tr", `sev-${r.severity || ""}`);
    tr.append(el("td", null, r.ref_id || ""), el("td", null, r.severity || ""), el("td", null, r.reason || ""), el("td", "num", String(r.details || 1)));
    tr.title = (r.codes || []).join(", ");
    body.append(tr);
  }
}

function renderItems(v) {
  const table = $("itTable"), head = table.tHead, body = table.tBodies[0];
  head.replaceChildren(); const hr = el("tr");
  for (const n of ["#", "Category", "Description", "Status", "Confidence", "Seen at"]) hr.append(el("th", null, n));
  head.append(hr); body.replaceChildren();
  for (const it of v.items || []) {
    const tr = el("tr");
    tr.append(el("td", "num", it.i != null ? String(it.i) : ""), el("td", null, (it.category || "").replace(/_/g, " ")), el("td", null, it.description || ""),
      el("td", null, (it.status || "").replace(/_/g, " ")), el("td", "num", it.confidence != null ? `${Math.round(100 * it.confidence)}%` : "—"),
      el("td", "num", it.t_s != null ? clock(it.t_s) : ""));
    if (it.status_reason) tr.title = it.status_reason;
    body.append(tr);
  }
  $("itCount").textContent = `(${(v.items || []).length})`;
}

async function loadRaw(runId, rev) {
  if (rawRev === rev) return;
  rawRev = rev;
  const box = $("pkRawSections"); box.replaceChildren(el("p", "small", "Loading the packet…"));
  let env;
  try { env = await get(`/api/watch/sweeps/${enc(runId)}/packet.json`); } catch { rawRev = null; box.replaceChildren(el("p", "small", "Could not load the packet.")); return; }
  if (runId !== current) return;
  box.replaceChildren();
  const section = (name, value) => {
    const d = el("details"), sm = el("summary", null, name);
    const size = Array.isArray(value) ? `${value.length} entries` : value && typeof value === "object" ? `${Object.keys(value).length} keys` : JSON.stringify(value);
    sm.append(el("span", "small", size)); d.append(sm);
    d.addEventListener("toggle", () => { if (d.open && !d.querySelector("pre")) d.append(el("pre", null, JSON.stringify(value, null, 2))); });
    box.append(d);
  };
  for (const [k, val] of Object.entries(env)) {
    if (k === "data" && val && typeof val === "object") for (const [dk, dv] of Object.entries(val)) section(`data.${dk}`, dv);
    else section(k, val);
  }
}

$("pkRawToggle").addEventListener("click", () => {
  rawOpen = !rawOpen; $("pkRawToggle").setAttribute("aria-pressed", String(rawOpen));
  $("pkBody").hidden = !packetView || rawOpen; $("pkRaw").hidden = !packetView || !rawOpen;
  if (rawOpen && current) loadRaw(current, packetRev);
});
$("bkSearch").addEventListener("input", (e) => { bk.q = e.target.value; renderBooks(); });
$("bkFilter").addEventListener("change", (e) => { bk.filter = e.target.value; renderBooks(); });
$("bkThumbs").addEventListener("change", () => renderBooks());
const sortBy = (k) => { if (bk.sort === k) bk.dir = -bk.dir; else { bk.sort = k; bk.dir = 1; } renderBooks(); };
$("bkTable").tHead.addEventListener("click", (e) => { const th = e.target.closest("th[data-sort]"); if (th) sortBy(th.dataset.sort); });
$("bkTable").tHead.addEventListener("keydown", (e) => { const th = e.target.closest("th[data-sort]"); if (th && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); sortBy(th.dataset.sort); } });

// ?run=sweep_<id> opens one sweep (no auto-follow); the picker's first entry returns to following
const asked = new URLSearchParams(location.search).get("run");
if (asked && /^sweep_[0-9a-f]{16}$/.test(asked)) { follow = false; show(asked); }
pollList();
setInterval(pollList, 3000);
setInterval(() => { if (current) { const c = $("conn"); if (lastOk && Date.now() - lastOk > 5000) c.textContent = "connection lost — retrying"; } }, 1000);
