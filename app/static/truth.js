// Ground-truth labelling page. Loopback server only. Reads the sweep's packet / frames / crops through the read-only watch routes,
// saves answers (debounced) to ground_truth/ground_truth_labels.json via POST /api/truth/<run>/labels, never anything under runs/.
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const params = new URLSearchParams(location.search);
  let RUN = params.get("run") || "";
  let C = null;                  // cards from /api/truth/<run>
  let L = null;                  // this sweep's labels
  const pending = {};            // patch not yet saved
  let saveTimer = null, saving = false;
  const imgs = new Map();
  const S = { tab: "titles", t: 0, i: 0, rotT: 0, tUnans: false, tIdent: false, iUnans: false };

  // ------------------------------------------------------------------ fetch / save
  async function api(path, opt = {}) {
    const r = await fetch(path, { credentials: "omit", cache: "no-store", ...opt });
    if (!r.ok) {
      let msg = `${r.status}`;
      try { msg = (await r.json()).detail || msg; } catch (_) { /* not json */ }
      const e = new Error(msg); e.status = r.status; throw e;
    }
    return r.json();
  }
  function showErr(msg) { const e = $("err"); e.textContent = msg || ""; e.hidden = !msg; }
  function queue(section, key, value) {
    if (key === null) pending[section] = value;
    else (pending[section] ||= {})[key] = value;
    $("saveState").textContent = "Unsaved…";
    clearTimeout(saveTimer); saveTimer = setTimeout(flush, 600);
  }
  async function flush(keepalive = false) {
    clearTimeout(saveTimer);
    if (saving && !keepalive) { saveTimer = setTimeout(flush, 300); return; }
    const keys = Object.keys(pending);
    if (!keys.length) return;
    const body = {};
    for (const k of keys) { body[k] = pending[k]; delete pending[k]; }
    saving = true; $("saveState").textContent = "Saving…";
    try {
      const r = await api(`/api/truth/${RUN}/labels`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), keepalive });
      $("saveState").textContent = `Saved ${new Date().toLocaleTimeString()}`; showErr("");
      void r;
    } catch (e) {
      for (const k of keys) {               // keep the answers for the next try (newer edits win)
        if (body[k] && typeof body[k] === "object" && !Array.isArray(body[k])) pending[k] = { ...body[k], ...(pending[k] || {}) };
        else if (!(k in pending)) pending[k] = body[k];
      }
      $("saveState").textContent = "Not saved";
      showErr(`Could not save: ${e.message}`);
      if (e.status !== 400) saveTimer = setTimeout(flush, 3000);
    } finally { saving = false; }
  }
  addEventListener("pagehide", () => flush(true));
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") flush(true); });

  // ------------------------------------------------------------------ images
  function img(url) {
    if (!imgs.has(url)) {
      const p = new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = url; });
      imgs.set(url, p);
      if (imgs.size > 80) imgs.delete(imgs.keys().next().value);
    }
    return imgs.get(url);
  }
  const frameUrl = (f) => `/api/watch/sweeps/${RUN}/frames/${f}`;
  const cropUrl = (c) => `/api/watch/sweeps/${RUN}/crops/${c}`;

  function fitCanvas(cv, w, h) {
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.max(1, Math.round(w * dpr)); cv.height = Math.max(1, Math.round(h * dpr));
    cv.style.width = `${w}px`; cv.style.height = `${h}px`;
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return ctx;
  }
  function blank(cv, text) {
    const w = cv.parentElement.clientWidth || 200;
    const ctx = fitCanvas(cv, w, 60); ctx.fillStyle = "#777"; ctx.font = "13px system-ui"; ctx.fillText(text, 8, 34);
  }
  // draw src (image) region [sx,sy,sw,sh] into the canvas, rotated by rot*90°, fitted into maxW x maxH
  function drawRegion(cv, im, sx, sy, sw, sh, rot, maxH) {
    const maxW = cv.parentElement.clientWidth || 240;
    const rw = rot % 2 ? sh : sw, rh = rot % 2 ? sw : sh;
    const k = Math.min(maxW / rw, maxH / rh, 4);
    const ctx = fitCanvas(cv, rw * k, rh * k);
    ctx.save(); ctx.translate((rw * k) / 2, (rh * k) / 2); ctx.rotate((rot * Math.PI) / 2);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(im, sx, sy, sw, sh, (-sw * k) / 2, (-sh * k) / 2, sw * k, sh * k); ctx.restore();
  }
  // a tilted spine outline (4 points): rotate so its long side is vertical, then cut it out (+ a small margin)
  function drawQuad(cv, im, q, rot, maxH) {
    const len = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);
    const e = [0, 1, 2, 3].map((j) => len(q[j], q[(j + 1) % 4]));
    const j = e[0] + e[2] >= e[1] + e[3] ? 0 : 1;
    let dx = q[j + 1][0] - q[j][0], dy = q[j + 1][1] - q[j][1];
    if (dy < 0) { dx = -dx; dy = -dy; }                       // long side pointing down the image: an upright book stays upright
    const theta = Math.atan2(dy, dx), m = 6;
    const H = (e[j] + e[j + 2]) / 2 + 2 * m, W = (e[j + 1] + e[(j + 3) % 4]) / 2 + 2 * m;
    const cx = q.reduce((s, p) => s + p[0], 0) / 4, cy = q.reduce((s, p) => s + p[1], 0) / 4;
    const maxW = cv.parentElement.clientWidth || 240;
    const rw = rot % 2 ? H : W, rh = rot % 2 ? W : H;
    const k = Math.min(maxW / rw, maxH / rh, 4);
    const ctx = fitCanvas(cv, rw * k, rh * k);
    ctx.save(); ctx.translate((rw * k) / 2, (rh * k) / 2); ctx.rotate((rot * Math.PI) / 2); ctx.scale(k, k);
    ctx.beginPath(); ctx.rect(-W / 2, -H / 2, W, H); ctx.clip();
    ctx.rotate(Math.PI / 2 - theta); ctx.translate(-cx, -cy); ctx.imageSmoothingQuality = "high"; ctx.drawImage(im, 0, 0); ctx.restore();
  }
  async function drawCard(card, cropCv, frameCv, cropCap, frameCap, rot, token) {
    blank(cropCv, "loading…"); blank(frameCv, "loading…");
    const fr = card.frame ? await img(frameUrl(card.frame)) : null;
    const cr = card.crop ? await img(cropUrl(card.crop)) : null;
    if (token !== drawToken) return;
    // crop: the packet's reading crop, else the line's box cut from its frame
    if (cr) { drawRegion(cropCv, cr, 0, 0, cr.naturalWidth, cr.naturalHeight, rot, 440); if (cropCap) cropCap.textContent = "spine crop (reading evidence)"; }
    else if (fr && card.polygon && card.polygon.length === 4) {
      drawQuad(cropCv, fr, card.polygon, rot, 440);
      if (cropCap) cropCap.textContent = "cut from the frame by the line's outline (straightened)";
    } else if (fr && card.bbox) {
      const [x0, y0, x1, y1] = card.bbox, m = Math.max(6, 0.15 * Math.max(x1 - x0, y1 - y0));   // a little context around small items
      const sx = Math.max(0, x0 - m), sy = Math.max(0, y0 - m), sw = Math.min(fr.naturalWidth, x1 + m) - sx, sh = Math.min(fr.naturalHeight, y1 + m) - sy;
      if (sw > 2 && sh > 2) drawRegion(cropCv, fr, sx, sy, sw, sh, rot, 440); else blank(cropCv, "box too small");
      if (cropCap) cropCap.textContent = "cut from the frame by the line's box";
    } else blank(cropCv, "no crop or box");
    if (!fr) { blank(frameCv, card.frame ? "frame not found" : "no frame"); frameCap.textContent = ""; return; }
    const W = frameCv.parentElement.clientWidth || 600, k = W / fr.naturalWidth;
    const ctx = fitCanvas(frameCv, W, fr.naturalHeight * k);
    ctx.drawImage(fr, 0, 0, W, fr.naturalHeight * k);
    const shape = () => {
      if (card.polygon) { card.polygon.forEach(([x, y], j) => (j ? ctx.lineTo(x * k, y * k) : ctx.moveTo(x * k, y * k))); ctx.closePath(); }
      else if (card.bbox) { const [x0, y0, x1, y1] = card.bbox; ctx.rect(x0 * k, y0 * k, (x1 - x0) * k, (y1 - y0) * k); }
    };
    if (card.polygon || card.bbox) {
      ctx.beginPath(); ctx.rect(0, 0, W, fr.naturalHeight * k); shape();          // dim everything outside the line's outline
      ctx.fillStyle = "rgba(0,0,0,0.45)"; ctx.fill("evenodd");
      ctx.beginPath(); shape(); ctx.lineJoin = "round";
      ctx.lineWidth = 5; ctx.strokeStyle = "rgba(0,0,0,0.85)"; ctx.stroke();
      ctx.lineWidth = 2.5; ctx.strokeStyle = "#ffd60a"; ctx.stroke();
    }
    frameCap.textContent = `${card.frame}${card.t_s != null ? ` · ${Number(card.t_s).toFixed(1)} s` : ""}`;
  }
  let drawToken = 0;

  function savePos() {                 // resume point: saved only when the card changes
    const pos = { titles: C.books[S.t]?.id || null, items: C.items[S.i]?.id || null }, old = L.position || {};
    if (pos.titles === old.titles && pos.items === old.items) return;
    L.position = pos; queue("position", null, pos);
  }

  // ------------------------------------------------------------------ titles
  const tLab = (b) => (L.titles || {})[b.id];
  const tDone = (b) => { const a = tLab(b); return !!a && (a.legible === false || (!!a.hand_title && !!a.correct)); };
  const tPass = (b) => (!S.tUnans || !tDone(b)) && (!S.tIdent || b.status === "identified");
  function tStep(dir, fromAnswer = false) {
    const B = C.books;
    for (let j = S.t + dir; j >= 0 && j < B.length; j += dir) if (tPass(B[j])) { S.t = j; return renderTitles(); }
    if (fromAnswer) $("tVerdict").textContent = dir > 0 ? "That was the last card in this filter." : "First card.";
  }
  function setTitle(b, v) {
    L.titles ||= {};
    L.titles[b.id] = v; queue("titles", b.id, v); renderProgressT();
  }
  function tAct(k) {
    const b = C.books[S.t]; if (!b) return;
    const a = { hand_title: "", correct: null, legible: null, ...(tLab(b) || {}) };
    const typed = $("tTitle").value.trim();
    if (k === "y") {
      if (!b.title) { $("tVerdict").textContent = "The system has no title for this line: type the title (Enter) or press L."; return; }
      setTitle(b, { hand_title: typed || b.title, correct: "Y", legible: true }); return tStep(1, true);
    }
    if (k === "n") {
      setTitle(b, { hand_title: typed, correct: "N", legible: true });
      if (!typed) { $("tVerdict").textContent = "Wrong — type the title you can read, then Enter."; $("tTitle").focus(); return; }
      return tStep(1, true);
    }
    if (k === "l") { setTitle(b, { hand_title: "", correct: null, legible: false }); return tStep(1, true); }
    if (k === "save") {
      if (!typed) { $("tTitle").blur(); return; }
      const correct = !b.title ? "N" : a.correct;
      setTitle(b, { hand_title: typed, correct, legible: true }); $("tTitle").blur();
      if (correct) return tStep(1, true);
      $("tVerdict").textContent = "Is the system title correct? Y / N"; return;
    }
    if (k === "clear") { delete L.titles[b.id]; queue("titles", b.id, null); renderTitles(); return; }
    if (k === "prev") return tStep(-1);
    if (k === "next") return tStep(1);
    if (k === "rot") { S.rotT = (S.rotT + 1) % 4; return renderTitles(); }
    if (k === "jump") {
      const n = prompt(`Go to line # (1–${C.books.length})`); if (!n) return;
      const j = C.books.findIndex((x) => String(x.i) === n.trim()); if (j < 0) return alert(`No line #${n}`);
      S.t = j; return renderTitles();
    }
  }
  function renderProgressT() {
    const B = C.books, done = B.filter(tDone).length, ident = B.filter((b) => b.status === "identified");
    const legible = B.filter((b) => tLab(b)?.hand_title).length, correct = B.filter((b) => tLab(b)?.hand_title && tLab(b)?.correct === "Y").length;
    $("tProg").textContent = `${done} of ${B.length} answered · ${ident.filter(tDone).length} of ${ident.length} identified · legible ${legible}, system correct ${correct}`;
    $("tBar").style.width = `${(100 * done) / Math.max(1, B.length)}%`;
  }
  function renderTitles() {
    renderProgressT();
    const B = C.books, b = B[S.t];
    if (!b) { $("tSys").innerHTML = "<dd>No book lines.</dd>"; return; }
    const filt = B.filter(tPass), pos = filt.indexOf(b);
    $("tPos").textContent = `card ${pos >= 0 ? pos + 1 : "–"} of ${filt.length}${pos < 0 ? " (outside the filter)" : ""} · line #${b.i}`;
    const a = tLab(b);
    const st = !a ? "unanswered" : a.legible === false ? "not legible" : a.correct === "Y" ? "system title correct" : a.correct === "N" ? "system title wrong" : "title typed, Y/N missing";
    $("tSys").innerHTML = `
      <div><dt>System title</dt><dd class="gt-title">${b.title ? esc(b.title) : "<span class=muted>— (none)</span>"}</dd></div>
      <div><dt>Author</dt><dd>${esc(b.author || "—")}</dd></div>
      <div><dt>Status</dt><dd><span class="gt-chip ${esc(b.status)}">${esc(b.status)}</span> ${b.confidence != null ? `confidence ${Number(b.confidence).toFixed(2)}` : ""}</dd></div>
      ${b.visible_text ? `<div><dt>Spine reads (model)</dt><dd class="small">${esc(b.visible_text)}</dd></div>` : ""}
      <div><dt>Your answer</dt><dd class="gt-ans ${a ? "set" : ""}">${esc(st)}</dd></div>`;
    $("tTitle").value = a?.hand_title || "";
    $("tVerdict").textContent = "";
    const token = ++drawToken;
    drawCard(b, $("tCrop"), $("tFrame"), $("tCropCap"), $("tFrameCap"), S.rotT, token);
    const nx = B[S.t + 1]; if (nx?.frame) img(frameUrl(nx.frame));   // prefetch
    savePos();
  }

  // ------------------------------------------------------------------ items
  const iLab = (x) => (L.items || {})[x.id];
  const iDone = (x) => { const a = iLab(x); return !!a && (a.real === "N" || (a.real === "Y" && !!a.category_correct)); };
  const iPass = (x) => !S.iUnans || !iDone(x);
  function iStep(dir, fromAnswer = false) {
    const X = C.items;
    for (let j = S.i + dir; j >= 0 && j < X.length; j += dir) if (iPass(X[j])) { S.i = j; return renderItems(); }
    if (fromAnswer) { $("iVerdict").textContent = dir > 0 ? "That was the last item in this filter." : "First item."; renderItems(true); }
  }
  function setItem(x, v) { L.items ||= {}; L.items[x.id] = v; queue("items", x.id, v); renderProgressI(); }
  function iAct(k) {
    const x = C.items[S.i]; if (!x) return;
    const a = { real: null, true_category: "", category_correct: null, ...(iLab(x) || {}) };
    if (k === "y") {
      setItem(x, { ...a, real: "Y" });
      if (a.category_correct) return iStep(1, true);
      renderItems(true); $("iVerdict").textContent = "Category correct? C / X"; return;
    }
    if (k === "n") { setItem(x, { real: "N", true_category: "", category_correct: null }); return iStep(1, true); }
    if (k === "c") { setItem(x, { real: "Y", true_category: x.category || "", category_correct: "Y" }); return iStep(1, true); }
    if (k === "x") {
      setItem(x, { real: "Y", true_category: a.category_correct === "N" ? a.true_category : "", category_correct: "N" });
      renderItems(true); $("iVerdict").textContent = "Type or pick the true category, then Enter."; $("iCat").focus(); return;
    }
    if (k === "save") {
      const cat = $("iCat").value.trim(); $("iCat").blur();
      if (!cat) return;
      const same = cat.toLowerCase() === String(x.category || "").toLowerCase();
      setItem(x, { real: "Y", true_category: cat, category_correct: a.category_correct || (same ? "Y" : "N") });
      return iStep(1, true);
    }
    if (k === "clear") { delete L.items[x.id]; queue("items", x.id, null); return renderItems(); }
    if (k === "prev") return iStep(-1);
    if (k === "next") return iStep(1);
    if (k === "jump") {
      const n = prompt(`Go to item # (1–${C.items.length})`); if (!n) return;
      const j = C.items.findIndex((y) => String(y.i) === n.trim()); if (j < 0) return alert(`No item #${n}`);
      S.i = j; return renderItems();
    }
  }
  function renderProgressI() {
    const X = C.items, done = X.filter(iDone).length, ok = X.filter((x) => iLab(x)?.real === "Y" && iLab(x)?.category_correct === "Y").length;
    const t = L.true_item_count;
    $("iProg").textContent = `${done} of ${X.length} answered · ${ok} real with the right category${t ? ` · ${(100 * ok / t).toFixed(0)}% of ${t} (bar 80%)` : ""}`;
    $("iBar").style.width = `${(100 * done) / Math.max(1, X.length)}%`;
  }
  function renderItems(keepImages = false) {
    renderProgressI();
    const X = C.items, x = X[S.i];
    if (!x) { $("iSys").innerHTML = "<dd>No non-book items in this packet.</dd>"; return; }
    const filt = X.filter(iPass), pos = filt.indexOf(x);
    $("iPos").textContent = `card ${pos >= 0 ? pos + 1 : "–"} of ${filt.length}${pos < 0 ? " (outside the filter)" : ""} · item #${x.i}`;
    const a = iLab(x);
    const st = !a ? "unanswered" : a.real === "N" ? "not a real non-book item" : a.real === "Y" && a.category_correct === "Y" ? "real, category correct"
      : a.real === "Y" && a.category_correct === "N" ? `real, true category ${a.true_category || "?"}` : "real, category not checked";
    $("iSys").innerHTML = `
      <div><dt>System category</dt><dd class="gt-title">${esc(x.category || "—")}</dd></div>
      <div><dt>Status</dt><dd>${esc(x.status || "—")} ${x.confidence != null ? `· confidence ${Number(x.confidence).toFixed(2)}` : ""}</dd></div>
      ${x.description ? `<div><dt>Description</dt><dd class="small">${esc(x.description)}</dd></div>` : ""}
      <div><dt>Your answer</dt><dd class="gt-ans ${a ? "set" : ""}">${esc(st)}</dd></div>`;
    $("iCat").value = a?.true_category || "";
    if (!keepImages) {
      $("iVerdict").textContent = "";
      const token = ++drawToken;
      drawCard(x, $("iCrop"), $("iFrame"), null, $("iFrameCap"), 0, token);
      const nx = X[S.i + 1]; if (nx?.frame) img(frameUrl(nx.frame));
      savePos();
    }
  }

  // ------------------------------------------------------------------ missed lists
  function renderMissed(kind) {
    const key = kind === "t" ? "missed_titles" : "missed_items", list = L[key] || [];
    $(`${kind}MissedN`).textContent = list.length;
    $(`${kind}Missed`).innerHTML = list.map((m, j) => `<li><b>${esc(kind === "t" ? m.title : m.category)}</b>${m.frame ? ` · ${esc(m.frame)}` : ""}${m.notes ? ` · <span class="muted">${esc(m.notes)}</span>` : ""}
      <button data-del="${j}" aria-label="Remove">×</button></li>`).join("");
  }
  function bindMissed(kind) {
    const key = kind === "t" ? "missed_titles" : "missed_items";
    $(`${kind}MissedForm`).addEventListener("submit", (ev) => {
      ev.preventDefault();
      const f = new FormData(ev.target), row = Object.fromEntries([...f.entries()].map(([k, v]) => [k, String(v).trim()]));
      L[key] = [...(L[key] || []), row]; queue(key, null, L[key]); ev.target.reset(); renderMissed(kind);
      ev.target.querySelector("input").focus();
    });
    $(`${kind}Missed`).addEventListener("click", (ev) => {
      const j = ev.target.dataset.del; if (j == null) return;
      L[key] = (L[key] || []).filter((_, n) => n !== Number(j)); queue(key, null, L[key]); renderMissed(kind);
    });
  }

  // ------------------------------------------------------------------ prices
  const today = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };
  function priceErr(p, v) { const h = Number(v); return v !== "" && v != null && h > 0 ? Math.abs(p.amount - h) / h : null; }
  function renderPrices() {
    $("pIntro").textContent = `The ${C.prices.length} books the sheet samples (every k-th of the ${C.priced} priced books in shelf order, same code as scripts/ground_truth_sheet.py). ` +
      "Check each at the same kind of source (new copy, same edition where you can), in USD, and paste the listing URL.";
    $("pTable").querySelector("tbody").innerHTML = C.prices.map((p) => {
      const a = (L.prices || {})[p.id] || {}, e = priceErr(p, a.price);
      return `<tr data-id="${esc(p.id)}">
        <td>${esc(p.i)}</td>
        <td><b>${esc(p.title || "—")}</b><br><span class="small muted">${esc(p.author || "")}${p.isbn ? ` · ISBN ${esc(p.isbn)}` : ""}</span></td>
        <td class="num">$${Number(p.amount).toFixed(2)}${p.converted && p.original ? `<br><span class="small muted">from ${esc(p.original[1])} ${esc(p.original[0])}</span>` : ""}</td>
        <td class="small">${esc(p.source || "—")}${p.condition ? ` · ${esc(p.condition)}` : ""}${p.url ? `<br><a href="${esc(p.url)}" target="_blank" rel="noopener noreferrer">open listing ↗</a>` : ""}</td>
        <td class="small">${esc((p.retrieved_at || "").slice(0, 10))}</td>
        <td><input class="gt-num" data-f="price" type="number" min="0" step="0.01" value="${a.price ?? ""}" aria-label="Hand-checked price for #${esc(p.i)}" /></td>
        <td><input data-f="url" type="url" value="${esc(a.url || "")}" placeholder="https://…" aria-label="Hand URL for #${esc(p.i)}" />${a.url ? ` <a href="${esc(a.url)}" target="_blank" rel="noopener noreferrer">↗</a>` : ""}</td>
        <td><input data-f="date" type="date" value="${esc(a.date || "")}" aria-label="Date checked for #${esc(p.i)}" /></td>
        <td class="num ${e == null ? "" : e <= 0.25 ? "ok" : "bad"}">${e == null ? "—" : `${(100 * e).toFixed(1)}%`}</td></tr>`;
    }).join("");
  }
  function bindPrices() {
    $("pTable").addEventListener("change", (ev) => {
      const tr = ev.target.closest("tr[data-id]"); if (!tr) return;
      const id = tr.dataset.id, v = {};
      tr.querySelectorAll("input[data-f]").forEach((inp) => { v[inp.dataset.f] = inp.value.trim(); });
      v.price = v.price === "" ? null : Number(v.price);
      if (v.price != null && !v.date) v.date = today();
      if (v.url && !/^https?:\/\//.test(v.url)) { showErr("The URL must start with http:// or https://"); return; }
      L.prices ||= {}; L.prices[id] = v; queue("prices", id, v);
      const dateIn = tr.querySelector('input[data-f="date"]'); if (dateIn.value !== v.date) dateIn.value = v.date;
      const p = C.prices.find((x) => x.id === id), e = priceErr(p, v.price), td = tr.lastElementChild;   // update in place: keeps Tab focus
      td.className = `num ${e == null ? "" : e <= 0.25 ? "ok" : "bad"}`; td.textContent = e == null ? "—" : `${(100 * e).toFixed(1)}%`;
    });
  }

  // ------------------------------------------------------------------ write to sheet
  async function writeSheet(force) {
    await flush(); $("writeMsg").textContent = "Writing…"; $("writeForce").hidden = true;
    try {
      const r = await api(`/api/truth/${RUN}/sheet${force ? "?force=1" : ""}`, { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
      $("writeMsg").textContent = `Wrote ${r.out} at ${new Date().toLocaleTimeString()}: ${r.titles} title rows (+${r.missed_titles} missed), ${r.prices} prices, ${r.items} items (+${r.missed_items} missed).`;
      $("excelWarn").hidden = true;
      renderResults(r.results);
    } catch (e) {
      $("writeMsg").textContent = "";
      if (e.status === 409) { $("excelWarn").textContent = e.message; $("excelWarn").hidden = false; $("writeForce").hidden = false; }
      else showErr(`Could not write the sheet: ${e.message}`);
    }
  }
  function renderResults(results) {
    const fmt = (v) => (v == null ? "—" : typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(2)) : esc(v));
    $("rTable").querySelector("tbody").innerHTML = results.map((x) => `<tr><td><b>${esc(x.measure)}</b>${x.note ? `<br><span class="small muted">${esc(x.note)}</span>` : ""}</td><td class="small">${esc(x.pass_bar)}</td>
      <td class="num">${fmt(x.ground_truth)}</td><td class="num">${fmt(x.system)}</td>
      <td class="num">${x.score == null ? "—" : x.measure === "Time to packet" ? `${x.score} s` : `${(100 * x.score).toFixed(1)}%`}</td>
      <td><span class="gt-res ${/^PASS/.test(x.result) ? "ok" : /^FAIL/.test(x.result) ? "bad" : "ne"}">${esc(x.result)}</span></td></tr>`).join("");
    $("rTable").hidden = false;
  }

  // ------------------------------------------------------------------ tabs + keys
  function setTab(t) {
    S.tab = t;
    document.querySelectorAll("[data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === t)));
    for (const n of ["titles", "items", "prices", "results"]) $(`tab-${n}`).hidden = n !== t;
    try { localStorage.setItem("gt.tab", t); } catch (_) { /* private mode */ }
    if (t === "titles") renderTitles(); else if (t === "items") renderItems(); else if (t === "prices") renderPrices();
  }
  document.addEventListener("keydown", (ev) => {
    if (!C || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const tg = ev.target, inText = tg.matches?.("input:not([type=checkbox]), textarea, select");
    if (tg.id === "tTitle") {
      if (ev.key === "Enter") { ev.preventDefault(); tAct("save"); } else if (ev.key === "Escape") tg.blur();
      return;
    }
    if (tg.id === "iCat") {
      if (ev.key === "Enter") { ev.preventDefault(); iAct("save"); } else if (ev.key === "Escape") tg.blur();
      return;
    }
    if (inText) return;
    const k = ev.key.length === 1 ? ev.key.toLowerCase() : ev.key;
    if (S.tab === "titles") {
      const map = { y: "y", n: "n", l: "l", j: "jump", r: "rot", ArrowLeft: "prev", ArrowRight: "next" };
      if (k === "Enter") { ev.preventDefault(); $("tTitle").focus(); return; }
      if (map[k]) { ev.preventDefault(); tAct(map[k]); }
    } else if (S.tab === "items") {
      const map = { y: "y", n: "n", c: "c", x: "x", j: "jump", ArrowLeft: "prev", ArrowRight: "next" };
      if (k === "Enter") { ev.preventDefault(); $("iCat").focus(); return; }
      if (map[k]) { ev.preventDefault(); iAct(map[k]); }
    }
  });
  document.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
  $("tCard").addEventListener("click", (ev) => { const k = ev.target.closest("button[data-k]")?.dataset.k; if (k) tAct(k); });
  $("iCard").addEventListener("click", (ev) => { const k = ev.target.closest("button[data-k]")?.dataset.k; if (k) iAct(k); });
  $("tUnanswered").addEventListener("change", (e) => { S.tUnans = e.target.checked; if (!tPass(C.books[S.t])) tStep(1); renderTitles(); });
  $("tIdentified").addEventListener("change", (e) => { S.tIdent = e.target.checked; if (!tPass(C.books[S.t])) tStep(1); renderTitles(); });
  $("iUnanswered").addEventListener("change", (e) => { S.iUnans = e.target.checked; if (!iPass(C.items[S.i])) iStep(1); renderItems(); });
  $("iTrue").addEventListener("change", (e) => {
    const v = e.target.value.trim(), n = v === "" ? null : Math.round(Number(v));
    L.true_item_count = n; queue("true_item_count", null, n); renderProgressI();
  });
  $("writeBtn").addEventListener("click", () => writeSheet(false));
  $("writeForce").addEventListener("click", () => writeSheet(true));
  let resizeT = null;
  addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => { if (S.tab === "titles") renderTitles(); else if (S.tab === "items") renderItems(); }, 200); });
  bindMissed("t"); bindMissed("i"); bindPrices();

  // ------------------------------------------------------------------ boot
  (async () => {
    try {
      if (!RUN) RUN = (await api("/api/truth/default")).run;
      if (!/^sweep_[0-9a-f]{16}$/.test(RUN)) throw new Error("bad ?run= (sweep_<16 hex>)");
      C = await api(`/api/truth/${RUN}`);
      L = C.labels || {};
      const sw = C.sweep || {};
      $("sweepLine").textContent = `${sw.id || RUN} · ${C.books.length} book lines · ${C.items.length} items · ${C.prices.length} price checks · saves to ${C.labels_file}`;
      document.title = `Ground truth · ${RUN}`;
      $("sheetName").textContent = C.sheet_file;
      if (C.excel_open) { $("excelWarn").textContent = `Excel has ${C.sheet_file} open: save and close it before writing (values typed there are kept).`; $("excelWarn").hidden = false; }
      $("iCats").innerHTML = C.categories.map((c) => `<option value="${esc(c)}"></option>`).join("");
      $("iTrue").value = L.true_item_count ?? "";
      const pos = L.position || {};
      let t = C.books.findIndex((b) => b.id === pos.titles);
      if (t < 0) t = Math.max(0, C.books.findIndex((b) => !tDone(b)));
      let i = C.items.findIndex((x) => x.id === pos.items);
      if (i < 0) i = Math.max(0, C.items.findIndex((x) => !iDone(x)));
      S.t = t; S.i = i;
      renderMissed("t"); renderMissed("i");
      if (L.last_sheet?.results) {
        renderResults(L.last_sheet.results);
        $("writeMsg").textContent = `Last written ${String(L.last_sheet.written_at).replace("T", " ").slice(0, 19)} to ${L.last_sheet.out}. Answers changed since then count after the next write.`;
      }
      let tab = params.get("tab") || "titles"; if (!params.get("tab")) { try { tab = localStorage.getItem("gt.tab") || "titles"; } catch (_) { /* private mode */ } }
      setTab(["titles", "items", "prices", "results"].includes(tab) ? tab : "titles");
      delete pending.position; $("saveState").textContent = L.updated_at ? `Last saved ${L.updated_at.replace("T", " ").slice(0, 19)}` : "Nothing saved yet";
      clearTimeout(saveTimer);
    } catch (e) { showErr(`Could not load ${RUN || "the sweep"}: ${e.message}`); $("sweepLine").textContent = "—"; }
  })();
})();
