// Title labelling view: keyboard-first, resumable, undoable. Talks only to /api/label/* (held-out truth, never read by the pipeline).
(() => {
  const $ = (id) => document.querySelector(id);
  const api = (p, o) => fetch(p, { headers: { "Content-Type": "application/json" }, ...o }).then(async (r) => {
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || `HTTP ${r.status}`);
    return j;
  });
  let S = null, index = 0, rotation = 0, nativeView = false, currentPrefill = null;
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[c]));
  const msg = (t) => { $("#labelMsg").textContent = t || ""; };

  async function loadRuns() {
    try {
      const d = await api("/api/label/runs");
      const runs = d.runs.filter((r) => r.scan);
      $("#labelRun").innerHTML = runs.map((r) => `<option value="${esc(r.run_id)}"${r.default_for_scan ? " data-default='1'" : ""}>${esc(r.scan)} · ${esc(r.run_id)} · ${r.crops} crops${r.default_for_scan ? " · default" : ""}${r.scan_inferred_from_duration ? " · scan inferred from duration" : ""}</option>`).join("");
      const first = runs.find((r) => r.default_for_scan && r.scan === "YugoTest2") || runs.find((r) => r.default_for_scan);
      if (first) $("#labelRun").value = first.run_id;
      $("#labelN").value = d.n_default;
      const last = localStorage.getItem("labelAnnotator");
      if (last && !$("#labelName").value) $("#labelName").value = last;
    } catch (e) { msg(`Could not list runs: ${e.message}`); }
  }

  function showCurrent() {
    if (!S) return;
    const it = S.items[index];
    $("#labelWork").hidden = false;
    $("#labelProgress").textContent = `${S.done} / ${S.n} labelled`;
    const c = S.counts;
    $("#labelCounts").textContent = ` · partial ${c.partial ?? 0} · unreadable ${c.unreadable} · not a book ${c.not_a_book} · duplicate ${c.duplicate_of_previous}`;
    $("#labelMeta").textContent = ` · ${S.scan.label || "scan?"} · ${S.run_id} · saved to ${S.saved_to}`;
    const base = `/api/label/${encodeURIComponent(S.label_id)}`;
    $("#labelImg").src = `${base}/crop/${it.index}?view=${nativeView ? "native" : "preview"}`;
    $("#labelCtx").src = `${base}/context/${it.index}`;
    [0, 1].forEach((k) => { const el = $(`#labelS${k}`); if (k < (it.other_sightings || 0)) { el.hidden = false; el.src = `${base}/sighting/${it.index}/${k}`; } else { el.hidden = true; el.removeAttribute("src"); } });
    rotation = (it.label && it.label.rotation_seen_deg) || 0;
    applyRotation();
    $("#labelCurrent").textContent = `crop ${index + 1} of ${S.n} · ${it.frame_id} · ${it.stratum}${it.label ? " · already: " + it.label.status : ""}`;
    currentPrefill = it.prefill || null;
    if (it.label && (it.label.status === "labeled" || it.label.status === "partial")) {
      $("#labelTitle").value = it.label.title || ""; $("#labelAuthor").value = it.label.author || "";
      if (it.label.language) $("#labelLang").value = it.label.language;
      $("#labelPrefillNote").textContent = `already labelled${it.label.prefilled ? (it.label.edited ? " (pre-filled, edited)" : " (pre-filled, accepted as shown)") : ""}`;
    } else if (currentPrefill) {
      $("#labelTitle").value = currentPrefill.title || ""; $("#labelAuthor").value = currentPrefill.author || "";
      $("#labelPrefillNote").textContent = `pre-filled from ${currentPrefill.source} — model text, may be wrong: fix it or press Enter to accept as shown`;
    } else {
      $("#labelTitle").value = ""; $("#labelAuthor").value = "";
      $("#labelPrefillNote").textContent = "no model reading for this book: type the title as printed, or mark it";
    }
    $("#labelTitle").focus(); $("#labelTitle").select();
  }

  function applyRotation() { $("#labelImg").style.transform = `rotate(${rotation}deg)`; }

  async function start() {
    try {
      const name = $("#labelName").value.trim();
      if (!name) throw new Error("Enter your name first (annotator).");
      localStorage.setItem("labelAnnotator", name);
      msg("Sampling crops…");
      S = await api("/api/label/session", { method: "POST", body: JSON.stringify({ run_id: $("#labelRun").value, annotator: name, n: Number($("#labelN").value) }) });
      index = S.next_index ?? 0;
      msg(`Sample ready: ${S.n} crops, ${S.done} already done.`);
      showCurrent();
    } catch (e) { msg(e.message); }
  }

  function goto(i) { if (!S) return; index = Math.max(0, Math.min(S.n - 1, i)); showCurrent(); }
  function gotoNextUnlabelled(from) {
    const n = S.items.findIndex((it, i) => i > from && !it.label);
    index = n >= 0 ? n : (S.next_index ?? from);
    showCurrent();
  }

  async function submit(status) {
    if (!S) return;
    const it = S.items[index];
    try {
      const body = { observation_id: it.observation_id, status, rotation_seen: rotation };
      if (status === "labeled" || status === "partial") Object.assign(body, { title: $("#labelTitle").value, author: $("#labelAuthor").value, language: $("#labelLang").value, prefill: currentPrefill });
      S = await api(`/api/label/${encodeURIComponent(S.label_id)}/answer`, { method: "POST", body: JSON.stringify(body) });
      msg(status === "labeled" ? "Saved." : `Marked ${status.replace(/_/g, " ")}.`);
      if (S.remaining === 0) { msg(`All ${S.n} crops labelled. Saved to ${S.saved_to}.`); showCurrent(); return; }
      gotoNextUnlabelled(index);
    } catch (e) { msg(`Not saved: ${e.message}`); }
  }

  async function undo() {
    if (!S) return;
    try {
      const r = await api(`/api/label/${encodeURIComponent(S.label_id)}/undo`, { method: "POST", body: "{}" });
      S = r;
      const i = S.items.findIndex((x) => x.observation_id === r.undone_observation_id);
      index = i >= 0 ? i : index;
      msg("Undone.");
      showCurrent();
    } catch (e) { msg(e.message); }
  }

  const inField = (e) => e.target && e.target.tagName && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT");
  document.addEventListener("keydown", (e) => {
    const view = $("#view-label");
    if (!view || view.hidden || $("#labelRoot").hidden || !S) return;
    const meta = e.metaKey || e.ctrlKey;
    if (meta && !e.shiftKey && (e.key === "z" || e.key === "Z")) { e.preventDefault(); undo(); return; }     // Cmd+Z / Ctrl+Z anywhere
    if (e.altKey || meta) return;
    if (inField(e)) {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit("labeled"); }
      else if (e.key === "Escape") { e.preventDefault(); e.target.blur(); }
      return;
    }
    const k = e.key;
    const map = { Enter: () => submit("labeled"), p: () => submit("partial"), u: () => submit("unreadable"), n: () => submit("not_a_book"), d: () => submit("duplicate_of_previous"), z: undo,
      ArrowLeft: () => goto(index - 1), ArrowRight: () => goto(index + 1), "1": () => { $("#labelLang").value = "es"; }, "2": () => { $("#labelLang").value = "en"; },
      "3": () => { $("#labelLang").value = "other"; }, "[": () => { rotation = (rotation + 270) % 360; applyRotation(); }, "]": () => { rotation = (rotation + 90) % 360; applyRotation(); },
      t: () => { $("#labelTitle").focus(); $("#labelTitle").select(); }, a: () => { $("#labelAuthor").focus(); } };
    const fn = map[k] || map[k.toLowerCase && k.toLowerCase()];
    if (fn) { e.preventDefault(); fn(); }
  });
  $("#labelStart").addEventListener("click", start);
  $("#labelSave").addEventListener("click", () => submit("labeled"));
  $("#labelUnreadable").addEventListener("click", () => submit("unreadable"));
  $("#labelPartial").addEventListener("click", () => submit("partial"));
  $("#labelNotBook").addEventListener("click", () => submit("not_a_book"));
  $("#labelDup").addEventListener("click", () => submit("duplicate_of_previous"));
  $("#labelUndo").addEventListener("click", undo);
  $("#labelPrev").addEventListener("click", () => goto(index - 1));
  $("#labelNext").addEventListener("click", () => goto(index + 1));
  $("#labelRotL").addEventListener("click", () => { rotation = (rotation + 270) % 360; applyRotation(); });
  $("#labelRotR").addEventListener("click", () => { rotation = (rotation + 90) % 360; applyRotation(); });
  $("#labelNative").addEventListener("click", () => { nativeView = !nativeView; $("#labelNative").textContent = nativeView ? "show reader view" : "show native crop"; showCurrent(); });
  window.labelOnShow = () => { if (!$("#labelRun").options.length) loadRuns(); };
  window.__labelState = () => ({ S, index, rotation });   // test hook
  if (location.hash.startsWith("#label")) loadRuns();
})();
