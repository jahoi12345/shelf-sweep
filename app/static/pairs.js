// Blinded pair browser/importer. No labels or private control key are fetched.
(() => {
  const $ = (id) => document.getElementById(id);
  let manifest = null, index = 0;
  const msg = (s) => { $("pairMsg").textContent = s; };
  const api = async (url, options) => {
    const r = await fetch(url, options), d = await r.json();
    if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
    return d;
  };
  const base = () => `/api/pairs/${encodeURIComponent(manifest.run_id)}`;
  function show() {
    if (!manifest || !manifest.pairs.length) return;
    const p = manifest.pairs[index], url = `${base()}/sheet/${encodeURIComponent(p.pair_id)}`;
    $("pairPosition").textContent = `${index + 1} / ${manifest.pairs.length} · ${p.pair_id}`;
    $("pairImage").src = url; $("pairDownload").href = url;
    $("pairContact").href = `${base()}/contact/${Math.floor(index / 12) + 1}`;
    $("pairSummary").href = `${base()}/summary`;
    $("pairPrev").disabled = index === 0; $("pairNext").disabled = index + 1 === manifest.pairs.length;
  }
  async function load() {
    try { manifest = await api(`/api/pairs/${encodeURIComponent($("pairRun").value)}/manifest`); index = 0; show(); msg("Sheets ready. Share images and the prompt only."); }
    catch (e) { msg(e.message); }
  }
  $("pairView").addEventListener("click", async () => {
    $("labelRoot").hidden = true; $("pairRoot").hidden = false;
    if ($("pairRun").options.length) return;
    try {
      const d = await api("/api/pairs/runs");
      d.runs.forEach((r) => { const o = document.createElement("option"); o.value = r.run_id; o.textContent = `${r.run_id} · ${r.pairs} pairs`; $("pairRun").appendChild(o); });
      if (d.runs.length) await load(); else msg("No pair sheets generated yet. Use the generation CLI in the prompt document.");
    } catch (e) { msg(e.message); }
  });
  $("pairTitles").addEventListener("click", () => { $("labelRoot").hidden = false; $("pairRoot").hidden = true; });
  $("pairLoad").addEventListener("click", load);
  $("pairPrev").addEventListener("click", () => { if (manifest) { index = Math.max(0, index - 1); show(); } });
  $("pairNext").addEventListener("click", () => { if (manifest) { index = Math.min(manifest.pairs.length - 1, index + 1); show(); } });
  $("pairImport").addEventListener("click", async () => {
    if (!manifest) return msg("Load a sheet set first.");
    try {
      const body = { annotator: $("pairAnnotator").value, kind: $("pairKind").value, answers: $("pairAnswers").value };
      const d = await api(`${base()}/import`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      msg(`Imported ${d.imported} labels as ${d.kind}. ${d.summary.honesty}`);
      $("pairAnswers").value = "";
    } catch (e) { msg(`Not imported: ${e.message}`); }
  });
})();
