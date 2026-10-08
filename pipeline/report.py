"""Stage ``report``: readable HTML rendered ONLY from claim_packet.json.

    report.render(packet_path, run_dir, config) -> report_path   (report.html)

No model calls. Frame evidence is linked by run-relative path and shown with a CSS box overlay
(percent coordinates), so the original frames are never modified. Unknown values print as "unknown".
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

from pipeline import contracts as C

E = html.escape
UNKNOWN = '<span class="unk">unknown</span>'


def v(x: Any) -> str:
    return UNKNOWN if x is None or x == "" else E(str(x))


def money(offer):
    if not isinstance(offer, dict): return UNKNOWN
    conv = ""
    if offer.get("converted") is True:          # a non-US seller's asking price converted to USD: always labelled, with the native amount and the cited ECB rate
        fx = offer.get("fx") or {}
        conv = (f"<br><small><b>CONVERTED</b> from {v(offer.get('original_currency'))} {v(offer.get('original_amount'))} at {v(fx.get('rate'))} USD/{v(offer.get('original_currency'))} "
                f"(<a href='{E(str(fx.get('url') or ''), quote=True)}'>ECB reference rate</a> dated {v(fx.get('rate_date'))}); seller market {v(offer.get('market'))}</small>")
    return (f"{v(offer.get('amount'))} {v(offer.get('currency'))}{conv}<br><a href='{E(str(offer.get('url') or ''), quote=True)}'>{v(offer.get('source'))}</a><br>"
            f"<small>{v(offer.get('retrieved_at'))} · {v(offer.get('condition_assumed'))} · {v(offer.get('proxy_assumptions'))}</small>")


def price_cell(b: dict[str, Any], key: str) -> str:
    """One priced value of a book line. Work-level lines always say so and name the chosen ISBN (never shown as an exact-edition price); nulls show their reason."""
    offer = b.get(key)
    if not isinstance(offer, dict):
        why = (b.get("pricing_null_reasons") or {}).get(key)
        return UNKNOWN + (f"<br><small>{E(str(why))}</small>" if why else "")
    if offer.get("price_basis") == "work_level_equivalent" or b.get("price_basis") == "work_level_equivalent":
        isbn = offer.get("equivalent_isbn") or b.get("equivalent_isbn")
        return (money(offer) + f"<br><small><b>work-level equivalent (not exact edition)</b> &middot; equivalent ISBN {v(isbn)}</small>")
    return money(offer)


def second_market_cell(b: dict[str, Any]) -> str:
    """Spain / native-EUR line (second market): the EUR amount is the price, the USD figure is a labelled conversion with its dated ECB rate; never a US price."""
    sm = b.get("second_market")
    if not isinstance(sm, dict) or not (sm.get("replacement_cost") or sm.get("used_value")):
        return ""
    parts = []
    for key, label in (("replacement_cost", "new"), ("used_value", "used")):
        o = sm.get(key)
        if not isinstance(o, dict):
            continue
        usd = f" &asymp; {v(o.get('usd_equivalent'))} USD" if o.get("usd_equivalent") is not None else ""
        parts.append(f"{label} {v(o.get('amount'))} EUR{usd} <small>(<a href='{E(str(o.get('url') or ''), quote=True)}'>Iberlibro</a>, {v(o.get('condition_assumed'))})</small>")
    fx = sm.get("fx") or {}
    basis = f"ECB {v(fx.get('rate'))} of {v(fx.get('rate_date'))}" if fx else "no FX evidence"
    return ("<br><small><b>Second market, Spain (native EUR, equivalent ISBN " + v((sm.get("equivalent_edition") or {}).get("isbn")) + "):</b></small><br>" + "; ".join(parts)
            + f"<br><small>USD figure = EUR x {basis}; not a US price</small>")


def item_price_cell(i: dict[str, Any]) -> str:
    rc = i.get("replacement_cost")
    if not isinstance(rc, dict):
        return UNKNOWN
    lo, hi = rc.get("low", rc.get("amount")), rc.get("high", rc.get("amount"))
    return (f"{v(lo)} - {v(hi)} {v(rc.get('currency'))}<br><a href='{E(str(rc.get('url') or ''), quote=True)}'>{v(rc.get('source'))}</a>"
            f"<br><small>{v(rc.get('retrieved_at'))}</small>")


def review_details(r: dict[str, Any]) -> str:
    det = r.get("reasons") or []
    if len(det) <= 1:
        return ""
    lis = "".join(f"<li>{E(str(d.get('severity')))} &middot; {E(str(d.get('code')))}: {E(str(d.get('reason')))}</li>" for d in det)
    return f"<details><summary class='small'>{len(det)} details</summary><ul class='small'>{lis}</ul></details>"


def second_country_html(sc: Any) -> str:
    """The same 10 books priced for the second country (native currency, sources and dates), next to their US/USD lines."""
    if not isinstance(sc, dict):
        return ""
    head = f"<h2>Second country: the same books priced in {v(sc.get('country'))} / {v(sc.get('currency'))} ({E(str(sc.get('status')))})</h2>"
    if not sc.get("books"):
        return head + f"<p class=small>{E(str(sc.get('note') or 'not run'))}</p>"

    def cell(o):
        if not isinstance(o, dict):
            return UNKNOWN
        return (f"{v(o.get('amount'))} {v(o.get('currency'))}<br><a href='{E(str(o.get('url') or ''), quote=True)}'>{v(o.get('source'))}</a>"
                f"<br><small>{v(o.get('retrieved_at'))} &middot; {v(o.get('condition_assumed'))}</small>")
    rows = "".join(
        f"<tr><td>{v(b.get('display_index'))}</td><td>{v(b.get('title'))}<br><small>{v(b.get('author'))} &middot; ISBN {v(b.get('isbn'))}</small>"
        + (f"<br><small><b>stale:</b> {E(str(b['stale']))}</small>" if b.get("stale") else "") + "</td>"
        f"<td>{cell((b.get('us') or {}).get('replacement_cost'))}</td><td>{cell((b.get('us') or {}).get('used_value'))}</td>"
        f"<td>{cell((b.get('second') or {}).get('replacement_cost'))}</td><td>{cell((b.get('second') or {}).get('used_value'))}</td>"
        f"<td><small>{E('; '.join(str(x) for x in b.get('issues') or []))}</small></td></tr>" for b in sc["books"])
    t = sc.get("totals") or {}
    cur = E(str(sc.get("currency")))
    foot = (f"<tr><th colspan=2>Totals (computed from these lines)</th><td>{v(t.get('us_replacement_cost_same_books_usd'))} USD</td><td>{v(t.get('us_used_value_same_books_usd'))} USD</td>"
            f"<td>{v(t.get('replacement_cost'))} {cur} ({v(t.get('priced_replacement'))} priced)</td><td>{v(t.get('used_value'))} {cur} ({v(t.get('priced_used'))} priced)</td><td></td></tr>")
    return (head + f"<p class=small>{E(str(sc.get('rule') or ''))}. {E(str(sc.get('note') or ''))} Setting: {E(str(sc.get('setting_source')))}; computed {v(sc.get('computed_at'))}.</p>"
            f"<table class='lines'><tr><th>#</th><th>Book</th><th>US new (USD)</th><th>US used (USD)</th><th>{v(sc.get('country'))} new ({cur})</th>"
            f"<th>{v(sc.get('country'))} used ({cur})</th><th>Notes</th></tr>{rows}{foot}</table>")


def evidence_thumb(line: dict[str, Any]) -> str:
    ev = line.get("evidence") or {}
    ref, bb, w, h = line.get("frame_ref"), ev.get("bbox_xyxy"), ev.get("frame_width_px"), ev.get("frame_height_px")
    if not ref:
        return UNKNOWN
    box = ""
    if bb and w and h:
        x0, y0, x1, y1 = bb
        box = (f'<span class="box" style="left:{100*x0/w:.2f}%;top:{100*y0/h:.2f}%;'
               f'width:{100*(x1-x0)/w:.2f}%;height:{100*(y1-y0)/h:.2f}%"></span>')
    return (f'<a class="thumb" href="{E(ref)}" title="{E(ref)} t={ev.get("timestamp_s")}s"><img loading="lazy" src="{E(ref)}" alt="">{box}</a>'
            f'<div class="small">{E(str(ev.get("frame_id")))} @ {v(ev.get("timestamp_s"))}s</div>')


def _rows(lines: list[dict[str, Any]], cols: list[tuple[str, Any]]) -> str:
    out = []
    for ln in lines:
        tds = "".join(f"<td>{fn(ln)}</td>" for _, fn in cols)
        out.append(f"<tr>{tds}</tr>")
    return "".join(out)


def evidence_text(line: dict[str, Any]) -> str:
    """Image-free evidence cell for the printable PDF: frame id, time, path and box (the files stay in the bundle)."""
    ev = line.get("evidence") or {}
    if not line.get("frame_ref"):
        return UNKNOWN
    bb = ev.get("bbox_xyxy")
    return f'{E(str(ev.get("frame_id")))} @ {v(ev.get("timestamp_s"))}s<br><small>{E(line["frame_ref"])}{(" box " + E(str([round(x) for x in bb]))) if bb else ""}</small>'


def render_html(packet_env: dict[str, Any], thumbs: bool = True) -> str:
    p = packet_env["data"]
    t, room, sweep = p["totals"], p["room"], p["sweep"]
    banner = ""
    if p.get("fixture"):
        banner += '<div class="banner fixture">FIXTURE DATA: these lines are test fixtures, not real detections.</div>'
    banner += f'<div class="banner prov">{E(p["provisional_notice"])}</div>'
    wl_n = sum(1 for b in p.get("books", []) if b.get("price_basis") == "work_level_equivalent")
    wl_tag = " (work-level equivalent, not exact-edition)" if wl_n else ""
    if wl_n:
        banner += (f'<div class="banner prov">Prices on {wl_n} line(s) are WORK-LEVEL EQUIVALENT: each is priced through the equivalent ISBN shown on the line, '
                   'not the exact edition on the shelf; US/USD asking prices, shipping and tax excluded.</div>')

    conv = (p.get("completeness") or {}).get("converted_lines") or {}
    conv_tag = lambda k: f" (USD; includes {conv[k]} line(s) converted from a non-US market)" if conv.get(k) else ""
    if conv.get("books_replacement_cost") or conv.get("books_used_value"):
        banner += f'<div class="banner prov">{E(conv.get("note") or "")}</div>'
    est = t.get("count_estimator")
    head_pan = t.get("headline_estimator") == "panorama"
    n_lines = t.get("book_lines", t["book_count"])
    est_text = (f'physical instances: {t.get("book_count_track")} track groups merged into {n_lines} lines by panorama overlap and cross-segment registration'
                + (f' (panorama cross-check: {t["panorama_count"]})' if t.get("panorama_count") is not None else "") if est == "instances" else
                f'2D panorama detector (the book lines below are track groups: {t.get("book_count_track")}, which count duplicates)' if est == "panorama" else
                ("track association" + (f' (panorama cross-check: {t["panorama_count"]})' if t.get("panorama_count") is not None else "")) if est else None)
    totals_rows = [
        ("Books (headline: 2D panorama count)" if head_pan else "Book instances (deduplicated)", t["book_count"]),
        ("Panorama range (may not contain the true count)" if head_pan else "Plausible range (lower - upper)",
         None if t["book_count_lower"] is None else f'{t["book_count_lower"]} - {t["book_count_upper"]}'),
        ("Book lines listed (some may be duplicates)", n_lines),
        *([("Book line range (association bounds)", f'{t["book_lines_lower"]} - {t["book_lines_upper"]}')] if head_pan and t.get("book_lines_lower") is not None else []),
        *([("Count estimator", est_text)] if est_text else []),
        *([("Why this headline", (p.get("count") or {}).get("headline", {}).get("reason"))] if (p.get("count") or {}).get("headline") else []),
        ("Identified (accepted catalogue match)", t["books_identified"]),
        ("Unidentified (counted, no title guessed)", t["books_unidentified"]),
        ("Non-book item instances", t["item_count"]),
        ("Uncertain cross-frame associations", t["uncertain_associations"]),
        ("Needs appraisal (books)", t.get("books_needs_appraisal")),
        ("Excluded by user (not counted)", t.get("user_excluded_count")),
        ("Shelf run (m)", t["shelf_run_m"]),
        ("Books replacement cost" + wl_tag + conv_tag("books_replacement_cost"), t["books_replacement_cost"]),
        ("Books used value" + wl_tag + conv_tag("books_used_value"), t["books_used_value"]),
        ("Items replacement cost (low-high)", None if t.get("items_replacement_cost_low") is None else f'{t["items_replacement_cost_low"]} - {t["items_replacement_cost_high"]}'),
        ("Items without a sourced price (needs appraisal, excluded)", t.get("items_unpriced")),
        ("Excluded from totals", t["excluded_from_totals"]),
    ]
    room_rows = [(k.replace("_", " "), room.get(k)) for k in ("status", "length_m", "width_m", "height_m", "floor_area_m2", "floor_area_ft2",
                                                              "wall_area_m2", "wall_area_ft2", "shelved_wall_area_m2", "shelved_wall_area_ft2",
                                                              "scale_method", "shape", "validation_state", "confidence")]
    room_rows.append(("height provenance", (room.get("height_provenance") or {}).get("label")))
    room_rows.append(("shelved wall provenance", room.get("shelved_wall_provenance")))
    room_rows.append(("uncertainty (declared terms)", None if not room.get("uncertainty") else
                      "; ".join(f"{k}={x}" for k, x in room["uncertainty"].items() if k != "note")))
    room_rows.append(("not established because", "; ".join(room.get("abstention_reasons") or []) or None))
    room_rows.append(("still needed", ", ".join(room.get("coverage_missing") or []) or None))
    me = (p.get("measurement_evidence") or {})
    cal_rows = "".join(
        f"<tr><td>{E(str(c['calibration_id']))}</td><td>{E(c['kind'])}</td><td>{E(str((c.get('plane') or {}).get('id')))}/{E(str((c.get('plane') or {}).get('role')))}</td>"
        f"<td>{E(c['status'])}{(' - ' + E('; '.join(c['problems']))) if c['problems'] else ''}</td><td>{E(json.dumps(c.get('metric')))}</td>"
        f"<td>{v((c.get('provenance') or {}).get('source'))}: {v((c.get('provenance') or {}).get('utterance') or (c.get('provenance') or {}).get('note'))}</td></tr>"
        for c in ((me.get("scale") or {}).get("calibrations") or []))
    meas_html = ("<table class='lines'><tr><th>ID</th><th>Kind</th><th>Plane</th><th>Status</th><th>Metric (user-supplied)</th><th>Provenance</th></tr>" + cal_rows + "</table>"
                 if cal_rows else "<p class=small>No calibration supplied: spine dimensions and shelf run are unknown.</p>")
    sc_sum = ((me.get("scale") or {}).get("summary") or {})
    meas_html += (f"<p class=small>Spines measured {sc_sum.get('measured')} / abstained {sc_sum.get('abstained')}; reasons: "
                  + E(", ".join(f"{k} x{n}" for k, n in (sc_sum.get('abstention_reasons') or {}).items())) + "</p>") if sc_sum else ""
    meas_html += "<p class=small>All measurement values are software_only_unvalidated: no tape-measured comparison has been made. Intervals are stated sums of declared terms, not statistical guarantees.</p>"
    kv = lambda rows: "".join(f"<tr><th>{E(str(k))}</th><td>{v(x)}</td></tr>" for k, x in rows)

    book_cols = [
        ("ID", lambda b: E(b["id"])), ("Status", lambda b: E(b["status"])),
        ("Title / author", lambda b: (v(b["title"]) + "<br><small>" + v(b["author"]) + (" &middot; from user statement" if b.get("identity_source") == "user_statement" else "") + "</small>")),
        ("Visible text (literal)", lambda b: v(b["visible_text"])),
        ("Shelf / pos", lambda b: f'{v(b["shelf"])} / {v(b["position"])}'),
        ("Det. conf.", lambda b: v(b["detection_confidence"])),
        ("Spine h x t (cm)", lambda b: f'{v(b["spine_height_cm"])} x {v(b["spine_thickness_cm"])}' + (
            f'<br><small>{b["measurement"]["spine_height_interval_cm"]} / {b["measurement"]["spine_thickness_interval_cm"]}</small>' if (b.get("measurement") or {}).get("status") == "measured" else
            (f'<br><small>abstained: {E(", ".join(b["measurement"]["reasons"][:2]))}</small>' if (b.get("measurement") or {}).get("status") == "abstained" else ""))),
        ("Replacement / used", lambda b: f'{price_cell(b, "replacement_cost")} / {price_cell(b, "used_value")}' + second_market_cell(b)),
        ("Dedup", lambda b: E("; ".join(str(x) for x in b["dedup_reasons"])) + (" <b>(uncertain)</b>" if b["uncertain_association"] else "")),
        ("Evidence", evidence_thumb if thumbs else evidence_text),
    ]
    item_cols = [
        ("ID", lambda i: E(i["id"])), ("Category", lambda i: v(i["category"])),
        ("Description", lambda i: v(i["description"])),
        ("Status", lambda i: v(i["status"]) + (f'<br><small>{E(str(i.get("status_reason")))}</small>' if i.get("status_reason") else "")),
        ("Conf.", lambda i: v(i["confidence"])), ("Price (low-high)", item_price_cell),
        ("Evidence", evidence_thumb if thumbs else evidence_text),
    ]
    head = lambda cols: "".join(f"<th>{E(c)}</th>" for c, _ in cols)
    review = "".join(
        f'<tr class="sev-{E(str(r.get("severity")))}"><td>{E(str(r["ref_id"]))}</td><td>{E(str(r.get("severity")))}</td><td>{E(", ".join(r.get("codes") or [str(r.get("code"))]))}</td>'
        f'<td>{E(str(r.get("reason")))}{review_details(r)}</td></tr>'
        for r in p["review_queue"])
    n_details = sum(int(r.get("detail_count") or 1) for r in p["review_queue"])
    gaps = "".join(f"<li><b>{E(g['type'])}</b>"
                   + (f" ({g['start_s']:.1f}-{g['end_s']:.1f}s)" if g.get("start_s") is not None else "")
                   + f": {E(g['detail'])}</li>" for g in p["coverage_gaps"])
    excl = "<table><tr><th>Instance</th><th>Reason</th><th>Correction</th><th>Evidence frame</th></tr>" + "".join(
        f"<tr><td>{E(x['instance_id'])}</td><td>{E(str(x.get('reason')))}</td><td>{E(str(x.get('correction_id')))}</td><td>{v(x.get('frame_ref'))}</td></tr>"
        for x in p.get("excluded_by_user", [])) + "</table>" if p.get("excluded_by_user") else "<p class=small>None.</p>"
    prov = "".join(f"<tr><th>{E(str(k))}</th><td>{v(x)}</td></tr>" for k, x in (p.get("provenance") or {}).items())
    lb = p.get("language_breakdown") or {}
    lang_html = ("<table class='lines'><tr><th>Language</th><th>Groups</th><th>Identified</th><th>Unidentified</th><th>Needs appraisal</th><th>With visible text</th><th>Human-legible denominator</th><th>Title accuracy</th></tr>" +
                 "".join(f"<tr><td>{E(k)}</td><td>{g['books']}</td><td>{g['identified']}</td><td>{g['unidentified']}</td><td>{g['needs_appraisal']}</td><td>{g['with_visible_text']}</td>"
                         f"<td>{E((lb.get('human_legible_spines', {}).get(k) or {}).get('status', 'n/a'))}</td><td>{E((lb.get('title_accuracy', {}).get(k) or {}).get('status', 'n/a'))}</td></tr>"
                         for k, g in (lb.get("counts") or {}).items()) + "</table><p class=small>" + E(lb.get("note", "")) + "</p>") if lb else "<p class=small>n/a</p>"
    cp = p.get("completeness") or {}
    def frow(label, d):
        return f"<tr><th>{E(label)}</th><td>{d.get('known', '')}</td><td>{d.get('null', '')}</td></tr>"
    comp_html = ("<table class='lines'><tr><th>Field</th><th>Known</th><th>Null / abstained</th></tr>" +
                 "".join(frow(f"books.{k}", d) for k, d in (cp.get("books") or {}).items() if isinstance(d, dict) and "known" in d) +
                 "".join(frow(f"items.{k}", d) for k, d in (cp.get("items") or {}).items() if isinstance(d, dict) and "known" in d) + "</table>" +
                 "<table class='lines'><tr><th>Room / total</th><th>Value</th><th>State</th></tr>" +
                 "".join(f"<tr><th>{E(k)}</th><td>{v(d.get('value'))}</td><td>{E(d.get('state') or 'value')}{(' - ' + E(d['why'])) if d.get('why') else ''}</td></tr>"
                         for k, d in {**(cp.get('room') or {}), **(cp.get('totals') or {})}.items()) + "</table>") if cp else "<p class=small>n/a</p>"
    stages = "".join(f"<tr><td>{E(s)}</td><td>{E(st['status'])}</td><td>{v(st.get('elapsed_s'))}</td><td>{E(str(st.get('detail') or ''))}</td></tr>"
                     for s, st in p.get("stage_status", {}).items())
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Provisional inventory {E(sweep['id'])}</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#1b1b1b;background:#fff}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;margin:8px 0}} th,td{{border:1px solid #ddd;padding:4px 6px;vertical-align:top;text-align:left}}
th{{background:#f4f4f4}} .unk{{color:#999;font-style:italic}} .small,small{{color:#666;font-size:11px}}
.banner{{padding:8px 12px;margin:8px 0;border-radius:4px;font-weight:600}} .prov{{background:#fff3cd;border:1px solid #e0b100}}
.fixture{{background:#f8d7da;border:1px solid #c0392b}} .thumb{{position:relative;display:inline-block;width:240px}}
.thumb img{{width:100%;display:block}} .box{{position:absolute;border:2px solid #e11;box-sizing:border-box}}
.sev-blocking td{{background:#fdeaea}} .sev-warning td{{background:#fffbea}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style>
<h1>Library contents inventory - PROVISIONAL</h1>{banner}
<p>Run <b>{E(sweep['id'])}</b> &middot; source <b>{v(sweep.get('source_video') or sweep.get('source'))}</b> &middot; duration {v(sweep['duration_s'])}s &middot;
captured {v(sweep['captured_at'])} &middot; device {v(sweep['device'])} &middot; country {v(sweep['country'])} &middot; currency {v(sweep['currency'])}</p>
<p class="small">Generated from claim_packet.json only. Source video sha256 {v(sweep.get('source_video_sha256'))}.</p>
<div class="grid"><div><h2>Totals (computed in code from the lines)</h2><table>{kv(totals_rows)}</table></div>
<div><h2>Room</h2><table>{kv(room_rows)}</table><p class="small">{E(room.get('evidence_audit') or '')}</p></div></div>
<h2>Capture gaps and missing evidence</h2><ul>{gaps}</ul>
<h2>Review queue ({len(p['review_queue'])} lines, {n_details} details)</h2><p class="small">One entry per line or topic, most severe first; every detail is kept under it.</p><table><tr><th>Ref</th><th>Severity</th><th>Code</th><th>Reason</th></tr>{review}</table>
<h2>Books ({len(p['books'])})</h2><table><tr>{head(book_cols)}</tr>{_rows(p['books'], book_cols)}</table>
{second_country_html(p.get("second_country"))}
<h2>Non-book items ({len(p['items'])})</h2><table><tr>{head(item_cols)}</tr>{_rows(p['items'], item_cols)}</table>
<h2>Calibration inputs and measurement scope</h2>{meas_html}
<h2>Per-language breakdown (es / en)</h2>{lang_html}
<h2>Completeness: known vs null / abstained</h2>{comp_html}
<h2>Excluded by user ({len(p.get('excluded_by_user', []))})</h2>{excl}
<h2>Provenance of results</h2><table>{prov}</table>
<h2>Stage status</h2><table><tr><th>Stage</th><th>Status</th><th>Elapsed (s)</th><th>Detail</th></tr>{stages}</table>
</html>"""


def render_pdf(html_path: str | Path, pdf_path: str | Path, timeout_s: float = 90.0) -> dict[str, Any]:
    """Best-effort PDF through a locally installed headless Chrome/Chromium. Never required; failure is reported, not hidden."""
    import shutil
    import subprocess
    html_path, pdf_path = Path(html_path).resolve(), Path(pdf_path).resolve()
    candidates = [shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser", "chrome")] + [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Chromium.app/Contents/MacOS/Chromium"]
    chrome = next((c for c in candidates if c and Path(c).exists()), None)
    if not chrome:
        return {"status": "unavailable", "reason": "no Chrome/Chromium found; open report.html and print to PDF"}
    t0 = __import__("time").perf_counter()
    try:
        r = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                           capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"status": "unavailable", "reason": f"PDF render exceeded {timeout_s:.0f}s"}
    ok = pdf_path.is_file() and pdf_path.stat().st_size > 1000
    return {"status": "ok" if ok else "unavailable", "path": pdf_path.name if ok else None, "bytes": pdf_path.stat().st_size if ok else None,
            "elapsed_s": round(__import__("time").perf_counter() - t0, 2), "reason": None if ok else (r.stderr.strip()[-200:] or "no output")}


def render_print_pdf(packet_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """PDF from an image-free print variant (evidence cited by path + hash index) so the PDF stays small."""
    run_dir = Path(run_dir)
    tmp = run_dir / "report_print.html"
    tmp.write_text(render_html(C.read_json(packet_path), thumbs=False), encoding="utf-8")
    try:
        return render_pdf(tmp, run_dir / "report.pdf")
    finally:
        tmp.unlink(missing_ok=True)


def render(packet_path: str | Path, run_dir: str | Path, config: dict[str, Any] | None = None) -> Path:
    env = C.read_json(packet_path)
    out = Path(run_dir) / C.ARTIFACTS["report"]
    out.write_text(render_html(env), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render report.html from claim_packet.json")
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    print(render(Path(a.run) / C.ARTIFACTS["packet"], a.run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
