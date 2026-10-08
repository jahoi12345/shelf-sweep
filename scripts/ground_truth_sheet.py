"""Ground truth and results workbook for one sweep (ground_truth/ground_truth_and_results.xlsx).

    python scripts/ground_truth_sheet.py runs/<sweep>/claim_packet.json ground_truth/ground_truth_and_results.xlsx \
        [--labels ground_truth/ground_truth_labels.json]

System numbers come from the claim packet (never edited). The yellow input cells hold the hand-collected ground truth:
  * from the labels file written by the /truth page (titles, non-book items, the 15 price checks; one entry per sweep id), and
  * carried over from the existing workbook at OUT (same sweep only) for every yellow cell the labels file does not answer, so values typed
    straight into the sheet (spines, room tape, prices) survive a rebuild.
`build()` also returns a Python preview of the Results tab, computed the same way as its formulas.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

DEFAULT_TRUE_BOOK_COUNT = 458      # the user's hand count of the whole Yugo shelf (entered by the user, 2026-10-07)
EXTRA_ROWS = 30                    # blank rows for missed spines / missed items (more when the labels list more)
F = 'Arial'
H = Font(name=F, bold=True); N = Font(name=F); BLUE = Font(name=F, color='0000FF'); B = Font(name=F, bold=True, size=13)
GREEN = Font(name=F, color='008000'); IT = Font(name=F, italic=True, color='555555')
YEL = PatternFill('solid', fgColor='FFFF00'); HDR = PatternFill('solid', fgColor='DDDDDD')
wrap = Alignment(wrap_text=True, vertical='top')


# ------------------------------------------------------------------------------------------------ samples (shared with the /truth page)
def sorted_books(data: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(data['books'], key=lambda b: b.get('display_index') or 0)


def spine_sample(books: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """20 lines evenly spaced (every k-th in shelf order) over the lines the system measured."""
    meas = [b for b in books if b.get('spine_height_cm') is not None and b.get('spine_thickness_cm') is not None]
    return ([meas[round(i * (len(meas) - 1) / 19)] for i in range(20)] if len(meas) >= 20 else meas), len(meas)


def price_sample(books: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """15 books evenly spaced (every k-th in shelf order) over the priced lines."""
    priced = [b for b in books if (b.get('replacement_cost') or {}).get('amount') is not None]
    return ([priced[round(i * (len(priced) - 1) / 14)] for i in range(15)] if len(priced) >= 15 else priced), len(priced)


# ------------------------------------------------------------------------------------------------ inputs: labels file + existing workbook
def load_labels(path: str | Path | None, sweep_id: str) -> dict[str, Any]:
    if not path or not Path(path).is_file():
        return {}
    doc = json.loads(Path(path).read_text())
    return ((doc.get('sweeps') or {}).get(sweep_id)) or {}


def _is_yellow(c) -> bool:
    try:
        return c.fill is not None and c.fill.fill_type == 'solid' and str(c.fill.fgColor.rgb).upper().endswith('FFFF00')
    except AttributeError:
        return False


def read_existing(path: str | Path | None, sweep_id: str) -> dict[str, Any]:
    """Yellow-cell values already in the workbook at `path`, keyed so a rebuild can put them back: (tab, row key, column). Row key = the line /
    item id (column B) on the per-line tabs, the measure label (column A) on Book count / Room, ('extra', n) for the missed-spine / missed-item
    rows. Only for a workbook built from the same sweep (its Read me names the sweep id)."""
    out: dict[str, Any] = {}
    if not path or not Path(path).is_file():
        return out
    try:
        wb = load_workbook(path)
    except Exception:            # not a workbook we can read: nothing to carry
        return out
    if 'Read me' not in wb.sheetnames or sweep_id not in str(wb['Read me']['A2'].value or ''):
        return out
    for tab in ('Book count', 'Titles', 'Spines', 'Prices', 'Items', 'Room'):
        if tab not in wb.sheetnames:
            continue
        ws = wb[tab]; extra = -1; last_id = None
        for row in ws.iter_rows(min_row=1):
            vals = [c for c in row if _is_yellow(c) and c.value not in (None, '')]
            rid = row[1].value if len(row) > 1 else None
            if tab in ('Book count', 'Room') or (tab == 'Items' and row[0].row == 1):
                key = str(row[0].value)
            elif isinstance(rid, str) and rid and rid != 'Item id' and rid != 'Line id':
                key = rid; last_id = rid
            elif any(_is_yellow(c) for c in row) and last_id is not None:
                extra += 1; key = ('extra', extra)
            else:
                continue
            if tab in ('Titles', 'Items') and isinstance(key, tuple):
                cols = (3, 4, 8, 9, 12) if tab == 'Titles' else (3, 8, 10)
                m = {c.column: c.value for c in row if c.value not in (None, '') and c.column in cols}
                if m.get(8) not in (None, ''):          # a missed spine / item row is one with a title / category
                    out.setdefault((tab, 'extra_rows'), []).append(m)
            for c in vals:
                out[(tab, key, c.column)] = c.value
    return out


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v in (None, ''):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ the workbook
def build(packet_path: str | Path, out_path: str | Path, labels: dict[str, Any] | None = None,
          carry_from: str | Path | None = None) -> dict[str, Any]:
    """Write the workbook atomically to out_path. labels = this sweep's entry of the labels file (or {}); carry_from = an earlier workbook whose
    yellow cells are kept where the labels say nothing (default: out_path itself). Returns counts and the Results preview."""
    d = json.loads(Path(packet_path).read_text())['data']
    books = sorted_books(d)
    items = d['items']; t = d['totals']; sweep = d['sweep']
    labels = labels or {}
    carry = read_existing(out_path if carry_from is None else carry_from, sweep['id'])
    L_titles = labels.get('titles') or {}; L_items = labels.get('items') or {}; L_prices = labels.get('prices') or {}

    def kept(tab, key, col):
        return carry.get((tab, key, col))

    wb = Workbook()

    def header(ws, row, cols, widths=None):
        for i, c in enumerate(cols, 1):
            x = ws.cell(row=row, column=i, value=c); x.font = H; x.fill = HDR; x.alignment = wrap
        if widths:
            for i, w in enumerate(widths, 1): ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w

    def put(ws, r, c, v, font=N, fill=None, fmt=None):
        x = ws.cell(row=r, column=c, value=v); x.font = font
        if fill: x.fill = fill
        if fmt: x.number_format = fmt
        return x

    def inp(ws, r, c, v, fmt=None):                     # a yellow ground-truth input cell
        return put(ws, r, c, None if v in (None, '') else v, BLUE, YEL, fmt)

    yn = DataValidation(type='list', formula1='"Y,N"', allow_blank=True)

    # ---------------- Read me
    rm = wb.active; rm.title = 'Read me'
    lines = [
     ('Ground truth and results — demo-video sweep', B),
     (f"Sweep {sweep['id']} · captured {sweep['captured_at']} · {sweep['duration_s']} s · {sweep.get('country')} / {sweep.get('currency')}", N),
     ('System numbers come from claim_packet.json of that sweep (not edited). Ground truth is entered by hand.', N),
     ('', N),
     ('How to use', H),
     ('Yellow cells with blue text are hand-entered ground truth. Everything else is the system or a formula.', N),
     ('Blank yellow cell = not collected. The Results tab then says NOT EVALUATED for that bar instead of inventing a value.', N),
     ('Titles: type the title you can read on the spine (blank if not legible) and Y/N whether the system title is correct.', N),
     ('Spines / Prices: enter your tape-measured cm or hand-checked price (same source type) next to the system value.', N),
     ('Items: mark each system item Y/N (real non-book item, correct category); add items the system missed; enter the true item total.', N),
     ('Room: enter tape-measured length, width and ceiling height in metres; floor and wall areas are computed.', N),
     ('Titles, items and price checks can also be entered on http://127.0.0.1:4182/truth (saved to ground_truth/ground_truth_labels.json); '
      '"Write to sheet" there rebuilds this file and keeps every value typed here that the page does not answer.', N),
     ('', N),
     ('Example of the expected format (not data)', H),
     ('Spines row: system height 21.3, system thickness 2.4 → you enter hand height 20.8, hand thickness 2.6.', IT),
     ('Prices row: system $15.97 (AbeBooks US, New) → you enter 14.99 from the same kind of listing, with its URL and date.', IT),
     ('', N),
     ('Assumptions (stated where the specification is ambiguous)', H),
     ('Spine dimensions and prices pass when the MEAN absolute error over the filled sample is within the bar (15% / 25%).', N),
     ('Title rate = correct system titles / hand-legible spines; confidently wrong = identified lines marked N / hand-legible spines.', N),
     ('Spine and price samples are chosen evenly across the shelf order (every k-th measured/priced line), not picked.', N),
     ('Wall area = gross area of four walls, 2 × (length + width) × height; openings not subtracted (same as the system).', N),
    ]
    for i, (s, f) in enumerate(lines, 1): put(rm, i, 1, s, f)
    rm.column_dimensions['A'].width = 130

    # ---------------- Book count
    bc = wb.create_sheet('Book count')
    header(bc, 1, ['Measure', 'Value', 'Source'], [34, 14, 90])
    true_count = labels.get('true_book_count') if _num(labels.get('true_book_count')) is not None else kept('Book count', 'True book count (hand count)', 2)
    true_count = DEFAULT_TRUE_BOOK_COUNT if true_count is None else true_count
    put(bc, 2, 1, 'True book count (hand count)'); inp(bc, 2, 2, true_count); put(bc, 2, 3, "The user's hand count of the whole Yugo shelf (entered by the user, 2026-10-07)")
    put(bc, 3, 1, 'System headline count'); put(bc, 3, 2, t['book_count']); put(bc, 3, 3, f"claim_packet totals.book_count ({t.get('headline_estimator')}; range {t.get('book_count_range')})")
    put(bc, 4, 1, 'System book lines listed'); put(bc, 4, 2, t.get('book_lines')); put(bc, 4, 3, 'totals.book_lines (some may be duplicates)')
    put(bc, 5, 1, 'Error vs truth'); put(bc, 5, 2, '=(B3-B2)/B2', fmt='0.0%'); put(bc, 5, 3, 'headline count')

    # ---------------- Titles
    ti = wb.create_sheet('Titles')
    header(ti, 1, ['#', 'Line id', 'Frame', 'System status', 'System title', 'System author', 'System id confidence',
                   'Hand-read title (blank = not legible)', 'System title correct? (Y/N)', 'Correct (formula)', 'Confidently wrong (formula)', 'Notes'],
           [6, 18, 20, 14, 40, 26, 12, 40, 14, 12, 14, 30])
    ti.add_data_validation(yn)
    title_rows: list[tuple[str, str, str]] = []          # (hand title, Y/N, system status) for the preview
    r = 2
    for b in books:
        put(ti, r, 1, b.get('display_index')); put(ti, r, 2, b['id']); put(ti, r, 3, b.get('frame_ref')); put(ti, r, 4, b['status'])
        put(ti, r, 5, b.get('title')); put(ti, r, 6, b.get('author')); put(ti, r, 7, b.get('id_confidence'))
        lab = L_titles.get(b['id'])
        if lab is not None:
            hand = '' if lab.get('legible') is False else (lab.get('hand_title') or '')
            ok = (lab.get('correct') or '') if hand else ''
        else:
            hand = kept('Titles', b['id'], 8) or ''; ok = kept('Titles', b['id'], 9) or ''
        inp(ti, r, 8, hand); inp(ti, r, 9, ok); yn.add(f'I{r}')
        put(ti, r, 10, f'=IF(AND(H{r}<>"",I{r}="Y"),1,0)'); put(ti, r, 11, f'=IF(AND(H{r}<>"",I{r}="N",D{r}="identified"),1,0)')
        title_rows.append((str(hand), str(ok), b['status'])); r += 1
    missed = [m for m in (labels.get('missed_titles') or []) if (m or {}).get('title')]
    rows_missed = [{3: m.get('frame') or None, 4: 'missed', 8: m['title'], 9: 'N', 12: m.get('notes') or None} for m in missed]
    if 'missed_titles' not in labels:                    # the page never touched this list: keep what was typed in the sheet
        rows_missed = carry.get(('Titles', 'extra_rows'), [])
    put(ti, r + 1, 1, 'Legible spines the system did not list: add rows below with status "missed", the hand-read title, and N.', IT)
    n_extra = max(EXTRA_ROWS, len(rows_missed) + 10)
    for j, k in enumerate(range(r + 2, r + 2 + n_extra)):
        m = rows_missed[j] if j < len(rows_missed) else {}
        if m.get(3): put(ti, k, 3, m[3])
        inp(ti, k, 4, m.get(4)); inp(ti, k, 8, m.get(8)); inp(ti, k, 9, m.get(9)); yn.add(f'I{k}')
        if m.get(12): put(ti, k, 12, m[12])
        put(ti, k, 10, f'=IF(AND(H{k}<>"",I{k}="Y"),1,0)'); put(ti, k, 11, f'=IF(AND(H{k}<>"",I{k}="N",D{k}="identified"),1,0)')
        if m.get(8): title_rows.append((str(m[8]), str(m.get(9) or ''), str(m.get(4) or '')))
    TEND = r + 1 + n_extra

    # ---------------- Spines (20, evenly spaced over measured lines)
    sp = wb.create_sheet('Spines')
    pick, n_meas = spine_sample(books)
    header(sp, 1, ['#', 'Line id', 'Frame', 'System title', 'System height cm', 'System thickness cm', 'Hand height cm', 'Hand thickness cm',
                   'Height error', 'Thickness error'], [6, 18, 20, 36, 12, 12, 12, 12, 12, 12])
    spine_vals = []
    for i, b in enumerate(pick, 2):
        put(sp, i, 1, b.get('display_index')); put(sp, i, 2, b['id']); put(sp, i, 3, b.get('frame_ref')); put(sp, i, 4, b.get('title'))
        put(sp, i, 5, b['spine_height_cm'], fmt='0.0'); put(sp, i, 6, b['spine_thickness_cm'], fmt='0.0')
        hh, ht = kept('Spines', b['id'], 7), kept('Spines', b['id'], 8)
        inp(sp, i, 7, hh, '0.0'); inp(sp, i, 8, ht, '0.0')
        put(sp, i, 9, f'=IF(G{i}="","",ABS(E{i}-G{i})/G{i})', fmt='0.0%'); put(sp, i, 10, f'=IF(H{i}="","",ABS(F{i}-H{i})/H{i})', fmt='0.0%')
        spine_vals.append((b['spine_height_cm'], b['spine_thickness_cm'], _num(hh), _num(ht)))
    SPEND = 1 + len(pick)
    put(sp, SPEND + 2, 1, f'Sample: 20 of the {n_meas} lines the system measured, every k-th in shelf order. System scale: {(d.get("measurement_evidence") or {}).get("scale", {}).get("calibrations", [{}])[0].get("kind")}.', IT)

    # ---------------- Prices (15, evenly spaced over priced lines)
    pr = wb.create_sheet('Prices')
    pp, n_priced = price_sample(books)
    header(pr, 1, ['#', 'Line id', 'Title', 'ISBN', 'System replacement (USD)', 'Converted?', 'System source', 'System URL', 'Retrieved',
                   'Hand-checked price (USD, same source type)', 'Hand URL', 'Hand date', 'Error'], [6, 18, 32, 16, 12, 10, 26, 40, 22, 14, 34, 12, 10])
    price_vals = []
    for i, b in enumerate(pp, 2):
        rc = b['replacement_cost']
        put(pr, i, 1, b.get('display_index')); put(pr, i, 2, b['id']); put(pr, i, 3, b.get('title')); put(pr, i, 4, rc.get('isbn') or b.get('isbn'))
        put(pr, i, 5, rc['amount'], fmt='$#,##0.00'); put(pr, i, 6, 'yes' if rc.get('converted') else 'no'); put(pr, i, 7, rc.get('source'))
        put(pr, i, 8, rc.get('url')); put(pr, i, 9, rc.get('retrieved_at'))
        lab = L_prices.get(b['id'])
        if lab is not None:
            hp, hu, hd = _num(lab.get('price')), lab.get('url') or None, lab.get('date') or None
        else:
            hp, hu, hd = kept('Prices', b['id'], 10), kept('Prices', b['id'], 11), kept('Prices', b['id'], 12)
        inp(pr, i, 10, hp, '$#,##0.00'); inp(pr, i, 11, hu); inp(pr, i, 12, hd)
        put(pr, i, 13, f'=IF(J{i}="","",ABS(E{i}-J{i})/J{i})', fmt='0.0%')
        price_vals.append((rc['amount'], _num(hp)))
    PREND = 1 + len(pp)
    put(pr, PREND + 2, 1, f'Sample: 15 of the {n_priced} priced books, every k-th in shelf order. Replacement cost = lowest validated new listing (see packet).', IT)

    # ---------------- Items
    im = wb.create_sheet('Items')
    true_items = labels.get('true_item_count') if 'true_item_count' in labels else kept('Items', 'True number of non-book items in the room (hand count)', 5)
    put(im, 1, 1, 'True number of non-book items in the room (hand count)', H); inp(im, 1, 5, true_items)
    header(im, 3, ['#', 'Item id', 'Frame', 'System category', 'System status', 'System confidence', 'Real non-book item? (Y/N)', 'True category',
                   'Category correct? (Y/N)', 'Notes'], [6, 18, 20, 18, 16, 12, 14, 20, 14, 30])
    im.add_data_validation(yn)
    item_vals = []
    for i, it in enumerate(items, 4):
        put(im, i, 1, it.get('display_index')); put(im, i, 2, it['id']); put(im, i, 3, it.get('frame_ref')); put(im, i, 4, it.get('category'))
        put(im, i, 5, it.get('status')); put(im, i, 6, it.get('confidence'), fmt='0.00')
        lab = L_items.get(it['id'])
        vals = ([lab.get('real') or None, lab.get('true_category') or None, lab.get('category_correct') or None] if lab is not None
                else [kept('Items', it['id'], c) for c in (7, 8, 9)])
        for c, v in zip((7, 8, 9), vals): inp(im, i, c, v)
        yn.add(f'G{i}'); yn.add(f'I{i}')
        item_vals.append((vals[0], vals[2]))
    IEND = 3 + len(items)
    put(im, IEND + 2, 1, 'Items the system missed: add them below (true category only).', IT)
    m_items = [m for m in (labels.get('missed_items') or []) if (m or {}).get('category')]
    rows_mi = [{3: m.get('frame') or None, 8: m['category'], 10: m.get('notes') or None} for m in m_items]
    if 'missed_items' not in labels:
        rows_mi = carry.get(('Items', 'extra_rows'), [])
    n_mi = max(EXTRA_ROWS, len(rows_mi) + 10)
    for j, k in enumerate(range(IEND + 3, IEND + 3 + n_mi)):
        m = rows_mi[j] if j < len(rows_mi) else {}
        if m.get(3): put(im, k, 3, m[3])
        inp(im, k, 8, m.get(8))
        if m.get(10): put(im, k, 10, m[10])

    # ---------------- Room
    ro = wb.create_sheet('Room')
    header(ro, 1, ['Measure', 'Tape measurement', 'System', 'Error'], [34, 18, 18, 12])
    room = d['room']
    rows = [('Length m', room.get('length_m')), ('Width m', room.get('width_m')), ('Ceiling height m', room.get('height_m'))]
    tape = {}
    for i, (k, v) in enumerate(rows, 2):
        tape[k] = kept('Room', k, 2)
        put(ro, i, 1, k); inp(ro, i, 2, tape[k], '0.00'); put(ro, i, 3, v, fmt='0.00')
        put(ro, i, 4, f'=IF(OR(B{i}="",C{i}=""),"",ABS(C{i}-B{i})/B{i})', fmt='0.0%')
    put(ro, 5, 1, 'Floor area m²'); put(ro, 5, 2, '=IF(OR(B2="",B3=""),"",B2*B3)', fmt='0.00'); put(ro, 5, 3, room.get('floor_area_m2'), fmt='0.00')
    put(ro, 6, 1, 'Wall area m² (gross)'); put(ro, 6, 2, '=IF(OR(B2="",B3="",B4=""),"",2*(B2+B3)*B4)', fmt='0.00'); put(ro, 6, 3, room.get('wall_area_m2'), fmt='0.00')
    put(ro, 7, 1, 'Shelved wall area m²'); inp(ro, 7, 2, kept('Room', 'Shelved wall area m²', 2), '0.00'); put(ro, 7, 3, room.get('shelved_wall_area_m2'), fmt='0.00')
    for i in (5, 6, 7): put(ro, i, 4, f'=IF(OR(B{i}="",C{i}=""),"",ABS(C{i}-B{i})/B{i})', fmt='0.0%')
    put(ro, 9, 1, f"System room status: {room.get('status')} — the system left every room figure empty rather than guess (see failure log).", IT)

    # ---------------- Results (first tab after Read me)
    rs = wb.create_sheet('Results', 1)
    header(rs, 1, ['Measure', 'Pass bar', 'Ground truth', 'System (this sweep)', 'Score', 'Result', 'How it is scored'], [26, 30, 16, 18, 12, 22, 70])
    NE = '"NOT EVALUATED (no ground truth)"'
    T = f"Titles!$H$2:$H${TEND}"
    rows = [
     ('Book count', 'Within 5% of the true count', "='Book count'!B2", "='Book count'!B3", '=ABS(D2-C2)/C2', '=IF(C2="",' + NE + ',IF(E2<=0.05,"PASS","FAIL"))', 'abs(system − truth) / truth'),
     ('Title identification', '≥ 70% of legible spines correct; ≤ 3% confidently wrong', f'=COUNTA({T})', f'=SUM(Titles!$J$2:$J${TEND})',
      '=IF(C3=0,"",D3/C3)', f'=IF(C3=0,{NE},IF(AND(E3>=0.7,SUM(Titles!$K$2:$K${TEND})/C3<=0.03),"PASS","FAIL"))',
      'C = hand-legible spines, D = correct system titles; confidently wrong = identified lines marked N'),
     ('Spine dimensions', 'Within 15% on a sample of 20', f'=COUNT(Spines!G2:G{SPEND})', f'=COUNT(Spines!E2:E{SPEND})',
      f'=IF(C4=0,"",MAX(AVERAGE(Spines!I2:I{SPEND}),AVERAGE(Spines!J2:J{SPEND})))', '=IF(C4=0,' + NE + ',IF(E4<=0.15,"PASS","FAIL"))',
      'C = hand-measured spines, D = system-measured; score = worse of mean height / thickness error'),
     ('Book prices', 'Within 25% on a sample of 15', f'=COUNT(Prices!J2:J{PREND})', f'=COUNT(Prices!E2:E{PREND})',
      f'=IF(C5=0,"",AVERAGE(Prices!M2:M{PREND}))', '=IF(C5=0,' + NE + ',IF(E5<=0.25,"PASS","FAIL"))', 'C = hand-checked prices; score = mean absolute error'),
     ('Non-book items', '≥ 80% found, correct category', '=Items!E1', f'=COUNTIFS(Items!G4:G{IEND},"Y",Items!I4:I{IEND},"Y")',
      '=IF(OR(C6="",C6=0),"",D6/C6)', '=IF(OR(C6="",C6=0),' + NE + ',IF(E6>=0.8,"PASS","FAIL"))', 'C = true item count, D = system items that are real with the right category'),
     ('Floor area', 'Within 10% of tape', '=Room!B5', '=IF(Room!C5="","abstained",Room!C5)', '=IF(OR(C7="",D7="abstained"),"",ABS(D7-C7)/C7)',
      '=IF(C7="",' + NE + ',IF(D7="abstained","FAIL (system abstained)",IF(E7<=0.1,"PASS","FAIL")))', 'tape length × width'),
     ('Wall area', 'Within 15% of tape', '=Room!B6', '=IF(Room!C6="","abstained",Room!C6)', '=IF(OR(C8="",D8="abstained"),"",ABS(D8-C8)/C8)',
      '=IF(C8="",' + NE + ',IF(D8="abstained","FAIL (system abstained)",IF(E8<=0.15,"PASS","FAIL")))', '2 × (length + width) × height'),
    ]
    ft = sweep.get('finalization_timings') or {}
    ttp = round(sum(v for v in ft.values() if isinstance(v, (int, float))), 1)
    rows.append(('Time to packet', 'Under 5 minutes after the sweep ends', 300, ttp, '=D9', '=IF(D9<C9,"PASS","FAIL")', 'seconds from End to packet + report (sum of finalization_timings in the packet); C = the 300 s limit'))
    for i, row in enumerate(rows, 2):
        for c, v in enumerate(row, 1):
            x = put(rs, i, c, v); x.alignment = wrap
            if c in (3,) and isinstance(v, str) and v.startswith('='): x.font = GREEN
        rs.cell(row=i, column=5).number_format = '0.0%'
    rs.cell(row=9, column=5).number_format = '0.0'
    put(rs, 11, 1, f"System figures from {sweep['id']} claim_packet.json. Book count truth = the user's hand count; every other truth cell links to a yellow input on its tab and reads NOT EVALUATED while empty.", IT)
    put(rs, 12, 1, f"Other system figures: {t.get('books_identified')} books identified, {t.get('books_unidentified')} unidentified; {n_priced} books priced, replacement ${t.get('books_replacement_cost')}, used ${t.get('books_used_value')}; {len(items)} non-book items, all needs_appraisal; shelf run {t.get('shelf_run_m')}.", IT)
    for ws in wb.worksheets:
        ws.freeze_panes = 'A2' if ws.title not in ('Read me', 'Items') else ws.freeze_panes
    wb['Items'].freeze_panes = 'A4'

    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + out.name + '.', suffix='.tmp', dir=out.parent); os.close(fd)
    try:
        wb.save(tmp); os.chmod(tmp, 0o644); os.replace(tmp, out)      # mkstemp makes 0600; a workbook is an ordinary repo file
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

    preview = results_preview(true_count=_num(true_count), system_count=t['book_count'], titles=title_rows, spines=spine_vals,
                              prices=price_vals, true_items=_num(true_items), items=item_vals,
                              tape=[_num(tape['Length m']), _num(tape['Width m']), _num(tape['Ceiling height m'])],
                              room_floor=_num(room.get('floor_area_m2')), room_wall=_num(room.get('wall_area_m2')), ttp=ttp)
    return {'sweep': sweep['id'], 'out': str(out), 'titles': len(books), 'missed_titles': len(rows_missed), 'spines': len(pick), 'measured': n_meas,
            'prices': len(pp), 'priced': n_priced, 'items': len(items), 'missed_items': len(rows_mi), 'ttp': ttp, 'results': preview}


# ------------------------------------------------------------------------------------------------ Results preview (same rules as the formulas)
NOT_EVAL = 'NOT EVALUATED (no ground truth)'


def _bar(measure, bar, truth, system, score, result, note=None):
    return {'measure': measure, 'pass_bar': bar, 'ground_truth': truth, 'system': system, 'score': score, 'result': result, 'note': note}


def results_preview(*, true_count, system_count, titles, spines, prices, true_items, items, tape, room_floor, room_wall, ttp) -> list[dict]:
    out = []
    if true_count:
        e = abs(system_count - true_count) / true_count
        out.append(_bar('Book count', 'Within 5% of the true count', true_count, system_count, e, 'PASS' if e <= 0.05 else 'FAIL'))
    else:
        out.append(_bar('Book count', 'Within 5% of the true count', true_count, system_count, None, NOT_EVAL))
    legible = [x for x in titles if x[0] != '']
    correct = sum(1 for h, ok, _ in legible if ok == 'Y')
    wrong = sum(1 for h, ok, st in legible if ok == 'N' and st == 'identified')
    if legible:
        rate = correct / len(legible); wr = wrong / len(legible)
        out.append(_bar('Title identification', '≥ 70% of legible spines correct; ≤ 3% confidently wrong', len(legible), correct, rate,
                        'PASS' if rate >= 0.7 and wr <= 0.03 else 'FAIL', f'confidently wrong {wrong} ({wr:.1%})'))
    else:
        out.append(_bar('Title identification', '≥ 70% of legible spines correct; ≤ 3% confidently wrong', 0, 0, None, NOT_EVAL))
    he = [abs(sh - hh) / hh for sh, st, hh, ht in spines if hh]
    te = [abs(st - ht) / ht for sh, st, hh, ht in spines if ht]
    if not he:
        out.append(_bar('Spine dimensions', 'Within 15% on a sample of 20', 0, len(spines), None, NOT_EVAL))
    elif not te:
        out.append(_bar('Spine dimensions', 'Within 15% on a sample of 20', len(he), len(spines), None, '#DIV/0! in the sheet', 'enter hand thickness too'))
    else:
        sc = max(sum(he) / len(he), sum(te) / len(te))
        out.append(_bar('Spine dimensions', 'Within 15% on a sample of 20', len(he), len(spines), sc, 'PASS' if sc <= 0.15 else 'FAIL'))
    pe = [abs(sp - hp) / hp for sp, hp in prices if hp]
    if pe:
        sc = sum(pe) / len(pe)
        out.append(_bar('Book prices', 'Within 25% on a sample of 15', len(pe), len(prices), sc, 'PASS' if sc <= 0.25 else 'FAIL'))
    else:
        out.append(_bar('Book prices', 'Within 25% on a sample of 15', 0, len(prices), None, NOT_EVAL))
    found = sum(1 for real, cat in items if real == 'Y' and cat == 'Y')
    if true_items:
        sc = found / true_items
        out.append(_bar('Non-book items', '≥ 80% found, correct category', true_items, found, sc, 'PASS' if sc >= 0.8 else 'FAIL'))
    else:
        out.append(_bar('Non-book items', '≥ 80% found, correct category', true_items, found, None, NOT_EVAL))
    L, W, Hh = tape
    for name, bar, lim, truth, sysv in (('Floor area', 'Within 10% of tape', 0.10, L * W if L and W else None, room_floor),
                                        ('Wall area', 'Within 15% of tape', 0.15, 2 * (L + W) * Hh if L and W and Hh else None, room_wall)):
        if truth is None:
            out.append(_bar(name, bar, None, sysv if sysv is not None else 'abstained', None, NOT_EVAL))
        elif sysv is None:
            out.append(_bar(name, bar, truth, 'abstained', None, 'FAIL (system abstained)'))
        else:
            e = abs(sysv - truth) / truth
            out.append(_bar(name, bar, truth, sysv, e, 'PASS' if e <= lim else 'FAIL'))
    out.append(_bar('Time to packet', 'Under 5 minutes after the sweep ends', 300, ttp, ttp, 'PASS' if ttp < 300 else 'FAIL', 'score in seconds'))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('packet'); ap.add_argument('out')
    ap.add_argument('--labels', help='ground_truth/ground_truth_labels.json (the /truth page); this sweep\'s answers fill the yellow cells')
    a = ap.parse_args(argv)
    sweep_id = json.loads(Path(a.packet).read_text())['data']['sweep']['id']
    s = build(a.packet, a.out, load_labels(a.labels, sweep_id))
    print('titles', s['titles'], 'spines', s['spines'], 'of', s['measured'], 'prices', s['prices'], 'of', s['priced'], 'items', s['items'], 'ttp', s['ttp'])
    for r in s['results']:
        print(f"  {r['measure']:<22} {r['result']}" + (f"  score {r['score']:.3f}" if isinstance(r['score'], float) else ''))


if __name__ == '__main__':
    main()
