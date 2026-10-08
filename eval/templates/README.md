# Human annotation protocol v1 — Phase 2 review decision

This is an annotation structure, not ground truth. The existing full-frame spike
predates reviewed labels. Do not infer labels from its output, crops, catalogue,
or model answers. Do not fill `eval/ground_truth.json` from these tools.

1. Annotator opens the original frames in the viewer, draws **every** book (also
   unreadable, leaning, flat, stacked and occluded), every non-book object and
   ambiguous object. Books use `kind=book, category=book`; text-bearing signs,
   decorative letters, plants, pots, frames, shelving and other clutter use
   `kind=non_book` with consistent categories. Occluded objects remain labelled
   if their existence is visible; do not invent invisible books.
2. Read each title independently in native evidence. `title_legible=true` means
   the full title can be read; supply human `accepted_titles`. False means
   unreadable; null means the human cannot determine legibility. Preserve literal
   `visible_text` separately. Edition/ISBN evidence is literal printed wording or
   null. Do not look at predictions during labelling/review.
3. Assign `instance_id` across views based on visible physical location/appearance.
   Two copies of the same title get different IDs. Uncertain association stays
   null, and `physical_instances_complete=false`. Count/association scores remain
   unavailable until this is completed. The annotation subset count is never a
   physical whole-room count.
4. One named human can annotate, self-review and freeze labels: set
   `review_mode=self_reviewed`, annotator name, optional same-name reviewer,
   approval and review/freeze timestamps. An optional second **human** reviewer
   records `review_mode=independent` with a different reviewer name. Review every
   object and missed/clutter region, then set `reviewed=true`. Tools do not declare
   review, approval or freeze on your behalf; declarations must be genuine.
5. Freeze labels before tuning. Keep previous JSON immutable. Scoring records a
   canonical SHA-256 hash and refuses to overwrite results. Pending/model-created
   labels yield unavailable metrics. Never rename a synthetic fixture as human
   labels. Corrections create a new labelled version with provenance.

Prepare a blank viewer from any run manifest:

```bash
python3 -m eval.templates.prepare --manifest runs/<run>/manifest.json --run-dir runs/<run> --out eval/annotations/<run>_pending_v1.json --viewer runs/<run>_review/annotations.html
```

The viewer is a blank annotation aid, contains **no candidate boxes or model text**,
and exports JSON for a named human to review and freeze. It defaults all coverage flags
and review declarations to incomplete. Draw a box, then edit its fields on the
right. Click a box to edit/delete it. Supply physical IDs across frames manually.
Reopen completed JSON using the viewer's import control; a human must verify
boxes/text against frames and update the review fields explicitly.

Scoring definitions (thresholds fixed before tuning):

- IoU ≥0.5, one-to-one class-independent matching that maximizes cardinality then
  total IoU. A pot labelled as a book is a false book positive and a non-book miss.
  Extra overlapping predictions count as duplicates; invalid boxes remain in
  precision's denominator. Every exhaustive reviewed frame is scored, even if
  detection emitted no candidates for it.
- Detection precision = correct book observations / all predicted book observations
  in scored frames; recall = correct book observations / all human book observations.
- Title correctness = correct full titles / human-legible books, including missed
  books and abstentions in the denominator. Literal and catalogue observation
  metrics are separate; final inventory title metrics use physical instances.
- Confidence ≥0.85 defines confident. Confident-wrong = wrong confident titles /
  all human-legible books. Also report correct, incorrect, missed and abstained.
- Book count relative error = |predicted physical count − human physical count| /
  human physical count. Whole-room eligibility also needs complete physical truth.
- Split = human instance associated to >1 predicted instance, denominator all
  human instances; duplicate instance excess = sum(max(0, predicted associations−1))
  / all predicted instances. Merge = prediction associated to >1 human instance /
  all predicted instances. No text-based instance matching. Missed instances do
  not count as correctly deduplicated. Mapping tables expose errors.
- Non-book correct-category coverage = unique human non-book instances found with
  correct category / all human non-book instances. Observation coverage is also
  reported. Clutter FP categories each use all book predictions as denominator.
- Missing/unknown truth or zero denominator is unavailable, never a pass. Assignment
  bars (5%, 70%, 3%, 80%) only receive pass/fail on complete whole-video/physical truth.

The t4/t12 subset spans upper/middle/lower shelf regions, upright/leaning/horizontal
books, foliage/pots, framed text, decorative letters, globe/sculpture and shelf
structure. These are coverage prompts, **not human labels or a count**. Both frames
are from one development video; adjacent views are not an independent test set.
Annotate the whole available video before full-video scores. A different shelf/room
or new uninterrupted take is required for generalization evaluation. Physical
measurements/prices/room truth remain a separate human deliverable.

Phase 2 adds human `language_code` (es/en/mul/null), sample sizes and separate
correctness/abstention/miss/confident-wrong rates for literal, catalogue and physical
inventory outputs. Title comparisons preserve accents in NFC; translated titles
are incorrect unless that exact native visible title is human-labelled. Unknown
language has its own group. Model-predicted language never selects a scoring group.
Optional independent
review is accurately distinguished from sufficient human self-review.
