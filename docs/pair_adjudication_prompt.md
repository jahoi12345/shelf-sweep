# Blinded duplicate-pair adjudication

Send only the individual PNG sheets or contact-sheet PNGs. Do not send
`pairs.json`, `pairs_key.json`, inventory titles, grouping proposals, scores,
previous model answers, or evaluation summaries. Contact sheets contain up to
12 pairs. Use the individual sheet when a contact panel is too small.

Paste this exact text with each sheet/page:

```text
For each pair ID in the attached image, decide whether panels A and B show the
same physical book. Each panel contains a spine crop and a cutout of its saved
frame; the yellow outline marks the spine to compare. Frame IDs and seconds
identify the sightings. Compare visible spine markings, wear, shape and nearby
books. Two copies of the same title are different physical books. Matching
title text alone is insufficient. Use cannot_tell if the image does not let you
decide confidently; do not guess or assume the camera followed the same book.

Return one JSON object per line for every pair shown, with no Markdown fences
or other text. Use exactly these fields:
{"pair_id":"pair-00001","answer":"same","reason":"One short line explaining the visible evidence."}

The answer must be exactly same, different, or cannot_tell. Copy each pair ID
exactly. Do not infer an answer from the pair order or frame timing. Do not
provide catalogue titles, confidence scores, counts or corrections.
```

## Generate and import

```sh
nice .venv/bin/python -m pipeline.pair_sheets
# Optional: all three named runs (11, 7, 5); run once in a fresh output directory
nice .venv/bin/python -m pipeline.pair_sheets --all-defaults

.venv/bin/python -m app.pair_labels import \
  --run sweep_3af346a2f3f14f72 --answers answers.jsonl \
  --annotator chatgpt-MODEL-VERSION --kind model

.venv/bin/python -m app.pair_labels import \
  --run sweep_3af346a2f3f14f72 --answers human_spot_check.jsonl \
  --annotator human-NAME --kind human

.venv/bin/python -m app.pair_labels summary --run sweep_3af346a2f3f14f72
```

Replace `MODEL-VERSION` with the actual external model name/version. Import only
the answers that annotator actually gave. Human files use the same JSONL format
and contain only pairs independently checked by that human. A blank response
is not a human check. Keep model answers hidden during the human check.

Files are stored in `runs/<run>/pair_labels/<annotator>.json`, with annotator,
kind, import timestamp, answer-file hash, sheet hashes and manifest/key hashes.
Same-answer reimports are idempotent; conflicting reimports require a new
annotator/version name. Nothing is written into inventory or packet artifacts.
The summary is `runs/<run>/pair_labels_summary.md`. Never replace existing
sheet sets in place: labels refer to their specific bytes.

The Label page's Pairs sub-view lets you browse/export sheets and paste JSONL
for import. It hides proposed groups, source kinds, control kinds and previous
answers while browsing. Results are available separately via
`/api/pairs/<run>/summary` (and a Summary link).

## Evaluation limits

The fixed seed is 20261007. One pair represents each unordered group
combination from the union of audit links, all saved panorama versions' merge
evidence, and uncertain associations between book groups. All source links are
retained in the private key. The audit's two confirmed duplicate examples retain
their exact observation IDs. Control pairs are mixed into the same random order,
with randomly assigned sides and identical image formatting; no titles, groups,
tiers, source tags or scores are printed on any sheet.

Approximately 20% of the final set are different-book controls backed by
adjacent, non-overlapping same-frame detections, shown using separate-frame
sightings where possible. Approximately 10% are same-book controls backed by
two sightings of one proposed group. These are **automatic proxy expectations**:
detector/grouping errors can invalidate them. Agreement with them must not be
described as independently measured accuracy. The remaining pairs are sampled
hypotheses, not known duplicates.

Native crops preserve saved source pixels (with an explicit quarter-turn when
needed to put the long axis upright). Sheets display crops up to 300 pixels
tall without upscaling; larger originals are retained in `native_crops/`.
Context uses up to three detected neighbours on each side in the same row,
outlined in the saved frame. Upright means geometric spine orientation, not
OCR-inferred text direction; the adjudicator may rotate the image to read text.

Model labels are **model-read, never truth**. Only human-checked overlap may
back a claim about that checked subset. Summaries retain abstentions in the
answered denominator and report decisive agreement separately. Conflicting
human answers are excluded from the consensus reference. A convenient human
spot check does not establish whole-run accuracy or a corrected physical-book
count. No pipeline, engine, packet or dedup stage reads imported labels.
