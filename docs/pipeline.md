# Pipeline

The phone, a browser replay and `app.replay` all feed the same engine (`app/engine.py`). Stages write versioned JSON artifacts into the
run directory (`runs/<run>/`); post-sweep builds write new sibling directories (`runs/<run>_panorama_vN`, `_recon3d_vN`, `_measure_vN`)
and never overwrite earlier ones.

## During the sweep

| Stage | Code | What it does |
| --- | --- | --- |
| Capture guidance | `app/capture_quality.py`, `app/nudge_policy.py` | sharpness, motion and distance per frame; nudges `slow_down`, `recapture`, `too_far`, `revisit`; a capture-quality strip and a count-reliability figure |
| Detection | `providers/local_detector.py`, `pipeline/detect.py` | YOLO oriented boxes for spines (1280 px) plus a YOLO-World pass for non-book objects (640 px), about 2.5 detections per second in the background |
| Association | `pipeline/cross_view.py`, `pipeline/dedup.py`, `pipeline/tracker.py` | follows each spine from frame to frame with tracked motion and anisotropic matching; titles are never used to link |
| Room pan | `app/room_sweep.py`, `pipeline/room_pan.py` | after the shelf pass, a short pan of the room; frames of that pan are never counted as books |

## When the sweep ends

| Stage | Code | What it does |
| --- | --- | --- |
| Reading | `app/group_reader.py`, `providers/readers.py` | the best 1-2 sightings of each physical book are read (local OCR, or Gemini when enabled) within the packet's time budget; skipped groups are listed |
| Resolution | `pipeline/resolve.py`, `providers/openlibrary.py`, `providers/googlebooks.py` | literal spine text is matched to a catalogue work only when the visible words support it; otherwise the line keeps the literal text |
| Pricing | `pipeline/price.py`, `providers/book_prices.py`, `providers/fx.py` | US/USD physical-edition offers with the raw source, date and hash; when no US offer exists, a EUR/GBP offer converted at the ECB rate (labelled); `pipeline/second_country.py` prices the same 10 books in a second country |
| Packet and report | `pipeline/packet.py`, `pipeline/report.py` | `claim_packet.json`, `report.html`, `report.pdf`, `evidence_index.json`; every item is `priced`, `range` or `needs_appraisal`; one review-queue entry per line |

## Post-sweep queue (automatic, `app/post_sweep.py`)

Shown on the watch page in this order. Each step is also a command:

| Step | Command |
| --- | --- |
| 2D panorama | `python -m pipeline.panorama --source-run runs/<run> --out runs/<run>_panorama_v1` |
| Panorama count | `python -m pipeline.panorama_count --run runs/<run> --panorama runs/<run>_panorama_v1 --attach` |
| Duplicate merging (physical instances) | `python -m pipeline.instances --run runs/<run> --attach` |
| 3D map (and joining split maps) | `python -m pipeline.recon3d --source-run runs/<run> --out runs/<run>_recon3d_v1` |
| Measurement | `python -m app.measurement --run <run>` |
| Finish reading | `python -m app.finish_reading --run runs/<run>` |

- **Panorama count**: spines detected on the stitched shelf-face pieces; used as the headline count when the panorama is usable.
- **Duplicate merging**: one packet line per physical book. Track groups are merged only on geometric evidence (panorama overlap,
  frame registration across tracking breaks, short gaps and revisits, shelf-plane position in the 3D map); two groups seen in the
  same frame are never merged, and every merge carries its evidence line.
- **3D map**: pycolmap structure-from-motion, rotated into a shelf-aligned frame; a join of split maps must keep one shelf plane.
- **Measurement**: metric scale from catalogue spine dimensions of identified books on the 3D map (`pipeline/auto_scale.py`), room
  size from the room pan when floor and walls reconstruct (`pipeline/auto_room.py`); otherwise it abstains.
- **Finish reading**: reads the groups the time budget left unread, resolves and prices them, and updates the packet.

## Ground truth and evaluation

- `http://127.0.0.1:4182/truth?run=<run>`: hand-label titles, non-book items and 15 price checks for a sweep. Answers are saved to
  `ground_truth/ground_truth_labels.json` (git-ignored, `LIBRARY_GROUND_TRUTH_DIR` overrides it).
- `python scripts/ground_truth_sheet.py runs/<run>/claim_packet.json ground_truth/ground_truth_and_results.xlsx --labels ground_truth/ground_truth_labels.json`
  builds the results workbook (also the page's "Write to sheet" button).
- `python -m app.score_titles` scores readings against held-out human title labels; `python -m eval.count_vs_truth` compares a run's
  count with a hand count. Model-read labels are only ever used as development checks, never as truth.
- Pair sheets for duplicate checks: `python -m pipeline.pair_sheets`, prompt in [pair_adjudication_prompt.md](pair_adjudication_prompt.md).

## Other commands

```bash
.venv/bin/python -m pipeline.run --video sweeps/my_shelf.mov --out runs/offline_baseline   # staged offline pipeline
.venv/bin/python -m pipeline.price --run runs/<run> --online                              # pricing only
.venv/bin/python -m pipeline.scale --run runs/<run> --calibration calibration.json        # spine sizes from a calibration (docs/templates/)
.venv/bin/python -m pipeline.room  --run runs/<run> --evidence room_evidence.json         # room sizes from room evidence (docs/templates/)
.venv/bin/python -m app.smoke live                                                        # checks the voice setup (needs a key and LIBRARY_LIVE_MODEL)
```
