# Configuration

Settings come from the environment or `.env` (copy `.env.example`; real environment variables win). Every default below is what a plain
run does.

## Keys (all optional)

| Variable | Used for |
| --- | --- |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Gemini: hosted reading/detection (also needs `LIBRARY_ALLOW_HOSTED=1` or an explicit `gemini` selection), pair adjudication, voice |
| `LIBRARY_LIVE_MODEL` | the Gemini Live model id for voice; none is assumed. Check with `python -m app.smoke live` |
| `LIBRARY_OL_CONTACT` | a contact for Open Library's User-Agent (allows 3 requests/s instead of 1) |
| `GOOGLE_BOOKS_API_KEY` | Google Books lookups with your own quota (anonymous requests share a daily quota) |

Hosted calls go through a kill-switch (Connect / Disconnect on the site), a disk cache and a call counter. `GEMINI_THINKING_BUDGET`
(default 512 tokens) caps thinking per image request; `GEMINI_PRICE_IN_PER_MTOK` / `GEMINI_PRICE_OUT_PER_MTOK` turn on cost estimates.

## Stages

| Stage | Default | Override |
| --- | --- | --- |
| Reader | `auto`: Gemini only with a key and `LIBRARY_ALLOW_HOSTED=1`, else RapidOCR, else EasyOCR (if its weights exist), else none | `LIBRARY_READER=rapidocr\|easyocr\|apple\|gemini\|none`; `LIBRARY_READER_SECOND=florence` reads weak groups a second time (optional Florence-2 runtime, not installed by setup) |
| Reading policy | `group`: best 1-2 sightings per physical book after association, parallel, 150 s budget | `LIBRARY_READ_POLICY=observation`, `LIBRARY_READ_BUDGET_S`, `LIBRARY_READ_WORKERS` (3) |
| Detector | `auto`: Gemini only with a key and `LIBRARY_ALLOW_HOSTED=1`, else the local YOLO detector, else none | `LIBRARY_DETECTOR=local\|gemini\|none\|external:<candidates.json>` |
| Detector profile | `fast_items`: books at 1280 px plus a 26-prompt non-book pass at 640 px on every frame | `LIBRARY_LOCAL_DETECTOR_PROFILE=baseline` |
| Detection rate | `dense`: 2.5 detections/s, blurry frames included | `LIBRARY_DETECT_POLICY=keyframes`, `LIBRARY_DETECT_MAX_FPS`, `LIBRARY_DETECT_SKIP_BLURRY=1` |
| Capture guidance | thresholds for the nudges | `LIBRARY_CQ_*` |
| Association | `cross_view` (tracked motion, titles never used) | `LIBRARY_ASSOC=dedup` |
| Duplicate merging | revisit and plane rules as shipped | `LIBRARY_REVISIT_MERGE=0\|1`, `LIBRARY_PLANE_MERGE=0\|1`, `LIBRARY_PAIR_ORACLE=1` (hosted pair adjudication, model-read, off by default) |
| 3D map | `auto` profile (fast settings from 100 frames, all frames kept) | `LIBRARY_RECON3D_PROFILE=auto\|fast\|default`, `LIBRARY_RECON3D_FRAMES=all\|diet\|keyframes`, `LIBRARY_RECON3D_MAX_FRAMES`, `LIBRARY_RECON3D_IMAGE_SIZE`, `LIBRARY_RECON3D_MAX_FEATURES_SIFT`, `LIBRARY_RECON3D_MERGE=0\|1` |
| Post-sweep builds | on after every sweep | `LIBRARY_POST_SWEEP_BUILD=0` |
| Catalogue and pricing | US/USD; public HTTP lookups are opt-in | `LIBRARY_CATALOGUE_ONLINE=1`, `LIBRARY_PRICING_ONLINE=1`, `LIBRARY_SECOND_COUNTRY` (default `GB`), `LIBRARY_SECOND_MARKET=1` (Spain/EUR comparison) |

## Paths

| Variable | Default |
| --- | --- |
| `LIBRARY_DETECTOR_PYTHON`, `LIBRARY_DETECTOR_WEIGHTS` | `.runtime/venv/bin/python`, `.runtime/models` (written to `.env` by the setup script) |
| `LIBRARY_OCR_MODEL_DIR`, `LIBRARY_OCR_PACKAGE_DIR` | `.runtime/easyocr_models`, `.runtime/easyocr_packages` |
| `LIBRARY_FLORENCE_PYTHON`, `LIBRARY_FLORENCE_MODEL_DIR` | `.runtime/florence/...` (optional, install yourself) |
| `LIBRARY_RUNS_DIR` | `runs/` |
| `LIBRARY_GROUND_TRUTH_DIR` | `ground_truth/` |
| `LIBRARY_TLS_DIR` | `runs/_server_state/tls/` |

Hosted inference is never used unless you opt in: `python -m pipeline.detect` refuses billable hosted detection without `--allow-hosted`.
