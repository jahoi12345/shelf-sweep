# Setup

## Requirements

- macOS or Linux, Python 3.12 via [`uv`](https://docs.astral.sh/uv/).
- `ffmpeg` / `ffprobe` (video metadata, browser previews), `openssl` (the phone's HTTPS certificate), Chrome or Chromium (PDF reports).
- Node 20+ only for the browser client tests.
- An iPhone (Safari) on the same Wi-Fi as the Mac for live sweeps. Any recorded video works for replay.

## Install

```bash
cp .env.example .env
./scripts/setup_local_models.sh            # full: app, RapidOCR, YOLO spine detector, YOLO-World/CLIP, EasyOCR + CRAFT
./scripts/setup_local_models.sh --minimal  # app + RapidOCR only; detection is reported "unavailable"
```

The script creates `.venv` (the app) and `.runtime/` (a separate detector runtime and the model weights, about 1.3 GB). Downloads are
pinned and sha256-checked; re-running the same command resumes an interrupted setup (partial downloads continue, verified weights are
reused). Each step has a 900 s limit (`--step-timeout` changes it). Per-step timings are written to `.runtime/setup_timings/`.
The script writes the local-only settings it owns into `.env` (`LIBRARY_DETECTOR`, `LIBRARY_READER`, runtime and weight paths) and
keeps every other line.

Without the local runtime the app still works: frames, capture guidance, the packet and the report are produced; detection and reading
are reported as unavailable (never as an empty shelf).

## Servers

| URL | What |
| --- | --- |
| `http://127.0.0.1:4182/` | the site on the Mac: Sweep, Frames, Debugger (`/debug/`), Measurements; `/api/health`, `/api/build` |
| `http://127.0.0.1:4182/watch` | read-only live view of the phone's sweep (loopback only) |
| `http://127.0.0.1:4182/truth` | ground-truth labelling page (loopback only) |
| `https://<Mac LAN IP>:4183/?k=<pairing key>` | the same app over HTTPS for the phone (`scripts/serve_https.sh`) |

Run both in the background with logs:

```bash
mkdir -p runs/_server_state/logs
nohup .venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 4182 --no-access-log > runs/_server_state/logs/site4182.log 2>&1 &
nohup scripts/serve_https.sh > runs/_server_state/logs/https4183.log 2>&1 &
grep -m1 '?k=' runs/_server_state/logs/https4183.log      # the phone's pairing link
```

Both servers share `runs/`, so a sweep recorded on the phone appears on the Mac's site. Python changes need a restart of both
(static JS/CSS are served with `no-store`, a reload is enough); restart between sweeps, not while a build is running.

## The phone (HTTPS on the LAN)

Safari only gives a page the camera on a secure origin, so `scripts/serve_https.sh` creates a local certificate authority and a pairing
key under `runs/_server_state/tls/` (never committed) and serves the app on port 4183. It prints the steps; in short:

1. Send `runs/_server_state/tls/ca.crt` to the iPhone (AirDrop), install the profile, then enable it under
   Settings > General > About > Certificate Trust Settings. This is needed once; the CA stays valid when the Mac's IP changes.
2. Open the printed `https://<IP>:4183/?k=<key>` link in Safari. The key is stored in a cookie; requests without it are refused.
3. Hold the phone in landscape. `?res=4k` asks the phone for 3840x2160.

Networks with client isolation block phone-to-Mac traffic; use the phone's Personal Hotspot instead. Overrides: `LAN_IP`, `HTTPS_PORT`,
`LIBRARY_TLS_DIR`, `PYTHON`; `scripts/serve_https.sh --print` refreshes the certificate and prints the link without starting a server.

## Recorded videos

Put videos in `sweeps/` (git-ignored) and replay them through the same ingestion path as the camera:

```bash
.venv/bin/python -m app.replay --video sweeps/my_shelf.mov --out runs/my_replay --speed 1.0
ffmpeg -i sweeps/my_shelf.mov -c:v libx264 -pix_fmt yuv420p -an sweeps/my_shelf_preview_h264.mp4   # optional browser preview
```

`--speed 0` replays as fast as possible; `--frames-from runs/<run>` re-runs the pipeline on the saved frames of an earlier sweep.

## Privacy

Packet ZIPs never contain a source video; JPEG frames are shipped with EXIF/XMP/GPS removed and `privacy_audit.json` records what was
removed. To share a video yourself: `.venv/bin/python -m pipeline.sanitize video IN.MOV OUT.mp4`.
