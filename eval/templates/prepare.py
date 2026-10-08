"""Prepare blank, model-independent frame annotation JSON and a local editor."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from pipeline.read_spines import inside, load, records
from pipeline.contracts import write_json


def prepare(manifest_path, run_dir, output, viewer):
    output, viewer = Path(output), Path(viewer)
    if output.exists() or viewer.exists():
        raise ValueError("Refusing to overwrite existing labels/viewer; choose a new version")
    manifest = load(manifest_path)
    template = load(Path(__file__).with_name("annotation_v1.json"))
    template["dataset_id"] = manifest["run_id"]
    frames = []
    for frame in records(manifest, "frames"):
        inside(run_dir, frame["image_path"])  # enforce original evidence bounds
        frames.append({**{k: frame[k] for k in ("frame_id", "timestamp_s", "image_path", "width_px", "height_px")},
                       "image_sha256": hashlib.sha256(inside(run_dir, frame["image_path"]).read_bytes()).hexdigest(),
                       "exhaustive": False, "reviewed": False, "objects": [], "notes": None})
    template["frames"] = frames
    write_json(output, template)
    viewer.parent.mkdir(parents=True, exist_ok=True)
    # Local original images; no resize/export into the inference path and no predictions.
    image_urls = {f["frame_id"]: inside(run_dir, f["image_path"]).as_uri() for f in frames}
    content = Path(__file__).with_name("review.html").read_text()
    payload = json.dumps({"annotations": template, "images": image_urls}, ensure_ascii=False).replace("<", "\\u003c")
    viewer.write_text(content.replace("__ANNOTATION_PAYLOAD__", payload))
    return output, viewer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--viewer", required=True)
    a = p.parse_args()
    print(*prepare(a.manifest, a.run_dir, a.out, a.viewer), sep="\n")

if __name__ == "__main__":
    main()
