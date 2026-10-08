"""Front and top views of a 3D map (recon3d.json) as a PNG (OpenCV only; matplotlib is not installed).

recon3d.json stores points, cameras and markers in the shelf-aligned frame: z = 0 is the fitted shelf plane, +z the camera side,
+y spine-up. FRONT = (x, y), what a person facing the shelf sees; TOP = (x, z), looking down: a flat shelf face is ONE horizontal
line at z = 0, so fragments joined at wrong rotations show as lines fanned out at different angles. Cameras are drawn coloured by
time (blue = first frame, red = last) so the path and its jumps show; book markers are black crosses (triangulated) or grey dots
(projected onto the plane). The band +-tol around z = 0 in the top view is the book-marker plane tolerance when recon3d.json has
one (`plane_consistency.tolerance`), else the marker placement tolerance.

    python scripts/recon3d_render.py runs/<run>_recon3d_v1 runs/<run>_recon3d_v2 --out views.png [--width 1600]
Several maps are stacked as rows (one row = front | top of one map).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

WHITE, BLACK, GREY, BAND = (255, 255, 255), (0, 0, 0), (150, 150, 150), (235, 225, 205)


def load(path):
    p = Path(path)
    data = json.loads((p / 'recon3d.json' if p.is_dir() else p).read_text())
    pts = np.asarray(data.get('points') or [], float).reshape(-1, 3)
    col = np.asarray(data.get('point_colors') or [], int).reshape(-1, 3)
    if len(col) != len(pts):
        col = np.full((len(pts), 3), 120, int)
    cams = data.get('cameras') or []
    cam_xyz = np.asarray([c['center'] for c in cams], float).reshape(-1, 3)
    cam_t = np.asarray([_frame_order(c['frame_id']) for c in cams], float)
    marks = data.get('markers') or []
    m_xyz = np.asarray([m['xyz'] for m in marks], float).reshape(-1, 3)
    m_tri = np.asarray([m.get('method') == 'triangulated' for m in marks], bool)
    return data, pts, col, cam_xyz, cam_t, m_xyz, m_tri


def _frame_order(frame_id):
    digits = ''.join(ch for ch in str(frame_id).rsplit('_', 1)[-1] if ch.isdigit())
    return float(digits) if digits else 0.0


def _title(name, data):
    pc = data.get('plane_consistency') or {}
    plane = data.get('plane') or {}
    bits = [name, data.get('status', '?'), f"reg {100 * float(data.get('registered_ratio') or 0):.0f}%"]
    if plane:
        bits.append(f"plane inliers {100 * float(plane.get('inlier_ratio') or 0):.0f}%")
    if pc.get('share') is not None:
        bits.append(f"books on plane {100 * pc['share']:.0f}% (n={pc.get('books_evaluated')})")
    mf = data.get('merged_from') or {}
    if mf.get('used'):
        bits.append(f"merge: {mf.get('used')}")
    return '  '.join(bits)


def _bounds(*arrays, lo=1, hi=99, pad=0.06):
    a = np.concatenate([x for x in arrays if len(x)]) if any(len(x) for x in arrays) else np.zeros((1, 2))
    mn, mx = np.percentile(a, lo, axis=0), np.percentile(a, hi, axis=0)
    span = np.maximum(mx - mn, 1e-6)
    return mn - pad * span, mx + pad * span


def _panel(w, h, xy_pts, col, xy_cam, cam_t, xy_mark, m_tri, label, band=None, bounds=None):
    img = np.full((h, w, 3), 255, np.uint8)
    top = 26
    mn, mx = bounds if bounds is not None else _bounds(xy_pts, xy_cam, xy_mark)
    scale = min((w - 20) / (mx[0] - mn[0]), (h - top - 10) / (mx[1] - mn[1]))
    ox = 10 + ((w - 20) - scale * (mx[0] - mn[0])) / 2
    oy = top + ((h - top - 10) - scale * (mx[1] - mn[1])) / 2

    def px(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        u = ox + (xy[:, 0] - mn[0]) * scale
        v = oy + (mx[1] - xy[:, 1]) * scale            # +y (or +z) upwards on the image
        return np.stack([u, v], 1).round().astype(int)

    if band is not None:
        y0, y1 = px([[mn[0], band]])[0][1], px([[mn[0], -band]])[0][1]
        cv2.rectangle(img, (0, max(top, y0)), (w - 1, min(h - 1, y1)), BAND, -1)
        yz = px([[mn[0], 0.0]])[0][1]
        cv2.line(img, (0, yz), (w - 1, yz), (200, 180, 140), 1)
    for (u, v), c in zip(px(xy_pts), col):
        if 0 <= u < w and top <= v < h:
            img[v, u] = (int(c[2]), int(c[1]), int(c[0]))   # RGB -> BGR
    if len(xy_mark):
        for (u, v), tri in zip(px(xy_mark), m_tri):
            if tri:
                cv2.drawMarker(img, (int(u), int(v)), BLACK, cv2.MARKER_CROSS, 7, 1)
            else:
                cv2.circle(img, (int(u), int(v)), 2, GREY, -1)
    if len(xy_cam):
        order = np.argsort(cam_t)
        t = (cam_t - cam_t.min()) / max(1e-9, np.ptp(cam_t))
        for (u, v), f in zip(px(xy_cam)[order], t[order]):
            cv2.circle(img, (int(u), int(v)), 3, (int(255 * (1 - f)), 0, int(255 * f)), -1)
    cv2.putText(img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1, cv2.LINE_AA)
    return img


def render(paths, out, width=1600, row_height=None):
    rows = []
    for p in paths:
        data, pts, col, cam, cam_t, mk, tri = load(p)
        name = Path(p).name.replace('sweep_', '')
        tol = (data.get('plane_consistency') or {}).get('tolerance') or data.get('plane_tolerance')
        half = width // 2
        bf = _bounds(pts[:, :2], cam[:, :2], mk[:, :2])
        bt = _bounds(pts[:, [0, 2]], cam[:, [0, 2]], mk[:, [0, 2]])
        bt = (np.array([bf[0][0], bt[0][1]]), np.array([bf[1][0], bt[1][1]]))    # same x range in both views
        h = row_height or int(half * 0.62)
        front = _panel(half, h, pts[:, :2], col, cam[:, :2], cam_t, mk[:, :2], tri,
                       'FRONT (x right, y up)  ' + _title(name, data), bounds=bf)
        topv = _panel(width - half, h, pts[:, [0, 2]], col, cam[:, [0, 2]], cam_t, mk[:, [0, 2]], tri,
                      'TOP (x right, up = distance from shelf plane, band = book tolerance)', band=tol, bounds=bt)
        row = np.concatenate([front, np.full((h, 2, 3), 200, np.uint8), topv], 1)
        rows.append(row[:, :width])
    sep = np.full((3, width, 3), 120, np.uint8)
    img = rows[0]
    for r in rows[1:]:
        img = np.concatenate([img, sep, r], 0)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('maps', nargs='+', help='recon3d dirs (or recon3d.json files), one row each')
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=1600)
    ap.add_argument('--row-height', type=int, default=None)
    a = ap.parse_args()
    print(render(a.maps, a.out, a.width, a.row_height))


if __name__ == '__main__':
    main()
