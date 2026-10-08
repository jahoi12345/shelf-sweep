"""One table row per 3D map / measurement dir (report helper; read-only).

  .venv/bin/python scripts/recon3d_merge_report.py runs/<run>_recon3d_vN[:runs/<run>_measure_vM] ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def row(spec):
    recon_dir, _, measure_dir = spec.partition(':')
    r = json.loads((Path(recon_dir) / 'recon3d.json').read_text())
    mf = r.get('merged_from') or {}
    out = {'recon': Path(recon_dir).name, 'profile': (r.get('library') or {}).get('profile'), 'frames': r.get('frames_total') or r.get('frames_used'),
           'models': r.get('models_found'), 'largest_before': (r.get('model_registered_sizes') or [None])[0],
           'registered': r.get('registered_frames'), 'ratio': r.get('registered_of_total', r.get('registered_ratio')),
           'merge_used': mf.get('used'), 'fragments_after': mf.get('fragments_after'), 'status': r.get('status'),
           'placed': r.get('markers_total'), 'books': r.get('book_count'), 'wall_s': (r.get('timings') or {}).get('total_s'),
           'merge_s': (r.get('timings') or {}).get('merge_s'), 'reproj_px': r.get('mean_reprojection_error_px')}
    if measure_dir:
        d = json.loads((Path(measure_dir) / 'auto_measure.json').read_text())['data']
        sc, sh, room = d.get('scale') or {}, d.get('shelf') or {}, d.get('room') or {}
        out.update(measure=Path(measure_dir).name, m_status=d.get('status'), scale_source=sc.get('source') if sc.get('cm_per_unit') else None,
                   cm_per_unit=sc.get('cm_per_unit'), scale_books=sc.get('n_inliers'),
                   spines_measured=(d.get('summary') or {}).get('measured'), shelf_run_m=sh.get('shelf_run_m'), rows=sh.get('rows'),
                   row_spans_m=sh.get('row_spans_m'), room=room.get('status'), room_reasons=room.get('reasons') or room.get('reason'))
    return out


if __name__ == '__main__':
    for spec in sys.argv[1:]:
        print(json.dumps(row(spec)))
