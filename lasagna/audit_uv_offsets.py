"""Compare saved runs with identical attachment samples; read-only JSON report.

Usage: python lasagna/audit_uv_offsets.py RUN_DIR [RUN_DIR ...]
Slope is added UV displacement per base-voxel source arc; slope-change is the
difference between consecutive slopes (same sample grid required for comparison).
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import tifffile
from fiber_surface_repair import restore_targets


def audit(paths):
    torch.set_num_threads(8)
    rows, reference = [], None
    for name in paths:
        p = Path(name)
        with np.load(p/'attachments.npz') as f:
            a = {k: f[k] for k in f.files}
        s = json.loads((p/'summary.json').read_text())
        if reference is None:
            reference = a
        for key in ('initial_grid_uv', 'piece', 'source', 'arc', 'crossing'):
            if not np.array_equal(a[key], reference[key]):
                raise ValueError(f'Unmatched input {key}: {name}')
        with np.load(p/'targets.npz') as f:
            data = {k: f[k] for k in f.files}
        step = float(data['grid_step'])
        d = (a['final_grid_uv']-a['initial_grid_uv'])*step
        slopes, changes = [], []
        for piece in np.unique(a['piece']):
            idx = np.flatnonzero(a['piece'] == piece)
            idx = idx[np.argsort(a['arc'][idx], kind='stable')]
            gap = np.diff(a['arc'][idx])
            slope = np.diff(d[idx], axis=0)/np.maximum(gap, 1)[:, None]
            slopes.extend(np.linalg.norm(slope, axis=1))
            changes.extend(np.linalg.norm(np.diff(slope, axis=0), axis=1))
        # Recompute common positional/tangent metrics for old runs too, using the
        # same sampler and weights rather than comparing unlike saved summaries.
        mesh = np.stack([tifffile.imread(p/'surface.tifxyz'/f'{axis}.tif') for axis in 'xyz'], -1)
        targets = restore_targets(data, a, mesh.shape[:2], step, 'cpu')
        fit = targets.statistics(torch.as_tensor(mesh[None]))
        rows.append(dict(run=str(p),
            local_p95_base=float(np.quantile(np.linalg.norm(a['local_delta']*step, axis=1), .95)),
            slope_rms=float(np.sqrt(np.mean(np.square(slopes)))),
            slope_p95=float(np.quantile(slopes, .95)),
            slope_change_rms=float(np.sqrt(np.mean(np.square(changes)))),
            u_p95_base=fit['shift_u_p50_p95_max_base'][1],
            v_p95_base=fit['shift_v_p50_p95_max_base'][1],
            h_normal_rms_base=fit['h_normal_rms_base'],
            v_offset_normal_rms_base=fit['v_offset_normal_rms_base'],
            h_tangent_rms_base=fit['h_tangent_rms_base'],
            v_offset_tangent_rms_base=fit['v_offset_tangent_rms_base'],
            h_position_rms_base=fit['h_position_rms_base'],
            v_offset_position_rms_base=fit['v_offset_position_rms_base'],
            folds=s['after']['geometry']['opposing_triangle_quads'],
            seconds=s['total_seconds'],
            crossing_error_grid=float(np.abs(a['final_grid_uv'][a['crossing_a']]-a['final_grid_uv'][a['crossing_b']]).max())))
    return rows


if __name__ == '__main__':
    print(json.dumps(audit(sys.argv[1:]), indent=2))
