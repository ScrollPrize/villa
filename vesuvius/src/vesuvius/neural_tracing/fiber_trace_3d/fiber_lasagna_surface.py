"""Read-only adapter from solved fiber maps to Lasagna's fixed-UV surface fit."""
import json
from pathlib import Path
import sys
import argparse

import numpy as np

from . import fiber_uv_layout as uv
from .fiber_uv_surface import PolylineBank


def load_h_curves(run_root, threshold=.5, deformation_root=None, apply_offsets=False):
    root = Path(run_root)
    cfg = json.loads((root/'config.json').read_text())
    with np.load(cfg['tensor_cache']) as f:
        data = {k: f[k] for k in f.files}
    with np.load(root/'solution.npz') as f:
        state, widths = f['state'], f['widths']
        for key in ('source_id', 'start', 'end', 'family'):
            if not np.array_equal(data[key], f[key]):
                raise ValueError(f'Solution/tensor mismatch: {key}')
    deformation = None
    if deformation_root is not None:
        import torch
        with np.load(Path(deformation_root)/'solution.npz') as f:
            deformation = {k: f[k] for k in f.files}
        for key in ('source_id', 'start', 'end', 'family'):
            if not np.array_equal(data[key], deformation[key]):
                raise ValueError(f'Deformation identity mismatch: {key}')
        if not np.array_equal(state, deformation['state']) or not np.array_equal(widths, deformation['widths']):
            raise ValueError('Matched original/deformed comparison requires identical base state and widths')
        if state.shape != (len(data['source_id']), 4):
            raise ValueError('Expected implicit winding state[N,4]')
        # Canonical shared winding-coordinate transform, with fixed assignments.
        w = torch.as_tensor(widths, dtype=torch.float64)
        q = torch.as_tensor(deformation['fixed_winding']+state[:, 1], dtype=torch.float64)
        u = uv.q_to_u(q, w, torch.cat((w.new_zeros(1), w.cumsum(0)))).numpy()
        v = state[:, 2]
        chosen = np.flatnonzero(deformation['retained'] & (data['family'] == 0))
        confidence = np.ones(len(state))  # Selection is the explicitly supplied frozen mask.
    else:
        p, _, keep, pick, u, v, _ = uv.hard_geometry(data, state, widths, cfg)
        confidence = p[np.arange(len(p)), pick]
        chosen = np.flatnonzero(keep & (data['family'] == 0) & (confidence > threshold))
    if not len(chosen):
        raise ValueError('No retained H pieces above joint confidence threshold')
    bank = PolylineBank(cfg['afv'], data['source_id'][chosen])
    curves = []
    for i in chosen:
        source = np.searchsorted(bank.ids, data['source_id'][i])
        start, end = float(data['start'][i]), float(data['end'][i])
        direction = np.sign(data['node_ds'][i, 2]-data['node_ds'][i, 0])
        if direction == 0 or end <= start:
            raise ValueError('Degenerate solved piece span/direction')
        knots = bank.keys[bank.first[source]:bank.last[source]+1]-bank.base[source]
        arc = np.unique(np.r_[start, knots[(knots > start) & (knots < end)], end])
        if deformation is not None:
            nodes = np.flatnonzero(deformation['v_offset_source'] == data['source_id'][i])
            oa = deformation['v_offset_arc'][nodes]
            if len(oa) and not np.all(np.diff(oa) > 0):
                raise ValueError('Offset arcs must be strictly increasing per source')
            arc = np.unique(np.r_[arc, oa[(oa > start) & (oa < end)]])
        xyz = bank.sample_xyz(np.full(len(arc), source), arc)
        cu = u[i]+direction*(arc-start)
        cv = np.full(len(arc), v[i], dtype=np.float64)
        if deformation is not None:
            cu += data['node_ds'][i, 0]
            if apply_offsets and len(nodes):
                cu += np.interp(arc, oa, deformation['u_offsets'][nodes])
                cv += np.interp(arc, oa, deformation['v_offsets'][nodes])
        if direction < 0:
            cu, cv, xyz, arc = cu[::-1], cv[::-1], xyz[::-1], arc[::-1]
        if deformation is not None:
            curves.append(dict(piece=int(i), source=int(data['source_id'][i]), confidence=1.,
                               u=cu, v=cv, xyz=xyz, arc=arc,
                               winding_width=float(widths[int(np.clip(deformation['fixed_winding'][i], 0, len(widths)-1))])))
            continue
        # Source endpoints and polyline knots can differ below one translated-U
        # float64 ULP. They are the same UV sample, not a negative/zero segment.
        cu, unique = np.unique(cu, return_index=True)
        xyz = xyz[unique]
        curves.append(dict(piece=int(i), source=int(data['source_id'][i]),
                           confidence=float(confidence[i]), u=cu, v=float(v[i]), xyz=xyz))
    provenance = dict(run_root=str(root.resolve()), solution_sha256=uv.sha256_file(root/'solution.npz'),
                      config_sha256=uv.sha256_file(root/'config.json'),
                      tensor_cache=cfg['tensor_cache'], tensor_sha256=uv.sha256_file(cfg['tensor_cache']),
                      afv=cfg['afv'], joint_threshold=threshold)
    if deformation is not None:
        provenance.update(deformation_sha256=uv.sha256_file(Path(deformation_root)/'solution.npz'),
                          selection='deformed retained H mask', apply_offsets=apply_offsets)
    return curves, provenance


def run_local(settings):
    # Lasagna's existing modules use script-local imports (model, fit_data,...).
    repo = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(repo/'lasagna'))
    if settings.get('render_surface'):
        from fiber_surface_render import run as render_run
        return render_run(settings)
    if settings.get('repair_from'):
        from fiber_surface_repair import run as repair_run
        return repair_run(settings)
    if settings.get('summarize_runs'):
        from fiber_surface import summarize_runs
        return summarize_runs(settings)
    from fiber_surface import solve
    curves, provenance = load_h_curves(settings['run_root'], settings.get('joint_threshold', .5),
                                     settings.get('deformation_root'), settings.get('apply_offsets', False))
    files = [Path(__file__), Path(__file__).with_name('fiber_uv_surface.py'),
             repo/'lasagna/fiber_surface.py', repo/'lasagna/opt_loss_fixed_uv.py',
             repo/'lasagna/model.py', repo/'lasagna/optimizer.py',
             repo/'lasagna/opt_loss_smooth.py', repo/'lasagna/opt_loss_step.py', repo/'lasagna/opt_loss_bend.py']
    provenance['code_sha256'] = {str(p.relative_to(repo)): uv.sha256_file(p) for p in files}
    return solve(curves, settings, provenance)


def run(settings):
    if settings.get('device', 'cpu') == 'cpu':
        return run_local(settings)
    from .managed_evaluation import launch_managed
    if not settings.get('gpu_manager'):
        raise ValueError('CUDA surface fitting requires gpu_manager settings')
    if ',' in str(settings.get('devices', '0')):
        raise ValueError('This surface solver uses one GPU')
    root = Path(settings['output_root'])
    if root.exists():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    frozen = root.with_name(root.name+'.launch.json')
    frozen.write_text(json.dumps(settings, indent=2)+'\n')
    command = [sys.executable, '-m', 'vesuvius.neural_tracing.fiber_trace_3d.fiber_lasagna_surface',
               '--config', str(frozen)]
    rc = launch_managed(command, devices=settings.get('devices', '0'), settings=settings['gpu_manager'])
    if rc:
        raise RuntimeError(f'Managed surface fit exited {rc}')
    return json.loads((root/'summary.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--unmanaged', action='store_true', help='Internal managed child entrypoint')
    parser.add_argument('--devices', default='0')
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    if args.unmanaged:
        if args.devices != '0':
            raise ValueError('Expected one manager-assigned local GPU')
        run_local(dict(settings, device='cuda:0'))
    else:
        run(settings)
