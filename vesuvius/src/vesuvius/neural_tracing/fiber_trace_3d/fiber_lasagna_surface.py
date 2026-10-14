"""Read-only adapter from solved fiber maps to Lasagna's fixed-UV surface fit."""
import json
from pathlib import Path
import sys
import argparse

import numpy as np

from . import fiber_uv_layout as uv
from .fiber_uv_surface import PolylineBank


def load_h_curves(run_root, threshold=.5, deformation_root=None, apply_offsets=False, include_v=False):
    if include_v and (deformation_root is None or not apply_offsets):
        raise ValueError('Connected H/V fitting requires the deformed layout with applied offsets')
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
        # Add in float64 too: casting an already-rounded float32 sum can shift
        # crossing coordinates by hundredths of a base voxel.
        q = torch.as_tensor(deformation['fixed_winding'], dtype=torch.float64)+torch.as_tensor(state[:, 1], dtype=torch.float64)
        u = uv.q_to_u(q, w, torch.cat((w.new_zeros(1), w.cumsum(0)))).numpy()
        v = state[:, 2]
        chosen = np.flatnonzero(deformation['retained'] & ((data['family'] == 0) | include_v))
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
        family = int(data['family'][i])
        ds = direction*(arc-start)
        cu = u[i]+(ds if family == 0 else np.zeros_like(ds))
        cv = np.full(len(arc), v[i], dtype=np.float64)
        if family == 1:
            cv += ds
        if deformation is not None:
            if family == 0:
                cu += data['node_ds'][i, 0]
            else:
                cv += data['node_ds'][i, 0]
            if apply_offsets and len(nodes):
                cu += np.interp(arc, oa, deformation['u_offsets'][nodes])
                cv += np.interp(arc, oa, deformation['v_offsets'][nodes])
        if direction < 0:
            cu, cv, xyz, arc = cu[::-1], cv[::-1], xyz[::-1], arc[::-1]
        if deformation is not None:
            curves.append(dict(piece=int(i), source=int(data['source_id'][i]), confidence=1.,
                               u=cu, v=cv, xyz=xyz, arc=arc, family=family,
                               winding_width=float(widths[int(np.clip(deformation['fixed_winding'][i], 0, len(widths)-1))])))
            continue
        # Source endpoints and polyline knots can differ below one translated-U
        # float64 ULP. They are the same UV sample, not a negative/zero segment.
        cu, unique = np.unique(cu, return_index=True)
        xyz = xyz[unique]
        curves.append(dict(piece=int(i), source=int(data['source_id'][i]),
                           confidence=float(confidence[i]), u=cu, v=float(v[i]), xyz=xyz))
    if include_v:
        # Offset graph arcs were serialized after source-arc arithmetic, which
        # can round at large base coordinates. Recover exact crossing UV from
        # the original signed event distances used by the deformation solve.
        events = deformation['fixed_event_indices']
        node_uv = {}
        for side, node_key, ds_key in [('i', 'v_offset_a', 'ds_a'), ('j', 'v_offset_b', 'ds_b')]:
            for owner, node, ds in zip(data[side][events], deformation[node_key], data[ds_key][events]):
                pos = np.array([u[owner], v[owner]], dtype=np.float64)
                pos[int(data['family'][owner])] += float(ds)
                pos += [deformation['u_offsets'][node], deformation['v_offsets'][node]]
                if node in node_uv and not np.allclose(node_uv[node], pos, atol=1e-5, rtol=0):
                    raise ValueError('Inconsistent saved crossing UV for one source node')
                node_uv[int(node)] = pos
        for c in curves:
            nodes = np.flatnonzero(deformation['v_offset_source'] == c['source'])
            for n in nodes:
                hits = np.flatnonzero(c['arc'] == deformation['v_offset_arc'][n])
                if len(hits):
                    c['u'][hits], c['v'][hits] = node_uv[int(n)]
    provenance = dict(run_root=str(root.resolve()), solution_sha256=uv.sha256_file(root/'solution.npz'),
                      config_sha256=uv.sha256_file(root/'config.json'),
                      tensor_cache=cfg['tensor_cache'], tensor_sha256=uv.sha256_file(cfg['tensor_cache']),
                      afv=cfg['afv'], joint_threshold=threshold)
    if deformation is not None:
        provenance.update(deformation_sha256=uv.sha256_file(Path(deformation_root)/'solution.npz'),
                          selection='deformed retained H/V mask' if include_v else 'deformed retained H mask', apply_offsets=apply_offsets)
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
                                     settings.get('deformation_root'), settings.get('apply_offsets', False),
                                     include_v=settings.get('connected_hv', False))
    files = [Path(__file__), Path(__file__).with_name('fiber_uv_surface.py'),
             repo/'lasagna/fiber_surface.py', repo/'lasagna/opt_loss_fixed_uv.py',
             repo/'lasagna/model.py', repo/'lasagna/optimizer.py',
             repo/'lasagna/opt_loss_smooth.py', repo/'lasagna/opt_loss_step.py', repo/'lasagna/opt_loss_bend.py']
    if settings.get('connected_hv'):
        files.append(repo/'lasagna/fiber_surface_connected.py')
    if settings.get('winding_barrier') is not None:
        files.append(repo/'lasagna/fiber_surface_barrier.py')
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
