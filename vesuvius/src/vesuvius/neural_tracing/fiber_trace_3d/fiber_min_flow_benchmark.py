"""Read-only adapter for villa2's seven cached fiber benchmarks."""
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from .fiber_min_flow import Seam, corrected_targets, integer_targets, fractional_targets, split_fractional_edges, solve


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class Paths:
    def __init__(self, config, data, angle=-np.pi/2):
        from .fiber_normal_sign_visualization import load_umbilicus_frame
        from vesuvius.neural_tracing.fiber_follow.data.afv import read_afv_polylines
        with sqlite3.connect(Path(config['afv']).resolve().as_uri()+'?mode=ro', uri=True) as db:
            root = json.loads(db.execute("SELECT value FROM metadata WHERE key='root'").fetchone()[0])
        self.seam = Seam(load_umbilicus_frame(config['umbilicus'], root), angle)
        self.paths = {}
        for row, points in read_afv_polylines(config['afv'], data['source_id']):
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            transport = self.seam.polyline(points)
            self.paths[int(row['id'])] = (arc, points, transport)
        self.data = data
        if len(data['links']) or len(np.unique(data['source_id'])) != len(data['source_id']):
            raise ValueError('Benchmark expects whole unsplit fibers')
        if np.any(data['start'] != 0):
            raise ValueError('Benchmark anchors must be original fiber starts')

    def at(self, source, arc):
        s, points, transport = self.paths[int(source)]
        # Saved endpoint offsets are float32, while original polyline arc is f64.
        if arc < -1e-4 or arc > s[-1]+max(1e-4, 2e-7*s[-1]):
            raise ValueError(f'Endpoint outside fiber {source}: {arc}, {s[-1]}')
        arc = np.clip(arc, 0, s[-1])
        j = min(np.searchsorted(s, arc, side='right')-1, len(s)-2)
        p = points[j]+(points[j+1]-points[j])*(arc-s[j])/max(s[j+1]-s[j], 1e-300)
        return p, int(transport[j])+self.seam.segment(points[j], p)

    def ends(self, side):
        d = self.data
        picks = d['i' if side == 'a' else 'j']
        direction = np.sign(d['node_ds'][picks, -1])
        if np.any(direction == 0):
            raise ValueError('Missing aligned arc direction')
        arcs = d['start'][picks]+d['ds_'+side]*direction
        samples = [self.at(d['source_id'][p], arc) for p, arc in zip(picks, arcs)]
        return np.array([s[0] for s in samples]), np.array([s[1] for s in samples])


def graph(config, d, paths, sign_only=False, target_mode="fractional"):
    if target_mode not in ("rounded", "fractional"):
        raise ValueError("Unknown target mode: "+target_mode)
    a, ta = paths.ends('a')
    b, tb = paths.ends('b')
    c = np.array([paths.seam.segment(x, y) for x, y in zip(a, b)])
    fd = d['family'][d['j']]-d['family'][d['i']]
    target_function = fractional_targets if target_mode == 'fractional' and not sign_only else integer_targets
    measured, valid = target_function(d['distance']*config.get('lasagna_winding_scale', 1.),
        d['sign'], fd, d['parallel'], config.get('perpendicular_offset', .5))
    target = corrected_targets(measured, ta, tb, c)
    ia, ib = d['i'].copy(), d['j'].copy()
    if sign_only:
        valid = d['sign'] != 0
        margin = ((~d['parallel']) & (d['sign']*fd < 0)).astype(int)
        target = margin+d['sign']*(c-tb+ta)
        reverse = d['sign'] < 0
        ia[reverse], ib[reverse] = ib[reverse].copy(), ia[reverse].copy()
    weights = np.where(d['parallel'], 1., config.get('perpendicular_relative_weight', 1.))
    g = dict(a=ia[valid], b=ib[valid], target=target[valid], weight=weights[valid],
             one_sided=np.full(valid.sum(), sign_only))
    event_index = np.flatnonzero(valid)
    if target_mode == 'fractional' and not sign_only:
        g, source = split_fractional_edges(g['a'], g['b'], g['target'], g['weight'])
        event_index = event_index[source]
    return g, dict(valid=valid, fiber_a=ta, fiber_b=tb, connector=c,
                   measured_target=measured, corrected_target=target, flow_event_index=event_index)


def summarize_errors(error, parallel):
    error, parallel = np.asarray(error), np.asarray(parallel, bool)
    def part(mask):
        v = abs(error[mask])
        return dict(events=int(len(v)), l1=float(v.mean()) if len(v) else None,
            correct_percent=float(100*np.mean(v == 0)) if len(v) else None,
            max_error=int(v.max()) if len(v) else None)
    return dict(all=part(np.ones(len(error), bool)), perpendicular=part(~parallel), parallel=part(parallel))


def small_reference(d, solution, reference, transport):
    offsets = {int(k): v for k, v in reference['fiber_offsets'].items()}
    use = np.array([int(d['source_id'][a]) in offsets and int(d['source_id'][b]) in offsets
                    for a, b in zip(d['i'], d['j'])])
    # Reference = fiber offset + original UNWRAPPED angle. Convert that to
    # the same seam chart as our endpoint labels; fractional phase is absent.
    anchor = np.floor(d['node_phase'][:, 0]+.25).astype(int)
    refa = np.array([offsets.get(int(d['source_id'][p]), 0) for p in d['i']])+anchor[d['i']]+transport['fiber_a']
    refb = np.array([offsets.get(int(d['source_id'][p]), 0) for p in d['j']])+anchor[d['j']]+transport['fiber_b']
    local = solution['winding'][d['j']]-solution['winding'][d['i']]+transport['fiber_b']-transport['fiber_a']
    result = summarize_errors((local-(refb-refa))[use], d['parallel'][use])
    result.update(excluded_reference_source_ids=reference['excluded_source_ids'],
                  scored_events=int(use.sum()), unscored_events=int((~use).sum()))
    return result


def large_reference(project, d, solution, paths, reference, details=None):
    root = project/'data'
    events = read(root/'neighbor_round10_pair_v1/prepared/gt_events.json')
    gc = read(root/'bbox_gt_solve_v3/uv_whole/config.json')
    gd = arrays(gc['tensor_cache'])
    gp = Paths(gc, gd)
    offsets = {int(k): v for k, v in reference['fiber_offsets'].items()}
    picks = {int(s): i for i, s in enumerate(d['source_id'])}
    required, par, components, orders, signs, margins = [], [], [], [], [], []
    skipped = 0
    scored = []
    for e in events:
        if e['gt_source'] not in offsets or e['test_source'] not in picks:
            skipped += 1
            continue
        m, valid = integer_targets(abs(e['signed_distance']), np.sign(e['signed_distance']),
            e['family_delta'], e['parallel'])
        if not valid:
            skipped += 1
            continue
        p = picks[e['test_source']]
        x, t = paths.at(e['test_source'], e['test_arc'])
        y, gt_transport = gp.at(e['gt_source'], e['gt_arc'])
        c = paths.seam.segment(x, y)
        truth = offsets[e['gt_source']]+int(np.floor(gd['node_phase'][e['gt_piece'], 0]+.25))+gt_transport
        delta = truth-(solution['winding'][p]+t)-c
        required.append(int(delta-m))
        par.append(e['parallel'])
        components.append(int(solution['component'][p]))
        signs.append(np.sign(e['signed_distance']))
        orders.append(delta)
        margins.append(int(not e['parallel'] and np.sign(e['signed_distance'])*e['family_delta'] < 0))
        if details is not None:
            scored.append(dict(e, connector_crossings=int(c), fiber_crossings=int(t),
                gt_fiber_crossings=int(gt_transport), measured_integer_target=int(m)))
    required = np.asarray(required)
    components = np.asarray(components)
    # Each disconnected component has an unidentifiable additive integer gauge.
    # Align separately for diagnostic scoring, explicitly reporting each gauge.
    aligned = required.copy()
    alignment = {}
    for component in np.unique(components):
        mask = components == component
        shift = int(np.floor(np.median(required[mask])))
        aligned[mask] -= shift
        alignment[str(component)] = dict(offset=shift, events=int(mask.sum()))
    result = summarize_errors(aligned, par)
    result.update(available_events=len(events), scored_events=len(required), unscored_events=skipped,
        component_gauges=alignment, gauge_policy='per-component integer median; GT used only for evaluation',
        reference='frozen gt_winding_reference.json; source 7 excluded')
    global_shift = int(np.floor(np.median(required)))
    result['single_global_gauge'] = dict(offset=global_shift, metrics=summarize_errors(required-global_shift, par))
    per_shift = np.array([alignment[str(c)]['offset'] for c in components])
    wrong = np.asarray(signs)*(np.asarray(orders)-per_shift) < np.asarray(margins)
    result['signs'] = {name: dict(events=int(mask.sum()), violations=int(wrong[mask].sum()))
        for name, mask in [('parallel', np.asarray(par)), ('perpendicular', ~np.asarray(par))]}
    if details is not None:
        for index, e in enumerate(scored):
            delta = int(orders[index]-per_shift[index])
            deficit = int(margins[index]-signs[index]*delta)
            details.append(dict(e, component=int(components[index]), gauge=int(per_shift[index]),
                predicted_gt_minus_test=delta, required_signed_minimum=int(margins[index]),
                sign_violation=bool(wrong[index]), violation_steps=max(0, deficit),
                integer_residual=int(aligned[index]),
                global_gauge_sign_violation=bool(signs[index]*(orders[index]-global_shift) < margins[index])))
    return result


def audit_free(config, d):
    """Recompute separation, classification and sign from original AFC rows."""
    from .fiber_normal_convention import inward_solution
    from .fiber_uv_layout import signed_order
    actual = digest(config['tensor_cache'])
    if actual != config['tensor_sha256']:
        raise ValueError('Tensor cache differs from the original SGD solve hash')
    if not Path(config['database']).exists():
        return dict(tensor_hash_matches_original_sgd_solve=True, tensor_sha256=actual,
            source_afc_recomputation='unavailable: original database path no longer exists',
            missing_database=config['database'], rows_verified=0)
    normal = inward_solution(arrays(config['stage1']))
    with sqlite3.connect(Path(config['database']).resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.execute('CREATE TEMP TABLE selected(id INTEGER PRIMARY KEY)')
        db.executemany('INSERT INTO selected VALUES (?)', ((int(i),) for i in d['event_ids']))
        rows = db.execute('SELECT id,fiber_a,fiber_b,ax,ay,az,bx,by,bz,parallel_score,perpendicular_score,'
            'winding_distance,parallel_winding_distance,normal_a,normal_b FROM constraints '
            'WHERE id IN (SELECT id FROM selected) ORDER BY id').fetchall()
    by_id = {r[0]: r for r in rows}
    rows = [by_id[int(i)] for i in d['event_ids']]
    nums = np.array([r[:13] for r in rows], float)
    par = nums[:, 9] > nums[:, 10]
    distance = np.where(par, nums[:, 12], nums[:, 11])
    ends = np.searchsorted(normal['ids'], nums[:, 1:3].astype(np.int64))
    def normals(column, side):
        records = [json.loads(r[column]) for r in rows]
        return np.array([r['xyz'] if r['valid'] else [np.nan]*3 for r in records])*normal['normal_sign'][ends[:, side], None]
    sign = signed_order(nums[:, 6:9]-nums[:, 3:6], normals(13, 0), normals(14, 1), config.get('sign_cosine_min', .1))
    if not np.array_equal(sign, d['sign']) or not np.array_equal(par, d['parallel']) or not np.allclose(distance, d['distance'], rtol=1e-7, atol=1e-8):
        raise ValueError('Source AFC measurements do not match cached constraints')
    return dict(rows_verified=len(rows), signs_identical=True, family_classification_identical=True,
                distance_max_roundoff=float(np.max(abs(distance-d['distance']))),
                sign_cosine_min=config.get('sign_cosine_min', .1))


def run(settings):
    target_mode = settings.get('target_mode', 'fractional')
    if target_mode not in ('rounded', 'fractional'):
        raise ValueError('Unknown target mode: '+target_mode)
    project = Path(settings['project'])
    output = Path(settings['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    reference = read(project/'gt_winding_reference.json')
    cases = []
    for name in ['full', 'half', 'double', 'sign', 'sign_sparse_control']:
        manifest = project/f'run_gt_{name}_torch_softmin_tiny.json'
        trial = read(manifest)['fiber_solver_trial']
        recipe = read(trial['recipe'])['fiber_gt_convergence_probe']
        config_path = Path(recipe['source'])/'config.json'
        config = read(config_path)
        config.update(recipe['parameters'])
        config.update(trial['parameters'])
        cases.append(dict(name=name, config=config, config_path=str(config_path), manifest=str(manifest),
            sign_only=name.startswith('sign'), baseline=trial['output_root']))
    for name in ['free', 'joint']:
        folder = project/'data/neighbor_round10_pair_v1'/name
        path = folder/'solve/config.json'
        cases.append(dict(name=name, config=read(path), config_path=str(path), manifest=str(project/'run_neighbor_pair.json'),
                          sign_only=False, baseline=str(folder)))
    for case in cases:
        case['tensor_sha256'] = digest(case['config']['tensor_cache'])
        if case['config'].get('tensor_sha256') and case['tensor_sha256'] != case['config']['tensor_sha256']:
            raise ValueError('Cache hash differs from saved solve: '+case['name'])
        case['manifest_sha256'] = digest(case['manifest'])
        case['config_sha256'] = digest(case['config_path'])
    dependencies = {project/'gt_winding_reference.json', project/'data/neighbor_round10_pair_v1/prepared/gt_events.json'}
    for case in cases:
        dependencies.update(Path(case['config'][k]) for k in ('afv', 'umbilicus', 'tensor_cache'))
    gt_config_path = project/'data/bbox_gt_solve_v3/uv_whole/config.json'
    dependencies.add(gt_config_path)
    gt_config = read(gt_config_path)
    dependencies.update(Path(gt_config[k]) for k in ('afv', 'umbilicus', 'tensor_cache'))
    input_hashes = {str(path): digest(path) for path in sorted(dependencies)}
    write(output/'manifest.json', dict(cases=cases, reference_sha256=digest(project/'gt_winding_reference.json'),
        input_hashes=input_hashes, target_mode=target_mode,
        source_branch='global-fiber-sgd-surface', source_commit='db07e60fd',
        notes='CPU integer-only model; no fitted widths/geometry, fiber disabling or uncertainty.'))
    results = {}
    geometry_cache = {}
    for case in cases:
        started = time.monotonic()
        name, config = case['name'], case['config']
        print('min-flow preparing '+name, flush=True)
        d = arrays(config['tensor_cache'])
        key = (config['afv'], config['tensor_cache'])
        if key not in geometry_cache:
            geometry_cache[key] = Paths(config, d)
        paths = geometry_cache[key]
        g, transport = graph(config, d, paths, case['sign_only'], target_mode)
        # Cached phases linearly interpolate ANGLE, whereas the physical seam
        # intersects linearly interpolated XYZ. They can disagree near a cut.
        anchor = np.floor(d['node_phase'][:, 0]+.25).astype(int)
        phase_audit = {}
        for side, picks in [('a', d['i']), ('b', d['j'])]:
            expected = np.floor(d['phase_'+side]+.25).astype(int)-anchor[picks]
            phase_audit[side] = int(np.sum(expected != transport['fiber_'+side]))
        result = solve(len(d['source_id']), **g)
        k = result['winding']
        physical = k[d['j']]-k[d['i']]+transport['fiber_b']-transport['fiber_a']-transport['connector']
        fd = d['family'][d['j']]-d['family'][d['i']]
        margin = (~d['parallel']) & (d['sign']*fd < 0)
        wrong = (d['sign'] != 0) & (d['sign']*physical < margin)
        metrics = dict(target_mode=target_mode, flow_arcs=len(g['a']), nodes=len(k), events=len(d['i']), used_events=int(transport['valid'].sum()),
            cached_interpolated_angle_seam_disagreements=phase_audit,
            unsigned_events=int(np.sum(d['sign'] == 0)),
            unsigned_unique_target_retained=int(np.sum((d['sign'] == 0)&transport['valid'])),
            excluded_ambiguous_events=int(np.sum(~transport['valid'])),
            signs={kind: dict(events=int(np.sum(mask&(d['sign'] != 0))), violations=int(np.sum(mask&wrong)))
                for kind, mask in [('parallel', d['parallel']), ('perpendicular', ~d['parallel'])]},
            seam=dict(fiber_endpoint_crossings=int(np.sum(abs(transport['fiber_a'])+abs(transport['fiber_b']))),
                      connectors_crossing=int(np.sum(transport['connector'] != 0))),
            certificate=result['certificate'], sign_only=case['sign_only'],
            reference=(large_reference(project, d, result, paths, reference) if name in ('free', 'joint')
                       else small_reference(d, result, reference, transport)),
            retained_fiber_percent=100., geometry_width_metrics='not applicable: no optimized geometry or widths')
        if name == 'free':
            metrics['source_measurement_audit'] = audit_free(config, d)
        folder = output/name
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'solution.npz', source_id=d['source_id'], family=d['family'],
            winding=k, component=result['component'], flow=result['flow'], **transport, **g)
        metrics['seconds_total'] = time.monotonic()-started
        write(folder/'metrics.json', metrics)
        results[name] = metrics
        write(output/'results.json', results)
        print(json.dumps(dict(case=name, objective=result['certificate']['primal'],
                              reference=metrics['reference']['perpendicular'], seconds=metrics['seconds_total'])), flush=True)
    return results
