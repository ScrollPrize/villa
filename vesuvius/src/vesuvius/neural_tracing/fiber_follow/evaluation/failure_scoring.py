"""Score traced predictions for three failure conditions: compressed sheets, a bending fiber, and the fiber turning
away from the crop's forward axis within the predicted planes.

Input: one or more evaluation folders written by long_trace_audit (or output eval_audit wrappers): every
``<source>/<cohort>/rows_*.json`` with its ``trace_NNN.npz`` (path, annotation, per-decision pos/frame). Each directed
trace gets at most one failure event, judged by the evaluation's own strict labels:
  divergence      the first sustained departure before the annotation ends; measured at its onset, the last decision
                  before it still within 1 vox of the annotation (the state the model was in when it began to leave)
  premature_stop  a confidence stop while still on the fiber (last match <= 3 vox) with > 16 vox of annotation left
Stage = trace length at the event: seed (< 16 vox), early (16-128), mid (128-1024), late (>= 1024).
Reference points: on-fiber decisions (match <= 1 vox) every ``--stride`` vox of each trace, >= 64 vox before its
event; they calibrate the thresholds per source and give each trace its exposure to the three conditions.

Measures at a decision (annotation point a at the matched arclength t, traversal sign, the decision's crop frame):
  compression  sheetness (l1 - l2)/l1 of the CT gradient structure tensor over 13^3 vox at a, and sheet modulation:
               (p90 - p10)/mean of the CT profile along the sheet normal (+-20 vox, mean of a 3x3 vox patch)
  bendiness    the annotation smoothed at sigma 4 vox: bend between its tangents 12 vox before and after a, the bend
               ahead between the tangent at a and 16 / 32 vox further in the direction of travel, and arc/chord of the
               raw annotation over 24 vox (a loop or doubling back >> 1)
  crop         the decision's crop: angle between its forward axis and the chord from the head to the annotation
               ``--planes`` vox ahead, and how far the annotation swings sideways (crop cross-plane) over those planes
               beyond the head's own offset
Scores (0-1, the measure's percentile among the source's reference points): compression = max(1 - pct(sheetness),
1 - pct(modulation)); bendiness = max(pct(bend), pct(bend ahead 16), pct(bend ahead 32), pct(arc/chord));
crop = max(pct(crop chord angle), pct(sideways swing)). Flag: the condition score is at or above the reference p90
of that same score, so each condition flags 10% of the reference points of its source.
``--thresholds`` reuses another run's thresholds.json (compare models on the same scale).

Outputs (--out): traces.csv (one row per trace: outcome, event, the three scores and flags at the event, exposure =
share of the trace's reference points flagged), points.json (every measured point), thresholds.json, summary.md
(per folder and source: share of failures flagged per condition and the excess over the reference rate:
(failures flagged - reference flagged)/(1 - reference flagged)).
Usage: python -m vesuvius.neural_tracing.fiber_follow.evaluation.failure_scoring RUN_DIR [RUN_DIR ...]
           --dataset-config CONFIG.json --out OUT_DIR [--planes 16] [--stride 32] [--workers 8] [--thresholds FILE]
"""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from ..shared.geometry import arclength, interp_at

STAGES = ((0., 16., 'seed'), (16., 128., 'early'), (128., 1024., 'mid'), (1024., np.inf, 'late'))
CONDITIONS = ('compression', 'bendiness', 'crop')
# Per condition: (measure, direction): 'low' = small values are the condition, 'high' = large values.
MEASURES = {'compression': (('sheetness', 'low'), ('sheet_modulation', 'low')),
            'bendiness': (('bend', 'high'), ('bend_ahead16', 'high'), ('bend_ahead32', 'high'), ('arc_chord', 'high')),
            'crop': (('crop_chord_angle', 'high'), ('crop_sideways_swing', 'high'))}
FLAG_QUANTILE = 90  # each condition flags this complement of the reference points


def stage(length):
    return next(name for lo, hi, name in STAGES if lo <= length < hi)


# ------------------------------------------------------------------------------------------------- CT measures
def sample_ct(vol, pts):
    """CT values (raw) at trace-coordinate points of any shape (..., 3); zero outside the volume."""
    pts = np.asarray(pts, float)
    zyx = pts.reshape(-1, 3)[:, ::-1]*vol.input_scale
    shape = np.asarray(vol.ct.shape)
    lo, hi = np.floor(zyx.min(0)).astype(int)-1, np.ceil(zyx.max(0)).astype(int)+2
    lo_c, hi_c = lo.clip(0, shape), hi.clip(0, shape)
    if not np.isfinite(zyx).all() or (hi_c <= lo_c).any():
        return np.zeros(pts.shape[:-1], np.float32)
    cube = np.zeros(hi-lo, np.float32)
    cube[tuple(slice(l-o, h-o) for l, h, o in zip(lo_c, hi_c, lo))] = vol.ct.read(lo_c, hi_c-lo_c)
    return map_coordinates(cube, (zyx-lo).T, order=1, mode='constant', cval=0.).reshape(pts.shape[:-1])


def sheet_normal(vol, centre, half=6.):
    """Largest eigenvector of the CT gradient structure tensor over a (2*half+1)^3 vox block, and sheetness."""
    g = np.arange(-half, half+.01, 1.)
    block = sample_ct(vol, centre+np.stack(np.meshgrid(g, g, g, indexing='ij'), -1))
    grads = np.stack(np.gradient(block), -1).reshape(-1, 3)
    w, v = np.linalg.eigh(grads.T @ grads)
    return v[:, -1], float((w[-1]-w[-2])/max(w[-1], 1e-9))


def sheet_modulation(vol, centre, normal, half=20.):
    """(p90 - p10)/mean of the CT profile along the sheet normal (3x3 vox patch mean): how clearly sheets separate."""
    e1 = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    offsets = np.arange(-half, half+.01, .5)
    profile = np.stack([sample_ct(vol, centre+u*e1+v*e2+offsets[:, None]*normal)
                        for u in (-1., 0., 1.) for v in (-1., 0., 1.)]).mean(0)
    return float(np.percentile(profile, 90)-np.percentile(profile, 10))/max(float(profile.mean()), 1e-6)


def compression(vol, a):
    normal, sheetness = sheet_normal(vol, a)
    return dict(sheetness=sheetness, sheet_modulation=sheet_modulation(vol, a, normal))


# --------------------------------------------------------------------------------------------- shape measures
def smoothed(annotation, s, sigma=4.):
    arcs = np.arange(s[0], s[-1]+1e-9, 1.)
    return gaussian_filter1d(interp_at(annotation, s, arcs), sigma, axis=0, mode='nearest'), arcs


def bendiness(annotation, s, t, sign):
    p, arcs = smoothed(annotation, s)

    def tan(at):
        k = int(np.clip(np.searchsorted(arcs, at), 1, len(p)-2))
        d = p[k+1]-p[k-1]
        return d/max(np.linalg.norm(d), 1e-9)
    angle = lambda u, v: float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))
    raw = interp_at(annotation, s, np.arange(max(s[0], t-12.), min(s[-1], t+12.)+1e-9, .5))
    chord = float(np.linalg.norm(raw[-1]-raw[0])) if len(raw) > 1 else 0.
    arc = float(np.linalg.norm(np.diff(raw, axis=0), axis=1).sum()) if len(raw) > 1 else 0.
    return dict(bend=angle(tan(t-12), tan(t+12)), bend_ahead16=angle(tan(t), tan(t+16*sign)),
                bend_ahead32=angle(tan(t), tan(t+32*sign)), arc_chord=arc/max(chord, 1e-6))


def crop_measures(annotation, s, t, sign, pos, frame, planes):
    fwd = frame[:, 2]
    ahead = interp_at(annotation, s, np.clip(t+sign*np.arange(0., planes+.01, 1.), s[0], s[-1]))
    local = (ahead-pos) @ frame
    lateral = np.hypot(local[:, 0], local[:, 1])
    chord = ahead[-1]-pos
    return dict(crop_chord_angle=float(np.degrees(np.arccos(np.clip(fwd @ chord/max(np.linalg.norm(chord), 1e-9), -1, 1)))),
                crop_sideways_swing=float(lateral.max()-lateral[0]), head_offset=float(lateral[0]))


# ------------------------------------------------------------------------------------------------ trace scanning
def matched(details, i, annotation, s, pos):
    t = details[i].get('matched_t')
    if t is None or not np.isfinite(t):
        t = float(s[np.argmin(np.linalg.norm(annotation-pos, axis=1))])
    return float(t)


def find_event(row):
    """(kind, decision index to measure at, trace length of the event) or None; see the module docstring."""
    details = row['decisions_detail']
    if not details:
        return None
    travelled = np.asarray([d['travelled'] for d in details])
    first = (row.get('local') or {}).get('first_sustained_departure_length')
    if row['diverged'] and first is not None:
        i = max(0, int(np.searchsorted(travelled, first, 'right'))-1)
        onset = [j for j in range(i+1) if details[j]['match_distance'] is not None and details[j]['match_distance'] <= 1.]
        return 'divergence', (onset[-1] if onset else 0), float(first)
    last = details[-1]['match_distance']
    if (not row['diverged'] and row['reason'] == 'confidence'
            and ((row.get('local') or {}).get('stop_remaining_annotation') or 0.) > 16.
            and last is not None and last <= 3.):
        return 'premature_stop', len(details)-1, float(row['length'])
    return None


_VOLUMES, _DOCUMENT = {}, None


def _init(config):
    global _DOCUMENT
    from ..data.datasets import read_dataset_config
    _DOCUMENT = read_dataset_config(config)[0]


def volume(name):
    if name not in _VOLUMES:
        from ..data.datasets import ct_source_spec, primary_source_spec
        from ..data.volume import FiberVolume
        source = next(s for s in _DOCUMENT['sources'] if s['name'] == name)
        spec = primary_source_spec(_DOCUMENT) if source['kind'] == 'paris4' else ct_source_spec(source, _DOCUMENT['cache_dir'])
        _VOLUMES[name] = FiberVolume(spec)
    return _VOLUMES[name]


def measure(vol, row, data, i, planes):
    details = row['decisions_detail']
    annotation, s, sign = data['annotation'], data['annotation_s'], float(row['sign'])
    pos, frame = data['pos'][i], data['frame'][i]
    t = matched(details, i, annotation, s, pos)
    a = interp_at(annotation, s, np.array([t]))[0]
    return dict(t=t, travelled=float(details[i]['travelled']), **compression(vol, a), **bendiness(annotation, s, t, sign),
                **crop_measures(annotation, s, t, sign, pos, frame, planes))


def scan(task):
    """Every measured point of one trace: its event (if any) and its reference points."""
    folder, rows_file, k, planes, stride = task
    row = json.loads(Path(rows_file).read_text())[k]
    npz = Path(rows_file).parent/f"trace_{row['seed_index']:03d}.npz"
    data = np.load(npz)
    if 'frame' not in data.files or not len(data['frame']):
        return []
    key = dict(folder=folder, source=row['source'], cohort=row['cohort'], seed_index=row['seed_index'],
               fiber_name=row['fiber_name'])
    vol = volume(row['source'])
    event = find_event(row)
    out = []
    limit = event[2]-64. if event else row['length']
    details = row['decisions_detail']
    next_at = 0.
    for j, d in enumerate(details[:len(data['frame'])]):
        if d['travelled'] >= limit:
            break
        if d['travelled'] >= next_at and d['match_distance'] is not None and d['match_distance'] <= 1.:
            out.append(dict(key, role='reference', **measure(vol, row, data, j, planes)))
            next_at = d['travelled']+stride
    if event and event[1] < len(data['frame']):
        out.append(dict(key, role='event', kind=event[0], stage=stage(event[2]), event_length=event[2],
                        **measure(vol, row, data, event[1], planes)))
    return out


# -------------------------------------------------------------------------------------------------- scoring
def thresholds_from(points):
    """Per source and measure: the sorted reference values (percentile lookup) and the p10/p90 flag thresholds."""
    out = {}
    for source in sorted({p['source'] for p in points}):
        ref = [p for p in points if p['source'] == source and p['role'] == 'reference']
        out[source] = {m: sorted(float(p[m]) for p in ref if np.isfinite(p[m]))
                       for cond in CONDITIONS for m, _ in MEASURES[cond]}
    return out


def score(point, reference):
    values = reference.get(point['source'])
    result = {}
    for cond in CONDITIONS:
        parts = []
        for m, direction in MEASURES[cond]:
            ref = values[m] if values else []
            if not ref or not np.isfinite(point[m]):
                continue
            pct = np.searchsorted(ref, point[m])/len(ref)
            parts.append(1-pct if direction == 'low' else pct)
        result[f'{cond}_score'] = float(max(parts)) if parts else np.nan
    return result


def flag_levels(points):
    """Per source and condition: the reference p90 of the condition score."""
    return {source: {cond: float(np.nanpercentile([p[f'{cond}_score'] for p in points
                                                   if p['source'] == source and p['role'] == 'reference'], FLAG_QUANTILE))
                     for cond in CONDITIONS}
            for source in sorted({p['source'] for p in points})}


def apply_flags(points, levels):
    for p in points:
        for cond in CONDITIONS:
            level = levels.get(p['source'], {}).get(cond, np.inf)
            p[f'{cond}_flag'] = bool(np.isfinite(p[f'{cond}_score']) and p[f'{cond}_score'] >= level)


def excess(failed, reference):
    if not failed or not reference:
        return np.nan, np.nan
    pe, pc = np.mean(failed), np.mean(reference)
    return pe, (pe-pc)/(1-pc) if pc < 1 else np.nan


def summarize(points, rows_by_trace):
    lines = ['# Failure conditions: compression, bendiness, crop', '',
             f'Flags: condition score >= its reference p{FLAG_QUANTILE} (each condition flags {100-FLAG_QUANTILE}% of '
             'reference points). "excess" = (failures flagged - reference flagged)/(1 - reference flagged).', '']
    for folder in sorted({p['folder'] for p in points}):
        for source in sorted({p['source'] for p in points if p['folder'] == folder}):
            sel = [p for p in points if p['folder'] == folder and p['source'] == source]
            ref = [p for p in sel if p['role'] == 'reference']
            ev = [p for p in sel if p['role'] == 'event']
            traces = [r for (f, s, _, _), r in rows_by_trace.items() if f == folder and s == source]
            c = sum(r['correct'] for r in traces)
            w = sum(r['offtrack'] for r in traces)
            fol, av = sum(r['followed'] for r in traces), sum(r['avail'] for r in traces)
            lines += [f'## {folder} / {source}', '',
                      f'{len(traces)} traces, coverage {fol/max(av, 1e-9):.3f}, precision {c/max(c+w, 1e-9):.3f}, '
                      f'{sum(e["kind"] == "divergence" for e in ev)} divergences, '
                      f'{sum(e["kind"] == "premature_stop" for e in ev)} premature stops, {len(ref)} reference points', '',
                      '| failures | n | compression | bendiness | crop | any of the three |', '|---|---|---|---|---|---|']
            for label, members in (('premature stops', [e for e in ev if e['kind'] == 'premature_stop']),
                                   ('divergences', [e for e in ev if e['kind'] == 'divergence']), ('all', ev)):
                cells = []
                for cond in (*CONDITIONS, 'any'):
                    get = (lambda p: any(p[f'{x}_flag'] for x in CONDITIONS)) if cond == 'any' else (lambda p: p[f'{cond}_flag'])
                    pe, ex = excess([get(p) for p in members], [get(p) for p in ref])
                    cells.append(f'{100*pe:.0f}% (excess {100*ex:.0f}%)' if np.isfinite(pe) else '-')
                lines.append(f'| {label} | {len(members)} | '+' | '.join(cells)+' |')
            refcells = [f"{100*np.mean([p[f'{c}_flag'] for p in ref]):.0f}%" for c in CONDITIONS]
            refcells.append(f"{100*np.mean([any(p[f'{x}_flag'] for x in CONDITIONS) for p in ref]):.0f}%" if ref else '-')
            lines.append(f'| reference rate | {len(ref)} | '+' | '.join(refcells)+' |')
            div = [e for e in ev if e['kind'] == 'divergence']
            wrong = {k: r['offtrack'] for k, r in rows_by_trace.items()}
            total = sum(wrong[(e['folder'], e['source'], e['cohort'], e['seed_index'])] for e in div)
            if total:
                flagged = sum(wrong[(e['folder'], e['source'], e['cohort'], e['seed_index'])] for e in div
                              if any(e[f'{x}_flag'] for x in CONDITIONS))
                lines.append(f'\nDivergence wrong length in flagged places: {100*flagged/total:.0f}% of {total/1e3:.1f}k vox')
            lines.append('')
    return '\n'.join(lines)+'\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='evaluation folders (searched recursively for rows_*.json)')
    ap.add_argument('--dataset-config', required=True, help='dataset configuration naming each source\'s CT volume')
    ap.add_argument('--out', required=True)
    ap.add_argument('--planes', type=int, default=16, help="the model's gate horizon (predicted planes judged)")
    ap.add_argument('--stride', type=float, default=32., help='trace length between reference points')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--thresholds', help='thresholds.json of an earlier run: score on its reference distributions')
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks, rows_by_trace = [], {}
    for run in args.runs:
        folder = str(Path(run).resolve())
        for rows_file in sorted(Path(run).rglob('rows_*.json')):
            rows = json.loads(rows_file.read_text())
            for k, row in enumerate(rows):
                rows_by_trace[(folder, row['source'], row['cohort'], row['seed_index'])] = row
                tasks.append((folder, str(rows_file), k, args.planes, args.stride))
    print(f'{len(tasks)} traces in {len(args.runs)} folder(s)', flush=True)
    points = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(Path(args.dataset_config).resolve()),)) as pool:
        for n, result in enumerate(pool.map(scan, tasks, chunksize=4)):
            points += result
            if (n+1) % 200 == 0:
                print(f'{n+1}/{len(tasks)} traces scored', flush=True)
    saved = json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
    reference = saved['reference'] if saved else thresholds_from(points)
    for p in points:
        p.update(score(p, reference))
    levels = saved['flag_levels'] if saved else flag_levels(points)
    apply_flags(points, levels)
    (out/'thresholds.json').write_text(json.dumps(dict(reference=reference, flag_levels=levels)))
    (out/'points.json').write_text(json.dumps(points, indent=1, default=float))
    by_trace = {}
    for p in points:
        by_trace.setdefault((p['folder'], p['source'], p['cohort'], p['seed_index']), []).append(p)
    fields = ['folder', 'source', 'cohort', 'seed_index', 'fiber_name', 'length', 'correct', 'wrong', 'coverage',
              'event', 'stage', 'event_length', *[f'{c}_{x}' for c in CONDITIONS for x in ('score', 'flag')],
              *[f'exposure_{c}' for c in CONDITIONS]]
    with open(out/'traces.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        for key, row in rows_by_trace.items():
            pts = by_trace.get(key, [])
            ev = next((p for p in pts if p['role'] == 'event'), None)
            ref = [p for p in pts if p['role'] == 'reference']
            entry = dict(folder=key[0], source=key[1], cohort=key[2], seed_index=key[3], fiber_name=row['fiber_name'],
                         length=round(row['length'], 1), correct=round(row['correct'], 1), wrong=round(row['offtrack'], 1),
                         coverage=round(row['coverage'], 3), event=ev['kind'] if ev else '', stage=ev['stage'] if ev else '',
                         event_length=round(ev['event_length'], 1) if ev else '')
            for c in CONDITIONS:
                entry[f'{c}_score'] = round(ev[f'{c}_score'], 3) if ev else ''
                entry[f'{c}_flag'] = int(ev[f'{c}_flag']) if ev else ''
                entry[f'exposure_{c}'] = round(float(np.mean([p[f'{c}_flag'] for p in ref])), 3) if ref else ''
            writer.writerow(entry)
    text = summarize(points, rows_by_trace)
    (out/'summary.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
