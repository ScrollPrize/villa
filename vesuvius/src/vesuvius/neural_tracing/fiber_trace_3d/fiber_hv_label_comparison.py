"""Compare independent H/V-stage assignments with source AFV family labels."""
import json
from pathlib import Path
import sqlite3

import numpy as np
from lasagna.inference_provenance import atomic_write


def comparison(ids, original, lengths, probabilities, labels, inherited, cutoffs):
    """Count/arc-weighted disagreement; exclusions never count as H/V errors."""
    ids = np.asarray(ids)
    original = np.asarray(original)
    lengths = np.asarray(lengths, dtype=np.float64)
    p = np.asarray(probabilities)
    labels = np.asarray(labels)
    inherited = np.asarray(inherited, dtype=bool)
    n = len(ids)
    if (n == 0 or original.shape != (n,) or lengths.shape != (n,)
            or p.shape != (n, 3) or labels.shape != (n,) or inherited.shape != (n,)):
        raise ValueError('Expected one source row and H/V/broken probability vector per fiber')
    if (not np.all(np.diff(ids) > 0) or not np.isfinite(lengths).all()
            or np.any(lengths < 0) or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(1), 1, atol=2e-7, rtol=0)
            or not np.array_equal(labels, p.argmax(1))
            or np.any(labels[inherited] != 2)):
        raise ValueError('Invalid IDs, lengths, probabilities, hard labels or inherited exclusions')
    known = (original == 0) | (original == 1)
    retained = labels != 2
    confidence = p.max(1)
    def amount(mask):
        return dict(fibers=int(mask.sum()), arc_base_voxels=float(lengths[mask].sum()))
    def percent(value, total):
        return 100 * value / total if total else None
    def score(mask):
        eligible = mask & known
        mismatched = eligible & (labels != original)
        total, wrong = amount(eligible), amount(mismatched)
        return dict(**total, mismatches=wrong['fibers'],
                    mismatch_arc_base_voxels=wrong['arc_base_voxels'],
                    mismatch_percent=percent(wrong['fibers'], total['fibers']),
                    mismatch_arc_percent=percent(wrong['arc_base_voxels'], total['arc_base_voxels']))
    thresholds = []
    for cutoff in cutoffs:
        if not 0 <= cutoff <= 1:
            raise ValueError('Confidence cutoffs must be in [0,1]')
        mask = retained & (confidence >= cutoff)
        thresholds.append(dict(minimum_joint_probability=float(cutoff), **score(mask)))
    confusion = {}
    for code, family in enumerate(('H', 'V')):
        source = original == code
        row = {name: amount(source & (labels == target)) for target, name in enumerate(('H', 'V', 'broken'))}
        row['normal_stage_broken'] = amount(source & inherited)
        row['hv_stage_broken'] = amount(source & (labels == 2) & ~inherited)
        confusion[family] = row
    return dict(total=amount(np.ones(n, bool)), original_unknown=amount(~known),
                normal_stage_broken=amount(inherited),
                hv_stage_broken=amount((labels == 2) & ~inherited),
                retained=score(retained), by_confidence=thresholds,
                confusion=confusion,
                interpretation='Disagreement with source labels, not anatomical accuracy; broken fibers excluded from mismatch denominators.')


def length_histogram(lengths, labels):
    """Source AFV arc-length histogram; half-open bins, native base voxels."""
    lengths = np.asarray(lengths, dtype=float)
    labels = np.asarray(labels)
    if (lengths.ndim != 1 or labels.shape != lengths.shape or not np.isfinite(lengths).all()
            or np.any(lengths < 0) or not np.isin(labels, [0, 1, 2]).all()):
        raise ValueError('Invalid lengths/labels')
    edges = np.array([0, 100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, np.inf])
    result = dict(units='native base voxels', bins_lower=edges[:-1].tolist(),
                  bins_upper=[None if np.isinf(v) else float(v) for v in edges[1:]], groups={})
    for name, mask in [('all', np.ones(len(lengths), bool)),
                       *[(name, labels == code) for code, name in enumerate(('H', 'V', 'broken'))]]:
        values = lengths[mask]
        counts = np.histogram(values, edges)[0]
        result['groups'][name] = dict(count=len(values), counts=counts.tolist(),
            percent=(100 * counts / len(values)).tolist() if len(values) else [0.] * len(counts),
            mean=float(values.mean()) if len(values) else None,
            quantiles=dict(zip(('min', 'p25', 'median', 'p75', 'p90', 'p95', 'p99', 'max'),
                              np.quantile(values, [0, .25, .5, .75, .9, .95, .99, 1]).tolist())) if len(values) else {})
    return result


def run_histogram(config):
    with np.load(config['comparison'], allow_pickle=False) as data:
        result = length_histogram(data['lengths_base'], data['labels'])
    result['input'] = str(Path(config['comparison']).resolve())
    destination = Path(config['report'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(destination, result, durable=False)
    print(json.dumps(result), flush=True)
    return result


def run(config):
    afv = Path(config['afv']).resolve()
    solution = Path(config['solution']).resolve()
    with sqlite3.connect(afv.as_uri() + '?mode=ro', uri=True) as db:
        rows = db.execute('SELECT id,family,length FROM fibers ORDER BY id').fetchall()
    ids = np.asarray([r[0] for r in rows], dtype=np.int64)
    original = np.asarray([{'H': 0, 'V': 1}.get(str(r[1]).strip().upper(), -1) for r in rows], dtype=np.int8)
    lengths = np.asarray([r[2] for r in rows], dtype=np.float64)
    del rows
    with np.load(solution, allow_pickle=False) as data:
        if not np.array_equal(ids, data['ids']):
            raise ValueError('Solution IDs do not match the complete source AFV')
        labels = data['labels']
        p = data['probabilities']
        inherited = data['inherited_broken']
        result = comparison(ids, original, lengths, p, labels, inherited,
                            config.get('probability_cutoffs', [.5, .9, .99]))
        result['inputs'] = dict(afv=str(afv), solution=str(solution))
        if config.get('per_fiber_output'):
            target = Path(config['per_fiber_output'])
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.partial.npz')
            np.savez(temporary, ids=ids, original_labels=original, lengths_base=lengths,
                     labels=labels, probabilities=p, inherited_broken=inherited)
            temporary.replace(target)
    destination = Path(config['report'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(destination, result, durable=False)
    print(json.dumps(result), flush=True)
    return result
