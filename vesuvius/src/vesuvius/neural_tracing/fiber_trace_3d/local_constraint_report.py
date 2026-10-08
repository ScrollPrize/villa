"""Report the local-constraint ablation using the existing benchmark metrics."""
import csv
import json
import re
from pathlib import Path


def geometry(rows):
    return [(r['original_trace'], r['source_piece'],
             float(r['begin_arc_base']), float(r['end_arc_base'])) for r in rows]


def validate_clean_geometry(rows):
    by_trace = {}
    for trace, _, begin, end in geometry(rows):
        if end <= begin:
            raise ValueError('nonpositive piece length')
        by_trace.setdefault(trace, []).append((begin, end))
    for intervals in by_trace.values():
        intervals.sort()
        if abs(intervals[0][0]) > 1e-6:
            raise ValueError('missing beginning of source trace')
        for left, right in zip(intervals, intervals[1:]):
            if abs(left[1] - right[0]) > 1e-6:
                raise ValueError('piece overlap or gap')


def report(config):
    import numpy as np
    from lasagna.inference_provenance import atomic_write, sha256_file
    from .density_penalty_sweep import metrics
    root = Path(config['output_root'])
    output = Path(config['local_event_report'])
    results, geometries = [], {}
    for name in ('legacy', 'local-parallel', 'local-orthogonal'):
        prefix = root / name
        record = json.loads(prefix.with_suffix('.run.json').read_text())
        if record['returncode'] != 0:
            raise ValueError(f'{name} did not finish successfully')
        row = dict(name=name, **metrics(prefix), provenance=record)
        with Path(str(prefix) + '_consistency.csv').open() as f:
            pieces = list(csv.DictReader(f))
        with Path(str(prefix) + '_winding_factors.csv').open() as f:
            factors = list(csv.DictReader(f))
        geometries[name] = geometry(pieces)
        if name != 'legacy':
            validate_clean_geometry(pieces)
        piece_to_trace = {r['piece']: r['original_trace'] for r in pieces}
        ordinary = [r for r in factors if piece_to_trace[r['piece_a']] != piece_to_trace[r['piece_b']]]
        parallel = [r for r in ordinary if float(r['parallel']) > float(r['perpendicular'])]
        row.update(pieces=len(pieces), ordinary=len(ordinary), parallel=len(parallel),
                   orthogonal=len(ordinary)-len(parallel),
                   continuity=len(factors)-len(ordinary),
                   parallel_evidence_mass=sum(float(r['evidence_weight']) for r in parallel))
        results.append(row)
    # The unchanged benchmark selects the largest active winding component
    # AFTER orientation BP. Parallel removal can legitimately change that set.
    # Audit extraction before BP, then check each retained piece against it.
    audit_geometry = {}
    for name in ('local-parallel', 'local-orthogonal'):
        path = root / ('audit-' + name + '_local_event_pieces.csv')
        with path.open() as f:
            rows = list(csv.DictReader(f))
        validate_clean_geometry(rows)
        audit_geometry[name] = geometry(rows)
        arc_keys = lambda entries: {(t, begin, end) for t, _, begin, end in entries}
        if not arc_keys(geometries[name]).issubset(arc_keys(audit_geometry[name])):
            raise ValueError('retained BP piece differs from extraction audit')
        audit_log = (root / ('audit-' + name + '.log')).read_text()
        line = next(line for line in audit_log.splitlines() if line.startswith('local constraint events:'))
        stats = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
        row = next(row for row in results if row['name'] == name)
        row['extraction'] = stats
        row['piece_length_base_p5_p50_p95'] = np.percentile(
            [end-begin for _, _, begin, end in audit_geometry[name]], [5, 50, 95]).tolist()
    if audit_geometry['local-parallel'] != audit_geometry['local-orthogonal']:
        raise ValueError('parallel ablation changed pre-BP segmentation')
    trace = Path(results[0]['provenance']['command'][2])
    document = dict(results=results, identical_local_geometry=True,
                    pre_bp_local_pieces=len(audit_geometry['local-parallel']),
                    trace_artifact=str(trace),
                    trace_files={str(p): sha256_file(p) for p in sorted(trace.rglob('*')) if p.is_file()})
    atomic_write(output, document, durable=False)
    print(json.dumps([{k: v for k, v in r.items() if k not in ('provenance', 'ladder')} for r in results], indent=2))
