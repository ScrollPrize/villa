"""CPU-only saved-trace auditing; no inference imports in worker processes."""
from collections import deque
import json
from pathlib import Path

import numpy as np


def bounded_map(executor, function, items, *, pending=256):
    """Ordered map with bounded submitted work (including on Python <3.14)."""
    if pending < 1:
        raise ValueError('pending work bound must be positive')
    iterator = iter(items)
    futures = deque()
    for _ in range(pending):
        try:
            futures.append(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while futures:
        yield futures.popleft().result()
        try:
            futures.append(executor.submit(function, next(iterator)))
        except StopIteration:
            pass


def audit_failure(document, target, grace):
    """Exact independent audit of first failure and saved outside travel."""
    replay = document['replay']
    if not replay['failures']:
        return dict(valid=False, reason='no_failure')
    failure = replay['failures'][0]
    if failure['segment_index'] != 0 or abs(failure['reference_arc_base']-target) > 100:
        return dict(valid=False, reason='different_first_failure', failure=failure)
    segment = replay['segments'][0]
    points = np.asarray(segment['trace_points_base_xyz'], dtype=np.float64)
    outside = None
    for match in segment['matches']:
        i = match['trace_point_index']
        if match['threshold_error_ratio'] > 1:
            outside = 0. if outside is None else outside + float(np.linalg.norm(points[i]-points[i-1]))
        else:
            outside = None
    valid = outside is not None and outside + 1e-7 >= grace
    return dict(valid=valid, reason='reproduced' if valid else 'terminal_unresolved_excursion',
                outside_travel_base=outside, failure=failure)


def audit_trace_batch(task):
    paths, grace, global_worklist, targets = task
    rows = []
    for path in paths:
        doc = json.loads(Path(path).read_text())
        provenance = doc['provenance']
        if global_worklist:
            if provenance.get('coordinate_contract') != 'original-fiber-arclength-v2':
                raise ValueError('legacy failure trial coordinate contract is not reusable')
            reference = doc['reference_points_base_xyz']
            for point, field in ((reference[0], 'expected_start_base_xyz'),
                                 (reference[-1], 'expected_end_base_xyz')):
                error = np.linalg.norm(np.asarray(point)-np.asarray(provenance[field]))
                if not np.isfinite(error) or error > 0.25:
                    raise ValueError('saved trial endpoint disagrees with requested base XYZ')
            key = (provenance['candidate_id'], provenance['direction'],
                   float(provenance['target_distance_base']))
            distance = key[2]
        else:
            key = provenance['fiber_path']
            distance = targets[key]
        audit = audit_failure(doc, distance, grace)
        if global_worklist and 'native_reproduced' in provenance:
            if provenance['native_reproduced'] != audit['valid']:
                raise ValueError('native pruning audit disagrees with saved-geometry audit')
        rows.append((key, str(path), audit))
    return rows


def load_trace_reference(path):
    """Do not transfer full replay geometry to the union coordinator."""
    doc = json.loads(Path(path).read_text())
    return {'provenance': doc['provenance'],
            'reference_points_base_xyz': doc['reference_points_base_xyz']}
