"""Read-only exact graph/geometry and normal-axis checks after sign alignment."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from vc_object_store import CollectionReader
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fibers_digest
from .fiber_graph_store import FiberGraph, PREFIX


def validate(original, oriented):
    started = time.monotonic()
    before, after = FiberGraph(original), FiberGraph(oriented)
    assert np.array_equal(before.ids, after.ids), 'Fiber identities changed'
    assert fibers_digest(original, before.ids) == fibers_digest(oriented, after.ids), 'Fiber geometry changed'
    for name in ('event_ids', 'edge_uuid', 'fiber_a', 'fiber_b', 'arc_a', 'arc_b', 'point_a', 'point_b',
                 'parallel_score', 'perpendicular_score', 'distance', 'parallel_distance', 'evidence_weight',
                 'valid_a', 'valid_b'):
        assert np.array_equal(getattr(before, name), getattr(after, name)), name
    # Check all typed diagnostic fields, not just columns consumed by the solve.
    excluded = {f'normal_{side}_{axis}' for side in ('a', 'b') for axis in ('x', 'y', 'z')}
    table = PREFIX+'constraints'
    with CollectionReader(str(original)) as a, CollectionReader(str(oriented)) as b:
        ac = [c['name'] for c in before.measurement_info['columns']]
        bc = [c['name'] for c in after.measurement_info['columns']]
        assert ac == bc, 'Measurement schema changed'
        keep = [i for i, name in enumerate(ac) if name not in excluded]
        ar = {row[0]: row for row in a.extension_rows(table)}
        br = {row[0]: row for row in b.extension_rows(table)}
        assert ar.keys() == br.keys(), 'Measurement relation identities changed'
        for key, row in ar.items():
            assert [row[i] for i in keep] == [br[key][i] for i in keep], key
        for edge in before.edge_uuid:
            assert a.edge(str(edge)) == b.edge(str(edge)), edge
    print('All original fibers and non-normal constraint fields match exactly.', flush=True)
    flips = valid_count = invalid_count = 0
    for side in ('a', 'b'):
        old, new = getattr(before, 'normal_'+side), getattr(after, 'normal_'+side)
        valid = getattr(after, 'valid_'+side)
        plus, minus = abs(new-old).sum(axis=1), abs(new+old).sum(axis=1)
        assert np.all(np.minimum(plus[valid], minus[valid]) <= 1e-12), side
        assert np.all(np.isfinite(new[valid])), side
        side_flips = int(np.count_nonzero(valid & (plus > 1e-12)))
        valid_count += int(valid.sum())
        invalid_count += int((~valid).sum())
        flips += side_flips
        print(f'Endpoint {side}: {side_flips:,} sign flips; all axes preserved.', flush=True)
    result = dict(integrity='ok', integrity_scope='native snapshot plus exact logical graph comparison',
                  exact_constraint_fields=True, valid_endpoints=valid_count,
                  invalid_endpoints=invalid_count, flipped_endpoints=flips,
                  seconds=time.monotonic()-started)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('oriented', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate(args.original, args.oriented)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
