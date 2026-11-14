"""Read-only compact-measurement and source object-store geometry checks."""
import argparse
import json
from pathlib import Path
import uuid

import numpy as np
from vc_object_store import (CollectionReader, fiber_constraint_extension,
                             decode_fiber_arc, decode_fiber_direction, decode_fiber_winding)
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import (
    fiber_catalog, fibers_digest, read_polylines,
)
from .fiber_graph_store import FiberGraph, PREFIX


def validate(path, fibers, all_measurements=False, *, header=None):
    """Check every persisted field; omitted XYZ is deliberately not reconstructed.

    The legacy header argument is accepted for CLI compatibility. The current
    storage contract is the native compact extension, not the extraction's
    larger transient candidate structure.
    """
    graph = FiberGraph(path)
    source_ids = {r['id'] for r in fiber_catalog(fibers, graph.volume)}
    assert set(graph.ids) <= source_ids, 'Graph refers to missing source fibers'
    assert fibers_digest(path, graph.ids) == fibers_digest(fibers, graph.ids), 'Source fiber version mismatch'
    expected = fiber_constraint_extension('')['columns']
    assert graph.measurement_info['format_version'] == 4
    assert graph.measurement_info['columns'] == expected, 'Unexpected compact measurement schema'
    count = len(graph.event_ids)
    step = 1 if all_measurements else max(1, count//1000)
    take = np.arange(0, count, step)
    table = PREFIX+'constraints'
    columns = [c['name'] for c in expected]
    endpoint_ids = np.unique(np.r_[graph.fiber_a[take], graph.fiber_b[take]])
    # Geometry is read only to determine the legal arc range, never to invent
    # a stored measurement XYZ value or validate it against a reconstruction.
    lengths = {row['id']: float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
               for row, xyz in read_polylines(fibers, endpoint_ids, volume=graph.volume)}
    with CollectionReader(str(path)) as store:
        for k in take:
            values = store.extension_rows(table, [uuid.UUID(graph.edge_uuid[k]).bytes])
            assert len(values) == 1, 'Missing or ambiguous measurement row'
            row = dict(zip(columns, values[0]))
            assert row['edge_uuid'] == uuid.UUID(graph.edge_uuid[k]).bytes
            for side in ('a', 'b'):
                arc = decode_fiber_arc(row['arc_'+side])
                assert arc == getattr(graph, 'arc_'+side)[k]
                length = lengths[getattr(graph, 'fiber_'+side)[k]]
                # Compare in the stored float32 domain: rounding the true end
                # position gives the greatest legal encoded arc, with no
                # additional tolerance admitting another representable value.
                maximum_arc = float(np.float32(length))
                assert np.isfinite(arc) and 0 <= arc <= maximum_arc, 'Arc outside source fiber'
            for field, validity in (('normal_a','valid_a'), ('normal_b','valid_b'),
                                    ('connector','valid_connector')):
                raw = row[field]
                assert len(raw) == 3
                vector = np.asarray(decode_fiber_direction(raw))
                valid = raw[2] != 0
                assert bool(getattr(graph, validity)[k]) == valid
                assert np.array_equal(vector, getattr(graph, field)[k]), field
                if valid:
                    assert np.isfinite(vector).all() and abs(np.linalg.norm(vector)-1) < 1e-12
                else:
                    assert np.array_equal(vector, np.zeros(3))
            winding = decode_fiber_winding(row['winding_distance'])
            assert winding == graph.distance[k]
            if row['extra'] is not None:
                assert isinstance(json.loads(row['extra']), dict), 'Extension extra must be a JSON object'
    degrees = np.bincount(np.r_[graph.i, graph.j], minlength=len(graph.ids))
    assert degrees.sum() == 2*count
    result = dict(integrity='ok', integrity_scope='native snapshot and all persisted compact measurement fields; not SQLite page integrity',
                  fibers=len(graph.ids), constraints=count, measurement_rows_checked=len(take),
                  arc_ranges_checked=2*len(take), geometry_endpoints_checked=0,
                  endpoint_xyz='absent; access fails explicitly', degree_sum=int(degrees.sum()))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('--fibers', type=Path, required=True)
    parser.add_argument('--header', type=Path)
    parser.add_argument('--all-measurements', action='store_true')
    args = parser.parse_args()
    validate(args.database, args.fibers, args.all_measurements, header=args.header)
