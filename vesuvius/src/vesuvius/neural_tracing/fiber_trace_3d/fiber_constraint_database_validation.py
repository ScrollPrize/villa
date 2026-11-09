"""Read-only typed-measurement and source object-store fiber geometry checks."""
import argparse
import json
from pathlib import Path
import re
import uuid

import numpy as np
from vc_object_store import CollectionReader
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import (
    fiber_catalog, fibers_digest, read_polylines,
)
from .fiber_graph_store import FiberGraph, PREFIX


# Assert coverage of the public C++ measurement contract in the typed extension;
# this maps schema names for validation, rather than decoding another format.
MEASUREMENT_FIELDS = dict(
    pieceA='piece_a', pieceB='piece_b', closestDistanceBaseVoxels='closest_distance',
    hardContinuity='hard_continuity', artificialTileSeam='artificial_tile_seam',
    allowMeasurementDisable='allow_measurement_disable', localEvent='local_event',
    localOrientation='local_orientation', supportABegin='support_a_begin', supportAEnd='support_a_end',
    supportBBegin='support_b_begin', supportBEnd='support_b_end',
    parallelCorrespondenceSamples='parallel_correspondence_samples',
    parallelMeanAdvanceResidualFraction='parallel_advance_residual',
    parallelMeanConnectorTangentAbsDot='parallel_connector_tangent_dot',
    parallelMeanConnectorLengthChangeFraction='parallel_connector_length_change',
    parallelMeanConnectorDirectionChange='parallel_connector_direction_change',
    parallelLimitHitFraction='parallel_limit_hit_fraction', signedWindingDelta='signed_winding_delta',
    perpendicularNormalAlignment='perpendicular_normal_alignment', windingNormalComponent='winding_normal_component',
    signedParallelWindingDelta='signed_parallel_winding_delta', parallelNormalAlignment='parallel_normal_alignment',
    arcABaseVoxels='arc_a', arcBBaseVoxels='arc_b', parallelScore='parallel_score',
    perpendicularScore='perpendicular_score', windingDistance='las_winding',
    parallelWindingDistance='parallel_las_winding', evidenceWeight='evidence_weight',
)


def validate(path, fibers, all_measurements=False, *, header):
    graph = FiberGraph(path)
    source_ids = {r['id'] for r in fiber_catalog(fibers, graph.volume)}
    assert set(graph.ids) <= source_ids, 'Graph refers to missing source fibers'
    assert fibers_digest(path, graph.ids) == fibers_digest(fibers, graph.ids), 'Source fiber version mismatch'
    body = Path(header).read_text().split('struct FiberTraceConstraint {', 1)[1].split('\n};', 1)[0]
    expected = set(re.findall(r'(?:std::\w+(?:<[^;]+?>)?|double|bool|int|cv::Vec3d)\s+(\w+)\s*(?:=|\{|;)', body))
    expected |= {'supportAEnd', 'supportBEnd'}
    count = len(graph.event_ids)
    step = 1 if all_measurements else max(1, count//1000)
    take = np.flatnonzero(graph.event_ids % step == 0)
    table = PREFIX+'constraints'
    columns = [c['name'] for c in graph.measurement_info['columns']]
    endpoint_ids = np.unique(np.r_[graph.fiber_a[take[:100]], graph.fiber_b[take[:100]]])
    curves = {row['id']: xyz for row, xyz in read_polylines(fibers, endpoint_ids, volume=graph.volume)}
    with CollectionReader(str(path)) as store:
        selected_rows = []
        for k in take:
            rows = store.extension_rows(table, [uuid.UUID(graph.edge_uuid[k]).bytes])
            assert len(rows) == 1, 'Missing or ambiguous measurement row'
            selected_rows.append(rows[0])
    for checked, (k, values) in enumerate(zip(take, selected_rows)):
        row = dict(zip(columns, values))
        extra = json.loads(row['extra'] or '{}').get('measurement', {})
        represented = {name for name, column in MEASUREMENT_FIELDS.items() if column in row}
        represented |= {'pointABaseXYZ', 'pointBBaseXYZ'} | extra.keys()
        assert expected <= represented, ('Unrepresented measurement fields', expected-represented)
        # Conflicting copies are retained by import in extra; do not silently
        # substitute them for the typed values or the core endpoint anchors.
        for name, value in extra.items():
            if name in MEASUREMENT_FIELDS:
                assert value == row[MEASUREMENT_FIELDS[name]], name
            elif name in ('pointABaseXYZ', 'pointBBaseXYZ'):
                assert value == getattr(graph, 'point_'+name[5].lower())[k].tolist(), name
        for side in ('a', 'b'):
            valid = getattr(graph, 'valid_'+side)[k]
            xyz = [row['normal_'+side+'_'+axis] for axis in ('x', 'y', 'z')]
            if valid:
                assert np.isfinite(xyz).all() and abs(np.linalg.norm(xyz)-1) < 1e-5
            else:
                assert xyz == [None, None, None]
            if checked < 100:
                points = curves[getattr(graph, 'fiber_'+side)[k]]
                lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
                remaining = float(getattr(graph, 'arc_'+side)[k])
                actual = None
                for a, b, length in zip(points[:-1], points[1:], lengths):
                    if remaining <= length+1e-7:
                        fraction = min(1, max(0, remaining/length)) if length else 0
                        actual = a+fraction*(b-a)
                        break
                    remaining -= length
                assert actual is not None and np.linalg.norm(actual-getattr(graph, 'point_'+side)[k]) < 1e-5
    degrees = np.bincount(np.r_[graph.i, graph.j], minlength=len(graph.ids))
    assert degrees.sum() == 2*count
    result = dict(integrity='ok', integrity_scope='native snapshot and logical typed measurements; not SQLite page integrity',
                  fibers=len(graph.ids), constraints=count, measurement_rows_checked=len(take),
                  geometry_endpoints_checked=2*min(len(take), 100), degree_sum=int(degrees.sum()))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('--fibers', type=Path, required=True)
    parser.add_argument('--header', type=Path, required=True)
    parser.add_argument('--all-measurements', action='store_true')
    args = parser.parse_args()
    validate(args.database, args.fibers, args.all_measurements, header=args.header)
