"""Native, immutable measured-graph fixtures shared by workflow regression tests."""
import uuid
import numpy as np
from vc_object_store import import_initial_records, fiber_constraint_extension
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_initial_fibers


def uid(number):
    return str(uuid.UUID(int=int(number)))


def measured_store(root, events, *, workers=2, packed=True, isolated_ids=(), field_overrides=None,
                   curves=None, volume=None, point_overrides=None):
    """Events: (source event ID, A, B, parallel, winding distance)."""
    ids=sorted({x for e in events for x in e[1:3]} | set(isolated_ids))
    fibers=write_initial_fibers(root/'fibers.vcobj',
        [(dict(id=uid(i),family='H' if i%2 else 'V'),
          np.asarray(curves[i]) if curves is not None else np.array([[0.,0.,0.],[10.,0.,0.]])) for i in ids],
        volume=volume or uid(999),workers=workers,packed=packed)['path']
    table=fiber_constraint_extension('')
    edges=[]
    for eid,a,b,parallel,distance in events:
        edge_uuid=uid(1000+eid)
        fields=dict(edge_uuid=uuid.UUID(edge_uuid).bytes,source_id=eid,source_fiber_a=a,source_fiber_b=b,
            source_reversed=int(a>b),arc_a=0.,arc_b=0.,parallel_score=float(parallel),
            perpendicular_score=float(not parallel),las_winding=float(distance),
            parallel_las_winding=float(distance),evidence_weight=1.,
            normal_a_valid=1,normal_b_valid=1,normal_a_x=1.,normal_a_y=0.,normal_a_z=0.,
            normal_b_x=1.,normal_b_y=0.,normal_b_z=0.)
        fields.update((field_overrides or {}).get(eid,{}))
        pa,pb=([0.,0.,0.],[1.,0.,0.]) if a<b else ([1.,0.,0.],[0.,0.,0.])
        if point_overrides is not None and eid in point_overrides:
            pa,pb=point_overrides[eid]
        edges.append(dict(record=dict(uuid=edge_uuid,a=uid(min(a,b)),b=uid(max(a,b)),
                          kind='parallel' if parallel else 'perpendicular',
                          a_anchor=dict(position=pa),b_anchor=dict(position=pb)),
                          extensions=[dict(table=table['name'],values=[fields.get(c['name']) for c in table['columns']])]))
    graph=import_initial_records(root/'graph.vcobj',[],edges=edges,extensions=[table],
                                 dependencies=[fibers],workers=workers,packed=packed)['path']
    return graph
