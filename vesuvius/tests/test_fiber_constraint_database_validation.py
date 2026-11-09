from pathlib import Path
import uuid

import numpy as np
import pytest
from vc_object_store import CollectionReader, fiber_constraint_extension, import_initial_records, write_extension_store
from vesuvius.neural_tracing.fiber_trace_3d.fiber_constraint_database_validation import validate
from vesuvius.neural_tracing.fiber_trace_3d.fiber_constraint_database_report import summarize
from vesuvius.neural_tracing.fiber_trace_3d.fiber_oriented_normals_validation import validate as validate_normals
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_initial_fibers
from .fiber_graph_fixture import uid, measured_store


HEADER = Path(__file__).resolve().parents[2]/'volume-cartographer/core/include/vc/fiber_tracer/FiberTraceConstraints.hpp'


def geometry_graph(root, *, corrupt_point=False, workers=2, packed=True):
    curves=[np.array([[0.,0,k],[10.,0,k],[10.,10,k]]) for k in range(3)]
    source=write_initial_fibers(root/'fibers.vcobj',[(dict(id=uid(k+1)),xyz) for k,xyz in enumerate(curves)],
                                volume=uid(999),workers=workers,packed=packed)['path']
    table=fiber_constraint_extension('')
    fields=dict(edge_uuid=uuid.UUID(uid(100)).bytes,source_id=0,source_fiber_a=1,source_fiber_b=2,
        source_reversed=0,arc_a=15.,arc_b=5.,parallel_score=1.,perpendicular_score=0.,
        las_winding=1.,parallel_las_winding=1.,evidence_weight=1.,local_event=1,hard_continuity=0,
        normal_a_valid=1,normal_b_valid=1,normal_a_x=0.,normal_a_y=0.,normal_a_z=1.,
        normal_b_x=0.,normal_b_y=0.,normal_b_z=1.)
    record=dict(uuid=uid(100),a=uid(1),b=uid(2),kind='parallel',
        a_anchor=dict(position=[10.,5.,0.]),b_anchor=dict(position=[6. if corrupt_point else 5.,0.,1.]))
    edge=dict(record=record,extensions=[dict(table=table['name'],values=[fields.get(c['name']) for c in table['columns']])])
    graph=import_initial_records(root/'graph.vcobj',[],edges=[edge],extensions=[table],
                                  dependencies=[source],workers=workers,packed=packed)['path']
    return graph,source


@pytest.mark.parametrize('workers,packed', [(1,False),(2,False),(2,True)])
def test_validation_reads_store_geometry(tmp_path,workers,packed):
    graph,source=geometry_graph(tmp_path/'good',workers=workers,packed=packed)
    result=validate(graph,source,header=HEADER)
    assert result['fibers']==3 and result['constraints']==1 and result['geometry_endpoints_checked']==2
    broken,source=geometry_graph(tmp_path/'bad',corrupt_point=True,workers=workers,packed=packed)
    with pytest.raises(AssertionError):
        validate(broken,source,header=HEADER)


def test_report_counts_repeated_events_and_both_endpoint_orders(tmp_path):
    graph=measured_store(tmp_path,[(0,1,2,True,.25),(1,2,1,True,.75),(2,1,3,False,.5)])
    result=summarize(graph)
    assert result['fibers']==3 and result['events']==3
    assert result['unique_fiber_pairs']==2
    assert result['neighbor_degree']==dict(mean=4/3,median=1.,p95=1,maximum=2)
    assert result['event_statistics']['parallel']==2
    assert [r['nearest_half_winding'] for r in result['winding_distance_bins']]==[.5,1.,.5]


def normal_overlay(source,destination,*,axis_error=False,measurement_error=False):
    with CollectionReader(str(source)) as store:
        table=next(t for t in store.extension_tables() if t['name']=='ext_fiber_graph_solve_constraints')
        rows=[list(r) for r in store.extension_rows(table['name'])]
        table['based_on']=[store.source]
        table.pop('table_uuid')
        table.pop('last_modified_commit')
    index={c['name']:i for i,c in enumerate(table['columns'])}
    rows[0][index['normal_a_z']]=-1.
    if axis_error:
        rows[0][index['normal_a_y']]=1.
    if measurement_error:
        rows[0][index['closest_distance']]=2.
    write_extension_store(str(destination),[str(source)],[table],{table['name']:rows})
    return destination


def test_oriented_validation_preserves_all_fields_and_axes(tmp_path):
    graph,unused=geometry_graph(tmp_path/'original')
    oriented=normal_overlay(graph,tmp_path/'oriented.vcobj')
    result=validate_normals(graph,oriented)
    assert result['flipped_endpoints']==1 and result['valid_endpoints']==2
    bad_axis=normal_overlay(graph,tmp_path/'axis.vcobj',axis_error=True)
    with pytest.raises(AssertionError):
        validate_normals(graph,bad_axis)
    bad_measurement=normal_overlay(graph,tmp_path/'measurement.vcobj',measurement_error=True)
    with pytest.raises(AssertionError):
        validate_normals(graph,bad_measurement)
