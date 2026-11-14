"""Native graph columns avoid corpus-sized Python records and UUID strings."""
import uuid
import numpy as np
import pytest
from vc_object_store import CollectionReader

from .fiber_graph_fixture import measured_store,uid


@pytest.mark.parametrize('workers,packed',[(1,False),(4,True)])
def test_numeric_graph_columns_match_legacy_measurements(tmp_path,workers,packed):
    source=measured_store(tmp_path,[(99,20,10,False,.27),(3,10,20,True,1.03)],
        workers=workers,packed=packed,isolated_ids=(30,),
        curves={i:np.array([[0.,0.,0.],[100.,0.,0.]]) for i in (10,20,30)},
        field_overrides={99:dict(arc_a=12.345678,arc_b=2.25)})
    with CollectionReader(str(source)) as reader:
        old=reader.fiber_constraints();columns=reader.fiber_graph_columns(workers)
        position=reader.source
        assert reader.source==position
    ids=columns['fiber_uuid_ascii'].view('S36').ravel().astype('U36')
    np.testing.assert_array_equal(ids,[uid(10),uid(20),uid(30)])
    np.testing.assert_array_equal(columns['lengths'],[100.,100.,100.])
    assert columns['volume']==uid(999)
    assert columns['edge_uuid'].shape==(2,16) and columns['edge_uuid'].dtype==np.uint8
    edge=[str(uuid.UUID(bytes=bytes(row))) for row in columns['edge_uuid']]
    assert edge==old['edge_uuid']
    np.testing.assert_array_equal(columns['event_ids'],np.frombuffer(old['ids'],dtype='=i8'))
    endpoints=columns['endpoints']
    np.testing.assert_array_equal(ids[endpoints[:,0]],old['fiber_a'])
    np.testing.assert_array_equal(ids[endpoints[:,1]],old['fiber_b'])
    old_values=np.frombuffer(old['values'],dtype='=f8').reshape(-1,16)
    np.testing.assert_array_equal(columns['arcs'],old_values[:,:2])
    np.testing.assert_array_equal(columns['normal_a'],old_values[:,2:5])
    np.testing.assert_array_equal(columns['normal_b'],old_values[:,5:8])
    np.testing.assert_array_equal(columns['connector'],old_values[:,8:11])
    np.testing.assert_array_equal(columns['distance'],old_values[:,11])
    np.testing.assert_array_equal(columns['parallel'],old_values[:,12])
    np.testing.assert_array_equal(columns['valid_a'],old_values[:,13])
    np.testing.assert_array_equal(columns['valid_b'],old_values[:,14])
    np.testing.assert_array_equal(columns['valid_connector'],old_values[:,15])
    assert not endpoints.flags.writeable
    assert all(value.dtype.kind not in 'UO' for value in columns.values() if isinstance(value,np.ndarray))


def test_native_graph_catalog_honors_extraction_run_subset(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import write_results
    source=measured_store(tmp_path,[(1,10,20,False,.5)],isolated_ids=(30,))
    selected=tmp_path/'selected.vcobj'
    write_results(selected,[source],'constraint_extraction','test',
        dict(fibers=[dict(fiber_uuid=uid(i)) for i in (10,20)]))
    with CollectionReader(str(selected)) as reader:
        columns=reader.fiber_graph_columns(4)
    np.testing.assert_array_equal(columns['fiber_uuid_ascii'].view('S36').ravel().astype('U36'),[uid(10),uid(20)])
    np.testing.assert_array_equal(columns['endpoints'],[[0,1]])
    other=tmp_path/'other.vcobj'
    write_results(other,[selected],'constraint_extraction','test',
        dict(fibers=[dict(fiber_uuid=uid(10))]))
    with CollectionReader(str(other)) as reader:
        with pytest.raises(ValueError,match='one extraction graph'):
            reader.fiber_graph_columns(2)


def test_native_graph_rejects_bad_worker_count(tmp_path):
    source=measured_store(tmp_path,[(1,10,20,False,.5)])
    with CollectionReader(str(source)) as reader:
        with pytest.raises(ValueError,match='workers'):
            reader.fiber_graph_columns(0)


def test_fiber_graph_import_never_uses_python_record_or_uuid_list_paths(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph,CACHE_COLUMNS
    source=measured_store(tmp_path,[(1,10,20,False,.5)])
    def forbidden(*args,**kwargs):raise AssertionError('Production graph import must use native numeric columns')
    monkeypatch.setattr(CollectionReader,'objects',forbidden)
    monkeypatch.setattr(CollectionReader,'fiber_constraints',forbidden)
    monkeypatch.setattr(CollectionReader,'extension_rows',forbidden)
    graph=FiberGraph(source,workers=2)
    assert set(CACHE_COLUMNS).issubset(vars(graph))
    assert 'fiber_a' not in vars(graph) and 'fiber_b' not in vars(graph) and 'edge_uuid' not in vars(graph)
    # Legacy small selections do not materialize all endpoint UUID strings.
    assert graph.fiber_a[0]==uid(10) and graph.fiber_b[0]==uid(20) and graph.edge_uuid[0]==uid(1001)
    assert graph.edge_uuid_bytes.nbytes==16*len(graph.event_ids)
    cache=tmp_path/'thin-cache';graph.save_cache(cache)
    assert not (cache/'fiber_a.npy').exists() and not (cache/'fiber_b.npy').exists()
    assert not (cache/'edge_uuid.npy').exists()
    assert np.load(cache/'edge_uuid_bytes.npy',mmap_mode='r').dtype==np.uint8
    saved=FiberGraph.cached(source,cache)
    for name in CACHE_COLUMNS:np.testing.assert_array_equal(getattr(saved,name),getattr(graph,name))
