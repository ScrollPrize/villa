"""Native event importer parity, ownership and unsupported-node regressions."""
from types import SimpleNamespace
import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_native_prepare import select_inputs,prepare_events


def inputs():
    return dict(source_id=np.array(['a','b','c','d','e']),family=np.array([0,1,0,1,0]),
                end=np.array([12.,13.,14.,15.,16.]),
                i=np.array([0,0,2,2,3,0,4]),j=np.array([1,2,3,4,1,3,0]),
                ds_a=np.arange(7,dtype=float),ds_b=np.arange(7,dtype=float)+1,
                parallel=np.array([False,True,False,True,True,True,True]))


@pytest.mark.parametrize('workers',[1,2,8])
def test_native_uv_prunes_unsupported_and_preserves_order(workers):
    original=inputs();data={k:v.copy() for k,v in original.items()}
    result,selected=select_inputs(data,workers=workers)
    # 0-1 and 2-3 establish perpendicular support. 4 has only parallels;
    # 0-3's claimed parallel relation is incompatible with its H/V family.
    np.testing.assert_array_equal(selected,[0,1,2,3])
    np.testing.assert_array_equal(result['i'],[0,0,2,3])
    np.testing.assert_array_equal(result['j'],[1,2,3,1])
    np.testing.assert_array_equal(result['ds_a'],[0.,1.,2.,4.])
    np.testing.assert_array_equal(result['ds_b'],[1.,2.,3.,5.])
    np.testing.assert_array_equal(result['parallel'],[False,True,False,True])
    for k in original:np.testing.assert_array_equal(original[k],data[k])
    assert not np.shares_memory(result['ds_a'],data['ds_a'])
    assert not result['i'].flags.writeable
    # Buffers remain valid after other views and the input go out of scope.
    copied=result['ds_a'];del result,data
    np.testing.assert_array_equal(copied,[0.,1.,2.,4.])


def test_native_uv_node_mask_precedes_perpendicular_support():
    result,selected=select_inputs(inputs(),node_mask=np.array([True,False,True,True,True]),workers=2)
    np.testing.assert_array_equal(selected,[2,3])
    np.testing.assert_array_equal(result['i'],[0]);np.testing.assert_array_equal(result['j'],[1])
    np.testing.assert_array_equal(result['ds_a'],[2.])


def test_native_uv_cached_graph_masks_and_mapping():
    d=inputs();graph=SimpleNamespace(i=d['i'],j=d['j'],arc_a=d['ds_a'],arc_b=d['ds_b'],parallel=d['parallel'],
        valid_a=np.array([True,True,True,True,True,True,True]),
        valid_b=np.array([False,True,True,True,True,True,True]),valid_connector=np.ones(7,bool))
    # Reject the only event supporting fiber 0/1 via the endpoint mask.
    paths=SimpleNamespace(graph_index=np.arange(5),ids=d['source_id'],catalog=dict(family=d['family']),
        keys={'arc':np.column_stack((np.zeros(5),d['end'])).ravel()},offset=np.arange(0,11,2))
    result,selected=prepare_events(paths,graph,workers=2)
    np.testing.assert_array_equal(selected,[2,3])
    np.testing.assert_array_equal(result['source_id'],['c','d'])
    np.testing.assert_array_equal(result['ds_a'],[2.])
    paths.graph_index=np.array([-1,1,2,3,4])
    result,selected=prepare_events(paths,graph,workers=2)
    np.testing.assert_array_equal(selected,[2,3])


def test_native_uv_empty_and_invalid_input():
    d=inputs();d['parallel'][:]=True
    result,selected=select_inputs(d,workers=1)
    assert selected.size==0 and result['i'].size==0
    with pytest.raises(ValueError,match='worker'):
        select_inputs(inputs(),workers=0)
    d=inputs();d['i'][0]=999
    with pytest.raises(ValueError,match='catalog'):
        select_inputs(d,workers=2)


def test_native_uv_parallel_block_order():
    # Exercise prefix offsets across several event blocks, not only one task.
    n=150003;edge=np.arange(n)
    data=dict(source_id=np.array(['h','v']),family=np.array([0,1]),end=np.array([3.,4.]),
              i=np.zeros(n,np.int64),j=np.ones(n,np.int64),ds_a=edge.astype(float),
              ds_b=edge.astype(float)+.25,parallel=np.zeros(n,bool))
    one,chosen=select_inputs(data,workers=1)
    many,other=select_inputs(data,workers=8)
    np.testing.assert_array_equal(chosen,other)
    for key in one:np.testing.assert_array_equal(one[key],many[key])


@pytest.mark.parametrize('workers',[1,4])
def test_native_uv_result_columns_match_typed_schema(tmp_path,workers):
    import uuid
    from vc_object_store import CollectionReader,fiber_graph_solve_table,export_fixed_uv_columns,fiber_uuid_bytes
    from .fiber_graph_fixture import measured_store,uid
    source=measured_store(tmp_path,[(0,10,20,False,.5)])
    with CollectionReader(str(source)) as reader:position=reader.source
    ids=np.array([uid(10),uid(20)])
    text=np.char.encode(ids,'ascii').view(np.uint8).reshape(-1,36)
    binary=fiber_uuid_bytes(text,workers)
    np.testing.assert_array_equal(binary,np.array([list(uuid.UUID(i).bytes) for i in ids],np.uint8))
    assert not binary.flags.writeable
    destination=tmp_path/'uv.vcobj';run=uid(100);input_run=uid(99)
    family=np.array([0,1],np.int64);winding=np.array([3.,4.]);direction=np.array([1.,-1.])
    owner=np.array([0,0,1,1,1]);arc=np.array([0.,10.,0.,-5.,-10.])
    q=np.array([3.1,3.2,4.3,4.3,4.3]);x=np.array([1.,2.,3.,3.,3.]);v=np.array([2.,2.,1.,5.,10.])
    result=export_fixed_uv_columns(destination,str(source),position,run,input_run,'gauss_newton_cg',
        dict(global_integer_shift=2,batch_size=32768,assignments_fixed=True),
        binary,family,winding,direction,owner,arc,q,x,v,np.array([1000.,1100.]),workers)
    assert result['run_uuid']==run and destination.exists()
    def rows(reader,role,prefix=()):
        spec=fiber_graph_solve_table(role)
        return [dict(zip([c['name'] for c in spec['columns']],r)) for r in reader.extension_rows(spec['name'],list(prefix))]
    with CollectionReader(str(destination)) as reader:
        fibers=rows(reader,'fibers');assert len(fibers)==2
        for i,fid in enumerate(ids):
            record,=rows(reader,'fibers',[uuid.UUID(fid).bytes,uuid.UUID(run).bytes])
            assert record['family']==family[i] and record['enabled']==1 and record['selected_winding']==winding[i]
            positions=rows(reader,'positions',[uuid.UUID(fid).bytes,uuid.UUID(run).bytes])
            actual=sorted(positions,key=lambda p:p['node_index']);selected=np.flatnonzero(owner==i)
            assert [p['node_index'] for p in actual]==list(range(len(selected)))
            np.testing.assert_array_equal([p['arc'] for p in actual],arc[selected]*direction[i])
            np.testing.assert_array_equal([p['u_turns'] for p in actual],q[selected]-winding[i])
            np.testing.assert_array_equal([p['u_vx'] for p in actual],x[selected])
            np.testing.assert_array_equal([p['v_vx'] for p in actual],v[selected])
            assert all(p['layer']==3 and p['hypothesis']==int(winding[i]) for p in actual)
        input_row,=rows(reader,'inputs')
        assert input_row['store_uuid']==uuid.UUID(position['store_uuid']).bytes
        assert input_row['generation_uuid']==uuid.UUID(position['generation']).bytes
        assert input_row['commit_sequence']==position['commit_sequence']
        assert input_row['input_run_uuid']==uuid.UUID(input_run).bytes
        assert len(rows(reader,'runs'))==1 and len(rows(reader,'widths'))==2
    stale=dict(position,commit_sequence=position['commit_sequence']+1)
    failed=tmp_path/'stale.vcobj'
    with pytest.raises(ValueError,match='snapshot'):
        export_fixed_uv_columns(failed,str(source),stale,uid(101),input_run,'gauss_newton_cg',{},
            binary,family,winding,direction,owner,arc,q,x,v,np.array([1000.]),workers)
    assert not failed.exists()


def test_native_uuid_column_canonicalization():
    from vc_object_store import canonical_uuid_ascii
    text=np.array(['12345678-9ABC-DEF0-1234-56789ABCDEF0','00000000-0000-0000-0000-000000000001'],dtype='S36')
    result=canonical_uuid_ascii(text.view(np.uint8).reshape(-1,36),2)
    assert not result.flags.writeable
    np.testing.assert_array_equal(result.view('S36').ravel(),np.char.lower(text))
    invalid=text.copy();invalid[0]=b'x'*36
    with pytest.raises(ValueError,match='UUID'):
        canonical_uuid_ascii(invalid.view(np.uint8).reshape(-1,36),2)
    with pytest.raises(ValueError,match='36'):
        canonical_uuid_ascii(np.zeros((2,37),np.uint8),2)
