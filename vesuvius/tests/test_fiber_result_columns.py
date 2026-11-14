"""Typed result exports match existing extension rows without Python corpus rows."""
import json
import numpy as np
import pytest
from vc_object_store import CollectionReader,fiber_graph_solve_table

from .fiber_graph_fixture import measured_store,uid
from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import write_results,write_fiber_result_columns


@pytest.mark.parametrize('stage',['classification','normal_sign','winding'])
@pytest.mark.parametrize('workers',[1,4])
def test_fiber_result_column_export_matches_named_row_writer(tmp_path,stage,workers):
    source=measured_store(tmp_path,[(1,10,20,False,.5)],isolated_ids=(30,))
    with CollectionReader(str(source)) as reader:pin=reader.source
    ids=np.array([uid(10),uid(20),uid(30)])
    if stage=='classification':
        columns=dict(family=np.array([0,1,2],np.int8),enabled=np.array([True,True,False]),
                     p_h=np.array([.8,.1,.3],np.float32),p_v=np.array([.1,.8,.2],np.float32),
                     p_broken=np.array([.1,.1,.5],np.float32))
    elif stage=='normal_sign':
        columns=dict(normal_sign=np.array([1,-1,1]),sign_probability=np.array([.9,.8,.95]))
    else:
        columns=dict(family=np.array([0,1,0]),enabled=np.ones(3,np.int64),
                     selected_winding=np.array([-3,5,2]),winding_mean=np.array([-3.1,5.2,2.05]),component=np.array([0,0,1]))
    metrics=[dict(iteration=15,scope='all',name='mean_residual',value=.25,units='turns')] if stage=='winding' else []
    # The reference writer receives only this three-fiber fixture. Production
    # callers instead supply the numeric columns directly.
    rows=[dict(fiber_uuid=str(fid),**{k:float(v[i]) if v.dtype.kind=='f' else int(v[i]) for k,v in columns.items()}) for i,fid in enumerate(ids)]
    tables=dict(fibers=rows)
    if metrics:tables['metrics']=metrics
    reference=tmp_path/'reference.vcobj';target=tmp_path/'columns.vcobj';run=uid(1000)
    write_results(reference,[source],stage,'test',tables,settings=dict(weight=.25),run_uuid=run,input_runs=[uid(999)])
    timings=tmp_path/'export.json'
    result=write_fiber_result_columns(target,source,pin,stage,'test',ids,columns,
        settings=dict(weight=.25),input_run=uid(999),metrics=metrics,workers=workers,run_uuid=run,timings_path=timings)
    assert result==run and json.loads(timings.read_text())['bytes']==target.stat().st_size
    for role in ('fibers','inputs','runs',*(['metrics'] if metrics else [])):
        spec=fiber_graph_solve_table(role);names=[c['name'] for c in spec['columns']]
        with CollectionReader(str(reference)) as before,CollectionReader(str(target)) as after:
            expected=before.extension_rows(spec['name']);actual=after.extension_rows(spec['name'])
        if role=='runs':
            settings_column=names.index('settings')
            for row in expected:row[settings_column]=json.loads(row[settings_column])
            for row in actual:row[settings_column]=json.loads(row[settings_column])
        assert sorted(actual,key=lambda r:r[:2])==sorted(expected,key=lambda r:r[:2])
    with CollectionReader(str(target)) as reader:
        columns_read=reader.fiber_result_columns(run)
        assert columns_read['fiber_uuid'].shape==(3,16)


def test_fiber_result_column_export_rejects_invalid_values_atomically(tmp_path):
    source=measured_store(tmp_path,[(1,10,20,False,.5)])
    with CollectionReader(str(source)) as reader:pin=reader.source
    ids=np.array([uid(10),uid(20)])
    for name,values in [('family',np.array([0,3])),('normal_sign',np.array([1,0])),
                        ('enabled',np.array([1,2])),('p_h',np.array([.5,1.01])),
                        ('winding_mean',np.array([1.,np.nan]))]:
        target=tmp_path/(name+'.vcobj')
        with pytest.raises(ValueError):
            write_fiber_result_columns(target,source,pin,'test','test',ids,{name:values},workers=2)
        assert not target.exists()
    with pytest.raises(ValueError,match='snapshot'):
        write_fiber_result_columns(tmp_path/'stale.vcobj',source,dict(pin,commit_sequence=pin['commit_sequence']+1),
                                  'test','test',ids,{},workers=2)
    with pytest.raises(ValueError,match='integer'):
        write_fiber_result_columns(tmp_path/'fractional.vcobj',source,pin,'test','test',ids,
                                  dict(family=np.array([0.,1.5])),workers=2)
