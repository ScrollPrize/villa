import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_error_breakdown import (
    event_features, summarize, interval_masks, tables)


def fixture():
    solution = dict(source_id=np.array([10,11]), start=np.zeros(2),
        end=np.array([500.,2000.]), family=np.array([0,1]),
        state=np.array([[2.,.2,0.,4.],[3.,.2,0.,4.]],dtype=np.float32),
        winding_sigma=np.array([.2,2.],dtype=np.float32))
    events = [dict(test_piece=i%2, test_source=10+i%2, gt_source=1,
        residual=r, parallel=i%2==0, enabled=True, normal_cosine=1)
        for i,r in enumerate([-.5,.5,.2,1.])]
    return solution,events,dict(cavity_winding_sigma=True,sigma_max=8)


def test_saved_sigma_correctness_boundary_and_weighted_rows():
    s,e,c=fixture()
    f=event_features(e,s,c,np.array([0.,-.5,1.,1.]))
    assert f['correct'].tolist()==[True,False,True,False]
    assert f['joint_probability'][0]>.9
    assert f['joint_probability'][1]<.25
    row=summarize(f,np.ones(4,bool))
    assert row['events']==4 and row['fibers']==2 and row['violations']==1
    assert row['sign_percent']==25 and row['correct_percent']==50
    assert row['l1']==pytest.approx(.55)
    rows=tables(f)
    assert sum(r['events'] for r in rows if r['kind']=='all' and r['grouping']=='joint_probability')==4


def test_source_join_and_unsplit_checks():
    s,e,c=fixture()
    e[0]['test_source']=99
    with pytest.raises(ValueError,match='identity'):
        event_features(e,s,c,np.zeros(4))
    s['source_id'][1]=10
    with pytest.raises(ValueError,match='unsplit'):
        event_features(e,s,c,np.zeros(4))


def test_bins_and_empty_groups():
    masks=interval_masks(np.array([0.,.25,.5,.9,1.]),[0,.25,.5,.9])
    assert [int(m.sum()) for _,m in masks]==[1,1,1,2]
    assert np.all(np.stack([m for _,m in masks]).sum(0)==1)
    s,e,c=fixture()
    f=event_features(e,s,c,np.zeros(4))
    row=summarize(f,np.zeros(4,bool))
    assert row['events']==0 and row['l1'] is None and row['sign_percent'] is None


def test_outputs_are_real_markdown_csv_and_metrics(tmp_path):
    import json
    import csv
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_error_breakdown import run
    s,e,c=fixture()
    for i,event in enumerate(e):
        event['event_id']=i
    root=tmp_path/'input';(root/'solve').mkdir(parents=True)
    np.savez(root/'solve/solution.npz',**s)
    saved=dict(score=dict(gt_global_offset=-2,
        gt_parallel=dict(events=2,l1=.35,correct_percent=100),
        gt_perpendicular=dict(events=2,l1=.75,correct_percent=0)))
    for path,value in [('gt_events.json',e),('results.json',saved),('solve/config.json',c)]:
        (root/path).write_text(json.dumps(value))
    out=tmp_path/'output'
    run(root,e,c,np.array([0.,-.5,1.,1.]),saved,out)
    assert (out/'report.md').read_text().startswith('# GT constraint')
    with (out/'events.csv').open() as stream:
        records=list(csv.DictReader(stream))
    assert len(records)==4 and records[0]['event_id']=='0'
    assert json.loads((out/'breakdown.json').read_text())['global_offset']==-2
