import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_selected_dataset_eval import remap_events
from vesuvius.neural_tracing.fiber_trace_3d.fiber_selected_dataset_eval import initialization_config
from vesuvius.neural_tracing.fiber_trace_3d.fiber_selected_dataset_eval import aligned_widths


def U(n):
    return f'00000000-0000-4000-8000-{n:012d}'


def test_width_alignment_and_unmapped_tails():
    radial=np.array([10.,20.,30.,40.]);donor=np.array([100.,200.,300.])
    result,count=aligned_widths(radial,donor,-1)
    np.testing.assert_array_equal(result,[10,100,200,300]);assert count==3
    result,count=aligned_widths(radial,donor,1)
    np.testing.assert_array_equal(result,[200,300,30,40]);assert count==2
    np.testing.assert_array_equal(radial,[10,20,30,40])
    with pytest.raises(ValueError):aligned_widths(radial,[0,1],0)


def test_zero_width_learning_rate_is_supported():
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    soft.validate(dict(width_lr=0))
    for value in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):soft.validate(dict(width_lr=value))


def test_initialization_keeps_weights_not_old_gt_or_state():
    template=dict(lr=.1,steps=1000,initial_solution='old',reference_subset='old',
        fixed_reference_events='gt',tensor_sha256='old',soft_samples='old',code_provenance={})
    config={k:'new' for k in ('tensor_cache','fibers','umbilicus','devices','gpu_manager','layout_mode','filter_constraint_families')}
    result=initialization_config(template,config,'output')
    assert result['lr']==.1 and result['steps']==1000
    assert not any(k in result for k in ('reference_subset','fixed_reference_events','initial_solution','soft_samples','tensor_sha256','code_provenance'))
    assert result['tensor_cache']=='new'
    assert initialization_config(template,config,'output','new_stage')['initial_solution']=='new_stage'


def test_prepared_soft_run_does_not_rebuild_graph(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    tensor=tmp_path/'tensors.npz';np.savez(tensor,family=np.array([0]))
    monkeypatch.setattr(soft,'validate',lambda c:None)
    monkeypatch.setattr(soft.uv,'prepare',lambda c:pytest.fail('Must not rebuild selected graph'))
    monkeypatch.setattr(soft,'prepare_samples',lambda c,d:dict(sample=np.array([1])))
    monkeypatch.setattr(soft.uv,'launch_tensor_solve',lambda c,*args:c)
    result=soft.run_prepared(dict(output_root=str(tmp_path/'run'),tensor_cache=str(tensor)))
    assert result['tensor_sha256'] and result['samples_sha256']


def test_remap_preserves_full_source_events_and_later_piece_boundary():
    d=dict(source_id=np.array([U(10),U(10),U(20)]),start=np.array([0.,5.,0.]),end=np.array([5.,10.,20.]))
    events=[dict(test_source=U(10),test_arc=5.,test_piece=99),dict(test_source=U(30),test_arc=3.,test_piece=100),
            dict(test_source=U(20),test_arc=19.,test_piece=101)]
    result=remap_events(events,d)
    assert [e['test_piece'] for e in result]==[1,2]
    assert events[0]['test_piece']==99


def test_selected_source_missing_arc_raises():
    d=dict(source_id=np.array([U(10)]),start=np.array([0.]),end=np.array([5.]))
    with pytest.raises(ValueError,match='lost GT event arc'):
        remap_events([dict(test_source=U(10),test_arc=6.)],d)


def test_checkpoint_report_map_scoring_and_unpublished_tail(tmp_path,monkeypatch):
    import json
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_selected_dataset_eval as mod
    root=tmp_path/'run';solve=root/'solve';(solve/'checkpoints').mkdir(parents=True)
    gt=tmp_path/'gt';(gt/'uv_whole').mkdir(parents=True)
    ds=tmp_path/'dataset';(ds/'cross').mkdir(parents=True)
    data=dict(source_id=np.array([U(10)]),start=np.array([0.]),end=np.array([10.]),family=np.array([0]))
    np.savez(root/'tensors.npz',**data)
    np.savez(gt/'tensors.npz',**dict(data,source_id=np.array([U(20)])))
    np.savez(gt/'uv_whole/solution.npz',state=np.zeros((1,4)),widths=np.ones(3)*1000)
    (gt/'uv_whole/config.json').write_text(json.dumps(dict(tensor_cache=str(gt/'tensors.npz'))))
    (solve/'config.json').write_text(json.dumps(dict(tensor_cache=str(root/'tensors.npz'))))
    (solve/'history.json').write_text(json.dumps([dict(iteration=0),dict(iteration=100)]))
    np.savez(solve/'checkpoints/000000.npz',state=np.array([[3.2,.1,0,4]]),widths=np.ones(3)*1000)
    (ds/'cross/cross_events.json').write_text(json.dumps(dict(events=[dict(test_source=U(10),
        test_arc=5,gt_piece=0,gt_source=U(20),parallel=False)])))
    monkeypatch.setattr(mod.uv,'hard_assignment',lambda *a,**kw:(None,None,np.array([True]),np.array([0])))
    def residual(events,d,state,*args,**kwargs):
        assert state[0,0]==3.;return np.array([.1])
    monkeypatch.setattr(mod,'event_residuals',residual)
    settings=dict(root=str(root),gt_root=str(gt),dataset=str(ds))
    rows=mod.checkpoint_scores(settings)
    assert len(rows)==1 and rows[0]['gt']['gt_perpendicular']['l1']==pytest.approx(.1)
    (solve/'summary.json').write_text('{}')
    with pytest.raises(FileNotFoundError):mod.checkpoint_scores(settings)
