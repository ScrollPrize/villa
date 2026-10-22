import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_winding_history import summarize


def test_attachment_support_excludes_current_non_same_sheet_and_parallel():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_winding_history import attachment_support
    d=dict(i=np.array([0,0,0]),j=np.array([1,2,3]),
           family=np.array([0,1,1,0,1]),parallel=np.array([False,False,True]))
    h=dict(iteration=np.arange(3),labels=np.zeros((3,5,1),dtype=int),joint=np.ones((3,5,1)))
    h['joint'][1:,2]=0
    score,edges,activation,count=attachment_support(h,d,np.zeros(3),{},2)
    np.testing.assert_allclose(activation,[1])
    np.testing.assert_allclose(score,[1,1,0,0,0])
    np.testing.assert_array_equal(count,[1,1,0,0,0])


def test_attachment_support_keeps_early_inactive_history_for_current_match():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_winding_history import attachment_support
    d=dict(i=np.array([0]),j=np.array([1]),family=np.array([0,1]),parallel=np.array([False]))
    h=dict(iteration=np.array([1,2]),labels=np.array([[[0],[1]],[[0],[0]]]),joint=np.ones((2,2,1)))
    score,edges,activation,_=attachment_support(h,d,np.zeros(1),{},2)
    np.testing.assert_array_equal(edges,[0])
    np.testing.assert_allclose(activation,[.5])
    np.testing.assert_allclose(score,[.5,.5])


def test_attachment_support_sums_matching_hypotheses_with_gate_and_carry():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_winding_history import attachment_support
    d=dict(i=np.array([0]),j=np.array([1]),family=np.array([0,1]),parallel=np.array([False]))
    h=dict(iteration=np.array([1]),labels=np.array([[[2,3],[1,2]]]),joint=np.array([[[.8,.2],[.8,.2]]]))
    cfg=dict(geometry_confidence_gated=True,geometry_confidence_low=.5,geometry_confidence_high=.6)
    score,_,activation,_=attachment_support(h,d,np.array([1]),cfg,1)
    np.testing.assert_allclose(score,[.64,.64])
    np.testing.assert_allclose(activation,[.64])


def test_attachment_support_rejects_incomplete_window():
    import pytest
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_winding_history import attachment_support
    with pytest.raises(ValueError,match='complete consecutive'):
        attachment_support(dict(iteration=np.array([1,3])),{},None,{},2)


def test_attachment_support_selection_is_joint_mass_not_each_side():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_winding_history import attachment_support
    d=dict(i=np.array([0]),j=np.array([1]),family=np.array([0,1]),parallel=np.array([False]))
    h=dict(iteration=np.array([1]),labels=np.zeros((1,2,1)),joint=np.full((1,2,1),.6))
    score,edges,_,count=attachment_support(h,d,np.zeros(1),{},1)
    assert len(edges)==0
    np.testing.assert_array_equal(count,[0,0])
    np.testing.assert_array_equal(score,[0,0])


def test_entry_and_probability_history():
    h=dict(iteration=np.arange(4),labels=np.tile([[[2,3]]],(4,1,1)),
           joint=np.array([[[.7,.3]],[[.4,.6]],[[.8,.2]],[[.9,.1]]]))
    r=summarize(h,np.array([2]))[0]
    assert r['entry']==2 and r['age']==1
    assert r['visible100']==3 and r['switches100']==2 and r['map_switches100']==2
    assert np.isclose(r['mean100'],.7)


def test_absent_label_has_zero_probability():
    h=dict(iteration=np.arange(2),labels=np.array([[[0,1]],[[1,2]]]),
           joint=np.array([[[.6,.4]],[[.3,.7]]]))
    r=summarize(h,np.array([2]))[0]
    assert r['entry']==1 and r['age']==0 and r['min100']==0
    assert np.isclose(r['mean100'],.35)


def test_variable_window_export(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_winding_history import WindingHistory
    recorder=WindingHistory()
    recorder.rows=[(0,np.array([[1,2]]),np.array([[.3,.7]]),np.ones(3)),
                   (1,np.array([[0,1,2,3]]),np.array([[.1,.1,.7,.1]]),np.ones(3))]
    path=tmp_path/'history.npz'
    recorder.save(path,np.array([100]))
    with np.load(path) as h:
        assert h['labels'].shape==(2,1,4)
        r=summarize(h,np.array([2]))[0]
        assert r['age']==1 and np.isclose(r['mean100'],.7)


def test_capture_does_not_change_state_or_rng():
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_winding_history import WindingHistory
    class Extension:
        def local_hypotheses(self,state):
            return state[:,None,:],state.new_zeros((len(state),1))
    state=torch.zeros(2,4,requires_grad=True)
    original=state.detach().clone();rng=torch.get_rng_state().clone()
    recorder=WindingHistory();recorder.capture(0,state,Extension(),torch.ones(3))
    torch.testing.assert_close(state,original)
    assert torch.equal(torch.get_rng_state(),rng) and state.grad is None
    np.testing.assert_allclose(recorder.rows[0][2],.5)


def test_batched_history_variable_windows_and_final_partial_flush(tmp_path):
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_winding_history import WindingHistory
    recorder=WindingHistory(100)
    for t,k in enumerate([2,4,1]):
        recorder.pending.append((t,torch.arange(k,dtype=torch.int32)[None],
                                 torch.full((1,k),.25),torch.ones(3)*t))
    recorder.save(tmp_path/'history.npz',np.array([123]))
    assert not recorder.pending
    with np.load(tmp_path/'history.npz') as h:
        np.testing.assert_array_equal(h['iteration'],[0,1,2])
        np.testing.assert_array_equal(h['labels'][1],[[0,1,2,3]])
        np.testing.assert_array_equal(h['joint'][2],[[.25,0,0,0]])
        np.testing.assert_array_equal(h['widths'],[[0]*3,[1]*3,[2]*3])


def test_history_capture_snapshots_mutable_widths():
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_winding_history import WindingHistory
    class Extension:
        def local_hypotheses(self,state):
            return state[:,None,:],state.new_zeros((len(state),1))
    recorder=WindingHistory();widths=torch.ones(3)
    recorder.capture(0,torch.zeros(2,4),Extension(),widths)
    widths.add_(10)
    np.testing.assert_array_equal(recorder.rows[0][3],np.ones(3))
