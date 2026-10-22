import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_constraint_score import offset_votes,piece_at,coarse_at
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import hard_assignment
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_constraint_score import enabled_l1
import pytest


def test_gt_solve_width_metric_alignment():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_constraint_score import gt_solve_width_metrics
    result=gt_solve_width_metrics(np.array([10.,20.,110.,180.]),np.array([100.,200.]),[0,0,1],-2)
    assert result['test_intervals']==[2,3]
    assert result['mean_normalized_factor']==pytest.approx(1.)
    assert result['mean_absolute_relative_error_percent']==pytest.approx(10.)
    assert result['mean_absolute_error_base_vx']==pytest.approx(15.)
    assert result['max_absolute_error_base_vx']==pytest.approx(20.)
    assert gt_solve_width_metrics([1.],[1.],[0],2)['compared_windings']==0


def test_enabled_l1_mask_and_integer_gauge():
    score=enabled_l1([2.1,2.3,100],[True,True,False],[True,False,True])
    assert score['gt_global_offset']==2
    assert score['gt_enabled_l1']==pytest.approx(.2)
    assert score['gt_enabled_events']==2
    assert score['gt_parallel']['l1']==pytest.approx(.1)
    shifted=enabled_l1([12.1,12.3,-100],[True,True,False],[True,False,True])
    assert shifted['gt_global_offset']==12
    assert shifted['gt_enabled_l1']==pytest.approx(.2)


@pytest.mark.parametrize('values,offset', [([1,10],1),([-10,10],0),([-10,-1],-1),([.6,.8],1),([.1,.3],0)])
def test_l1_median_interval(values,offset):
    assert enabled_l1(values,[True,True],[True,True])['gt_global_offset']==offset


def test_l1_empty_enabled_invalid():
    with pytest.raises(ValueError,match='No finite enabled'):
        enabled_l1([0],[False],[True])


def test_offset_votes_and_integer_gauge():
    values=np.array([3.1,2.8,3.2,4.1,-2.])
    offset,ok,votes=offset_votes(values)
    assert offset==3 and ok.tolist()==[True,True,True,False,False]
    off2,ok2,_=offset_votes(values+12)
    assert off2==15
    np.testing.assert_array_equal(ok,ok2)


def test_piece_boundary_and_gap():
    d=dict(source_id=np.array([4,4,4]),start=np.array([0.,10.,25.]),end=np.array([10.,20.,30.]))
    assert piece_at(d,4,10)==1
    assert piece_at(d,4,22) is None
    assert piece_at(d,5,5) is None


def test_disabled_placement_not_last_bucket():
    state=np.array([[1.,7.,2.,20.],[0.,0.,8.,1.]])
    _,labels,keep,pick=hard_assignment(state,3,{},best_nonbroken=True)
    assert labels.tolist()==[3,2] and keep.tolist()==[False,True]
    assert pick.tolist()==[1,2]


def test_coarse_phase_and_family_vote():
    d=dict(node_phase=np.array([[2.9]]),phase=np.array([.9]))
    state=np.zeros((1,8));state[0,5]=.3
    q=coarse_at(d,state,np.array([1]),0,3.1,4)
    assert abs(q-2.4)<1e-8
    # test H and GT V, same coarse position: outward layer separation +.5.
    difference=q-q+.5
    assert offset_votes([difference-.5])[0]==0


def test_cross_coarse_matches_training_correction():
    d=dict(node_phase=np.array([[2.95],[.05]]),phase=np.array([.95,.05]),
           i=np.array([0]),j=np.array([1]),phase_a=np.array([3.1]),
           phase_b=np.array([.1]),phase_delta=np.array([0.]))
    state=np.zeros((2,8));state[:,5]=[.4,-.2];pick=np.array([2,5])
    qa=coarse_at(d,state,pick,0,3.1,4);qb=coarse_at(d,state,pick,1,.1,4)
    actual=qb-qa-d['phase_delta'][0]+.5
    expected=pick[1]-pick[0]+uv.bucket_event_shift(d)[0]+state[1,5]-state[0,5]+.5
    assert abs(actual-expected)<1e-12


def test_soft_and_bucket_continuous_arc_metric_agree():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_constraint_score import event_residuals
    k=3; widths=np.array([100.,200.,300.]); pick=np.array([0,1])
    d=dict(start=np.array([10.,20.]),end=np.array([110.,120.]),
           node_ds=np.array([[0.,50.,100.],[0.,-50.,-100.]]),
           family=np.array([0,1]),phase=np.array([.9,.2]))
    bucket=np.zeros((2,3*k+3));bucket[:,k+1]=[.1,-.1]
    soft=np.column_stack((pick,d['phase']+bucket[:,k+1],np.zeros(2),np.ones(2)))
    events=[dict(test_piece=0,test_arc=40.,gt_piece=1,gt_arc=60.,family_delta=1,signed_distance=.5),
            dict(test_piece=1,test_arc=80.,gt_piece=0,gt_arc=100.,family_delta=-1,signed_distance=-.7)]
    gs=dict(state=bucket,widths=widths)
    expected=event_residuals(events,d,bucket,pick,k,d,gs,pick,
        config=dict(unified_winding_loss=True),widths=widths)
    actual=event_residuals(events,d,soft,None,k,d,gs,pick,
        config=dict(soft_winding_model=True),widths=widths)
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    soft[:,0]+=[3,-2];soft[:,1]-=[3,-2]
    shifted=event_residuals(events,d,soft,None,k,d,gs,pick,
        config=dict(soft_winding_model=True),widths=widths)
    np.testing.assert_allclose(shifted,actual,atol=1e-12)
