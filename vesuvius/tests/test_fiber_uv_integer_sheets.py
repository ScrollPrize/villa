import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_integer_sheets as sheets


CFG=dict(integer_sheet_constraints=True,bucket_position_split=True,layout_mode='shared_anchor',
    distance_objective='squared',distance_weight=.25,position_distance_weight=2.,distance_sigma=.5,
    perpendicular_layer_offset=.5,metric_weight=0.,sign_weight=20.)


def test_map_targets_and_margin():
    np.testing.assert_array_equal(sheets.target(np.array([.3,.7,1.]),np.ones(3),np.array([1,-1,0]),
        np.array([False,False,True]),.3),[0,1,1])


def test_gt_integer_scoring_phase_independence_and_seam():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_constraint_score import event_residuals
    d=dict(node_phase=np.array([[1.05]]),phase=np.array([.05]))
    gd=dict(node_phase=np.array([[.05]]),phase=np.array([.05]))
    state=np.zeros((1,12));state[0,4]=7.3
    gs=dict(state=np.zeros((1,12)),widths=np.ones(3));gs['state'][0,4]=-4.2
    event=dict(gt_piece=0,test_piece=0,gt_angle=.05,test_angle=1.05,
               phase_delta=0.,family_delta=1.,signed_distance=.5,parallel=False)
    # Same physical angle across an anchor seam; already-converted GT labels.
    actual=event_residuals([event],d,state,np.array([1]),3,gd,gs,np.array([1]),CFG,CFG)
    np.testing.assert_array_equal(actual,[0.])
    event.update(family_delta=-1.,signed_distance=-.5)
    np.testing.assert_array_equal(event_residuals([event],d,state,np.array([1]),3,gd,gs,np.array([1]),CFG,CFG),[0.])
    # A legacy GT's whole-winding shared shift must survive the conversion.
    gs['state'][0,4]=2.3
    np.testing.assert_array_equal(event_residuals([event],d,state,np.array([3]),3,gd,gs,np.array([1]),CFG,{}),[0.])
