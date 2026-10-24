import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_joint_gt_eval import configure_gt_enablement


def test_gt_selectable_removes_all_force_flags_and_saturated_logits():
    cfg=dict(force_keep_fibers=True,force_keep_source_ids=[20],rejection_cost=2.5)
    state=np.array([[1.,2.,3.,4.],[5.,6.,7.,100.]])
    configure_gt_enablement(cfg,state,np.array([10,20]),[20],False)
    np.testing.assert_array_equal(state,[[1,2,3,4],[5,6,7,4]])
    assert cfg==dict(rejection_cost=2.5)


def test_gt_forced_remains_default_and_does_not_force_ordinary_fibers():
    cfg={};state=np.zeros((2,4))
    configure_gt_enablement(cfg,state,np.array([10,20]),[20])
    assert cfg['force_keep_source_ids']==[20]
    np.testing.assert_array_equal(state[:,3],[0,100])
