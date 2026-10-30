import json

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_inpainting import score_cohorts


def test_deformation_settings_are_caller_supplied(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_inpainting import deformation_config
    solved=dict(output_root='prior',lr=.07,event_offset_anchor_weight=3,steps=42)
    settings=dict(deformation_parameters={'event_offset_anchor_weight':.4,'event_offset_merge_distance':0})
    cfg=deformation_config(solved,settings,tmp_path)
    assert cfg['event_offset_anchor_weight']==.4
    assert cfg['event_offset_merge_distance']==0
    assert cfg['lr']==.07 and cfg['steps']==42


@pytest.mark.parametrize('parallel', [True, False])
@pytest.mark.parametrize('enabled', [True, False])
def test_empty_and_rejected_cohorts(parallel, enabled):
    result=score_cohorts(np.array([.25]),np.array([enabled]),np.array([parallel]),
        np.array([0.]),np.array([0.]))
    present=result['parallel' if parallel else 'perpendicular']
    absent=result['perpendicular' if parallel else 'parallel']
    assert absent['total']==0 and absent['coverage_percent'] is None
    assert absent['p95'] is None and absent['max'] is None
    assert present['coverage_percent']==(100 if enabled else 0)
    assert present['l1']==(.25 if enabled else None)
    assert present['sign_violation_percent'] is None
    assert result['postfit_offset_diagnostic']==(0 if enabled else None)
    json.dumps(result,allow_nan=False)


def test_no_events():
    empty=np.array([],dtype=float);mask=np.array([],dtype=bool)
    result=score_cohorts(empty,mask,mask,empty,empty)
    assert result['all']['enabled']==0 and result['postfit_offset_diagnostic'] is None
    json.dumps(result,allow_nan=False)


def test_nonempty_metrics_unchanged():
    result=score_cohorts(np.array([.25,-.75,1.]),np.ones(3,dtype=bool),np.array([True,False,False]),
        np.array([1.,-1.,0.]),np.array([1.,-1.,0.]))
    all_events=result['all']
    assert all_events['l1']==pytest.approx(2/3)
    assert all_events['correct_percent']==pytest.approx(100/3)
    assert all_events['median']==.75 and all_events['max']==1
    assert all_events['p95']==pytest.approx(.975)
    assert all_events['sign_violations']==1 and all_events['sign_violation_percent']==50
