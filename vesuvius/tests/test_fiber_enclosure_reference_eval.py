import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_enclosure_reference_eval import stats,cross_metrics,integer_offset


def test_cross_offset_and_family_signs():
    target=np.array([0.,1.,2.]);local=target+7
    shift=np.median(local-target)
    r=cross_metrics(local-shift,target,target,np.ones(3),np.array([1,-1,0]),np.array([False,False,True]))
    assert r['all']['l1']==0 and r['all']['correct_step_percent']==100
    assert r['all']['sign_violations']==0
    r=cross_metrics(np.array([-1.,0.,-2.]),target,target,np.ones(3),np.array([1,-1,0]),np.array([False,False,True]))
    assert r['perpendicular']['sign_violations']==2
    assert r['parallel']['sign_violations']==1


def test_stats_coverage_and_weighting():
    r=stats(np.array([0.,2.,np.nan]),np.array([3.,1.,9.]))
    assert r['count']==2 and r['l1']==.5 and r['within_half_percent']==75
    assert stats([])['l1'] is None


def test_integer_gauge_minimizes_l1_without_fractional_sign_shift():
    v=np.array([3.1,3.2,3.3,4.1])
    assert integer_offset(v)==3
    assert integer_offset(v-9)==-6
    assert integer_offset([]) is None
    assert integer_offset([-.5,.5])==0
