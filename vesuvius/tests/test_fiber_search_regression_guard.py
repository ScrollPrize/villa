from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_weight_search import regression_guard
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_weight_search import needs_gt_validation


def test_only_potential_improvements_need_validation():
    good=dict(perp_l1=.4,gt_guard={'passed':True})
    bad=dict(perp_l1=.3,gt_guard={'passed':False})
    skipped=dict(perp_l1=.8,gt_guard=None)
    assert needs_gt_validation(.5,[bad,skipped],1e-4)
    assert not needs_gt_validation(.5,[good,bad],1e-4)
    assert not needs_gt_validation(.39995,[good],1e-4)
    assert needs_gt_validation(.39,[good,bad],1e-4)


def test_guard_rejects_signs_and_distortion_not_reference_distance():
    row=dict(perp_sign_bad=0,parallel_sign_bad=0,reference_perp={'max':2},segments={
        'H':dict(reversals=0,length_error_percent={'max':3},angle_degrees={'max':5})})
    limits=dict(length_max_percent=10,angle_max_degrees=15)
    assert regression_guard({'test':{'final':row}},limits)['passed']
    row['perp_sign_bad']=1
    assert not regression_guard({'test':{'final':row}},limits)['passed']
    row['perp_sign_bad']=0;row['segments']['H']['length_error_percent']['max']=11
    assert not regression_guard({'test':{'final':row}},limits)['passed']
