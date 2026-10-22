import copy
import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_check import align,evaluate


def fixture():
    points=np.array([[x,y] for x in range(4) for y in (0.,100.,200.)])
    ident=np.column_stack((np.repeat(np.arange(4),3),np.tile([0,1,2],4)))
    ref=dict(consensus=points,scale=np.array([3.,200.]),identity=ident,
             source_id=np.arange(4),family=np.array([0,1,0,1]),direction=np.ones(4),
             a=np.array([0]),b=np.array([0]),events=np.array([0]))
    metrics={f:{k:dict(mean=0.,p95=0.,max=0.) for k in ('length_percent','angle_degrees')} for f in ('H','V')}
    meta=dict(v_tolerance_vx=1.,affine_tolerance=.1,
              limits={f:{k:dict(mean=1.,p95=1.,max=1.) for k in ('length_percent','angle_degrees')} for f in ('H','V')})
    for f in metrics:metrics[f]['reversals']=0
    c={k:v for k,v in ref.items() if k not in ('consensus','scale')}
    c.update(root='synthetic',layers=dict(bucket=points.copy()),p=points.copy(),
             metrics=metrics,prob=np.ones(4),solution_sha256='test')
    return c,ref,meta


def test_global_shift_allowed_but_single_fiber_shift_fails():
    c,r,m=fixture();c['layers']['bucket']+=np.array([10.,1000.])
    assert evaluate(c,r,m)['passed']
    c['layers']['bucket'][:3,0]+=1
    assert not evaluate(c,r,m)['checks']['winding']


def test_missing_meeting_or_bad_geometry_cannot_hide():
    c,r,m=fixture();c['events']=np.array([],dtype=int)
    assert not evaluate(c,r,m)['checks']['cohort']
    c,r,m=fixture();c['metrics']['H']['length_percent']['max']=2
    assert not evaluate(c,r,m)['checks']['H_length_percent_max']
    c,r,m=fixture();c['metrics']['V']['angle_degrees']['max']=2
    assert not evaluate(c,r,m)['checks']['V_angle_degrees_max']
    c,r,m=fixture();c['b']=np.array([1]);r['b']=np.array([1])
    assert not evaluate(c,r,m)['checks']['meetings']


def test_small_global_distortion_allowed_large_scale_rejected():
    c,r,m=fixture();c['layers']['bucket']*=1.02
    assert evaluate(c,r,m)['passed']
    c['layers']['bucket']*=2
    assert not evaluate(c,r,m)['checks']['global_distortion']


def test_explicit_exclusion_does_not_influence_alignment_or_confidence():
    c,r,m=fixture();m['excluded_source_ids']=[3]
    c['layers']['bucket'][9:]+=np.array([100.,100000.]);c['prob'][3]=0
    result=evaluate(c,r,m)
    assert result['passed']
    assert not result['per_fiber'][3]['scored']
    assert result['per_fiber'][3]['winding_error']['max']>1
