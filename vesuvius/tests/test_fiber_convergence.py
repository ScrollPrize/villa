from vesuvius.neural_tracing.fiber_trace_3d.fiber_convergence import ConvergenceMonitor


def row(i, loss=.1, events=(2,8)):
    return dict(iteration=i,loss=loss,bucket_uv_meeting_events=list(events),terms={},
        **{k:1. for k in ('bucket_uv_local_u_p0.5_max','bucket_uv_local_v_p0.5_max',
         'direction_bucket_angle_max','direction_intermediate_angle_max',
         'direction_shared_angle_max','shared_curve_length_relative_max')})


def test_stationary_and_improvement():
    m=ConvergenceMonitor(dict(block_steps=10,minimum_steps=60))
    assert not any(m.observe(row(i)) for i in range(1,60))
    assert m.observe(row(60))
    m=ConvergenceMonitor(dict(block_steps=10,minimum_steps=60))
    assert not any(m.observe(row(i,loss=1-i*.001)) for i in range(1,100))


def test_membership_not_only_count_and_no_empty_success():
    for events in ((),None):
        m=ConvergenceMonitor(dict(block_steps=10,minimum_steps=60))
        assert not any(m.observe(row(i,events=events if events is not None else
            ((2,8) if (i//10)%2 else (2,9)))) for i in range(1,100))


def test_geometry_drift_prevents_stop():
    m=ConvergenceMonitor(dict(block_steps=10,minimum_steps=60))
    for i in range(1,100):
        r=row(i);r['bucket_uv_local_u_p0.5_max']=i
        assert not m.observe(r)
