from vesuvius.neural_tracing.fiber_trace_3d.fiber_optimizer_probe import plateau, no_improvement_plateau


def rows():
    return [dict(step=s,block_median_loss=.1,length_max_pct=1.,
                 angles={'bucket':{'max':.03}},meetings=8) for s in range(6000,11000,1000)]


def test_plateau_requires_loss_and_geometry_stability():
    h=rows();assert plateau(h)
    h[-1]['block_median_loss']=.09;assert not plateau(h)
    h=rows();h[-1]['length_max_pct']=.9;assert not plateau(h)
    h=rows();h[-1]['angles']['bucket']['max']=.2;assert not plateau(h)
    h=rows();h[-1]['meetings']=0;assert not plateau(h)
    assert not plateau(rows(),minimum=20000)


def test_no_improvement_allows_noise_but_not_new_progress():
    h=rows()+[dict(rows()[-1],step=11000)]
    h[1]['block_median_loss']=.11
    assert no_improvement_plateau(h)
    h[-1]['block_median_loss']=.09
    assert not no_improvement_plateau(h)
