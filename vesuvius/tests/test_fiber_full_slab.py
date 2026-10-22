from vesuvius.neural_tracing.fiber_trace_3d.fiber_full_slab import clean_config
import numpy as np


def test_no_inherited_gt_or_initialization():
    base=dict(reference_projection='x',reference_subset='x',fixed_reference_events='x',
              initial_solution='x',soft_samples='x',initialization_geometry_cache='x',
              tensor_sha256='x',lr=.1,steps=1000)
    assert clean_config(base)==dict(lr=.1,steps=1000)


def test_density_reference_metric_uses_map_not_legacy_logits(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_full_slab import reference_metrics
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_reference_comparison as ref
    monkeypatch.setattr(ref,'reference_interval_widths',lambda cfg,starts:np.full(len(starts),100.))
    d=dict(source_id=np.array([1,2]),family=np.array([0,1]),start=np.zeros(2),
           lengths=np.ones(2),node_ds=np.array([[0,.5,1.],[0,.5,1.]]))
    sol=dict(state=np.array([[0.,.2,0.,4.],[1.,.2,0.,4.]]),widths=np.full(4,100.))
    sample=dict(owner=np.array([0,0,1,1]),arc=np.array([0.,1.,0.,1.]),
                weight=np.full(4,.5),offset=np.array([0,2,4]),chosen_winding=np.array([10.,11.]))
    result=reference_metrics(d,sol,{},sample,'/unused')
    assert result['global_offset']==10
    assert result['mean_absolute_winding_error']==0
    assert result['within_two_windings_percent']==100
    assert result['occupied_anchor_widths']['mean_normalized_factor']==1
    sol['state'][0,0]=.4
    continuous=reference_metrics(d,sol,{},sample,'/unused',continuous=True)
    mapped=reference_metrics(d,sol,{},sample,'/unused')
    assert mapped['mean_absolute_winding_error']==0
    assert continuous['mean_absolute_winding_error']==.5


def test_soft_trial_preserves_init_and_only_changes_requested_settings():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_full_slab import soft_trial_config
    original=dict(initial_solution='init',soft_samples='samples',tensor_cache='tensors',
        tensor_sha256='t',samples_sha256='s',gpu_manager={'exclusive':False,'cpu_cores':4},
        lr=.07348469228349533,rejection_cost=20,width_prior=0,width_smooth=.01,
        cavity_winding_sigma=True,winding_sigma=.5,soft_pair_weight=1,probability_pair_weight=.3,
        sign_violation_weight=30,steps=1000,batch_size=32768)
    cfg=soft_trial_config(original,dict(output_root='out',devices='2',width_lr=200))
    for key in ('initial_solution','soft_samples','tensor_cache','lr','rejection_cost','width_prior','width_smooth','steps','batch_size'):
        assert cfg[key]==original[key]
    assert cfg['width_lr']==200 and cfg['devices']=='2'
    assert not any(k.startswith('cavity_') for k in cfg)
    assert 'probability_pair_weight' not in cfg and 'sign_violation_weight' not in cfg


def test_soft_objective_is_existing_combination_soft_component():
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_winding as imp
    a=torch.tensor([[1.2,.2,40.,4.],[2.,.3,10.,4.]])
    b=torch.tensor([[1.8,.4,50.,4.],[2.2,.2,80.,4.]])
    fa=torch.tensor([0,0]);fb=torch.tensor([1,0]);ds=torch.tensor([3.,4.])
    widths=torch.tensor([100.,200.,300.,400.]);sign=torch.ones(2)
    distance=torch.tensor([.5,1.]);parallel=torch.tensor([False,True])
    config=dict(soft_pair_weight=1,probability_pair_weight=0,winding_sigma=.5,
        sign_violation_weight=30,same_sheet_match_multiplier=2.5,distance_sigma=.5,v_pair_scale=1000)
    result=imp.pair_cost(a,b,fa,fb,ds,ds,widths,sign,distance,parallel,config)
    # The current soft component uses smooth integer-separation energy, not
    # squared error against a fractional Lasagna distance.
    _,ua,va=imp.soft.points(a,fa,ds,widths)
    _,ub,vb=imp.soft.points(b,fb,ds,widths)
    calibration=(~parallel)*(fb-fa)*.5
    energy,_=imp.soft.integer_separation_energy(b[:,0]-a[:,0]+ub-ua,sign,distance,calibration,.5)
    for actual,expected in zip(result,(energy,imp.soft.circular(ub-ua).square(),((vb-va)/1000).square())):
        torch.testing.assert_close(actual,expected)
