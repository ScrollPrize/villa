import numpy as np
import torch

from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_density import (
    density_maps,density,refit_disabled,plot_density)


def fixture():
    d=dict(family=np.array([0,1]),node_ds=np.array([[0.,1500.,3000.],[0.,-1500.,-3000.]]),
           lengths=np.array([3000.,3000.]))
    u=np.array([[0.,4000.],[500.,4500.]]);v=np.array([[500.,500.],[3000.,3000.]])
    enabled=np.array([True,False]);pick=np.array([0,0]);new=np.array([0,1])
    return d,u,v,enabled,dict(enabled=pick,all_saved=pick,all_refit=new)


def test_mass_area_and_independent_counts():
    d,u,v,enabled,choices=fixture()
    grids,domain=density_maps(d,u,v,enabled,choices,1000,10000)
    for name,pair in grids.items():
        for target,g in pair.items():
            assert g.values['h_length'].sum()==3000
            assert g.values['v_length'].sum()==(0 if name=='enabled' else 3000)
            np.testing.assert_allclose(g.values['area'].sum(),(domain[2]-domain[0])*(domain[3]-domain[1]))
            assert np.isfinite(density(g,'h','length')[g.values['area']>0]).all()
    heat=grids['all_saved']['heat'];profile=grids['all_saved']['profile']
    assert heat.values['v_count'].sum()==3
    assert profile.values['v_count'].sum()==1
    assert density(heat,'h','length').max()==1000
    assert not np.array_equal(heat.values['v_length'],grids['all_refit']['heat'].values['v_length'])
    np.testing.assert_array_equal(heat.values['h_length'],grids['enabled']['heat'].values['h_length'])


def test_exact_outer_boundary_nonzero_origin():
    d,u,v,enabled,choices=fixture();u-=4500;v-=3000
    grids,domain=density_maps(d,u,v,enabled,choices,1000,1000)
    for pair in grids.values():
        for g in pair.values():
            assert np.all(g.values['h_length'][g.values['area']==0]==0)
            assert np.all(g.values['v_length'][g.values['area']==0]==0)


def constraint_fixture():
    n=4;state=np.zeros((n,9))
    state[:,:3]=[[0,0,8],[8,0,0],[0,0,8],[0,0,8]]
    d={key:np.zeros(2) for key in ('ds_a','ds_b','phase_a','phase_b','phase_delta','target_v','distance','sign')}
    d.update(i=np.array([0,1]),j=np.array([1,2]),family=np.array([0,1,0,0]),
        phase=np.array([.1,.2,.3,.4]),z=np.zeros(n),degree=np.array([1.,2.,1.,0.]),
        node_ds=np.zeros((n,3)),node_phase=np.zeros((n,3)),node_z=np.zeros((n,3)),
        links=np.array([[0.,1.,1.,2.]]))
    d['sign'][:]=1;d['distance'][:]=.5
    return d,state,np.array([200.,300.]),dict(layout_mode='shared_anchor',prior_weight=0.,batch_size=1)


def test_conditional_refit_vs_full_cost_oracle():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_categorical import conditional_costs
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
    d,s,w,cfg=constraint_fixture()
    original=s.copy();new,detail,stats=refit_disabled(d,s,w,cfg,batch=1)
    t=torch.as_tensor(s.copy(),dtype=torch.float32);t[:,:3]=-1000;t[[0,2,3],2]=0;t[1,0]=0
    data={key:torch.as_tensor(value) for key,value in d.items()}
    table=torch.as_tensor(uv.make_distance_table(float(d['distance'].max()),.5))
    oracle=conditional_costs(t,torch.as_tensor(w),data,table,cfg).numpy()
    np.testing.assert_allclose(detail['costs'][[0,2,3]],oracle[[0,2,3],:2],rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(s,original)
    assert new[1]==0 and new[3]==0
    assert stats['unsupported_fallback']==1 and stats['supported_disabled']==2
    assert new[0]==detail['costs'][0].argmin() and new[2]==detail['costs'][2].argmin()
    new2,detail2,_=refit_disabled(d,s,w,cfg,batch=9)
    np.testing.assert_array_equal(new,new2);np.testing.assert_allclose(detail['costs'],detail2['costs'])


def test_tied_supported_keeps_saved_choice():
    d,s,w,cfg=constraint_fixture();s[0,1]=1
    cfg.update(sign_weight=0.,distance_weight=0.,metric_weight=0.,sign_charge_multiplier=0.,continuation_weight=0.)
    updated,detail,stats=refit_disabled(d,s,w,cfg)
    assert updated[0]==1 and stats['tied_minimum_disabled']==2


def test_continuation_only_right_endpoint():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_categorical import conditional_costs
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
    d,s,w,cfg=constraint_fixture()
    # All events enabled-enabled: only the continuation can place disabled 2.
    d['i'][:]=1;d['j'][:]=1;d['sign'][:]=[-1,0]
    d['links']=np.array([[1.,2.,7.,-9.]])
    s[2,3:]=[.1,.04,.02,-.02,.03,-.03]
    updated,detail,stats=refit_disabled(d,s,w,cfg,batch=1)
    t=torch.as_tensor(s.copy(),dtype=torch.float32);t[:,:3]=-1000;t[[0,2,3],2]=0;t[1,0]=0
    table=torch.as_tensor(uv.make_distance_table(.5,.5))
    oracle=conditional_costs(t,torch.as_tensor(w),{k:torch.as_tensor(v) for k,v in d.items()},table,cfg).numpy()
    np.testing.assert_allclose(detail['costs'][2],oracle[2,:2],rtol=1e-6,atol=1e-6)
    assert detail['support'][2]==1 and stats['unsupported_fallback']==2


def test_density_plot_smoke(tmp_path):
    d,u,v,enabled,choices=fixture();g,_=density_maps(d,u,v,enabled,choices,1000,10000)
    plot_density(g,tmp_path,{'heatmap_panels':1},'test')
    assert (tmp_path/'length_profiles.svg').exists()
    assert (tmp_path/'all_refit_length_heatmaps.tif').exists()
    with np.load(tmp_path/'all_saved_heat.npz') as x:
        assert x['area'].sum()>0 and x['h_length'].sum()==3000
