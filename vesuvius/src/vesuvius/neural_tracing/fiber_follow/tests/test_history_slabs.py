"""Causal slab sampling, input isolation, replay parity, and both task gradients."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model_fixtures import config as cfg, coordinate_config, coordinate_batch
from vesuvius.neural_tracing.fiber_follow.data.history_slabs import selected_arcs, slab_layout, fitted_heading, observed_path, load_slabs, slabs_allowed
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.data.observations import ObservationBuilder, FiberTracer
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength
from vesuvius.neural_tracing.fiber_follow.data.data import OnPolicyStates, ZBand, SampleConfig, make_sample
from vesuvius.neural_tracing.fiber_follow.tracing.trace import trace_history


def observation(path):
    path = np.asarray(path, dtype=float)
    return dict(pos=path[-1].copy(), frame=np.eye(3), hist_local=np.zeros((32, 3)), hmask=np.zeros(32),
                observed_path=path, seed_pos=path[0].copy(), seed_tangent=np.array([0., 0., 1.]),
                seed_age=float(arclength(path)[-1]), seed_valid=True)


def test_selection_spacing_seed_recent_anchors_and_loops():
    for length, expected in ((0, [0]), (5, [0]), (63, [0]), (64, [0, 32]), (96, [0, 32, 64]),
                             (160, [0, 32, 64, 96, 128]), (256, [0, 32, 64, 96, 128, 160, 192, 224])):
        np.testing.assert_allclose(selected_arcs(length), expected)
    for length in np.linspace(0, 4096, 1000):
        chosen = selected_arcs(length)
        assert len(chosen) <= 8 and chosen[0] == 0
        assert np.all(np.diff(chosen) >= 32-1e-8)
        if length >= 256:
            np.testing.assert_allclose(chosen[-3:], length-np.array([96,64,32]))
    # Spatial repeats remain different historical observations on a loop.
    path = np.array([[0,0,0],[0,0,64],[0,0,0],[0,0,64]],float)
    slabs = slab_layout(observation(path))
    np.testing.assert_allclose([s['age'] for s in slabs], 192-selected_arcs(192))
    assert len(slabs) == 6


def test_four_point_heading_regresses_arclength_with_boundary_shift():
    t = np.linspace(0,np.pi,101)
    path = np.c_[12*np.sin(t),np.zeros(len(t)),12*(1-np.cos(t))]
    arc = arclength(path)
    for at in (0., 3., 8., 20.):
        start = np.clip(at-3,0,arc[-1]-6)
        points = np.stack([np.interp(start+np.arange(4)*2,arc,path[:,i]) for i in range(3)],1)
        slope = np.linalg.lstsq(np.c_[np.ones(4),np.arange(4)*2],points,rcond=None)[0][1]
        expected = slope/np.linalg.norm(slope)
        np.testing.assert_allclose(fitted_heading(path,arc,at,[0,1,0]),expected,atol=1e-12)
        reverse = fitted_heading(path[::-1],arclength(path[::-1]),arc[-1]-at,[0,1,0])
        np.testing.assert_allclose(reverse,-expected,atol=1e-12)


def test_short_degenerate_and_strict_prefix():
    path = np.array([[0.,0,0],[0,0,2],[0,0,2]])
    np.testing.assert_allclose(fitted_heading(path,arclength(path),1,[1,0,0]),[1,0,0])
    loop = np.array([[0.,0,0],[0,0,2],[0,0,2],[0,0,0]])
    np.testing.assert_allclose(fitted_heading(loop,np.array([0.,2,4,6]),3,[1,0,0]),[1,0,0])
    item = observation(path)
    assert len(observed_path(item)) == 2
    with pytest.raises(ValueError, match='end at'):
        slab_layout(dict(item, observed_path=np.r_[path, [[0,0,100]]]))
    with pytest.raises(ValueError, match='Complete observed'):
        slab_layout(dict(item, seed_pos=np.array([1.,2,3])))


def fake_ct(monkeypatch):
    calls = []
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.ct_tensor',
                        lambda vol,pos: np.outer(np.array([0., 1., 0.]), np.array([0., 1., 0.])))
    def scalar(items, vol, crop, pool=None, *, presence=False, **kwargs):
        assert not presence
        calls.extend(items)
        return torch.stack([torch.full((1,crop.depth,crop.width,crop.width),float(i['pos'][2])/512)
                            for i in items])
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.data.crop_sampling.scalar_crops',scalar)
    return calls


def test_only_valid_ct_reads_remote_slabs_and_annotation_independence(monkeypatch):
    calls = fake_ct(monkeypatch)
    path = np.c_[np.arange(401)*.1,np.zeros(401),np.arange(401)]
    item = observation(path)
    c = coordinate_config()
    first = load_slabs([item],None,c)
    assert len(calls) == 8 and first['history_slabs'].shape == (1,8,2,8,65,65)
    assert first['history_overlap'][0,0] == 0 and first['history_valid'][0,0]
    assert first['history_path_valid'][0,0].tolist() == [False,True,True]
    calls.clear()
    other = load_slabs([dict(item, gt_history=np.full((32,3),np.nan), terminal=1., offtrack=True,
                            reference_on_fiber=np.zeros(33))],None,c)
    for key in ('history_slabs','history_valid','history_pose',
                'history_path_points','history_path_tangents','history_path_valid'):
        torch.testing.assert_close(first[key],other[key],rtol=0,atol=0)
    calls.clear()
    short = load_slabs([observation(path[:5])],None,cfg())
    assert len(calls) == 1 and not short['history_slabs'][0,1:].any()


def test_heatmap_follows_committed_wrong_turn_and_not_a_seed_chord(monkeypatch):
    fake_ct(monkeypatch)
    path = np.array([[0.,0,0],[0,0,4],[4,0,4],[4,0,8]])
    slabs = load_slabs([observation(path)],None,cfg())
    # Seed heading is fitted, but renderer must mark every actual vertex in
    # the local +/-8 arclength segment. Test against the shared renderer input.
    seed = slab_layout(observation(path))[0]
    assert seed['hmask'][0] == 0
    world = seed['hist_local'][1:] @ seed['frame'].T+seed['pos']
    np.testing.assert_allclose(world,path[:3],atol=1e-12)
    assert slabs['history_slabs'][0,0,1].max() > .9
    assert not slabs_allowed(observation(np.array([[0.,0,90],[0,0,400]])),ZBand(90,110))


OBSERVED_HISTORY = ('history_slabs','history_pose','history_valid','history_path_points',
                    'history_path_tangents','history_path_valid','path_geometry','path_geometry_valid')


def replay_for(item):
    from replay_fixtures import replay_states
    row=dict(pos=item['pos'],frame=item['frame'],hist=item['hist_local'] @ item['frame'].T+item['pos'],
             hmask=item['hmask'],seq_start=0,seq_end=len(item['observed_path']),
             **{k:item[k] for k in ('seed_pos','seed_tangent','seed_age','seed_valid')})
    return replay_states([],[row],track=item['observed_path'])


def test_fresh_replay_inference_and_resume_inputs_identical(monkeypatch,tmp_path):
    fake_ct(monkeypatch)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.data.observations.image_crop',
        lambda items,vol,crop,pool=None,**kw:torch.zeros(len(items),2,crop.depth,crop.width,crop.width))
    item = observation(np.c_[np.sin(np.arange(301)/12),np.zeros(301),np.arange(301)])
    states = replay_for(item)
    states.save(tmp_path/'replay.npz')
    loaded = OnPolicyStates.load(tmp_path/'replay.npz')
    builder = ObservationBuilder(coordinate_config())
    fresh = builder.images([item],None)
    replay = builder.images([dict(item,observed_path=loaded.observed_prefix(0))],None)
    tracer = FiberTracer.__new__(FiberTracer)
    tracer.device,tracer.vol,tracer.pool,tracer.observations='cpu',None,None,builder
    direct = tracer.build_inputs(item['pos'][None],item['frame'][None],item['hist_local'][None],item['hmask'][None],paths=[item])
    resumed = tracer.build_inputs(item['pos'][None],item['frame'][None],item['hist_local'][None],item['hmask'][None],
        paths=[dict(item,observed_path=loaded.observed_prefix(0))])
    for key in OBSERVED_HISTORY:
        for other in (replay,direct,resumed):
            torch.testing.assert_close(fresh[key],other[key],atol=0,rtol=0)
    loaded.seq_end = loaded.seq_end.copy()-1
    with pytest.raises(ValueError,match='end at'):
        loaded.observed_prefix(0)


def test_both_heads_train_history_and_only_geometry_trains_generator():
    torch.manual_seed(55)
    m=build_model(coordinate_config())
    b=coordinate_batch(m.cfg,1)
    b['x']['history_slabs'].requires_grad_()
    for loss in ('geometry','generated_confidence'):
        m.zero_grad(set_to_none=True);b['x']['history_slabs'].grad=None
        out=m(b['x'],b['hist'],b['hmask'])
        {'geometry':out['points'].square().mean(),'generated_confidence':out['hazard_logits'].sum()}[loss].backward()
        for module in (m.history_encoder.convolution[0],m.encoder.patch_projection,m.encoder.stem.projection):
            assert module.weight.grad.abs().sum()>0,(loss,module)
        assert b['x']['history_slabs'].grad[0,:2].abs().sum()>0
        assert not b['x']['history_slabs'].grad[0,2:].any()
    assert m.coordinates.weight.grad is None


def test_invalid_slots_paths_and_geometry_are_inert_and_never_encoded():
    torch.manual_seed(9)
    m=build_model(coordinate_config()).eval();b=coordinate_batch(m.cfg,2)
    calls=[]
    handle=m.history_encoder.convolution.register_forward_pre_hook(lambda module,args:calls.append(len(args[0])))
    b['x']['history_path_valid'][0,1,0]=False
    b['x']['path_geometry_valid'][1,:5]=False
    first=m(b['x'],b['hist'],b['hmask'])
    x=b['x']
    x['history_slabs'][:,2:]=float('nan');x['history_pose'][:,2:]=float('nan')
    x['history_path_points'][0,1,0]=float('nan');x['history_path_points'][:,2:]=float('nan')
    x['path_geometry'][1,:5]=float('nan')
    other=m(x,b['hist'],b['hmask'])
    for key in ('points','hazard_logits','refinement_points'):
        assert torch.isfinite(other[key]).all()
        torch.testing.assert_close(first[key],other[key],atol=0,rtol=0)
    assert calls==[4,4]
    x['history_valid'][:]=False
    tokens,padding=m.encode_history(x)
    assert calls[-1]==0 and padding.all() and not tokens.any()
    handle.remove()


def test_complete_synthetic_prefix_precedes_local_history_truncation():
    from model_fixtures import line_fiber
    fiber=line_fiber(1500.)
    sample=SampleConfig(n_history=128,startup_shares=(0.,0.,0.,1.))
    lengths=[]
    for seed in range(20):
        item=make_sample(fiber,1000.,False,sample,np.random.default_rng(seed))
        path=item['observed_path'];lengths.append(len(path))
        assert item['hist_local'].shape==(128,3)
        np.testing.assert_allclose(path[0],item['seed_pos'])
        np.testing.assert_allclose(path[-1],item['pos'])
        # The tracer's history of the observed path: unit arclength steps back from the head.
        hist,mask=trace_history(list(path),128)
        np.testing.assert_array_equal(item['hmask'],mask)
        n=int(mask.sum())
        np.testing.assert_allclose(item['hist_local'][:n] @ item['frame'].T+item['pos'],hist[:n],atol=1e-9)
    assert max(lengths)>128 and len(set(lengths))>10
    no=make_sample(fiber,1000.,False,SampleConfig(startup_shares=(1.,0.,0.,0.)),np.random.default_rng(3))
    assert len(no['observed_path'])==1 and not no['hmask'].any()


def test_actual_trace_commits_and_resumed_slabs_use_same_prefix(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams
    fake_ct(monkeypatch)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.data.observations.image_crop',
        lambda items,vol,crop,pool=None,**kw:torch.zeros(len(items),1,crop.depth,crop.width,crop.width))
    c=coordinate_config()
    class Model(torch.nn.Module):
        cfg=c
        def forward(self,x,hist,hmask):
            p=hist.new_tensor([[[.2,0,1.],[-.3,0,2.],[.1,0,3.],[0,0,4.]]]).expand(len(hist),-1,-1)
            return dict(points=p,confidence=p.new_ones(len(hist),4))
    def tracer():
        t=FiberTracer.__new__(FiberTracer)
        t.model,t.device,t.n_history=Model(),'cpu',c.n_history
        t.vol,t.pool,t.observations=SimpleNamespace(shape=(1000,1000,1000)),None,ObservationBuilder(c)
        t.p=TraceParams(n_commit=4,max_len=12.,loop_radius=.01)
        return t
    original=tracer();seen=[];images=[]
    build=original.build_inputs
    def record(*args,**kwargs):
        x=build(*args,**kwargs);images.append(x);return x
    monkeypatch.setattr(original,'build_inputs',record)
    original._trace(np.array([[100.,100.,100.]]),np.array([[0.,0.,1.]]),None,None,
                    lambda i,state:seen.append(state))
    assert len(seen)>1 and len(seen[1]['observed_path'])>4
    resumed=tracer();again=[]
    build=resumed.build_inputs
    monkeypatch.setattr(resumed,'build_inputs',lambda *args,**kw:again.append(build(*args,**kw)) or again[-1])
    resumed._trace(seen[1]['pos'][None],seen[1]['frame'][:,2][None],None,None,None,initial_states=[seen[1]])
    for key in OBSERVED_HISTORY:
        torch.testing.assert_close(images[1][key],again[0][key],atol=0,rtol=0)


def test_cached_history_attention_matches_mha_and_reuses_attached_projections():
    device = 'cpu'
    from vesuvius.neural_tracing.fiber_follow.models.history_slabs import HistoryAttention
    torch.manual_seed(91)
    dtype = torch.float32 if device == 'cuda' else torch.float64
    original = HistoryAttention(32, 4).to(device=device, dtype=dtype)
    cached = copy.deepcopy(original)
    query = torch.randn(3, 4, 32, device=device, dtype=dtype)
    tokens = torch.randn(3, 162, 32, device=device, dtype=dtype)
    padding = torch.zeros(3, 162, device=device, dtype=torch.bool)
    padding[1, 80:] = True
    padding[2] = True
    tokens[2] = 0.
    outputs, gradients = [], []
    for model, use_cache in ((original, False), (cached, True)):
        q, t = query.clone().requires_grad_(), tokens.clone().requires_grad_()
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            if use_cache:
                projected = model.project_memory(t, padding)
            for _ in range(3):
                if use_cache:
                    q = model.forward_cached(q, *projected)
                else:
                    empty = padding.all(-1)
                    value = model.attention(model.norm(q), t, t,
                        key_padding_mask=padding & ~empty[:, None], need_weights=False)[0]
                    q = q+torch.where(empty[:, None, None], 0., value)
        q.square().mean().backward()
        outputs.append(q.detach())
        gradients.append((t.grad, *(p.grad for p in model.parameters())))
        assert t.grad[padding].eq(0).all()
        torch.testing.assert_close(q[2], query[2], rtol=0, atol=0)
    rtol, atol = (.03, .002) if device == 'cuda' else (1e-9, 1e-10)
    torch.testing.assert_close(*outputs, rtol=rtol, atol=atol)
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def test_slab_instance_norm_matches_native_and_supports_empty_backward():
    from vesuvius.neural_tracing.fiber_follow.models.history_slabs import SlabInstanceNorm
    native = torch.nn.InstanceNorm3d(8, affine=True)
    norm = SlabInstanceNorm(8)
    norm.load_state_dict(native.state_dict())
    for batch_size in (0, 2):
        x = torch.randn(batch_size, 8, 2, 9, 9, requires_grad=True)
        actual = norm(x)
        if batch_size:
            torch.testing.assert_close(actual, native(x), atol=1e-6, rtol=1e-5)
        actual.square().sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_first_unusable_slab_transports_current_ct_roll_without_dropping_history(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import FRAME_POLICY
    from vesuvius.neural_tracing.fiber_follow.shared.geometry import frame_from_heading
    calls = fake_ct(monkeypatch)
    path = np.c_[np.zeros(129), np.zeros(129), np.arange(129)]
    item = observation(path)
    item['frame'] = frame_from_heading([0., 0., 1.], [1., 1., 0.])
    item['frame_policy'] = FRAME_POLICY
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.ct_tensor', lambda vol, pos: np.zeros((3,3)))
    expected = slab_layout(item)
    result = load_slabs([item], None, cfg())
    assert result['history_valid'].sum() == len(expected) == len(calls)
    np.testing.assert_array_equal(item['observed_path'], path)
    for before, after in zip(expected, item['_sampled_slabs']):
        np.testing.assert_allclose(after['frame'], item['frame'], atol=1e-12)
        np.testing.assert_allclose(after['hist_local'] @ after['frame'].T,
                                   before['hist_local'] @ before['frame'].T, atol=1e-12)
        np.testing.assert_array_equal(after['hmask'], before['hmask'])
    assert (result['history_frame_source'][result['history_valid']] == 1).all()
    # Without a validated anchor the first slab chooses deterministic roll.
    other = load_slabs([observation(path)], None, cfg())
    assert other['history_frame_source'][0,0] == 2
    assert (other['history_frame_source'][other['history_valid']][1:] == 1).all()


def test_valid_first_slab_keeps_independent_sign_despite_opposite_current_roll(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import FRAME_POLICY, transverse_frame
    fake_ct(monkeypatch)
    item = observation([[0., 0., 0.], [0., 0., 4.]])
    expected = transverse_frame(np.diag([0., 1., 0.]), [0., 0., 1.])
    item['frame'] = expected @ np.diag([-1., -1., 1.])
    item['frame_policy'] = FRAME_POLICY
    load_slabs([item], None, cfg())
    np.testing.assert_allclose(item['_sampled_slabs'][0]['frame'], expected, atol=1e-12)
