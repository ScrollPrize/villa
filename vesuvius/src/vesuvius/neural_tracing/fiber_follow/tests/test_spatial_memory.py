"""Spatial evidence survives history encoding and is read causally by candidates."""
import copy
from dataclasses import replace

import pytest
import torch
import numpy as np

from test_unified import scene, state
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectFollower, SPATIAL_MEMORY_ARCHITECTURE
from vesuvius.neural_tracing.fiber_follow.regression.train import checkpoint_config, initialize_encoder, compile_training_model


def spatial_scene(b=1, steps=4, **options):
    cfg,data = scene(b=b,steps=steps,grad=steps)
    options = dict(spatial_recent=1,spatial_archive=2,spatial_retrieve=1,**options)
    return replace(cfg,memory_version=4,**options),data


def test_full_grid_is_retained_and_remote_spatial_evidence_affects_predictions():
    torch.manual_seed(41)
    cfg,data = spatial_scene()
    model = DirectFollower(cfg).eval()
    with torch.no_grad():
        out = model(data['x'],data['hist'],data['hmask'])
        memory = state(out,model)
        assert memory['bank'].shape[2] == model.spatial_count+27
        context,valid,refs,ref_valid,roles = model.retained(memory,memory['position'],memory['frame'])
        obs = model.observation(data['x']['fine'])
        context.update(current=model.spatial_tokens(obs),current_valid=torch.ones(obs.shape[:2],dtype=torch.bool))
        points = torch.zeros(1,cfg.n_future,3)
        patches = model.head_patches(obs)[:,None].expand(-1,cfg.n_future,-1,-1)
        support = torch.ones(1,cfg.n_future,27,dtype=torch.bool)
        baseline = model.evaluate(points,patches,support,context,valid,refs,ref_valid,roles)
        # Change only spatial evidence away from the fixed 27 head descriptors.
        changed = copy.deepcopy(context)
        changed['state']['anchor'][:,:model.spatial_count] += torch.randn_like(memory['anchor'][:,:model.spatial_count])*3
        result = model.evaluate(points,patches,support,changed,valid,refs,ref_valid,roles)
        assert not torch.allclose(baseline,result)


def test_streaming_matches_unroll_through_archive_eviction_and_seed_stays_fixed():
    torch.manual_seed(43)
    cfg,data = spatial_scene(b=2,steps=6)
    model = DirectFollower(cfg).eval()
    x = data['x']
    with torch.no_grad():
        whole = model(x,data['hist'],data['hmask'])
        carry = None
        for j in range(cfg.memory_steps+1):
            step = dict(x,fine=x['history_crops'][:,j] if j < cfg.memory_steps else x['fine'],
                        history_crops=x['history_crops'][:,:0])
            for key in ('memory_mask','memory_positions','memory_frames'):
                step[key] = x[key][:,j:j+1]
            if j:
                step['seed_crop'] = torch.full_like(x['seed_crop'],torch.nan)
            out = model(step,data['hist'],data['hmask'],memory=carry)
            carry = state(out,model)
        for key,value in carry.items():
            torch.testing.assert_close(value,whole['memory_'+key],rtol=3e-5,atol=3e-6)
        torch.testing.assert_close(out['points'],whole['points'],rtol=3e-5,atol=3e-6)
        assert carry['bank_valid'].all()
        assert carry['bank_position'][0,:,2].tolist() == [6,5,4,3]


@pytest.mark.parametrize('checkpointing',[False,True])
def test_training_reaches_spatial_encoder_seed_and_archive_router(checkpointing):
    torch.manual_seed(45)
    cfg,data = spatial_scene(activation_checkpointing=checkpointing)
    model = DirectFollower(cfg)
    data['x']['seed_crop'].requires_grad_()
    data['x']['history_crops'].requires_grad_()
    out = model(data['x'],data['hist'],data['hmask'])
    (out['points'].square().sum()+out['confidence_logits'].square().sum()+out['memory_probe'].square().sum()).backward()
    for param in (model.encoder.compress.weight,model.retrieval_key.weight,model.retrieval_query.weight,
                  model.summary_query,model.archive_attention.in_proj_weight,model.write_gate.weight):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    assert data['x']['seed_crop'].grad.abs().sum() > 0
    assert data['x']['history_crops'].grad.abs().sum() > 0


def test_empty_seed_archive_and_nan_padding_have_finite_values_and_gradients():
    cfg,data = spatial_scene(b=2)
    x = data['x']
    x['memory_seed_valid'].zero_(); x['seed_crop'].fill_(torch.nan)
    x['memory_mask'][0,:-1] = False
    x['history_crops'][0] = torch.nan
    x['memory_positions'][0,:-1] = torch.nan
    x['memory_frames'][0,:-1] = torch.nan
    model = DirectFollower(cfg)
    out = model(x,data['hist'],data['hmask'])
    assert torch.isfinite(out['points']).all() and torch.isfinite(out['confidence']).all()
    (out['points'].square().sum()+out['confidence_logits'].square().sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_candidates_read_independently_without_writing_memory():
    cfg,data = spatial_scene()
    model = DirectFollower(cfg).eval()
    curves = torch.randn(1,2,cfg.n_future,3,requires_grad=True)
    out = model(data['x'],data['hist'],data['hmask'],candidates=curves)
    out['candidate_confidence_logits'].sum().backward()
    assert curves.grad is None
    with torch.no_grad():
        reversed_out = model(data['x'],data['hist'],data['hmask'],candidates=curves.flip(1))
        torch.testing.assert_close(out['candidate_confidence_logits'],reversed_out['candidate_confidence_logits'].flip(1))
        for key,value in state(out,model).items():
            torch.testing.assert_close(value,reversed_out['memory_'+key])


def test_checkpoint_version_and_explicit_encoder_transfer(tmp_path):
    cfg,data = spatial_scene()
    model = DirectFollower(cfg)
    source = DirectFollower(replace(cfg,memory_version=3))
    before = model.retrieval_key.weight.detach().clone()
    initialize_encoder(model,source)
    torch.testing.assert_close(model.encoder.compress.weight,source.encoder.compress.weight)
    torch.testing.assert_close(before,model.retrieval_key.weight)
    payload = dict(architecture=model.architecture,model_cfg=cfg.to_dict(),weights=model.state_dict())
    path = tmp_path/'model.pt'
    torch.save(payload,path)
    restored = torch.load(path,weights_only=False)
    assert checkpoint_config(restored) == cfg
    other = DirectFollower(checkpoint_config(restored)).eval()
    other.load_state_dict(restored['weights'])
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(data['x'],data['hist'],data['hmask'])['points'],
                                   other(data['x'],data['hist'],data['hmask'])['points'])
    restored['architecture'] = source.architecture
    with pytest.raises(ValueError,match='disagree'):
        checkpoint_config(restored)


def test_compilation_preserves_parameter_names_and_spatial_gradients():
    cfg,data = spatial_scene(steps=2)
    model = DirectFollower(cfg)
    keys = set(model.state_dict())
    assert compile_training_model(model,backend='eager') is model
    assert set(model.state_dict()) == keys
    out = model(data['x'],data['hist'],data['hmask'])
    out['points'].square().sum().backward()
    assert model.encoder.compress.weight.grad.abs().sum() > 0


def test_window_optimizer_reencodes_seed_and_observations_after_every_update():
    from vesuvius.neural_tracing.fiber_follow.regression.train import optimizer_update
    cfg,data = spatial_scene(steps=2,trajectory_window=3)
    x = data['x']
    window = []
    for j in range(3):
        step = dict(data)
        step['x'] = dict(x,fine=x['history_crops'][:,j] if j < 2 else x['fine'],history_crops=x['history_crops'][:,:0])
        for key in ('memory_mask','memory_positions','memory_frames'):
            step['x'][key] = x[key][:,j:j+1]
        window.append(step)
    wrapped = dict(data,trajectory_windows=[window])
    model = DirectFollower(cfg)
    ema = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(),lr=.01)
    encodings,seed_encodings,carries = [],[],[]
    in_forward = [False]
    def encoded(module,args,result):
        if not in_forward[0]:  # activation-checkpoint recomputation is not caching
            return
        encodings.append(1)
        if torch.equal(args[0],x['seed_crop']):
            seed_encodings.append(result[2].detach().clone())
    handle = model.encoder.register_forward_hook(encoded)
    def started(m,args,kwargs):
        in_forward[0] = True
        carries.append(kwargs.get('memory') is not None)
    before = model.register_forward_pre_hook(started,with_kwargs=True)
    after = model.register_forward_hook(lambda *args:in_forward.__setitem__(0,False))
    for update in (1,2):
        metrics = optimizer_update(model,ema,opt,[wrapped],update,.01,compute_metrics=False)
        assert metrics['supervised_decisions'] == 3
    handle.remove(); before.remove(); after.remove()
    assert len(encodings) == 8  # one seed and three distinct observations per update
    assert carries == [False,True,True,False,True,True]
    assert len(seed_encodings) == 2 and not torch.equal(*seed_encodings)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_real_window_builder_reuses_crop_views_and_preserves_holdout(monkeypatch):
    from test_identity import line_fiber, real_like_builder
    from vesuvius.neural_tracing.fiber_follow.regression import data as data_module
    from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig, make_sample, ZBand
    cfg,_ = spatial_scene(trajectory_window=3)
    fiber = line_fiber()
    builder = real_like_builder(cfg,[fiber],False)
    sample = SampleConfig(crop=cfg.fine,n_history=cfg.n_history,n_future=cfg.n_future,recent_history_points=cfg.n_history)
    item = make_sample(fiber,50.,False,sample,np.random.default_rng(2))
    track_pos = np.array([[100.,100.,z] for z in (240.,242.,244.,246.)])
    item.update(fiber_ref=(0,50.,False),source=0,source_step=-1,stratum=-1,
        seed_pos=np.array([100.,100.,220.]),seed_tangent=np.array([0.,0.,1.]),seed_valid=True,seed_age=30.,
        memory_track=dict(pos=track_pos,frame=np.tile(np.eye(3),(4,1,1)),offtrack=np.zeros(4),offset=np.zeros((4,3))))
    builder.prepare(item,fiber,np.random.default_rng(3))
    assert len(item['_trajectory_children']) == 2
    assert builder.footprint_allowed(item,None)
    assert not builder.footprint_allowed(item,ZBand(240.,245.))
    calls = []
    def crop(items,vol,crop,pool=None):
        calls.extend(items)
        return torch.rand(len(items),2,crop.depth,crop.width,crop.width)
    monkeypatch.setattr(data_module,'image_crop',crop)
    batch = builder([item],None)
    window = batch['trajectory_windows'][0]
    assert len(window) == 3 and len(calls) == 6  # current + four past + seed
    assert [w['x']['memory_mask'].shape[1] for w in window] == [3,1,1]
    assert [w['x']['history_crops'].shape[1] for w in window] == [2,0,0]
    assert window[0]['x']['fine'].untyped_storage().data_ptr() == batch['x']['history_crops'].untyped_storage().data_ptr()
    # Labels are outside model inputs and every window ends at the original state.
    assert not any('target' in key or 'offtrack' in key for key in window[0]['x'])
    torch.testing.assert_close(window[-1]['dense_ab'],batch['dense_ab'])
    assert all(w['memory_target_identity'].shape == w['x']['memory_mask'].shape for w in window)
    from vesuvius.neural_tracing.fiber_follow.regression.train import optimizer_update
    model = DirectFollower(cfg)
    metrics = optimizer_update(model,copy.deepcopy(model),torch.optim.SGD(model.parameters(),lr=0.),
                               [batch],1,0.,compute_metrics=True)
    assert metrics['supervised_decisions'] == 3 and metrics['fresh_fraction'] == 1.
    assert np.isfinite(metrics['loss'])


def test_global_frame_change_does_not_change_history_reads():
    cfg,data = spatial_scene()
    model = DirectFollower(cfg).eval()
    changed = copy.deepcopy(data)
    rotation = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    shift = torch.tensor([31.,-47.,19.])
    for key in ('memory_positions','memory_seed_position'):
        changed['x'][key] = data['x'][key] @ rotation.T+shift
    for key in ('memory_frames','memory_seed_frame'):
        changed['x'][key] = rotation @ data['x'][key]
    with torch.no_grad():
        a = model(data['x'],data['hist'],data['hmask'])
        b = model(changed['x'],changed['hist'],changed['hmask'])
    torch.testing.assert_close(a['points'],b['points'],rtol=3e-5,atol=3e-6)
    torch.testing.assert_close(a['confidence'],b['confidence'],rtol=3e-5,atol=3e-6)


def test_invalid_spatial_memory_budgets():
    cfg,_ = spatial_scene()
    for change in (dict(spatial_recent=0),dict(spatial_archive=0),dict(spatial_retrieve=3),
                   dict(memory_slots=0),dict(trajectory_window=0)):
        with pytest.raises(ValueError):
            replace(cfg,**change)


def test_memory_storage_matches_inside_and_outside_autocast():
    cfg,_ = spatial_scene()
    model = DirectFollower(cfg)
    outside = model.initial_memory(1,'cpu')
    with torch.autocast('cpu',dtype=torch.bfloat16):
        inside = model.initial_memory(1,'cpu')
    for key in model.memory_keys:
        assert inside[key].dtype == outside[key].dtype
        torch.testing.assert_close(inside[key],outside[key])
