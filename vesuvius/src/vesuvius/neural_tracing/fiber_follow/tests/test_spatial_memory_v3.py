"""Spatial identity routes, causal admission, supervision and checkpoint compatibility."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_learned_memory import memory_config, memory_batch
from test_neighbor_bank import make_bank, add_shard, publish
from test_neighbor_following import clean_sample
from vesuvius.neural_tracing.fiber_follow.regression.model import (
    build_model, DirectConfig, DirectFollower, MEMORY_ARCHITECTURE, SPATIAL_MEMORY_ARCHITECTURE,
)
from vesuvius.neural_tracing.fiber_follow.regression.identity_memory import IdentityMemory
from vesuvius.neural_tracing.fiber_follow.regression.spatial_model import connected_route, route_neighbors
from vesuvius.neural_tracing.fiber_follow.regression.spatial_supervision import route_loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.spatial_sequences import earlier_decision
from vesuvius.neural_tracing.fiber_follow.regression.identity_decisions import decision_pair
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, DirectTracer
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    optimizer_update, save_checkpoint, load_checkpoint, initialize_spatial_model, add_direction_inputs,
)
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.shared.trace import TraceParams


def cfg(**kw):
    return memory_config(memory_version=3,**kw)


def test_route_decoding_keeps_modes_connected_and_initial_step_reachable():
    xy = torch.cartesian_prod(torch.arange(-2.,3.),torch.arange(-2.,3.))
    neighbors, valid, costs = route_neighbors(5,1)
    logits = torch.zeros(1,4,25)
    logits[:,:,0] = 100  # unreachable at the first plane
    path = connected_route(logits,xy,neighbors,valid,costs*.25,0.)
    actual = xy[path][0]
    torch.testing.assert_close(actual[0],torch.zeros(2))
    assert (actual.diff(dim=0).abs() <= 1).all()
    assert (actual[-1] == torch.tensor([-2.,-2.])).all()
    # Two equal separated modes: output must be one mode, never their mean.
    logits.zero_(); logits[:,:,2] = 10; logits[:,:,22] = 10
    actual = xy[connected_route(logits,xy,neighbors,valid,costs*.25,3.)][0]
    assert actual[:,0].abs().eq(2).all()


def test_route_loss_trains_seed_writer_and_spatial_reader_without_confidence_loss():
    torch.manual_seed(19)
    model = build_model(cfg())
    batch = memory_batch(model.cfg)
    batch['x']['memory_patches'].requires_grad_()
    batch['x']['memory_seed_patch'].requires_grad_()
    out = model(batch['x'],batch['hist'],batch['hmask'])
    loss_terms(out,batch,model.cfg)['route_per_state'].mean().backward()
    assert batch['x']['memory_patches'].grad[:,0].abs().sum() > 0
    assert batch['x']['memory_seed_patch'].grad.abs().sum() > 0
    for parameter in (model.route_attention.in_proj_weight,model.recurrent_memory.gate.weight,
                      model.recurrent_memory.probe_head[-1].weight):
        assert parameter.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert (out['points'][:,:1].norm(dim=-1) <= model.cfg.max_recovery_distance+1e-5).all()
    half_cell = model.cfg.lateral_limit/(model.route_width-1)
    assert (out['points'][...,:2]-out['initial_points'][...,:2]).abs().max() <= half_cell+1e-5


def test_route_targets_can_localize_departed_heads_but_unknowns_have_no_gradient():
    c = cfg()
    model = build_model(c)
    batch = memory_batch(c)
    logits = torch.randn(2,c.n_future,len(model.route_xy),requires_grad=True)
    batch.update(offtrack=torch.ones(2),route_ab=torch.zeros(2,c.n_future,2),
                 route_mask=torch.tensor([[True]*c.n_future,[False]*c.n_future]))
    batch['route_ab'][1] = float('nan')
    terms = route_loss_terms(dict(route_logits=logits),batch,c)
    terms['route_per_state'].sum().backward()
    assert logits.grad[0].abs().sum() > 0 and logits.grad[1].abs().sum() == 0
    assert torch.isfinite(terms['route_per_state']).all()


def test_admission_is_prewrite_and_rejection_preserves_identity_but_keeps_recent():
    torch.manual_seed(6)
    memory = IdentityMemory(cfg())
    x = memory_batch(memory.cfg,1)['x']
    x = {k:v[:,:1] if k in ('memory_patches','memory_mask','memory_positions','memory_frames') else v for k,v in x.items()}
    # Hard rejection through the learned pre-write classifier.
    with torch.no_grad():
        memory.probe_head[-1].weight.zero_()
        memory.probe_head[-1].bias.zero_()
        memory.probe_head[-1].bias[0] = -100
    before = memory.initial_state(1,'cpu')
    out = memory.observe(x,before)
    torch.testing.assert_close(out['slots'],before['slots'],rtol=0,atol=0)
    assert out['recent'].abs().sum() > 0
    # A different write proposal cannot affect the admission judgment for that write.
    with torch.no_grad(): memory.proposal.bias.add_(100)
    torch.testing.assert_close(memory.observe(x,before)['probe'],out['probe'],rtol=0,atol=0)


def test_spatial_memory_streaming_matches_sequence_and_padding_is_inert():
    torch.manual_seed(8)
    memory = IdentityMemory(cfg()).eval()
    x = memory_batch(memory.cfg)['x']
    x['memory_mask'][:,1] = False
    expected = memory.observe(x)
    corrupted = copy.deepcopy(x)
    for key in ('memory_patches','memory_positions','memory_frames'):
        corrupted[key][:,1] = float('nan')
    for key,value in memory.observe(corrupted).items():
        torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
    state, probes = None, []
    for j in range(x['memory_mask'].shape[1]):
        part = {k:v[:,j:j+1] if k in ('memory_patches','memory_mask','memory_positions','memory_frames') else v for k,v in x.items()}
        state = memory.observe(part,state)
        probes.append(state.pop('probe'))
    for key,value in state.items():
        torch.testing.assert_close(value,expected[key],rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(torch.cat(probes,1),expected['probe'],rtol=2e-5,atol=2e-6)


def test_spatial_burn_in_is_gradient_free_but_preserves_state():
    memory = IdentityMemory(cfg(memory_steps=6,memory_grad_steps=2))
    x = memory_batch(memory.cfg)['x']
    x['memory_patches'].requires_grad_()
    out = memory.observe(x)
    (out['slots'].square().sum()+out['probe'].square().sum()).backward()
    g = x['memory_patches'].grad.flatten(2).abs().sum(-1)
    assert (g[:,:4] == 0).all() and (g[:,4:] > 0).all()


def test_matched_spatial_examples_require_out_of_crop_identity(tmp_path):
    bank, parent = make_bank(tmp_path)
    publish(tmp_path,[add_shard(tmp_path,0,x=4.,z_range=(20.,180.))])
    c = DirectConfig(memory_slots=4,memory_version=3)
    builder = IdentityObservationBuilder(c,[parent],negative_bank=bank,augment=True)
    rng = np.random.default_rng(7)
    counts = set()
    for _ in range(12):
        rows = decision_pair(bank,clean_sample(c),c,rng)
        assert rows is not None
        for row in rows:
            fiber = row.get('supervision_fiber',parent)
            builder.prepare(row,fiber,rng)
            assert not row['visible_seed_mask'].any()
            assert row['identity_observable']
            assert row['route_mask'].any()
            assert len(row['memory_track']['pos']) > 1
            assert builder.footprint_allowed(row,bank.band)
            earlier = earlier_decision(row,builder)
            assert earlier is not None
            assert len(earlier['memory_track']['pos']) < len(row['memory_track']['pos'])
            assert earlier['seed_age'] < row['seed_age']
            assert earlier['_route_sequence_member']
            assert earlier['route_mask'].any()
            counts.add(row['decision_kind'])
        a,b = rows
        assert a['photometric'] == b['photometric']
        assert a['identity_seed'] == b['identity_seed']
        for key in ('pos','frame','hist_local','hmask'):
            np.testing.assert_array_equal(a[key],b[key])
        assert not np.array_equal(a['memory_track']['pos'],b['memory_track']['pos'])
        assert not np.array_equal(a['route_ab'],b['route_ab'])
    assert counts == {1,2,3}


@pytest.mark.parametrize('version',[2,3])
def test_versions_roundtrip_and_remain_trainable(tmp_path,version):
    torch.manual_seed(32)
    c = memory_config(memory_version=version)
    model = build_model(c)
    batch = memory_batch(c)
    before = model.encoder.stem[0].weight.detach().clone()
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(),lr=.001)
    metrics = optimizer_update(model,ema,opt,[batch],1,.001,device='cpu',compute_metrics=False)
    assert np.isfinite(metrics['loss'])
    assert not torch.equal(before,model.encoder.stem[0].weight)
    assert ('route_loss' in metrics.get('memory',{})) == (version == 3)
    path = tmp_path/f'v{version}.pt'
    spec = FiberVolumeSpec('/tmp/presence',ct_zarr='/tmp/ct',inputs='ct+presence')
    sample = SampleConfig(crop=c.fine,n_history=c.n_history,n_future=c.n_future)
    save_checkpoint(path,model,ema,spec,sample)
    restored,*_ = load_checkpoint(path,'cpu')
    assert restored.architecture == (SPATIAL_MEMORY_ARCHITECTURE if version == 3 else MEMORY_ARCHITECTURE)
    with torch.no_grad():
        expected = ema.eval()(batch['x'],batch['hist'],batch['hmask'])
        actual = restored(batch['x'],batch['hist'],batch['hmask'])
    for key in ('points','confidence','memory_slots'):
        torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)


def test_explicit_upgrade_preserves_legacy_weights_and_supports_direction_inputs():
    old = DirectFollower(memory_config())
    upgraded = initialize_spatial_model(old,replace(old.cfg,memory_version=3))
    for key,value in old.state_dict().items():
        torch.testing.assert_close(upgraded.state_dict()[key],value,rtol=0,atol=0)
    assert old.cfg.memory_version == 2
    expanded = add_direction_inputs(upgraded)
    assert expanded.architecture == SPATIAL_MEMORY_ARCHITECTURE
    assert expanded.encoder.stem[0].weight.shape[1] == 8


def test_sequence_route_loss_updates_the_writer():
    torch.manual_seed(32)
    model = build_model(cfg())
    batch = memory_batch(model.cfg)
    earlier = memory_batch(model.cfg,1)
    batch['route_sequence'] = earlier
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(),lr=.001)
    metrics = optimizer_update(model,ema,opt,[batch],1,.001,device='cpu',compute_metrics=False)
    assert metrics['memory']['sequence_states'] == 1
    assert metrics['memory']['sequence_loss'] > 0
    assert model.recurrent_memory.probe_head[-1].weight.grad.abs().sum() > 0


def test_training_builder_includes_causal_sequence_and_labels_stay_out_of_inputs(tmp_path,monkeypatch):
    bank,parent = make_bank(tmp_path)
    publish(tmp_path,[add_shard(tmp_path,0,x=4.,z_range=(20.,180.))])
    c = cfg(fine=CropSpec(depth=40,width=25,behind=16,spacing=1.),memory_steps=16)
    builder = IdentityObservationBuilder(c,[parent],negative_bank=bank,augment=True)
    rng = np.random.default_rng(11)
    rows = decision_pair(bank,clean_sample(c),c,rng)
    assert rows is not None
    for row in rows:
        builder.prepare(row,row.get('supervision_fiber',parent),rng)
    def images(items,vol,crop,pool=None,**kwargs):
        return torch.ones(len(items),2,crop.depth,crop.width,crop.width)*.25
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.data.image_crop',images)
    b = builder(rows,None)
    assert 'route_sequence' in b and b['route_mask'].any()
    assert {'route_ab','route_mask','offtrack','memory_target_identity'}.isdisjoint(b['x'])
    assert b['x']['route_frame'].shape == (2,3,3)
    torch.testing.assert_close(b['x']['fine'][0],b['x']['fine'][1],rtol=0,atol=0)
    model = build_model(c)
    metrics = optimizer_update(model,copy.deepcopy(model),torch.optim.AdamW(model.parameters()),
                               [b],1,.001,device='cpu',compute_metrics=False)
    assert np.isfinite(metrics['loss']) and metrics['memory']['sequence_states'] > 0


def test_spatial_forward_and_backward_capture_without_graph_breaks():
    model = build_model(cfg())
    b = memory_batch(model.cfg,1)
    compiled = torch.compile(model,backend='eager',fullgraph=True)
    out = compiled(b['x'],b['hist'],b['hmask'])
    terms = loss_terms(out,b,model.cfg)
    (terms['route_per_state']+terms['geometry_per_state']+terms['confidence_per_state']).mean().backward()
    assert model.recurrent_memory.gate.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('version',[2,3])
def test_compiler_handles_streaming_then_long_training_sequence(version):
    model = build_model(memory_config(memory_version=version,memory_steps=6,memory_grad_steps=2))
    b = memory_batch(model.cfg,1)
    short = {k:v[:,:1] if k in ('memory_patches','memory_mask','memory_positions','memory_frames') else v for k,v in b['x'].items()}
    compiled = torch.compile(model,backend='eager',fullgraph=True)
    with torch.no_grad():
        compiled(short,b['hist'],b['hmask'])
    out = compiled(b['x'],b['hist'],b['hmask'])
    out['memory_probe'].square().mean().backward()
    assert model.recurrent_memory.patch_encoder[0].weight.grad.abs().sum() > 0


def test_spatial_tracer_carries_new_state_and_reads_seed_once(monkeypatch):
    model = build_model(cfg(correction=False))
    def images(items,vol,crop,pool=None,**kwargs):
        return torch.ones(len(items),2,crop.depth,crop.width,crop.width)*.25
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.data.image_crop',images)
    recorded = []
    observe = model.recurrent_memory.observe
    def capture(x,state=None):
        recorded.append((x['memory_mask'].clone(),x['memory_seed_valid'].clone(),state['recent'].clone()))
        return observe(x,state)
    monkeypatch.setattr(model.recurrent_memory,'observe',capture)
    tracer = DirectTracer(model,SimpleNamespace(shape=(1000,1000,1000)),model.cfg.fine,model.cfg.n_history,
                          TraceParams(n_commit=1,max_len=12,confidence=0.),device='cpu')
    try:
        paths,reasons = tracer.trace(np.array([[100.,100.,400.]]),np.array([[0.,0.,1.]]))
    finally:
        tracer.close()
    assert len(recorded) >= 2 and len(paths[0]) > 2
    assert recorded[0][1].all()
    assert all(not row[1].any() and row[0].shape[1] == 1 for row in recorded[1:])
    assert recorded[1][2].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_compiled_spatial_bf16_gradients_match_eager():
    from vesuvius.neural_tracing.fiber_follow.regression.train import compile_training_model,conv_memory_format
    torch.manual_seed(44)
    eager = build_model(cfg(memory_steps=6,memory_grad_steps=3)).to('cuda',memory_format=conv_memory_format('cuda'))
    compiled_base = copy.deepcopy(eager)
    b = memory_batch(eager.cfg,2)
    count = eager.cfg.memory_steps+1
    b.update(memory_target_identity=torch.ones(2,count),memory_target_identity_mask=torch.ones(2,count,dtype=torch.bool),
             memory_target_offset=torch.zeros(2,count,3),memory_target_offset_mask=torch.ones(2,count,dtype=torch.bool))
    results = []
    for model in (eager,compile_training_model(compiled_base)):
        results.append(optimizer_update(model,copy.deepcopy(eager),torch.optim.SGD(model.parameters(),lr=0.),
                       [b],1,0.,device='cuda',compute_metrics=False))
    assert results[1]['loss'] == pytest.approx(results[0]['loss'],rel=.02)
    for prefix in ('recurrent_memory.','route_','encoder.'):
        vectors = [torch.cat([p.grad.flatten() for n,p in m.named_parameters() if n.startswith(prefix) and p.grad is not None])
                   for m in (eager,compiled_base)]
        assert vectors[0].norm() > 0
        assert (vectors[1]-vectors[0]).norm()/vectors[0].norm() < .05
