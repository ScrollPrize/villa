"""Learned retention, causal observations, train/trace parity and legacy checkpoints."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_identity import config, batch, line_fiber
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectFollower, ARCHITECTURE, MEMORY_ARCHITECTURE
from vesuvius.neural_tracing.fiber_follow.regression.memory import LearnedMemory
from vesuvius.neural_tracing.fiber_follow.regression.memory_data import memory_layout, memory_images, memory_allowed
from vesuvius.neural_tracing.fiber_follow.regression.data import DirectTracer, IdentityObservationBuilder
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import optimizer_update, save_checkpoint, load_checkpoint, checkpoint_config
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig, ZBand
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.shared.trace import ModelTracer, TraceParams


def memory_config(**kwargs):
    defaults = dict(memory_slots=4, memory_steps=4, memory_stride=4, memory_patch_size=5)
    defaults.update(kwargs)
    return config(**defaults)


def memory_batch(cfg, b=2):
    data = batch(cfg, b)
    t, n = cfg.memory_steps+1, cfg.memory_patch_size
    positions = torch.zeros(b, t, 3)
    positions[..., 2] = torch.arange(t)*cfg.memory_stride
    data['x'].update(memory_patches=torch.rand(b, t, 2, n, n, n), memory_mask=torch.ones(b,t,dtype=torch.bool),
        memory_positions=positions, memory_frames=torch.eye(3).expand(b,t,-1,-1).clone(),
        memory_seed_patch=torch.rand(b,2,n,n,n), memory_seed_valid=torch.ones(b,dtype=torch.bool),
        memory_seed_position=torch.zeros(b,3), memory_seed_frame=torch.eye(3).expand(b,-1,-1).clone())
    return data


def open_read(model):
    """Version-2 reads start as identities; emulate a partly trained read."""
    torch.nn.init.xavier_uniform_(model.recurrent_memory.read_attention.out_proj.weight)
    return model


def take(value, sl):
    return {k: take(v, sl) for k,v in value.items()} if isinstance(value, dict) else value[sl]


def state_from(out):
    return {k.removeprefix('memory_'): v for k,v in out.items() if k.startswith('memory_')}


def test_geometry_and_confidence_train_the_writer_and_old_observations():
    torch.manual_seed(25)
    cfg = memory_config()
    model = open_read(DirectFollower(cfg))
    data = memory_batch(cfg)
    data['x']['memory_patches'].requires_grad_()
    data['x']['memory_seed_patch'].requires_grad_()
    for name in ('geometry_per_state', 'confidence_per_state'):
        model.zero_grad(set_to_none=True)
        data['x']['memory_patches'].grad = None
        data['x']['memory_seed_patch'].grad = None
        out = model(data['x'], data['hist'], data['hmask'])
        loss_terms(out, data, cfg)[name].mean().backward()
        assert data['x']['memory_patches'].grad[:,0].abs().sum() > 0
        assert data['x']['memory_seed_patch'].grad.abs().sum() > 0
        for module in (model.recurrent_memory.gate, model.recurrent_memory.proposal,
                       model.recurrent_memory.patch_encoder[0]):
            assert module.weight.grad.abs().sum() > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_seed_survives_later_updates_and_memory_changes_both_outputs():
    torch.manual_seed(2)
    cfg = memory_config()
    model = open_read(DirectFollower(cfg)).eval()
    data = memory_batch(cfg, 1)
    first = model(data['x'], data['hist'], data['hmask'])
    state = state_from(first)
    changed = copy.deepcopy(data)
    changed['x']['memory_seed_patch'].fill_(float('nan'))
    continued = model(changed['x'], changed['hist'], changed['hmask'], memory=state)
    torch.testing.assert_close(continued['memory_anchor'], first['memory_anchor'], rtol=0, atol=0)
    assert torch.isfinite(continued['points']).all()
    other = {k: v.clone() for k,v in state.items()}
    other['slots'] += torch.randn_like(other['slots'])*2
    alternative = model(data['x'], data['hist'], data['hmask'], memory=other)
    assert not torch.allclose(continued['points'], alternative['points'])
    assert not torch.allclose(continued['confidence_logits'], alternative['confidence_logits'])


def test_streaming_writes_equal_sequence_unroll_and_batches_are_independent():
    torch.manual_seed(3)
    cfg = memory_config()
    memory = LearnedMemory(cfg).eval()
    x = memory_batch(cfg)['x']
    sequence = memory.observe(x)
    state, probes = None, []
    for t in range(x['memory_patches'].shape[1]):
        chunk = {k: v[:,t:t+1] if k in ('memory_patches','memory_mask','memory_positions','memory_frames') else v
                 for k,v in x.items()}
        state = memory.observe(chunk, state)
        probes.append(state.pop('probe'))
    torch.testing.assert_close(torch.cat(probes, 1), sequence['probe'], rtol=2e-5, atol=2e-6)
    for k in state:
        torch.testing.assert_close(state[k], sequence[k], rtol=2e-5, atol=2e-6)
        split = torch.cat([memory.observe(take(x,slice(i,i+1)))[k] for i in range(2)])
        torch.testing.assert_close(split, sequence[k], rtol=2e-5, atol=2e-6)


def test_padding_is_inert_and_absolute_world_translation_does_not_change_memory():
    memory = LearnedMemory(memory_config()).eval()
    x = memory_batch(memory.cfg)['x']
    x['memory_mask'][:,1] = False
    x['memory_seed_valid'].zero_()
    expected = memory.observe(x)
    changed = copy.deepcopy(x)
    changed['memory_patches'][:,1] = float('nan')
    changed['memory_positions'][:,1] = float('nan')
    changed['memory_frames'][:,1] = float('nan')
    changed['memory_seed_patch'].fill_(float('nan'))
    for k,v in memory.observe(changed).items():
        torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
    changed = copy.deepcopy(x)
    changed['memory_positions'] += torch.tensor([100.,200.,300.])
    changed['memory_seed_position'] += torch.tensor([100.,200.,300.])
    torch.testing.assert_close(memory.observe(changed)['slots'], expected['slots'], rtol=0, atol=0)


def observation(cfg):
    hist = np.c_[np.zeros(cfg.n_history), np.zeros(cfg.n_history), -np.arange(1,cfg.n_history+1)]
    return dict(pos=np.array([100.,100.,400.]), frame=np.eye(3), hist_local=hist,
                hmask=np.ones(cfg.n_history), seed_pos=np.array([100.,100.,368.]),
                seed_tangent=np.array([0.,0.,1.]), seed_age=32., seed_valid=True)


def test_observation_layout_uses_only_observed_history_and_warm_calls_only_read_new_head():
    cfg = memory_config()
    item = observation(cfg)
    observations, seed = memory_layout(item, cfg)
    np.testing.assert_array_equal([o['pos'][2] for o in observations], [384,388,392,396,400])
    assert seed['pos'][2] == 368
    changed = copy.deepcopy(item)
    changed.update(gt_history=np.full((33,3),np.nan), dense_ab=np.full((13,2),np.nan), offtrack=True)
    after, _ = memory_layout(changed, cfg)
    for a,b in zip(observations, after):
        np.testing.assert_array_equal(a['pos'], b['pos'])
        np.testing.assert_array_equal(a['frame'], b['frame'])
    changed['memory_warm'] = True
    after, seed = memory_layout(changed, cfg)
    assert len(after) == 1 and seed is None
    changed = copy.deepcopy(item)
    changed['hmask'][5] = 0
    after, _ = memory_layout(changed, cfg)
    assert len(after) == 2  # no observations beyond an unknown history gap


def test_remote_seed_and_history_reads_obey_holdout_before_io():
    cfg = memory_config()
    item = observation(cfg)
    assert memory_allowed(item, cfg, ZBand(600,610))
    item['seed_pos'][2] = 605
    assert not memory_allowed(item, cfg, ZBand(600,610))
    builder = IdentityObservationBuilder(cfg)
    assert not builder.footprint_allowed(item, ZBand(600,610))
    item['seed_valid'] = False
    assert builder.footprint_allowed(item, ZBand(600,610))


def test_observation_sampler_training_and_streaming_patches_match():
    cfg = memory_config()
    item = observation(cfg)
    reads = []
    def sample(items, vol, crop, pool=None):
        reads.extend(items)
        return torch.stack([torch.full((2,crop.depth,crop.width,crop.width),float(i['pos'][2])) for i in items])
    x = memory_images([item], None, cfg, sample)
    assert len(reads) == 6
    torch.testing.assert_close(x['memory_patches'][0,:,0,0,0,0],torch.tensor([384.,388.,392.,396.,400.]))
    reads.clear()
    streamed = memory_images([dict(item,memory_warm=True)],None,cfg,sample)
    assert len(reads) == 1 and not streamed['memory_seed_valid'].any()
    torch.testing.assert_close(streamed['memory_patches'][:,0], x['memory_patches'][:,-1])


def test_memory_observability_changes_labels_without_filtering_writer_inputs():
    cfg = memory_config()
    fiber = line_fiber()
    item = observation(cfg)
    item.update(fiber_ref=(0,200.,False), offtrack=True, source=2)
    item['hist_local'][:,0] = 10.
    builder = IdentityObservationBuilder(cfg, [fiber])
    prepared = builder.prepare(copy.deepcopy(item), fiber, np.random.default_rng(1))
    assert not prepared['identity_reference_valid']
    assert prepared['identity_observable']  # remote original seed is still available
    observations, _ = memory_layout(prepared, cfg)
    assert all(o['pos'][0] == 110 for o in observations[:-1])  # wrong history still encoded
    legacy = IdentityObservationBuilder(replace(cfg,memory_slots=0),[fiber])
    assert not legacy.prepare(copy.deepcopy(item), fiber, np.random.default_rng(1))['identity_observable']
    item['seed_valid'] = False
    assert not builder.prepare(item, fiber, np.random.default_rng(1))['identity_observable']


def test_memory_checkpoint_and_legacy_checkpoint_roundtrip(tmp_path):
    cfg = memory_config()
    model = DirectFollower(cfg).eval()
    data = memory_batch(cfg)
    path = tmp_path/'memory.pt'
    save_checkpoint(path,model,model,FiberVolumeSpec('unused'),SampleConfig(crop=cfg.fine,n_history=cfg.n_history))
    restored,_,_,_,ck = load_checkpoint(path,'cpu')
    assert ck['architecture'] == MEMORY_ARCHITECTURE
    for k,v in model(data['x'],data['hist'],data['hmask']).items():
        torch.testing.assert_close(v,restored(data['x'],data['hist'],data['hmask'])[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='disagree'):
        checkpoint_config(dict(ck,architecture=ARCHITECTURE))
    legacy = DirectFollower(replace(cfg,memory_slots=0))
    save_checkpoint(path,legacy,legacy,FiberVolumeSpec('unused'),SampleConfig(crop=cfg.fine,n_history=cfg.n_history))
    ck = torch.load(path,weights_only=False)
    ck['model_cfg'] = {k:v for k,v in ck['model_cfg'].items() if not k.startswith('memory_')}
    torch.save(ck,path)
    restored,*_ = load_checkpoint(path,'cpu')
    assert restored.cfg.memory_slots == 0 and restored.architecture == ARCHITECTURE
    assert not any(k.startswith('recurrent_memory.') for k in restored.state_dict())


def test_optimizer_accumulation_preserves_sequence_loss_and_update():
    torch.manual_seed(23)
    cfg = memory_config()
    a = DirectFollower(cfg)
    b = copy.deepcopy(a)
    data = memory_batch(cfg, 2)
    updates = []
    for model, batches in ((a,[data]), (b,[take(data,slice(0,1)),take(data,slice(1,2))])):
        updates.append(optimizer_update(model,copy.deepcopy(model),torch.optim.SGD(model.parameters(),lr=.001),
                                       batches,1,.001))
    assert updates[0]['loss'] == pytest.approx(updates[1]['loss'],rel=1e-5)
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-7)


def test_memory_training_forward_can_be_captured_without_graph_breaks():
    torch.manual_seed(71)
    model = open_read(DirectFollower(memory_config(memory_grad_steps=2)))  # includes a no-grad burn-in
    data = memory_batch(model.cfg, 1)
    compiled = torch.compile(model, backend='eager', fullgraph=True)
    out = compiled(data['x'], data['hist'], data['hmask'])
    expected = model(data['x'], data['hist'], data['hmask'])
    torch.testing.assert_close(out['points'], expected['points'])
    loss_terms(out, data, model.cfg)['geometry_per_state'].mean().backward()
    assert model.recurrent_memory.gate.weight.grad.abs().sum() > 0


def test_writer_learns_to_retain_an_old_disambiguating_observation():
    """Identical current inputs; only a patch 32 voxels ago identifies the target."""
    torch.manual_seed(19)
    cfg = memory_config(memory_steps=8)
    x = memory_batch(cfg)['x']
    x['memory_seed_valid'].zero_()
    x['memory_patches'].fill_(.3)
    x['memory_patches'][0,0,0,:,:,:2] = 1.
    x['memory_patches'][1,0,0,:,:,-2:] = 1.
    memory = LearnedMemory(cfg)
    head = torch.nn.Linear(cfg.hidden,2)
    query = torch.zeros(2,1,cfg.hidden)
    target = torch.tensor([0,1])
    parameters = list(memory.parameters())+list(head.parameters())
    optimizer = torch.optim.Adam(parameters,lr=.002)
    for _ in range(45):
        optimizer.zero_grad(set_to_none=True)
        logits = head(memory.read(query,memory.observe(x))[:,0])
        loss = torch.nn.functional.cross_entropy(logits,target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.)
        optimizer.step()
    assert loss.item() < .02 and torch.equal(logits.argmax(-1),target)
    x['memory_patches'][:,0].fill_(.3)
    forgotten = head(memory.read(query,memory.observe(x))[:,0])
    torch.testing.assert_close(forgotten[0],forgotten[1],rtol=0,atol=0)


def test_trace_state_isolated_on_retirement_and_reset_between_calls():
    class Model(torch.nn.Module):
        cfg = SimpleNamespace(memory_slots=1,n_future=1,max_recovery_distance=6.)
        def initial_memory(self,n,device):
            return dict(seen=torch.zeros(n,dtype=torch.bool), slots=torch.zeros(n,1,1))
        def forward(self,x,hist,mask,memory):
            # Seed x=10 stops immediately; x=20 persists to max length.
            expected = (x[:,2]-10).clamp_min(0)
            torch.testing.assert_close(memory['slots'][:,0,0],expected)
            confidence = (x[:,0]>15).float()[:,None]
            return dict(points=torch.tensor([0.,0.,1.]).expand(len(x),1,3),confidence=confidence,
                        memory_slots=memory['slots']+1,memory_seen=torch.ones(len(x),dtype=torch.bool))
    class Tracer(ModelTracer):
        def build_inputs(self,pos,frames,hist,hmask):
            return torch.as_tensor(pos,dtype=torch.float32)
    cfg = memory_config()
    tracer = Tracer(Model(),SimpleNamespace(shape=(100,100,100)),cfg.fine,cfg.n_history,
                    TraceParams(n_commit=1,max_len=3),device='cpu')
    try:
        for _ in range(2):
            paths,reasons = tracer.trace(np.array([[10.,10.,10.],[20.,10.,10.]]),np.array([[0.,0.,1.]]*2))
            assert reasons == ['confidence','max_len']
            assert len(paths[0]) == 1 and len(paths[1]) == 4
    finally:
        tracer.close()


def test_direct_tracer_reads_seed_once_and_streams_one_observation_per_decision(monkeypatch):
    cfg = memory_config(correction=False)
    model = DirectFollower(cfg)
    with torch.no_grad():
        model.coordinates.weight.zero_(); model.coordinates.bias.zero_()
        model.confidence_head[-1].weight.zero_(); model.confidence_head[-1].bias.fill_(10.)
    def images(items,vol,crop,pool=None):
        return torch.stack([torch.full((2,crop.depth,crop.width,crop.width),float(i['pos'][2])/1000.) for i in items])
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.data.image_crop',images)
    inputs = []
    original = model.recurrent_memory.observe
    def observe(x,state=None):
        inputs.append((x['memory_mask'].clone(),x['memory_seed_valid'].clone()))
        return original(x,state)
    monkeypatch.setattr(model.recurrent_memory,'observe',observe)
    tracer = DirectTracer(model,SimpleNamespace(shape=(1000,1000,1000)),cfg.fine,cfg.n_history,
                          TraceParams(n_commit=4,max_len=8),device='cpu')
    try:
        paths,reasons = tracer.trace(np.array([[100.,100.,400.]]),np.array([[0.,0.,1.]]))
    finally:
        tracer.close()
    assert reasons == ['max_len'] and paths[0][-1,2] == 408.
    assert len(inputs) == 2
    assert inputs[0][0].shape == (1,cfg.memory_steps+1) and inputs[0][1].all()
    assert inputs[1][0].shape == (1,1) and not inputs[1][1].any()


@pytest.mark.parametrize('kwargs', [dict(memory_slots=-1),dict(memory_grad_steps=0),dict(memory_stride=0),
                                    dict(memory_patch_size=4),dict(memory_version=4)])
def test_invalid_memory_config(kwargs):
    with pytest.raises(ValueError): memory_config(**kwargs)
