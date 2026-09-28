"""Causal full-crop memory input and streaming trace contracts."""
import copy
from types import SimpleNamespace
import numpy as np
import torch
from test_identity import config, line_fiber
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectFollower
from vesuvius.neural_tracing.fiber_follow.regression.data import DirectTracer, IdentityObservationBuilder
from vesuvius.neural_tracing.fiber_follow.regression.memory_data import memory_layout, memory_images, memory_allowed
from vesuvius.neural_tracing.fiber_follow.shared.data import ZBand
from vesuvius.neural_tracing.fiber_follow.shared.trace import ModelTracer, TraceParams


def memory_config(**kwargs):
    return config(**dict(dict(memory_steps=4), **kwargs))

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


def test_observation_sampler_training_and_streaming_crops_match():
    cfg = memory_config()
    item = observation(cfg)
    reads = []
    def sample(items, vol, crop, pool=None):
        reads.extend(items)
        return torch.stack([torch.full((2,crop.depth,crop.width,crop.width),float(i['pos'][2])) for i in items])
    x = memory_images([item], None, cfg, sample)
    assert len(reads) == 5
    torch.testing.assert_close(x['history_crops'][0,:,0,0,0,0],torch.tensor([384.,388.,392.,396.]))
    reads.clear()
    streamed = memory_images([dict(item,memory_warm=True)],None,cfg,sample)
    assert not reads and not streamed['memory_seed_valid'].any()
    assert streamed['history_crops'].shape[1] == 0


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
    item['seed_valid'] = False
    assert not builder.prepare(item, fiber, np.random.default_rng(1))['identity_observable']


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
    original = model.observe
    def observe(x,*args,**kwargs):
        inputs.append((x['memory_mask'].clone(),x['memory_seed_valid'].clone()))
        return original(x,*args,**kwargs)
    monkeypatch.setattr(model,'observe',observe)
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
