"""Bounded geometry planning preserves sampling feedback and realized batches."""
from types import SimpleNamespace

import numpy as np
import torch

from model_fixtures import config
from test_neighbor_bank import make_bank, add_shard, publish
from sampling_fixtures import clean_sample
from vesuvius.neural_tracing.fiber_follow.shared import data as module
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, IdentitySampling


class RecordingClient:
    def __init__(self):self.windows=[]
    def ensure_metadata(self,spec):pass
    def submit(self,reader,bounds):pass
    def ensure(self,reader,bounds):pass
    def lookahead(self,reader,batches,**kwargs):self.windows.append(batches)


def test_plan_count_is_bounded_and_first_batch_does_not_wait_for_horizon(monkeypatch):
    planned=[];built=[]
    monkeypatch.setattr(module,'FiberVolume',lambda *a,**kw:SimpleNamespace(ct=None))
    monkeypatch.setattr(module,'crop_local_grid',lambda crop:np.zeros((1,3)))
    def sample(f,t,rev,cfg,rng,**options):
        planned.append(t)
        return dict(at=t,reverse=rev)
    monkeypatch.setattr(module,'make_sample',sample)
    class Builder:
        def prefetch_bounds(self,item,vol):return [(item['at'],item['reverse'])]
        def __call__(self,items,vol):
            built.append(items)
            return items
    def stream(lookahead):
        ds=module.FollowDataset([SimpleNamespace(length=100.)],None,
            SimpleNamespace(crop=None,future_s=[16.]),None,chunk=2,seed=17,batch_builder=Builder(),
            budget=module.TaskBudget.parse(['fresh=1','live=0','dagger_pre_excursion=0','dagger_recoverable=0',
                'dagger_terminal=0','dagger_premature_stop=0','dagger_ordinary=0','synthetic_terminal=0']))
        ds.state_allowed=lambda item:True
        ds.remote_prefetch=RecordingClient()
        ds.remote_prefetch_lookahead=lookahead
        return ds,iter(ds)
    baseline,plain=stream(0)
    expected=[next(plain) for _ in range(10)]
    planned.clear();built.clear()
    ds,ahead=stream(3)
    assert next(ahead)==expected[0]
    assert len(planned)==2 and len(built)==1
    for i in range(1,10):
        assert next(ahead)==expected[i]
        assert len(planned)==2*(i+1+3) and len(built)==i+1
    assert max(map(len,ds.remote_prefetch.windows))==4
    ahead.close();plain.close()


def test_lookahead_preserves_real_bank_feedback_labels_and_sample_sequence(tmp_path,monkeypatch):
    bank,fiber=make_bank(tmp_path,refresh_seconds=1e6)
    publish(tmp_path,[add_shard(tmp_path,0,x=4.,z_range=(0.,200.))])
    bank._next_refresh=0.
    cfg=config(fine=CropSpec(depth=40,width=25,behind=16,spacing=1.))
    sampling=IdentitySampling(lateral_fraction=.8)
    monkeypatch.setattr(module,'FiberVolume',lambda *a,**kw:SimpleNamespace(ct=None,input_scale=1.))
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.oriented_seed_heading',
                        lambda vol, pos, family, direction: np.asarray(direction))
    class GeometryBuilder(IdentityObservationBuilder):
        def __call__(self,items,vol):
            targets=self.bank_targets(items)
            targets['positions']=torch.from_numpy(np.stack([i['pos'] for i in items]))
            targets['frames']=torch.from_numpy(np.stack([i['frame'] for i in items]))
            targets['photometric']=torch.tensor([i['photometric'] for i in items])
            targets['seeds']=torch.tensor([i['identity_seed'] for i in items])
            return targets
    def stream(lookahead):
        builder=GeometryBuilder(cfg,[fiber],sampling,augment=True,negative_bank=bank)
        ds=module.FollowDataset([fiber],None,clean_sample(cfg),None,chunk=2,seed=29,batch_builder=builder,
                                budget=module.TaskBudget.parse(['live=0','fresh=.65']))
        ds.remote_prefetch=RecordingClient()
        ds.remote_prefetch_lookahead=lookahead
        return builder,iter(ds)
    full,plain=stream(0)
    planned,ahead=stream(4)
    for _ in range(12):
        expected=next(plain);actual=next(ahead)
        assert actual.keys()==expected.keys()
        for key in expected:
            torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    assert len(full.lateral)>0
    # Planning advances the same feedback sequence, never appending it twice
    # when the old planned batch is finally assembled.
    assert list(planned.lateral)[:len(full.lateral)]==list(full.lateral)
    ahead.close();plain.close()
