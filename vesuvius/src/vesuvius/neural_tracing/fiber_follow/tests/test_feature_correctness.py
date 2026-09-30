import copy
import numpy as np
import torch
from test_neighbor_bank import make_bank,add_shard,publish
from test_neighbor_following import clean_sample
from test_trajectory_memory import cfg
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder
from vesuvius.neural_tracing.fiber_follow.regression.feature_sequences import stream_rows
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import wrong_continuation,continuation_track
from vesuvius.neural_tracing.fiber_follow.shared.data import make_sample
from vesuvius.neural_tracing.fiber_follow.shared.geometry import frame_from_heading,CropSpec
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.diagnostics import decision_rows

def setup(tmp):
    bank,fiber=make_bank(tmp)
    publish(tmp,[add_shard(tmp,0,x=4.,z_range=(20.,180.))])
    c=cfg(n_history=64,memory_steps=64,fine=CropSpec(depth=40,width=25,behind=16))
    return c,IdentityObservationBuilder(c,[fiber],negative_bank=bank),fiber,bank

def test_generated_original_history_retains_supervision_and_causal_heading(tmp_path):
    c,b,f,_=setup(tmp_path);rng=np.random.default_rng(1)
    item=make_sample(f,150.,False,clean_sample(c),rng)
    item.update(fiber_ref=(0,150.,False),source=0,source_step=-1,stratum=-1)
    b.prepare(item,f,rng)
    other=copy.deepcopy(item)
    world=item['hist_local']@item['frame'].T+item['pos']
    other['frame']=frame_from_heading(np.array([.5,0.,np.sqrt(.75)]))
    other['hist_local']=(world-other['pos'])@other['frame']
    a,z=stream_rows(item,b),stream_rows(other,b)
    assert len(a)==len(z)>2
    assert all(r['identity_observable'] and not r['offtrack'] for r in a)
    np.testing.assert_allclose(a[0]['frame'][:,2],item['seed_tangent'])
    for x,y in zip(a[:-1],z[:-1]):
        for key in ('pos','frame','hist_local','hmask'):
            np.testing.assert_allclose(x[key],y[key],atol=1e-10)
    item['seed_valid']=other['seed_valid']=False
    for x,y in zip(stream_rows(item,b)[:-1],stream_rows(other,b)[:-1]):
        np.testing.assert_allclose(x['frame'],y['frame'],atol=1e-10)

def test_synthetic_neighbor_labels_certified_without_fabricated_recovery(tmp_path):
    c,b,f,bank=setup(tmp_path);rng=np.random.default_rng(3)
    item=wrong_continuation(bank,clean_sample(c),rng,tail_length_range=(32.,32.))
    b.prepare(item,f,rng);rows=stream_rows(item,b)
    neighbor=[r for r in rows[:-1] if abs(r['pos'][0]-4)<1e-4]
    assert neighbor and all(r['offtrack'] and r['identity_observable'] for r in neighbor)
    assert any(r['identity_observable'] and not r['offtrack'] for r in rows[:-1])
    assert any(not r['identity_observable'] for r in rows[:-1])

def test_unlabeled_replay_history_masks_losses_but_updates_memory(tmp_path,monkeypatch):
    c,b,f,bank=setup(tmp_path);rng=np.random.default_rng(3)
    item=wrong_continuation(bank,clean_sample(c),rng,tail_length_range=(32.,32.))
    # Reproduce a legacy replay cache without generator membership metadata.
    for key in list(item):
        if key.startswith('_'):item.pop(key)
    b.prepare(item,f,rng);rows=stream_rows(item,b)
    assert all(not r['identity_observable'] for r in rows[:-1])
    assert rows[-1]['offtrack'] and rows[-1]['identity_observable']
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.data.image_crop',
        lambda items,vol,crop,pool=None,**kw:torch.ones(len(items),2,crop.depth,crop.width,crop.width)*.25)
    batch=b([rows[0]],None);model=build_model(c)
    output=model(batch['x'],batch['hist'],batch['hmask'])
    terms=loss_terms(output,batch,c)
    assert all(terms[k].eq(0).all() for k in ('geometry_per_state','confidence_per_state'))
    assert output['memory_cache_valid'].any()
    diagnostic=decision_rows(output,batch,c)[0]
    assert not diagnostic['departed']
    assert diagnostic['known_prefixes']==diagnostic['annotated_points']==0
    assert diagnostic['gate_0.5']['departed_continues']==0

def test_synthetic_track_heading_is_backward_looking():
    path=np.array([[0.,0.,0.],[0.,0.,4.],[4.,0.,4.],[8.,0.,4.]])
    tr=continuation_track(path,np.array([0.,4.,8.,12.]),1,np.array([0.,4.,8.]),0.,4.,4.)
    j=np.flatnonzero(np.linalg.norm(tr['pos']-[0,0,4],axis=1)<1e-6)[0]
    np.testing.assert_allclose(tr['frame'][j,:,2],[0,0,1])
