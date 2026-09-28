"""Axial crop alignment, visibility, gradients and train/trace observation contracts."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_identity import config,batch,forward,line_fiber
from vesuvius.neural_tracing.fiber_follow.regression.model import (
    ARCHITECTURE,DirectConfig,DirectFollower,AxialBlock,ResidualConv,feature_grid,sample_features,TOKEN_OFFSET,TOKEN_STRIDE,
)
from vesuvius.neural_tracing.fiber_follow.regression.data import ObservationBuilder,IdentityObservationBuilder,reference_layout
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import save_checkpoint,load_checkpoint
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec


def test_production_grid_and_single_encoder():
    cfg=DirectConfig()
    assert (cfg.fine.depth,cfg.fine.width,cfg.fine.behind)==(120,101,48)
    assert cfg.token_shape==(15,51,51) and np.prod(cfg.token_shape)==39015
    m=DirectFollower(cfg)
    assert cfg.channels==32 and cfg.hidden==128 and cfg.layers==4
    assert m.encoder.stem[0].out_channels==32
    assert m.encoder.down[0].out_channels==64 and m.encoder.down[2].out_channels==128
    assert isinstance(m.encoder.down[-1],ResidualConv)
    assert m.encoder.down[-1].net[2].in_channels==128
    assert m.encoder.dense_projection.out_channels==32
    assert sum(p.numel() for p in m.parameters())==4464453
    assert m.encoder.compress.kernel_size==(4,1,1)
    assert m.encoder.compress.stride==(4,1,1)
    assert not hasattr(m,'appearance') and not hasattr(m,'coarse_encoder')
    assert [a.axis for a in m.encoder.blocks[0].axes]==[3,2,1]


def test_one_axial_cycle_connects_distant_positions_in_both_directions():
    torch.manual_seed(47)
    block=AxialBlock(8,2)
    x=torch.randn(1,5,6,7,8,requires_grad=True)
    out=block(x)
    g=torch.autograd.grad(out[0,0,0,0,0],x,retain_graph=True)[0]
    assert g[0,-1,-1,-1].abs().sum()>0
    g=torch.autograd.grad(out[0,-1,-1,-1,0],x)[0]
    assert g[0,0,0,0].abs().sum()>0


def test_compressed_lattice_and_dense_decoder_are_physically_aligned():
    cfg=config()
    m=DirectFollower(cfg)
    xyz=m.encoder.token_xyz.reshape(*cfg.token_shape,3)
    deep=xyz.permute(3,0,1,2)[None]
    point=xyz[1,3,4].reshape(1,1,3)
    sampled,supported=sample_features(deep,point,cfg.fine,TOKEN_STRIDE,TOKEN_OFFSET)
    assert supported.all()
    torch.testing.assert_close(sampled,point)
    dense=F.grid_sample(deep,m.encoder.decode_grid,align_corners=True,padding_mode='border')
    # Dense sample z=11 corresponds exactly to compressed token z=1, centre 8+3.
    expected=torch.tensor([(8-(cfg.fine.width-1)/2)*cfg.fine.spacing,
                           (6-(cfg.fine.width-1)/2)*cfg.fine.spacing,
                           (11-cfg.fine.behind)*cfg.fine.spacing])
    torch.testing.assert_close(dense[0,:,11,6,8],expected)


def test_learned_compression_preserves_forward_order():
    m=DirectFollower(config())
    with torch.no_grad():
        m.encoder.compress.weight.zero_();m.encoder.compress.bias.zero_()
        m.encoder.compress.weight[0,0,:,0,0]=torch.tensor([1.,2.,4.,8.])
    x=torch.zeros(1,4*m.cfg.channels,4,1,1)
    x[0,0,:,0,0]=torch.tensor([1.,0.,0.,0.])
    a=m.encoder.compress(x)
    b=m.encoder.compress(x.flip(2))
    assert a[0,0,0,0,0]==1 and b[0,0,0,0,0]==8


def test_outside_history_and_seed_cannot_influence_predictions():
    torch.manual_seed(12)
    m=DirectFollower(config()).eval()
    b=batch(m.cfg,1)
    b['x']['seed'][:]=torch.tensor([100.,100.,100.])
    before=forward(m,b)
    b['hist'][:,10:]=float('nan')
    b['x']['seed'][:]=float('nan')
    b['x']['seed_tangent'][:]=float('nan')
    b['x']['seed_age'][:]=float('nan')
    after=forward(m,b)
    for key in before:
        assert torch.isfinite(after[key]).all()
        torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)


def test_visible_reference_changes_context_without_oracle_inputs():
    torch.manual_seed(1)
    m=DirectFollower(config()).eval()
    b=batch(m.cfg,1)
    a=m.context(b['x'],b['hist'],b['hmask'])
    b['x']['seed'][:,:,0]=4.
    other=m.context(b['x'],b['hist'],b['hmask'])
    assert not torch.equal(a['fine'],other['fine'])
    # Changing annotation-only tensors cannot alter the model's input or output.
    before=forward(m,b)
    b['reference_on_fiber'].zero_();b['foreign'].fill_(1);b['dense_ab'].fill_(float('nan'))
    after=forward(m,b)
    for key in before:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)


def test_query_order_and_outside_support():
    m=DirectFollower(config()).eval();b=batch(m.cfg,1)
    b['identity_points'][0,-1]=torch.tensor([100.,0.,0.])
    first=forward(m,b)
    order=torch.randperm(b['identity_points'].shape[1])
    b['identity_points']=b['identity_points'][:,order]
    second=forward(m,b)
    torch.testing.assert_close(first['query_embedding'][:,order],second['query_embedding'])
    torch.testing.assert_close(first['query_support'][:,order],second['query_support'])
    assert not first['query_support'][0,-1]


def test_checkpointing_preserves_outputs_and_gradients():
    torch.manual_seed(9)
    a=DirectFollower(config());b=DirectFollower(replace(a.cfg,activation_checkpointing=True))
    b.load_state_dict(a.state_dict());data=batch(a.cfg,1)
    for m in (a,b):
        out=forward(m,data)
        loss=out['points'].square().mean()+out['query_embedding'][...,0].mean()
        loss.backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,atol=2e-6,rtol=2e-5)


def test_unobservable_identity_has_no_supervision_gradient():
    m=DirectFollower(config());data=batch(m.cfg,1)
    data['identity_observable']=torch.zeros(1,dtype=torch.bool)
    out=forward(m,data);terms=loss_terms(out,data,m.cfg)
    for name in ('geometry_per_state','confidence_per_state','identity_per_state'):
        assert terms[name].eq(0).all()
    sum(terms[k].sum() for k in ('geometry_per_state','confidence_per_state','identity_per_state')).backward()
    assert all(p.grad is None or p.grad.eq(0).all() for p in m.parameters())


def test_reference_visibility_masks_seed_geometry_and_labels():
    cfg=config();fiber=line_fiber()
    item=dict(pos=np.array([106.,100.,400.]),frame=np.eye(3),
        hist_local=np.c_[np.zeros(cfg.n_history),np.zeros(cfg.n_history),-np.arange(1,cfg.n_history+1)],
        hmask=np.ones(cfg.n_history),seed_pos=np.array([100.,100.,395.]),seed_tangent=np.array([0.,0.,1.]),
        seed_age=5.,seed_valid=True,fiber_ref=(0,200.,False),source=2,offtrack=True)
    builder=IdentityObservationBuilder(cfg,[fiber])
    a=builder.prepare(copy.deepcopy(item),fiber,np.random.default_rng(1))
    assert a['visible_seed_mask'].all() and a['identity_observable']
    item['seed_pos'][2]=200.
    b=builder.prepare(item,fiber,np.random.default_rng(1))
    assert not b['visible_seed_mask'].any() and not b['identity_observable']
    assert not b['visible_seed'].any() and not b['visible_seed_tangent'].any() and b['visible_seed_age']==0


def test_observation_builder_reads_only_one_ct_and_presence_crop(monkeypatch):
    cfg=config();calls=[]
    def sample(items,vol,crop,pool=None,*,presence=False):
        calls.append((crop,presence))
        return torch.zeros(len(items),1,crop.depth,crop.width,crop.width)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.data.scalar_crops',sample)
    item=dict(pos=np.zeros(3),frame=np.eye(3),hist_local=np.zeros((cfg.n_history,3)),hmask=np.zeros(cfg.n_history))
    x=ObservationBuilder(cfg).images([item],None)
    assert calls==[(cfg.fine,False),(cfg.fine,True)]
    assert set(x)=={'fine','seed','seed_mask','seed_age','seed_tangent'}


@pytest.mark.parametrize('old_architecture',['direct_identity_v1','axial_fiber_v1','axial_fiber_v2'])
def test_new_checkpoint_roundtrip_and_reject_old_architecture(tmp_path,old_architecture):
    cfg=config();m=DirectFollower(cfg).eval();data=batch(cfg,1)
    sample=SampleConfig(crop=cfg.fine,n_history=cfg.n_history,n_future=cfg.n_future)
    path=tmp_path/'model.pt'
    save_checkpoint(path,m,m,FiberVolumeSpec('/unused'),sample)
    loaded,*_=load_checkpoint(path,'cpu')
    for key,value in forward(m,data).items():torch.testing.assert_close(value,forward(loaded,data)[key])
    ck=torch.load(path,weights_only=False);ck['architecture']=old_architecture;torch.save(ck,path)
    with pytest.raises(ValueError):load_checkpoint(path,'cpu')


def test_history_cues_use_nearest_physical_token_centres():
    cfg=config();model=DirectFollower(cfg)
    refs=torch.zeros(1,cfg.n_history+1,3)
    mask=torch.zeros(1,cfg.n_history+1,dtype=torch.bool);mask[:,-1]=True
    # Input coordinates (x,y,z)=(1.5,1.5,7.5) are closer to token (1,1,1)
    # with centre (2,2,11) than token (0,0,0) with centre (0,0,3).
    refs[0,-1]=torch.tensor([-6.5,-6.5,-.5])
    rendered=model.encoder.conditioning(refs,mask)
    assert rendered[0,1,1,1,2]==1 and rendered[...,2].sum()==1
