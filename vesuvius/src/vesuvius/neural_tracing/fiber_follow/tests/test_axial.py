"""Axial crop alignment, visibility, gradients and train/trace observation contracts."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_identity import config,batch,forward,line_fiber
from label_fixtures import set_unknown
from vesuvius.neural_tracing.fiber_follow.regression.model import (
    ARCHITECTURE,DirectConfig,DirectFollower,AxialBlock,ResidualConv,feature_grid,sample_features,TOKEN_OFFSET,TOKEN_STRIDE,
)
from vesuvius.neural_tracing.fiber_follow.regression.data import ObservationBuilder,IdentityObservationBuilder,reference_layout
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import save_checkpoint,load_checkpoint
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec




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




@pytest.mark.parametrize('encoder', ['conv', 'patch4'])
def test_checkpointing_preserves_outputs_and_gradients(encoder):
    torch.manual_seed(9)
    a=DirectFollower(config(encoder=encoder));b=DirectFollower(replace(a.cfg,activation_checkpointing=True))
    b.load_state_dict(a.state_dict());data=batch(a.cfg,1)
    for m in (a,b):
        out=forward(m,data)
        loss=out['points'].square().mean()+out['hazard_logits'].square().mean()
        loss.backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,atol=2e-6,rtol=2e-5)


def test_unobservable_identity_has_no_supervision_gradient():
    m=DirectFollower(config());data=batch(m.cfg,1)
    set_unknown(data,0)
    out=forward(m,data);terms=loss_terms(out,data,m.cfg)
    for name in ('geometry_per_state','confidence_per_state'):
        assert terms[name].eq(0).all()
    sum(terms[k].sum() for k in ('geometry_per_state','confidence_per_state')).backward()
    assert all(p.grad is None or p.grad.eq(0).all() for p in m.parameters())








def test_history_cues_use_nearest_physical_token_centres():
    cfg=config();model=DirectFollower(cfg)
    refs=torch.zeros(1,cfg.n_history+1,3)
    mask=torch.zeros(1,cfg.n_history+1,dtype=torch.bool);mask[:,-1]=True
    # Input coordinates (x,y,z)=(1.5,1.5,7.5) are closer to token (1,1,1)
    # with centre (2,2,11) than token (0,0,0) with centre (0,0,3).
    refs[0,-1]=torch.tensor([-6.5,-6.5,-.5])
    rendered=model.encoder.conditioning(refs,mask)
    assert rendered[0,1,1,1,2]==1 and rendered[...,2].sum()==1
