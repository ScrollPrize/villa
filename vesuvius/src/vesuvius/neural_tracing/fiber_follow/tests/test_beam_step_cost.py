"""Regression tests for frontier scoring, independently of native bindings."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.beam.native import Pool
from vesuvius.neural_tracing.fiber_follow.beam.states import BeamStateConfig, pool_state, supported_states
from vesuvius.neural_tracing.fiber_follow.beam.model import BeamRankNet, BeamNetConfig
from vesuvius.neural_tracing.fiber_follow.beam.supervision import beam_loss
from vesuvius.neural_tracing.fiber_follow.beam.hook import ModelBeamHook
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, arclength


def pool(paths):
    n = len(paths)
    return Pool(0, len(paths[0])-1, 'trace', paths[0][0], paths[0][-1], paths,
                np.arange(n, dtype=np.float32), np.zeros(n, np.float32), np.ones(n, np.float32),
                np.full(n, len(paths[0])-1), np.full(n, len(paths[0])-1.), np.zeros(n, bool),
                np.tile([1.,0.,0.], (n,1)))


def line(n=40):
    return np.c_[np.arange(n, dtype=float), np.zeros(n), np.zeros(n)]


def cfg():
    return BeamStateConfig(CropSpec(depth=24, width=17, behind=12), n_history=16, k_back=8, pool_size=2)


def test_live_frontier_and_candidate_specific_histories():
    a = line(); b = a.copy(); b[10:, 1] = np.linspace(0, 5, len(b)-10)
    p = pool([a, b])
    gt = SimpleNamespace(points=a, s=arclength(a), endpoint_stop=(False, False))
    state = pool_state(p, cfg(), gt)
    np.testing.assert_array_equal(state['pos'], a[-2])
    assert not np.array_equal(state['candidates'][0], state['candidates'][1])
    assert state['onfiber'].tolist() == [1., 0.]
    # The training budget does not truncate inference proposals.
    assert len(pool_state(pool([a, b, a]), cfg())['candidates']) == 3


def test_spread_out_branches_are_recropped_not_dropped():
    a = line(); b = a.copy(); b[:,1] = 100
    groups = list(supported_states(pool([a,b]), cfg()))
    assert len(groups) == 2
    assert sorted(np.concatenate([ids for ids, _ in groups]).tolist()) == [0,1]
    assert all(item['supported'].all() for _,item in groups)


@pytest.mark.parametrize('all_unknown', [False, True])
def test_censored_ranking_has_finite_loss_and_gradients(all_unknown):
    logits = torch.zeros(2,3,requires_grad=True)
    mask = torch.zeros(2,3)
    if not all_unknown: mask[0] = 1
    batch = dict(label_mask=mask,cand_mask=torch.ones_like(mask),onfiber=torch.ones_like(mask),quality=torch.zeros_like(mask))
    loss,_ = beam_loss(dict(onfiber_logits=logits),batch)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[1].abs().sum() == 0


def test_incremental_cost_excludes_hand_cost_and_charges_only_new_length():
    p=pool([line(),line()]);p.parent_losses=np.array([2.,3.],np.float32)
    p.losses=np.array([100.,999.],np.float32);p.step_lengths=np.array([1.,.25],np.float32)
    hook=ModelBeamHook(None,None,cfg(),device='cpu')
    hook.scores=lambda _:torch.zeros(2)
    losses,stop=hook(p)
    np.testing.assert_allclose(losses,p.parent_losses+np.log(2)*p.step_lengths)
    assert not stop


def test_model_chunking_and_forward_context_gradient():
    torch.manual_seed(1)
    model=BeamRankNet(BeamNetConfig(depth=24,width=16,behind=12,widths=(4,8),hidden=8,
                                   hist_points=4,hist_stride=4,score_chunk=2)).eval()
    x=torch.randn(1,2,24,16,16,requires_grad=True)
    h=torch.zeros(1,16,3);hm=torch.ones(1,16)
    candidates=torch.zeros(1,5,10,3);candidates[:,:,:,2]=torch.arange(-8.,2.)
    pm=torch.ones(1,5,10)
    out=model(x,h,hm,candidates,pm)
    model.cfg.score_chunk=5
    other=model(x,h,hm,candidates,pm)
    torch.testing.assert_close(out['step_cost'],other['step_cost'])
    assert (out['step_cost']>=0).all()
    out['step_cost'].sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad[:,:,20:].abs().sum()>0
    assert not any('tube' in n or 'flow' in n for n,_ in model.named_parameters())


def test_defaults_single_level_large_context():
    from vesuvius.neural_tracing.fiber_follow.beam.train import build_parser
    args=build_parser().parse_args(['--fiber-zarrs','x','--fibers','x','--ct','x','--prediction-manifest','x','--name','x'])
    assert (args.ct_level,args.ct_grid_scale,args.crop_spacing)==(1,8.,1.)
    assert (args.crop_depth,args.crop_width,args.crop_behind)==(192,96,128)
