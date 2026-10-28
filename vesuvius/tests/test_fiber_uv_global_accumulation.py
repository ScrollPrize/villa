import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import (
    FullPassGradientAccumulator,pass_partition,validate_joint_rejection)


def test_accumulation_matches_monolithic_sparse_adam_two_updates():
    torch.manual_seed(4)
    initial=torch.randn(7,4,dtype=torch.float64)
    models=[torch.nn.Embedding.from_pretrained(initial.clone(),freeze=False,sparse=True) for _ in range(2)]
    widths=[torch.nn.Parameter(torch.tensor([1.3],dtype=torch.float64)) for _ in range(2)]
    opts=[torch.optim.SparseAdam(m.parameters(),lr=.03) for m in models]
    wopts=[torch.optim.Adam([w],lr=.2) for w in widths]
    accum=FullPassGradientAccumulator(models[1].weight,widths[1])
    ids=torch.tensor([1,1,2,3,2,0,5,1,4,5,0])
    nodes=torch.arange(7);parts=3
    # Last node participates explicitly but has zero gradient, matching full-node use.
    node_weight=torch.tensor([1,1,1,1,1,1,0],dtype=torch.float64)
    def terms(m,w,e,n):
        pair=((m(ids[e])*w-.4)**2).mean()
        reg=(m(nodes[n]).square().sum(-1)*node_weight[n]).mean()
        shared=w.square().mean()+.3*(m(torch.tensor([0,5])).sum()*w).square().mean()
        return pair,reg,shared
    for iteration in range(2):
        opts[0].zero_grad(set_to_none=True);wopts[0].zero_grad(set_to_none=True)
        whole=sum(terms(models[0],widths[0],slice(None),slice(None)))
        whole.backward()
        expected_grad=models[0].weight.grad.to_dense().clone()
        expected_width=widths[0].grad.clone()
        accum.reset();total=0.
        before=models[1].weight.detach().clone()
        for p in range(parts):
            e=pass_partition(len(ids),parts,p);n=pass_partition(len(nodes),parts,p)
            opts[1].zero_grad(set_to_none=True);wopts[1].zero_grad(set_to_none=True)
            pair,reg,shared=terms(models[1],widths[1],e,n)
            loss=(pair*((e.stop-e.start)*parts/len(ids))+
                  reg*((n.stop-n.start)*parts/len(nodes))+shared)/parts
            total+=float(loss.detach());loss.backward()
            accum.add(models[1].weight.grad,widths[1].grad)
            torch.testing.assert_close(models[1].weight,before)
        assert total==pytest.approx(float(whole.detach()),rel=1e-12)
        torch.testing.assert_close(accum.weight,expected_grad,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(accum.width,expected_width,rtol=1e-12,atol=1e-12)
        accum.install(models[1].weight,widths[1])
        assert models[1].weight.grad.is_coalesced()
        for opt,wopt in zip(opts,wopts):opt.step();wopt.step()
        torch.testing.assert_close(models[0].weight,models[1].weight,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(widths[0],widths[1],rtol=1e-12,atol=1e-12)


def test_global_mode_validation():
    valid=dict(update_mode='global_accumulated',traversal='full_pass',entropy_ramp=False)
    validate_joint_rejection(valid)
    with pytest.raises(ValueError,match='full_pass'):
        validate_joint_rejection(dict(valid,traversal='sampled'))
    with pytest.raises(ValueError,match='entropy_ramp'):
        validate_joint_rejection(dict(valid,entropy_ramp=True))
    with pytest.raises(ValueError,match='update_mode'):
        validate_joint_rejection(dict(update_mode='typo'))
