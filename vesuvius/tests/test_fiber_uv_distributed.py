from datetime import timedelta
from pathlib import Path
import pytest
import torch
import torch.distributed as dist

from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_distributed import SparseGradientReducer, shard, validate_config, validate_joint_gradient


def _gradient(rank, step, device='cpu'):
    # Row0 touched at step0 then absent. Row1 duplicate/cancelling across ranks.
    # Row4 touched with exact zero gradient at step1, must remain in union.
    rows=([0,1,1] if step==0 else [1,3,4]) if rank==0 else ([1,2,3] if step==0 else [1,2,4])
    values=([[1.,2.],[2.,1.],[-1.,-1.]] if step==0 else [[1.,-2.],[2.,1.],[0.,0.]]) if rank==0 else (
        [[-1.,0.],[3.,2.],[0.,0.]] if step==0 else [[-1.,2.],[1.,1.],[0.,0.]])
    return torch.tensor(rows,device=device),torch.tensor(values,dtype=torch.float64,device=device)


def _worker(rank,path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=Path(path).as_uri(),rank=rank,world_size=2,timeout=timedelta(seconds=60))
    try:
        initial=torch.arange(12,dtype=torch.float64).reshape(6,2)/10
        weight=torch.nn.Parameter(initial.clone());ref=torch.nn.Parameter(initial.clone())
        width=torch.nn.Parameter(torch.tensor([2.,3.],dtype=torch.float64))
        rw=torch.nn.Parameter(width.detach().clone())
        opt=torch.optim.SparseAdam([weight],lr=.03);ropt=torch.optim.SparseAdam([ref],lr=.03)
        wopt=torch.optim.Adam([width],lr=.1);rwopt=torch.optim.Adam([rw],lr=.1)
        reducer=SparseGradientReducer(weight)
        for step in range(2):
            rows,values=_gradient(rank,step)
            weight.grad=torch.sparse_coo_tensor(rows[None],values,weight.shape)
            width.grad=torch.tensor([rank+1.,step+1.],dtype=torch.float64)
            reducer.reduce(weight,width)
            a,av=_gradient(0,step);b,bv=_gradient(1,step)
            ref.grad=torch.sparse_coo_tensor(torch.cat((a,b))[None],torch.cat((av,bv))/2,ref.shape).coalesce()
            rw.grad=torch.tensor([1.5,step+1.],dtype=torch.float64)
            torch.testing.assert_close(weight.grad.indices(),ref.grad.indices(),rtol=0,atol=0)
            torch.testing.assert_close(weight.grad.values(),ref.grad.values(),rtol=0,atol=0)
            if step==1:
                assert 0 not in weight.grad.indices()[0]
                assert 4 in weight.grad.indices()[0]
            opt.step();ropt.step();wopt.step();rwopt.step()
            torch.testing.assert_close(weight,ref,rtol=0,atol=0)
            torch.testing.assert_close(width,rw,rtol=0,atol=0)
            for key in ['exp_avg','exp_avg_sq']:
                torch.testing.assert_close(opt.state[weight][key],ropt.state[ref][key],rtol=0,atol=0)
        validate_joint_gradient({'layout_mode':'shared_anchor'},rank,2,device='cpu')
    finally:dist.destroy_process_group()


def test_two_rank_sparse_adam_equivalence(tmp_path):
    torch.multiprocessing.spawn(_worker,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)


def test_global_shards():
    values=torch.arange(24)
    assert torch.equal(torch.cat([shard(values,r,6) for r in range(6)]),values)
    with pytest.raises(ValueError):shard(values[:23],0,6)


@pytest.mark.parametrize('bad',[
    {'batch_size':16,'distributed_world_size':6},
    {'batch_size':18,'distributed_world_size':6,'traversal':'full_pass'},
    {'batch_size':18,'distributed_world_size':6,'traversal':'sampled','update_mode':'global_accumulated'},
    {'batch_size':18,'distributed_world_size':6,'traversal':'sampled','active_fraction':.1},
])
def test_reject_unsupported(bad):
    with pytest.raises(ValueError):validate_config(bad)
