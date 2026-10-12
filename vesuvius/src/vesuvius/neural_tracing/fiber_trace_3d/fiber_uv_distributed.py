"""Data-parallel joint U/V solve, preserving SparseAdam touched-row semantics."""
from datetime import timedelta
import json
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist


def validate_config(config):
    world = int(config.get('distributed_world_size', 1))
    if world < 1 or int(config['batch_size']) % world:
        raise ValueError('Global batch_size must be divisible by distributed_world_size')
    if world > 1:
        if config.get('assignment_update')=='cost_target' or config.get('benchmark_pair_fusion') or config.get('experiment_matrix'):
            raise ValueError('Distributed solve requires the joint solver, not an alternate solver/matrix')
        if config.get('traversal', 'full_pass') != 'sampled' or config.get('update_mode', 'minibatch') != 'minibatch':
            raise ValueError('Distributed solve currently requires sampled minibatch updates')
        if config.get('active_fraction', 0) or config.get('stop_when_feasible', False):
            raise ValueError('Distributed active sampling/early stopping is not supported')
    return world


class SparseGradientReducer:
    """Mean-reduce gradients; absent rows remain absent from SparseAdam updates.

    Dense staging makes communication regular; the union mask is determined by
    sparse indices, NOT nonzero values. Zero/cancelled gradients still count as
    touched, just as in a concatenated single-device minibatch.
    """
    def __init__(self, weight):
        self.buffer = torch.empty_like(weight)
        self.mask = torch.empty(weight.shape[0], dtype=torch.int32, device=weight.device)

    def reduce(self, weight, width):
        local = weight.grad.coalesce()
        self.buffer.zero_()
        self.mask.zero_()
        rows = local.indices()[0]
        self.buffer.index_copy_(0, rows, local.values())
        self.mask.index_fill_(0, rows, 1)
        pending = [dist.all_reduce(self.buffer, async_op=True),
                   dist.all_reduce(self.mask, op=dist.ReduceOp.MAX, async_op=True),
                   dist.all_reduce(width.grad, async_op=True)]
        for work in pending: work.wait()
        world = dist.get_world_size()
        self.buffer.div_(world)
        width.grad.div_(world)
        rows = self.mask.nonzero(as_tuple=True)[0]
        weight.grad = torch.sparse_coo_tensor(rows[None], self.buffer.index_select(0, rows),
            size=weight.shape, is_coalesced=True, check_invariants=False)


def shard(values, rank, world):
    if len(values) % world:
        raise ValueError('Equal-sized distributed minibatch shards required')
    size = len(values)//world
    return values[rank*size:(rank+1)*size]


def _worker(rank, world, config, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank,
                            world_size=world, timeout=timedelta(minutes=30))
    try:
        validate_joint_gradient(config,rank,world)
        from .fiber_uv_layout import solve
        solve(dict(config, _distributed_rank=rank))
    finally:
        dist.destroy_process_group()


def validate_joint_gradient(config,rank,world,device=None):
    """CUDA/NCCL preflight against concatenated actual pair-loss gradients."""
    from . import fiber_uv_layout as uv
    device=f'cuda:{rank}' if device is None else device
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(19)
    k,n,b=7,19,world*8
    initial=torch.randn(n,3*k+3,device=device)*.01
    w0=torch.arange(k,device=device,dtype=torch.float32)*10+200
    weight=torch.nn.Embedding.from_pretrained(initial.clone(),freeze=False,sparse=True)
    reference=torch.nn.Embedding.from_pretrained(initial.clone(),freeze=False,sparse=True)
    raw=torch.nn.Parameter(uv.inverse_widths(w0))
    rraw=torch.nn.Parameter(raw.detach().clone())
    reducer=SparseGradientReducer(weight.weight)
    i=torch.arange(b,device=device)%n;j=(i+3)%n
    families=torch.arange(n,device=device)%2
    phase=torch.linspace(.1,.9,n,device=device)
    zeros=torch.zeros(n,device=device)
    table=torch.as_tensor(uv.make_distance_table(2.,config.get('distance_sigma',.5)),device=device)
    cfg=dict(config,distance_objective='squared',coordinate_backend='combined')
    event_weights=torch.as_tensor(uv.constraint_class_weights(
        [index%3!=0 for index in range(b)],cfg),device=device)
    evaluate=uv.pair_evaluator(cfg) if cfg.get('constraint_class_balance') else uv.pair_objective
    def loss(emb,r,sel,compiled=False):
        widths,bounds=uv.width_boundaries(r)
        ia,ib=i[sel],j[sel]
        sa,sb=emb(ia),emb(ib)
        ua,va,qa=uv.coordinates(sa,phase[ia],zeros[ia],families[ia],zeros[ia]+13,widths,bounds,'combined',config.get('coordinate_division','standard'))
        ub,vb,qb=uv.coordinates(sb,phase[ib],zeros[ib],families[ib],zeros[ib]-17,widths,bounds,'combined',config.get('coordinate_division','standard'))
        pa,pb=sa[:,:k+1].softmax(-1)[:,:k],sb[:,:k+1].softmax(-1)[:,:k]
        charge=selected=None
        if cfg.get('sign_charge_multiplier',0)>0:
            ones=torch.ones_like(phase[ia])
            charge=uv.calibrated_sign_charge(ones,ones,1.,b,n,
                cfg.get('rejection_cost',10.),cfg['sign_charge_multiplier'],
                cfg.get('rejection_weighting','fixed'))
            selected=uv.selected_ordering(sa,sb,phase[ia],phase[ib],zeros[ia],zeros[ib],
                families[ia],families[ib],zeros[ia]+13,zeros[ib]-17,widths,
                phase[ib]-phase[ia],ones)
        evaluator=evaluate if compiled else uv.pair_objective
        coarse_shift=((ia%3)-1).float() if cfg.get('bucket_position_split',False) else None
        if cfg.get('bucket_shared_phase',False):coarse_shift=coarse_shift+sb[:,k+1]-sa[:,k+1]
        return evaluator(pa,pb,qa,qb,va,vb,phase[ib]-phase[ia],families[ib]-families[ia],
            torch.ones_like(phase[ia]),torch.ones_like(phase[ia]),table,cfg,charge,
            event_weights[sel],selected,
            coarse_shift)+.01*uv.continuation_likelihood(pa,pb,ua,ub,va,vb,128.)
    loss(weight,raw,shard(torch.arange(b,device=device),rank,world),True).backward()
    reducer.reduce(weight.weight,raw)
    loss(reference,rraw,torch.arange(b,device=device)).backward()
    a,c=weight.weight.grad.coalesce(),reference.weight.grad.coalesce()
    torch.testing.assert_close(a.indices(),c.indices(),rtol=0,atol=0)
    torch.testing.assert_close(a.values(),c.values(),rtol=2e-4,atol=2e-5)
    torch.testing.assert_close(raw.grad,rraw.grad,rtol=2e-4,atol=2e-5)
    if rank==0:print('UV distributed gradient validation passed: '+str(world)+' ranks',flush=True)


def launch(config):
    """Spawn only inside the existing managed lease; never reserve independently."""
    world = validate_config(config)
    if torch.cuda.device_count() < world:
        raise ValueError('Manager lease exposes fewer GPUs than requested world size')
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    if (root/'summary.json').exists(): raise FileExistsError('Completed distributed solve exists')
    with tempfile.TemporaryDirectory(prefix='.distributed-', dir=root) as scratch:
        rendezvous = Path(scratch, 'rendezvous').resolve().as_uri()
        torch.multiprocessing.spawn(_worker, args=(world, config, rendezvous), nprocs=world, join=True)
    return json.loads((root/'summary.json').read_text())


def launch_suite(config):
    """Sequential 1/N-GPU trials inside one existing exclusive manager lease."""
    from lasagna.inference_provenance import atomic_write
    from .fiber_uv_layout import solve
    import gc
    if not config['gpu_manager'].get('exclusive'):
        raise ValueError('Distributed benchmark suite requires an exclusive lease')
    jobs=[]
    for path in config['distributed_suite']:
        child=json.loads(Path(path).read_text())['fiber_uv_layout']
        if child.get('distributed_suite') or child.get('experiment_matrix'):
            raise ValueError('Nested benchmark suites are not supported')
        child=dict(child,code_provenance=config.get('code_provenance'))
        root=Path(child['output_root']);root.mkdir(parents=True,exist_ok=True)
        if (root/'summary.json').exists():raise FileExistsError('Completed suite child exists')
        atomic_write(root/'config.json',child,durable=False)
        solve(child)
        jobs.append(dict(output_root=str(root),world_size=child.get('distributed_world_size',1)))
        print('DISTRIBUTED SUITE '+json.dumps(jobs[-1]),flush=True)
        gc.collect();torch.cuda.empty_cache()
    result=dict(config=config,jobs=jobs)
    atomic_write(Path(config['output_root'])/'summary.json',result,durable=False)
    return result
