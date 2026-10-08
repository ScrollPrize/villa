"""Data-parallel gradient pooling (train/distributed.py) across two CPU processes (gloo)."""
import os
import socket

import torch
import torch.multiprocessing as mp

from vesuvius.neural_tracing.tracers.fibers.train.distributed import DataParallel

# Per rank: decisions in its update, gradient of the shared parameter, gradient of a parameter only rank 1 used
# (rank 0 leaves it None, like an unexecuted refinement retry), and whether its loss was finite.
TOTALS, SHARED, ONLY_RANK1 = (2, 6), (torch.tensor([1., 2.]), torch.tensor([5., -2.])), torch.tensor([3., 3.])


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def worker(rank, port, finite, folder):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE='2', LOCAL_RANK=str(rank))
    parallel = DataParallel(backend='gloo')
    model = torch.nn.ParameterDict(dict(shared=torch.nn.Parameter(torch.zeros(2)), partial=torch.nn.Parameter(torch.zeros(2)),
                                        unused=torch.nn.Parameter(torch.zeros(2))))
    model['shared'].grad = SHARED[rank].clone()
    if rank == 1:
        model['partial'].grad = ONLY_RANK1.clone()
    total, all_finite, observed = parallel.average_gradients(model, TOTALS[rank], finite[rank], observed=10+rank)
    torch.save((total, all_finite, observed, model['shared'].grad, model['partial'].grad, model['unused'].grad),
               f'{folder}/{rank}.pt')
    parallel.close()


def pooled(finite, folder):
    mp.spawn(worker, args=(free_port(), finite, str(folder)), nprocs=2, join=True)
    return {rank: torch.load(folder/f'{rank}.pt') for rank in (0, 1)}


def test_gradients_pool_as_one_update_over_every_decision(tmp_path):
    out = pooled((True, True), tmp_path)
    expected_shared = (TOTALS[0]*SHARED[0]+TOTALS[1]*SHARED[1])/sum(TOTALS)
    expected_partial = TOTALS[1]*ONLY_RANK1/sum(TOTALS)  # zeros on rank 0
    for rank in (0, 1):
        total, all_finite, observed, shared, partial, unused = out[rank]
        assert total == sum(TOTALS) and all_finite and observed == 21
        assert torch.allclose(shared, expected_shared) and torch.allclose(partial, expected_partial)
        assert unused is None  # no process had a gradient: AdamW keeps skipping it


def test_a_nonfinite_loss_on_one_process_is_seen_by_all(tmp_path):
    out = pooled((True, False), tmp_path)
    assert not out[0][1] and not out[1][1]
