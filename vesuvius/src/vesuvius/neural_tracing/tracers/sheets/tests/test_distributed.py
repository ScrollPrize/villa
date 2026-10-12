"""Data-parallel sheet training (shared/distributed.py, train.optimize_step) across two CPU processes (gloo)."""
import os
import socket

import torch
import torch.multiprocessing as mp

from vesuvius.neural_tracing.tracers.sheets import config as C
from vesuvius.neural_tracing.tracers.sheets.data.patches import GridSpec
from vesuvius.neural_tracing.tracers.sheets.model import build_model
from vesuvius.neural_tracing.tracers.sheets.tests.test_geometry import SMALL, cylinder_grid
from vesuvius.neural_tracing.tracers.sheets.tracing.rollout import Tracer
from vesuvius.neural_tracing.tracers.sheets.train.train import optimize_step, train_rollout
from vesuvius.neural_tracing.tracers.shared.distributed import DataParallel
from vesuvius.neural_tracing.tracers.shared.training import clip_training_gradients

OVERRIDES = SMALL+['data.steps=2', 'train.grad_clip=0.05']  # a clip that binds: it must act on the pooled gradient


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def setup(rank):
    """This rank's config, grid spec, batch and rollout generator (each rank its own data)."""
    cfg = C.load(None, OVERRIDES)
    spec = GridSpec.for_episode(cfg.data, cfg.data.steps)
    gt, valid = cylinder_grid(spec)
    block = torch.rand(1, 1, 224, 360, 360, generator=torch.Generator().manual_seed(10+rank))*255
    return cfg, spec, dict(gt=gt, valid=valid, block=block, origin=torch.zeros(1, 3, dtype=torch.long)), \
        torch.Generator().manual_seed(20+rank)


def fresh_model(cfg, seed):
    torch.manual_seed(seed)
    model = build_model(cfg)
    torch.nn.init.normal_(model.head.weight, std=1e-2)
    return model


def worker(rank, port, folder):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE='2', LOCAL_RANK=str(rank))
    parallel = DataParallel(backend='gloo')
    cfg, spec, b, generator = setup(rank)
    model = fresh_model(cfg, rank)  # different initializations: the broadcast makes them rank 0's
    parallel.broadcast_module(model)
    opt = torch.optim.SGD(model.parameters(), lr=1.)  # linear in the gradient (AdamW's first step is ~lr*sign(g))
    optimize_step(Tracer(model, cfg, 'cpu'), opt, None, b, cfg, spec, generator, parallel=parallel)
    torch.save(model.state_dict(), f'{folder}/{rank}.pt')
    parallel.close()


def test_every_process_applies_the_update_of_the_averaged_batch_gradients(tmp_path):
    mp.spawn(worker, args=(free_port(), str(tmp_path)), nprocs=2, join=True)
    states = [torch.load(tmp_path/f'{rank}.pt') for rank in (0, 1)]
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])

    # One process: rank 0's initial weights, each rank's batch gradient on its own, their mean, then clip and step.
    cfg = C.load(None, OVERRIDES)
    grads = []
    for rank in (0, 1):
        _, spec, b, generator = setup(rank)
        model = fresh_model(cfg, 0)
        train_rollout(Tracer(model, cfg, 'cpu'), b, cfg, spec, generator)
        grads.append({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
    model = fresh_model(cfg, 0)
    for n, p in model.named_parameters():
        present = [g[n] for g in grads if n in g]
        if present:  # a parameter only one batch reached gets zeros from the other
            p.grad = sum(present)/2
    assert clip_training_gradients(model, cfg.train.grad_clip)['grad_clip_scale'] < 1
    torch.optim.SGD(model.parameters(), lr=1.).step()
    expected = model.state_dict()
    assert all(torch.allclose(states[0][k], expected[k], atol=1e-6) for k in expected)
