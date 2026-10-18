"""Data-parallel training across processes started by torchrun, shared by the tracers (fibers: runtime option ``gpus``
and scripts/launch_train.sh; sheets: ``train.gpus`` and ``torchrun --nproc_per_node=GPUS -m ...sheets.train.train``).

The tracers train through compiled functions or step-wise backward passes rather than one ``model.forward``, so the
model is not wrapped in ``DistributedDataParallel`` (whose gradient hooks run only through its own forward). Instead
every process computes the gradients of its own data, ``average_gradients`` combines them into exactly the gradient of
the pooled update, and every process then applies the same clipping, AdamW step and EMA update, so model, EMA and
optimizer stay identical. Anything else an update depends on (e.g. a learning rate following a time budget) must be
the same on every process: ``broadcast_value`` hands out rank 0's.

``average_gradients`` weights each process's gradient (of a loss averaged over its own ``total`` units: supervised
decisions for fibers, one batch for sheets) by ``total`` and divides the sum by the total over all processes. A
parameter without a gradient on every process keeps ``grad=None`` (AdamW skips it, as it does for unexecuted
refinement retries); one with a gradient on any process gets zeros where it had none. A nonfinite loss on any process
is reported on all of them before any update, so no process waits on the others.

Only rank 0 writes run files, logs, checkpoints and diagnostics (and, for fibers, collects DAgger states); each rank
draws its own training data (seed offset per rank, ``seed_offset``).
"""
from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist

RANK_SEED_STRIDE = 1_000_003  # data and RNG seed offset between ranks


class DataParallel:
    """This process's place in a torchrun process group."""

    def __init__(self, backend='nccl', timeout_minutes=60):
        self.rank, self.world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
        self.local_rank = int(os.environ.get('LOCAL_RANK', self.rank))
        if backend == 'nccl':
            torch.cuda.set_device(self.local_rank)
            self.device = f'cuda:{self.local_rank}'
        elif torch.cuda.is_available():  # gloo over CUDA tensors: ranks may share GPUs (testing on one GPU)
            self.device = f'cuda:{self.local_rank % torch.cuda.device_count()}'
            torch.cuda.set_device(self.device)
        else:
            self.device = 'cpu'
        dist.init_process_group(backend, timeout=datetime.timedelta(minutes=timeout_minutes))

    @property
    def is_main(self):
        return self.rank == 0

    def seed_offset(self):
        return self.rank*RANK_SEED_STRIDE

    def barrier(self):
        dist.barrier()

    def broadcast_object(self, value):
        """Rank 0's ``value`` on every rank."""
        holder = [value]
        dist.broadcast_object_list(holder, src=0)
        return holder[0]

    def broadcast_module(self, module):
        """Copy rank 0's parameters and buffers into ``module`` on every rank, so all processes start identical
        whatever their initialization."""
        with torch.no_grad():
            for tensor in module.state_dict().values():
                dist.broadcast(tensor, src=0)

    def broadcast_value(self, value):
        """Rank 0's float ``value`` on every rank (one small collective; cheap enough for every update)."""
        holder = torch.tensor([float(value)], device=self.device, dtype=torch.float64)
        dist.broadcast(holder, src=0)
        return float(holder)

    def average_gradients(self, model, total, finite, observed=0):
        """Pool the gradients of all processes (module docstring); returns (pooled total, all finite, pooled observed)."""
        params = [p for p in model.parameters() if p.requires_grad]
        device = params[0].device
        meta = torch.tensor([float(p.grad is not None) for p in params]+[float(total), float(bool(finite)), float(observed)],
                            device=device, dtype=torch.float64)
        dist.all_reduce(meta)
        present, pooled_total = meta[:len(params)] > 0, float(meta[-3])
        all_finite, pooled_observed = int(meta[-2]) == self.world_size, int(meta[-1])
        if not all_finite or pooled_total <= 0:
            return pooled_total, all_finite, pooled_observed
        chosen = [p for p, has in zip(params, present.tolist()) if has]
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().reshape(-1)*float(total)
                          for p in chosen])
        dist.all_reduce(flat)
        flat /= pooled_total
        offset = 0
        for p in chosen:
            value = flat[offset:offset+p.numel()].view(p.shape)
            offset += p.numel()
            if p.grad is None:
                p.grad = torch.empty_like(p)
            p.grad.copy_(value)
        return pooled_total, all_finite, pooled_observed

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()


def from_environment(expected=1):
    """The process group when started by torchrun with WORLD_SIZE > 1, else None; checks it against ``expected`` GPUs."""
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != expected:
        raise RuntimeError(f'The configuration asks for {expected} GPU process(es) but {world} were started; launch '
                           'with torchrun --nproc_per_node=GPUS (fibers: scripts/launch_train.sh does so when gpus > 1)')
    # TRACERS_DDP_BACKEND=gloo (testing only; formerly FIBER_FOLLOW_DDP_BACKEND) lets several processes share one GPU,
    # which NCCL refuses.
    backend = os.environ.get('TRACERS_DDP_BACKEND', os.environ.get('FIBER_FOLLOW_DDP_BACKEND', 'nccl'))
    return DataParallel(backend) if world > 1 else None


class NullLog:
    """The run log on ranks other than 0: records are dropped."""

    def record(self, values):
        pass

    def close(self):
        pass
