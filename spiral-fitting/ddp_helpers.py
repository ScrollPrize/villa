"""Explicit rank/world-size plumbing and rendezvous ownership for DDP runs.

Rank, world size, and local rank are values carried by a
:class:`DistributedContext`, not facts a helper rediscovers from the
environment. ``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` are read in exactly one
place, :meth:`DistributedContext.from_env`, which exists for the torchrun
rendezvous boundary of the headless CLI. Every other caller receives a
context and passes it on.

Deep call sites that cannot reasonably thread a context (loss kernels) read
the *process* context, which is installed explicitly by
:func:`maybe_init_distributed` — still a value someone passed in, never an
environment read at the point of use.
"""

import dataclasses
import datetime
import os

import torch
import torch.distributed as dist


# Bounded process-group timeout for collectives. The first collective
# (broadcast_model_params) runs after slow, per-rank startup work, so the
# window has to cover startup skew; but an interactive session must never sit
# inside a wedged collective for the length of it, which is why the parent
# watchdog in spiral_runtime is the primary fail-stop and this is only the
# backstop.
DEFAULT_COLLECTIVE_TIMEOUT_MIN = 60


def configure_torch_threads_from_env(env_var='FIT_SPIRAL_NUM_THREADS'):
    # Useful for multi-process GPU runs where each rank would otherwise use a full
    # host thread pool.
    num_threads = os.environ.get(env_var)
    if not num_threads:
        return
    try:
        torch.set_num_threads(int(num_threads))
        torch.set_num_interop_threads(int(num_threads))
    except (ValueError, RuntimeError):
        pass


@dataclasses.dataclass(frozen=True)
class DistributedContext:
    """Who this process is in the job, as an explicit value.

    Constructed by whoever knows the answer — the service spawning workers,
    or the CLI reading the torchrun environment — and passed down. It is
    hashable, picklable, and comparable, so a worker can be handed one
    through a spawn argument list.
    """

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    def __post_init__(self):
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank {self.rank} is outside world size {self.world_size}")
        if self.local_rank < 0:
            raise ValueError(f"local_rank must be >= 0, got {self.local_rank}")

    @property
    def is_distributed(self):
        return self.world_size > 1

    @property
    def is_main_process(self):
        return self.rank == 0

    @classmethod
    def from_env(cls, environ=None):
        """Read the torchrun rendezvous environment.

        The only environment read in the fitter. Call it at a rendezvous
        boundary (the headless CLI under torchrun) and pass the result on.
        """
        environ = os.environ if environ is None else environ
        return cls(
            rank=int(environ.get('RANK', '0')),
            world_size=int(environ.get('WORLD_SIZE', '1')),
            local_rank=int(environ.get('LOCAL_RANK', '0')),
        )


SINGLE_PROCESS = DistributedContext()

# The context this process was told it has. Installed explicitly by
# maybe_init_distributed(); never derived from the environment here.
_process_context = SINGLE_PROCESS


def set_process_context(context):
    """Install the process-wide context; returns the previous one."""
    global _process_context
    previous = _process_context
    _process_context = context
    return previous


def process_context():
    return _process_context


def get_world_size(context=None):
    return (context or _process_context).world_size


def get_rank(context=None):
    return (context or _process_context).rank


def get_local_rank(context=None):
    return (context or _process_context).local_rank


def is_distributed(context=None):
    return (context or _process_context).is_distributed


def is_main_process(context=None):
    return (context or _process_context).is_main_process


def maybe_init_distributed(context=None, *, rendezvous=None,
                           timeout_minutes=None, backend=None):
    """Join the process group described by ``context``.

    ``context`` defaults to the torchrun environment, which is the CLI's
    rendezvous boundary. ``rendezvous`` is any object exposing a
    ``store_path`` whose owner keeps it alive for the whole job (see
    :class:`spiral_runtime.FileStoreRendezvous`); without one the c10d
    ``env://`` rendezvous is used, which is how torchrun jobs meet. Returns
    the installed context.
    """
    if context is None:
        context = DistributedContext.from_env()
    set_process_context(context)
    if not context.is_distributed:
        return context
    if backend is None:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    if backend == 'nccl':
        torch.cuda.set_device(context.local_rank)
    if timeout_minutes is None:
        timeout_minutes = int(os.environ.get(
            'FIT_SPIRAL_DDP_TIMEOUT_MIN', str(DEFAULT_COLLECTIVE_TIMEOUT_MIN)))
    kwargs = {}
    if rendezvous is not None:
        # A file store has no discover-then-release window: the endpoint is a
        # path whose directory the caller owns for the session lifetime.
        kwargs['store'] = dist.FileStore(
            str(rendezvous.store_path), context.world_size)
    dist.init_process_group(
        backend=backend,
        rank=context.rank,
        world_size=context.world_size,
        timeout=datetime.timedelta(minutes=timeout_minutes),
        **kwargs,
    )
    # Force the communicator to be built now, while all ranks are still in
    # sync right after init, rather than lazily after divergent startup. This
    # ensures a real HW/link fault is reported immediately.
    if backend == 'nccl':
        dist.barrier(device_ids=[context.local_rank])
    else:
        dist.barrier()
    return context


def maybe_destroy_distributed(context=None):
    if is_distributed(context) and dist.is_initialized():
        dist.destroy_process_group()
    set_process_context(SINGLE_PROCESS)


def broadcast_model_params(model, context=None):
    if not is_distributed(context):
        return
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor.data, src=0)


def allreduce_grads_(params, world_size=None):
    world_size = get_world_size() if world_size is None else int(world_size)
    if world_size == 1:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def split_counts_across_ranks(config, count_keys,
                              split_key='optimizer_distributed_split_batch',
                              world_size=None):
    world_size = get_world_size() if world_size is None else int(world_size)
    if world_size == 1 or not config[split_key]:
        return 1
    for key in count_keys:
        config[key] = max(1, -(-config[key] // world_size))  # ceil-divide, floor of 1
    return world_size


class StepTimer:
    # Opt-in CUDA timing for distributed runs: FIT_SPIRAL_PROFILE_STEPS=1.
    def __init__(self, enabled, report, window=200):
        self.enabled = enabled
        self.report = report
        self.window = window
        self.totals = {}
        self.count = 0
        self._events = {}

    def start(self, name):
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._events.setdefault(name, []).append([event, None])

    def stop(self, name):
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._events[name][-1][1] = event

    def tick(self):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        for name, intervals in self._events.items():
            elapsed = sum(start.elapsed_time(stop) for start, stop in intervals)
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
        self._events.clear()
        self.count += 1

    def maybe_report(self, iteration):
        if not self.enabled or self.count == 0 or iteration % self.window != 0:
            return
        avgs = {k: v / self.count for k, v in self.totals.items()}
        if self.report:
            body = ', '.join(f'{k}={v:.2f}ms' for k, v in avgs.items())
            print(f'[profile step {iteration}] avg/step over {self.count}: {body}, sum={sum(avgs.values()):.2f}ms')
        self.totals.clear()
        self.count = 0
