import datetime
import os

import torch
import torch.distributed as dist


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


def get_world_size():
    return int(os.environ.get('WORLD_SIZE', '1'))


def get_rank():
    return int(os.environ.get('RANK', '0'))


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', '0'))


def is_distributed():
    return get_world_size() > 1


def is_main_process():
    return get_rank() == 0


def maybe_init_distributed():
    if not is_distributed():
        return
    torch.cuda.set_device(get_local_rank())
    # The default NCCL PG timeout is 10 min, and the ncclUniqueId is exchanged
    # via the c10d store on the *first* collective. Since that first collective
    # (broadcast_model_params) runs only after slow, per-rank startup work, any
    # startup skew > timeout makes the fastest ranks bail in store->get with a
    # 600000ms timeout. Give startup plenty of headroom.
    timeout_min = int(os.environ.get('FIT_SPIRAL_DDP_TIMEOUT_MIN', '60'))
    dist.init_process_group(
        backend='nccl',
        timeout=datetime.timedelta(minutes=timeout_min),
    )
    # Force the NCCL communicator to be built now, while all ranks are still in
    # sync right after init, rather than lazily after divergent startup. This
    # ensures a real HW/link fault is reported immediately.
    dist.barrier(device_ids=[get_local_rank()])


def maybe_destroy_distributed():
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_model_params(model):
    if not is_distributed():
        return
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor.data, src=0)


def allreduce_grads_(params):
    world_size = get_world_size()
    if world_size == 1:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def split_counts_across_ranks(config, count_keys, split_key='distributed_split_batch'):
    world_size = get_world_size()
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
