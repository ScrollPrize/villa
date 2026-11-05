"""CUDA/HIP graph replay of the eager RK4 slab integration (FIT_SPIRAL_GRAPH_RK4=1).

Where the fused Triton kernel is unavailable, flow_fields._RK4SparseFlowIntegrate
launches many small kernels per slab, and on some ROCm devices launch overhead
dominates. RK4GraphIntegrate runs the same forward and adjoint arithmetic, but
captures each as a graph keyed by (point-count bucket, field pointers) and
replays it. Point counts are padded to a bucket; padding points sit at 0.5 and
get zero gradient, so they add exactly 0 to the field accumulator.

Outputs are cloned out of the static buffers after each replay. All graphs
share one memory pool, which is safe because replays are sequential on one
stream and the cache holds every static tensor. Each cache keeps at most
FIT_SPIRAL_GRAPH_RK4_MAX graphs (default 256) and evicts the oldest first.
"""
import os

import torch

import flow_fields as ff
import prefetch

_POOL = None
_FWD = {}
_BWD = {}


def clear():
    # Drop every cached graph, e.g. when a field buffer is reallocated.
    _FWD.clear()
    _BWD.clear()


def _bucket(n):
    # Round up so that padding wastes <= 1/8 of the points.
    if n <= 1024:
        return 1024
    step = 1 << (n.bit_length() - 4)
    return ((n + step - 1) // step) * step


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = torch.cuda.graph_pool_handle()
    return _POOL


def _capture(fn):
    # The step prefetcher issues CUDA work from a worker thread, which is not
    # allowed during capture; wait for its pending jobs first.
    if prefetch._PREFETCHER is not None:
        prefetch._PREFETCHER.wait_idle()
    torch.cuda.synchronize()
    # Warm up on a side stream (required before capture), then capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=_pool()):
        fn()
    torch.cuda.synchronize()
    return g


def _forward_body(y_in, stages, y_out, low, high, h, n_steps):
    y, _ = ff._rk4_forward_stages(
        y_in, lambda p: ff.sample_field(p, low) + ff.sample_field(p, high),
        h, n_steps, stages)
    y_out.copy_(y)


def _backward_body(g_in, stages, g_out, low, high, acc, h, n_steps):
    g_out.copy_(ff._rk4_sparse_backward(
        g_in, stages, low, high, 1.0, acc, h, n_steps))


def _evict(cache):
    if len(cache) >= int(os.environ.get('FIT_SPIRAL_GRAPH_RK4_MAX', '256')):
        cache.pop(next(iter(cache)))


def _fwd_entry(nb, low, high, h, n_steps, dtype):
    key = (nb, low.data_ptr(), high.data_ptr(), tuple(low.shape), low.stride(), high.stride(),
           h, n_steps, dtype)
    e = _FWD.get(key)
    if e is None:
        _evict(_FWD)
        y_in = torch.full((nb, 3), 0.5, device=low.device, dtype=dtype)
        stages = torch.empty((4 * n_steps, nb, 3), device=low.device, dtype=dtype)
        y_out = torch.empty((nb, 3), device=low.device, dtype=dtype)
        g = _capture(lambda: _forward_body(y_in, stages, y_out, low, high, h, n_steps))
        # Entries hold the fields (and acc) so their memory can't be reused by
        # another tensor while a graph keyed on its address is cached.
        e = _FWD[key] = (g, y_in, stages, y_out, low, high)
    return e


def _bwd_entry(nb, low, high, acc, h, n_steps, dtype):
    key = (nb, low.data_ptr(), high.data_ptr(), None if acc is None else acc.data_ptr(),
           tuple(low.shape), low.stride(), high.stride(), h, n_steps, dtype)
    e = _BWD.get(key)
    if e is None:
        _evict(_BWD)
        g_in = torch.zeros((nb, 3), device=low.device, dtype=dtype)
        stages = torch.full((4 * n_steps, nb, 3), 0.5, device=low.device, dtype=dtype)
        g_out = torch.empty((nb, 3), device=low.device, dtype=dtype)
        # Warmup and capture run with zero gradient, so they add 0 to acc.
        g = _capture(lambda: _backward_body(g_in, stages, g_out, low, high, acc, h, n_steps))
        e = _BWD[key] = (g, g_in, stages, g_out, low, high, acc)
    return e


class RK4GraphIntegrate(torch.autograd.Function):
    # Drop-in for _RK4SparseFlowIntegrate.apply(y0, low, high, 1.0, acc, h, n_steps).
    # Graphs bake in the pointers of low, high and acc; keep their storage
    # across calls, or each new buffer is captured (and held) again.

    @staticmethod
    def forward(ctx, y0, low, high, acc, h, n_steps):
        ctx.set_materialize_grads(False)
        h, n_steps = float(h), int(n_steps)
        n = y0.shape[0]
        nb = _bucket(n)
        g, y_in, stages, y_out = _fwd_entry(nb, low, high, h, n_steps, y0.dtype)[:4]
        y_in[:n].copy_(y0)
        if nb > n:
            y_in[n:].fill_(0.5)
        g.replay()
        ctx.save_for_backward(stages.clone())
        ctx.low, ctx.high, ctx.acc = low, high, acc
        ctx.h, ctx.n_steps, ctx.n, ctx.nb = h, n_steps, n, nb
        return y_out[:n].clone()

    @staticmethod
    def backward(ctx, grad_y):
        if grad_y is None:
            return None, None, None, None, None, None
        (saved,) = ctx.saved_tensors
        n, nb = ctx.n, ctx.nb
        g, g_in, stages, g_out = _bwd_entry(
            nb, ctx.low, ctx.high, ctx.acc, ctx.h, ctx.n_steps, grad_y.dtype)[:4]
        stages.copy_(saved)
        g_in[:n].copy_(grad_y.reshape(n, 3))
        if nb > n:
            g_in[n:].zero_()
        g.replay()
        return g_out[:n].clone(), None, None, None, None, None
