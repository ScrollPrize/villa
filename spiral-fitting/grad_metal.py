"""Fused Metal kernel for the per-step nonfinite-gradient sanitize on MPS.

FitContext._sanitize_nonfinite_grads_ counts and zeroes NaN and +/-inf
gradient entries for every trainable parameter. Eagerly that is an aminmax,
two isfinite checks, the counter updates and an in-place nan_to_num_ per
parameter: about thirty MPS ops a step, which cost ~9 ms of host dispatch and
~10 ms of GPU time on an 18.8M-parameter fit, for what is one read of the
gradients. Here one dispatch per parameter reads its gradient once, writes
zero only where an entry is nonfinite and, the first time a parameter's
gradient has one, adds 1 to that parameter's counter; one final single-thread
dispatch adds 1 to the step counter when any parameter had one and clears the
flags for the next step. The result is the same as the eager path: the same
zeroed entries, and the same counts.

Compiled once per process with torch.mps.compile_shader (PyTorch's safe math
mode). Set FIT_SPIRAL_METAL=0 to use the eager path.
"""
import os

import torch

_GROUP = 256
# grid-stride launch: enough threads to fill the GPU, each walks the rest
_MAX_THREADS = 1 << 18
_LIB = None

_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

kernel void sanitize_grad(device float* grad [[buffer(0)]],
                          device atomic_uint* flags [[buffer(1)]],
                          device float* counter [[buffer(2)]],
                          constant long& n [[buffer(3)]],
                          constant long& slot [[buffer(4)]],
                          uint tid [[thread_position_in_grid]],
                          uint threads [[threads_per_grid]]) {
    bool bad = false;
    for (long i = long(tid); i < n; i += long(threads)) {
        if (!isfinite(grad[i])) {
            grad[i] = 0.0f;
            bad = true;
        }
    }
    // the first thread to see a nonfinite entry of this parameter counts it
    if (bad && atomic_exchange_explicit(&flags[slot], 1u, memory_order_relaxed) == 0u) {
        counter[0] += 1.0f;
    }
}

// dispatches on one encoder run in order, so every sanitize_grad has finished
kernel void count_nonfinite_step(device uint* flags [[buffer(0)]],
                                 device float* steps [[buffer(1)]],
                                 constant long& count [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]]) {
    if (tid != 0) {
        return;
    }
    uint any = 0u;
    for (long i = 0; i < count; ++i) {
        any |= flags[i];
        flags[i] = 0u;
    }
    if (any != 0u) {
        steps[0] += 1.0f;
    }
}
'''


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SOURCE)
    return _LIB


class NonfiniteGradSanitizer:
    """Counts and zeroes nonfinite gradient entries with one dispatch per
    parameter. ``available`` is False off MPS, with FIT_SPIRAL_METAL=0, or
    when a gradient or counter is not a contiguous float32 MPS tensor; the
    caller then uses its eager path."""

    def __init__(self, named_params, by_param, steps):
        self.named_params = named_params
        self.by_param = by_param
        self.steps = steps
        self.flags = None
        if self._enabled():
            self.flags = torch.zeros(len(named_params), dtype=torch.int32,
                                     device=steps.device)

    def _enabled(self):
        if os.environ.get('FIT_SPIRAL_METAL', '1') == '0':
            return False
        if not torch.backends.mps.is_available():
            return False
        counters = [self.steps, *self.by_param.values()]
        return all(c.device.type == 'mps' and c.dtype == torch.float32
                   for c in counters)

    def available(self):
        if self.flags is None:
            return False
        for _, p in self.named_params:
            g = p.grad
            if g is not None and not (g.device.type == 'mps'
                                      and g.dtype == torch.float32
                                      and g.is_contiguous()):
                return False
        return True

    def __call__(self):
        lib = _lib()
        for slot, (name, p) in enumerate(self.named_params):
            g = p.grad
            if g is None or g.numel() == 0:
                continue
            n = g.numel()
            lib.sanitize_grad(g, self.flags, self.by_param[name], int(n),
                              int(slot), threads=min(n, _MAX_THREADS),
                              group_size=_GROUP)
        lib.count_nonfinite_step(self.flags, self.steps,
                                 int(len(self.named_params)), threads=1,
                                 group_size=1)
