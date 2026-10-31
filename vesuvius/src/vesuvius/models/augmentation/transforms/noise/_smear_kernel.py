"""Compiled kernel for :class:`SmearTransform`.

The smear is a recurrence along one axis: every slice is blended with the mean
of the ``k`` slices before it, *after* those were themselves updated. The loop
therefore cannot be vectorised over the smear axis, and as a Python loop of tiny
torch ops it is dominated by dispatch overhead. This kernel runs the same
recurrence in compiled code with the same float32 operation order, so the result
is bit-for-bit identical to the torch implementation.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def smear_inplace(mf, B, R, k, shift0, shift1, keep, blend, k_float):
    """Smear ``mf`` in place along axis 0.

    Args:
        mf: float32, C-contiguous, shape ``(N, A, B * R)``. Axis 0 is the smear axis,
            ``(A, B)`` are the two axes the shift is applied to and ``R`` is the
            flattened remainder.
        B, R: sizes used to split the last axis of ``mf``.
        k: number of previous slices to aggregate.
        shift0, shift1: wrap-around shift along ``A`` and ``B`` (``torch.roll`` semantics).
        keep, blend: float32 ``1 - alpha`` and ``alpha``.
        k_float: float32 ``k``.
    """
    n, a_size, size = mf.shape
    buf = np.empty(size, np.float32)
    for i in range(k, n):
        for a in range(a_size):
            src_a = (a - shift0) % a_size
            for p in range(size):
                buf[p] = np.float32(0.0)
            for j in range(i - k, i):
                for p in range(size):
                    buf[p] = buf[p] + mf[j, src_a, p]
            if shift1 % B == 0:
                for p in range(size):
                    mf[i, a, p] = keep * mf[i, a, p] + blend * (buf[p] / k_float)
            else:
                for b in range(B):
                    src = ((b - shift1) % B) * R
                    for r in range(R):
                        mf[i, a, b * R + r] = keep * mf[i, a, b * R + r] + blend * (buf[src + r] / k_float)
