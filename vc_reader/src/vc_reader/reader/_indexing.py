"""Shared NumPy-style basic indexing for disk-backed and aligned readers."""

from typing import Any
import operator

import numpy as np


def parse_index(key: Any, shape: tuple[int, ...]) -> tuple[list[int], list[int], list[bool]]:
    """numpy basic indexing (ints, unit-step slices, Ellipsis) -> start, size, axes to drop."""
    if not isinstance(key, tuple):
        key = (key,)
    key = expand_index(key, len(shape))
    start, size, squeeze = [], [], []
    for k, dim in zip(key, shape):
        if isinstance(k, slice):
            a, b, step = k.indices(dim)
            if step != 1:
                raise IndexError("only unit-step slices are supported")
            start.append(a)
            size.append(max(b - a, 0))
            squeeze.append(False)
        else:
            if isinstance(k, (bool, np.bool_)):
                raise IndexError("boolean indexing is not supported")
            try:
                i = operator.index(k)
            except TypeError:
                raise IndexError("indices must be integers, unit-step slices, or ellipsis") from None
            if i < 0:
                i += dim
            if not 0 <= i < dim:
                raise IndexError(f"index {k} out of bounds for axis of size {dim}")
            start.append(i)
            size.append(1)
            squeeze.append(True)
    return start, size, squeeze


def expand_index(keys: tuple, ndim: int) -> tuple:
    if sum(k is Ellipsis for k in keys) > 1:
        raise IndexError("an index can only have a single ellipsis")
    if any(k is Ellipsis for k in keys):
        i = next(i for i, k in enumerate(keys) if k is Ellipsis)
        keys = keys[:i] + (slice(None),) * max(0, ndim - len(keys) + 1) + keys[i + 1:]
    if len(keys) > ndim:
        raise IndexError(f"too many indices for array of dimension {ndim}")
    if any(k is None for k in keys):
        raise IndexError("newaxis indexing is not supported")
    return keys + (slice(None),) * (ndim - len(keys))
