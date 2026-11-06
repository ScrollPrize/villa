"""Patch-lattice views for self-supervised training, on CPU or CUDA.

Arrays are C order, ZYX. A token is one cell of a patch lattice; crop origins
and shapes are in token units. Orientation ``id = perm_index * 8 + flip_bits``
(48 in all) over the lexicographic permutations of (0, 1, 2); an oriented crop
is ``numpy.flip(numpy.transpose(C, perm), axes)`` where flip bit ``k`` reverses
output axis ``k``. Every seed starts an independent SplitMix64 stream (see
``include/vc_reader/views.hpp`` for the exact draws).

``token_support``, ``correspondence`` and ``mask_patch_cells`` accept NumPy
arrays and PyTorch CPU or CUDA tensors and return the same kind of array on the
same device. CUDA work runs on the current PyTorch stream without
synchronizing, like any PyTorch operation. ``plan_view_sets``, ``patch_masks``
and ``draw_orientations`` run on the host and return NumPy arrays; masks are
small per-view grids generated sequentially, so upload them where needed.
"""
import operator
import sys

import numpy as np

from . import _views
from ._views import ORIENTATION_COUNT, orientation, orientation_id

__all__ = ["ORIENTATION_COUNT", "orientation", "orientation_id", "draw_orientations", "token_support",
           "plan_view_sets", "correspondence", "patch_masks", "mask_patch_cells"]


def _torch(value):
    torch = sys.modules.get("torch")
    return torch if torch is not None and isinstance(value, torch.Tensor) else None


def _triple(value):
    if isinstance(value, (int, np.integer)):
        return (operator.index(value),) * 3
    value = tuple(operator.index(v) for v in value)
    if len(value) != 3:
        raise ValueError("expected one integer or three (ZYX)")
    return value


def _native(value):
    """A contiguous array for the native call, and torch if it is a tensor."""
    torch = _torch(value)
    if torch is not None:
        if value.device.type not in ("cpu", "cuda"):
            raise ValueError("tensors must be on the CPU or a CUDA device")
        return value.detach().contiguous(), torch
    if hasattr(value, "__dlpack_device__") and not isinstance(value, np.ndarray):
        raise TypeError("expected a NumPy array or a PyTorch tensor")
    return np.ascontiguousarray(value), None


def _host(value, dtype):
    torch = _torch(value)
    if torch is not None:
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.size and value.dtype.kind not in "iu":
        raise TypeError("expected an integer array")
    return np.ascontiguousarray(value, dtype=dtype)


def _stream(value, torch):
    return torch.cuda.current_stream(value.device).cuda_stream if torch is not None and value.is_cuda else 0


def _empty(shape, dtype, like, torch):
    if torch is None:
        return np.empty(shape, dtype)
    return torch.empty(shape, dtype=getattr(torch, np.dtype(dtype).name), device=like.device)


def draw_orientations(seeds):
    """One orientation id per uint64 seed, uniform over [0, 48); int32 (B,)."""
    return _views._draw_orientations(_host(seeds, np.uint64).reshape(-1))


def token_support(valid, patch):
    """uint8 (..., D/P0, H/P1, W/P2): 1 iff every voxel of the cell is nonzero.

    valid: bool/uint8 (D,H,W) or (B,D,H,W); patch: int or three ints.
    """
    patch = _triple(patch)
    if any(p <= 0 for p in patch):
        raise ValueError("patch must be positive")
    valid, torch = _native(valid)
    shape = tuple(valid.shape[:-3]) + tuple(n // p for n, p in zip(valid.shape[-3:], patch))
    out = _empty(shape, np.uint8, valid, torch)
    _views._token_support(valid, patch, out, _stream(valid, torch))
    return out


def plan_view_sets(support, teacher_shape, student_shapes, *, n_sets, min_overlap_fraction, min_overlap_count,
                   max_student_attempts, max_teacher_attempts, seed):
    """Place a teacher and its students on a (T0,T1,T2) token support grid, per set.

    Origins are uniform per axis over [0, T - n]; a student proposal is accepted
    when O, the supported tokens shared with the teacher box, satisfies
    O >= min_overlap_count and O >= min_overlap_fraction * prod(student shape).
    A student out of attempts resamples its set's teacher. Sets consume one
    stream of ``seed`` in order. Returns NumPy arrays ``set_ok`` (n_sets,),
    ``teacher_origins`` int32 (n_sets,3), ``student_origins`` int32
    (n_sets,S,3), ``student_overlap`` int64 (n_sets,S), ``student_attempts``
    int32 (n_sets,S), ``teacher_attempts`` int32 (n_sets,), ``rejections``
    int64 (n_sets,), and ``ok`` (bool, every set placed). Failed sets have zero
    origins. Shapes that are not positive or do not fit raise ValueError.
    """
    torch = _torch(support)
    if torch is not None:
        support = support.detach().cpu().numpy()
    support = np.asarray(support)
    if support.dtype == np.bool_:
        support = support.view(np.uint8)
    return _views._plan_view_sets(np.ascontiguousarray(support, dtype=np.uint8), _triple(teacher_shape),
                                  [_triple(s) for s in student_shapes], operator.index(n_sets),
                                  float(min_overlap_fraction), operator.index(min_overlap_count),
                                  operator.index(max_student_attempts), operator.index(max_teacher_attempts),
                                  operator.index(seed) & 0xFFFFFFFFFFFFFFFF)


def correspondence(support, support_index, student_origins, student_shape, student_orientations,
                   teacher_origins, teacher_shape, teacher_orientations, *, lattice=False):
    """Map each oriented student token to its oriented teacher token, for N pairs.

    support: (Sg,T0,T1,T2) token grids (a single (T0,T1,T2) grid is also
    accepted); support_index (N,), origins (N,3) and orientation ids (N,) are
    small integer arrays on any device. Returns ``teacher_index`` int32
    (N, prod(student_shape)), -1 outside the teacher or where the student token
    is unsupported; ``student_valid`` uint8 (N, prod(student_shape)); and with
    ``lattice=True``, ``lattice`` int32 (N, prod(student_shape), 3), ZYX token
    coordinates. Outputs are on the device of ``support``.
    """
    support, torch = _native(support)
    if support.ndim == 3:
        support = support.reshape((1, *support.shape))
    support_index = _host(support_index, np.int64).reshape(-1)
    n = len(support_index)
    student_shape, teacher_shape = _triple(student_shape), _triple(teacher_shape)
    tokens = student_shape[0] * student_shape[1] * student_shape[2]
    result = {"teacher_index": _empty((n, tokens), np.int32, support, torch),
              "student_valid": _empty((n, tokens), np.uint8, support, torch)}
    if lattice:
        result["lattice"] = _empty((n, tokens, 3), np.int32, support, torch)
    _views._correspondence(support, support_index, _host(student_origins, np.int64), student_shape,
                           _host(student_orientations, np.int64).reshape(-1), _host(teacher_origins, np.int64),
                           teacher_shape, _host(teacher_orientations, np.int64).reshape(-1),
                           result["teacher_index"], result["student_valid"], result.get("lattice"),
                           _stream(support, torch))
    return result


def patch_masks(shape, seeds, *, mode, probability, fraction, min_block_tokens=4, max_block_tokens=None,
                min_aspect=0.3, max_aspect=None, attempts=10, cyclic_shift=True):
    """Token masks over a (n0,n1,n2) grid, one independent stream per uint64 seed.

    Returns NumPy ``masks`` uint8 (B,n0,n1,n2), ``applied`` bool (B,) and
    ``counts`` int64 (B,). An applied mask has exactly round_half_up(fraction * N)
    tokens. ``mode`` is ``random`` (uniform without replacement) or
    ``multiblock`` (3D DINOv3-style cuboids, then uniform fill of any shortfall,
    then an optional cyclic shift). Block dimensions round half to even.
    ``max_block_tokens`` defaults to N and ``max_aspect`` to 1 / min_aspect.
    """
    return _views._patch_masks(_triple(shape), _host(seeds, np.uint64).reshape(-1), mode, float(probability),
                               float(fraction), float(min_block_tokens),
                               None if max_block_tokens is None else float(max_block_tokens), float(min_aspect),
                               None if max_aspect is None else float(max_aspect), operator.index(attempts),
                               bool(cyclic_shift))


def mask_patch_cells(images, masks, patch, *, fill=0.0, out=None):
    """Set every voxel of masked token cells to ``fill`` in all channels.

    images: float32 (or float16/bfloat16 tensors) (B,C,D,H,W); masks: bool/uint8
    (B,D/P0,H/P1,W/P2) on the same device. Returns a new array unless ``out`` is
    given; ``out`` may be ``images`` itself for in-place masking. No autograd.
    """
    patch = _triple(patch)
    images, torch = _native(images)
    if torch is not None and _torch(masks) is None:
        masks = torch.as_tensor(np.asarray(masks), device=images.device)
    masks, _ = _native(masks)
    if out is None:
        out = torch.empty_like(images) if torch is not None else np.empty_like(images)
    target = out.detach() if torch is not None and _torch(out) is not None else out
    _views._mask_patch_cells(images, masks, patch, float(fill), target, _stream(images, torch))
    return out
