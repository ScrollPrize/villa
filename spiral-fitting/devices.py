"""The torch device a fit runs on, and the float precision it supports.

CUDA is used whenever it is available, exactly as before. Without CUDA the
fit runs on Apple's MPS backend, and without that on the CPU.
FIT_SPIRAL_DEVICE=cuda|mps|cpu overrides the choice.

MPS has no float64. Code that wants a float64 device tensor asks float_hi()
for the dtype instead: float64 on CUDA and the CPU, float32 on MPS.
FIT_SPIRAL_MAX_PRECISION_FLOAT=32 caps it at float32 on every device, so a
CPU run can use the same precision as an MPS one. It is read and checked
once, when fit_device() picks the device, so a bad value fails before any
input loads. This is the switch
lasagna/dtypes.py has (villa PR #1639), except that the default here keeps
float64 wherever the device supports it.

Counts are kept in count_dtype(): float64 where float_hi() is float64, as
before, and int64 where it is float32, whose integers are exact only up to
2**24.

scatter_reduce_() stands in for the int64 amax/amin scatter MPS also lacks.
"""

import functools
import os

import torch

DEVICE_TYPES = ('cuda', 'mps', 'cpu')


def fit_device(requested=None):
    """Return ``requested`` (a torch.device or a name such as 'cuda:1'),
    else $FIT_SPIRAL_DEVICE, else the first of CUDA, MPS and CPU that is
    available, as a torch.device.

    CUDA comes back as the bare 'cuda' device, so the current device a
    distributed driver selected still applies.
    """
    max_precision_bits()
    name = requested if isinstance(requested, torch.device) else str(
        requested or os.environ.get('FIT_SPIRAL_DEVICE') or '').strip().lower()
    if not name:
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    try:
        device = torch.device(name)
    except RuntimeError:
        device = None
    if device is None or device.type not in DEVICE_TYPES:
        raise ValueError(
            f'unsupported device {str(name)!r}: expected one of '
            f'{", ".join(DEVICE_TYPES)}, optionally with an index (cuda:1)')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('device cuda was requested but CUDA is not available')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('device mps was requested but MPS is not available')
    return device


@functools.cache
def max_precision_bits():
    """$FIT_SPIRAL_MAX_PRECISION_FLOAT as 32 or 64, read once per process."""
    bits = os.environ.get('FIT_SPIRAL_MAX_PRECISION_FLOAT', '64').strip()
    if bits not in ('32', '64'):
        raise ValueError(
            f'FIT_SPIRAL_MAX_PRECISION_FLOAT must be 32 or 64, got {bits!r}')
    return int(bits)


def float_hi(device=None):
    """The high-precision float dtype for tensors on ``device`` (None is
    torch's default device, the CPU)."""
    if max_precision_bits() == 32 or (device is not None and torch.device(device).type == 'mps'):
        return torch.float32
    return torch.float64


def count_dtype(device=None):
    """The dtype counts on ``device`` add up in: float64 where float_hi() is
    float64 (exact to 2**53), else int64, since float32 is exact only to
    2**24."""
    return torch.float64 if float_hi(device) == torch.float64 else torch.int64


def scatter_reduce_(target, index, source, reduce):
    """``target.scatter_reduce_(0, index, source, reduce=reduce)``. MPS has no
    int64 amax/amin, so those run on the CPU there and are copied back."""
    if (target.device.type == 'mps' and target.dtype == torch.int64
            and reduce in ('amax', 'amin')):
        return target.copy_(target.cpu().scatter_reduce_(
            0, index.cpu(), source.cpu(), reduce=reduce))
    return target.scatter_reduce_(0, index, source, reduce=reduce)
