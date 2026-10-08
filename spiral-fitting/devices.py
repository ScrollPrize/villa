"""The torch device a fit runs on, and the float precision it supports.

CUDA is used whenever it is available, exactly as before. Without CUDA the
fit runs on Apple's MPS backend, and without that on the CPU.
FIT_SPIRAL_DEVICE=cuda|mps|cpu overrides the choice.

MPS has no float64. Code that wants a float64 device tensor asks float_hi()
for the dtype instead: float64 on CUDA and the CPU, float32 on MPS.
FIT_SPIRAL_MAX_PRECISION_FLOAT=32 caps it at float32 on every device, so a
CPU run can use the same precision as an MPS one. This is the switch
lasagna/dtypes.py has (villa PR #1639), except that the default here keeps
float64 wherever the device supports it.

scatter_reduce_() stands in for the int64 amax/amin scatter MPS also lacks.
"""

import os

import torch

DEVICE_TYPES = ('cuda', 'mps', 'cpu')


def fit_device(requested=None):
    """Return ``requested``, else $FIT_SPIRAL_DEVICE, else the first of
    CUDA, MPS and CPU that is available, as a torch.device.

    CUDA comes back as the bare 'cuda' device, so the current device a
    distributed driver selected still applies.
    """
    name = str(requested or os.environ.get('FIT_SPIRAL_DEVICE') or '').strip().lower()
    if not name:
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    if name not in DEVICE_TYPES:
        raise ValueError(
            f'unsupported device {name!r}: expected one of {", ".join(DEVICE_TYPES)}')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('device cuda was requested but CUDA is not available')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('device mps was requested but MPS is not available')
    return torch.device(name)


def float_hi(device=None):
    """The high-precision float dtype for tensors on ``device`` (None is
    torch's default device, the CPU)."""
    bits = os.environ.get('FIT_SPIRAL_MAX_PRECISION_FLOAT', '64').strip()
    if bits not in ('32', '64'):
        raise ValueError(
            f'FIT_SPIRAL_MAX_PRECISION_FLOAT must be 32 or 64, got {bits!r}')
    if bits == '32' or (device is not None and torch.device(device).type == 'mps'):
        return torch.float32
    return torch.float64


def scatter_reduce_(target, index, source, reduce):
    """``target.scatter_reduce_(0, index, source, reduce=reduce)``. MPS has no
    int64 amax/amin, so those run on the CPU there and are copied back."""
    if (target.device.type == 'mps' and target.dtype == torch.int64
            and reduce in ('amax', 'amin')):
        return target.copy_(target.cpu().scatter_reduce_(
            0, index.cpu(), source.cpu(), reduce=reduce))
    return target.scatter_reduce_(0, index, source, reduce=reduce)
