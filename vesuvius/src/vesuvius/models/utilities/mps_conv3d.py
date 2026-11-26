"""Faster 3-D convolutions on Apple GPUs (MPS).

PyTorch's MPS backend runs ``nn.Conv3d`` through an explicit matrix product and has no float16/bfloat16
``nn.ConvTranspose3d`` at all. :func:`speed_up_3d_convs_on_mps` changes the class of every eligible 3-D
(transposed) convolution in a model, in place, to a subclass that computes the same result from operations
MPS runs natively:

* ``Conv3d`` as 2-D convolutions over its output depth planes: one convolution with the depth taps stacked into
  channels when that copy is small (the deeper layers), else one per depth tap, summed;
* ``ConvTranspose3d`` whose kernel equals its stride (the U-Net upsampling layer) as one ``einsum`` and a
  reshape, since each input voxel then writes its own disjoint output block.

The modules keep their class hierarchy (``isinstance(m, nn.Conv3d)`` still holds), their parameters (the same
objects, so an optimizer built before the swap keeps working) and their ``state_dict`` keys, so checkpoints
load and save unchanged. On any other device the swapped modules run PyTorch's own kernels. Set
``VESUVIUS_MPS_NATIVE_CONV3D=1`` to skip the swap.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

ENV_KEEP_NATIVE = "VESUVIUS_MPS_NATIVE_CONV3D"


STACK_LIMIT_BYTES = 256 * 2**20
CHUNK_BYTES = 64 * 2**20


def conv3d_as_conv2d(conv: nn.Conv3d, x: torch.Tensor, stack_limit_bytes: int = STACK_LIMIT_BYTES) -> torch.Tensor:
    """``conv(x)`` for a zero-padded ``nn.Conv3d``, computed with ``F.conv2d`` over the output depth planes.

    Output plane ``o`` reads input planes ``o * stride + k * dilation - padding`` for the ``kd`` depth taps ``k``. The
    input is read depth-major (``[D, B, C, H, W]``), so that the planes one tap reads form a slice. The output is
    returned contiguous in the usual ``[B, C, D, H, W]`` layout, as ``nn.Conv3d`` returns it, so callers that ``view``
    it (losses flattening the spatial axes, for example) work unchanged.

    Stacked channel-major (``[C, kd]``) the taps make one 2-D convolution with ``C * kd`` input channels (the weight
    ``[Co, C / groups, kd, kh, kw]`` reshaped to ``[Co, C / groups * kd, kh, kw]`` matches that order), which is the
    fastest form on MPS. MPS unfolds a 2-D convolution's input into a ``channels * kh * kw`` by ``pixels`` matrix, so
    when that matrix would pass ``stack_limit_bytes`` (the large shallow layers) the layer instead runs one 2-D
    convolution per depth tap, each over only the output planes whose input plane lies inside the volume (no padded
    copy), and adds them in place; the depth-major copy of the input is then made in chunks of about
    ``CHUNK_BYTES``.
    """
    unbatched = x.dim() == 4
    if unbatched:
        x = x.unsqueeze(0)
    kd, sd, pd, dd = conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.dilation[0]
    b, c, d, h, w = x.shape
    d_out = (d + 2 * pd - dd * (kd - 1) - 1) // sd + 1
    xd = x.permute(2, 0, 1, 3, 4)  # [D, B, C, H, W]
    stride, padding, dilation = conv.stride[1:], conv.padding[1:], conv.dilation[1:]
    unfolded = b * d_out * c * kd * conv.kernel_size[1] * conv.kernel_size[2] * h * w * x.element_size()
    if unfolded <= stack_limit_bytes:
        if pd:
            xd = torch.cat([xd.new_zeros(pd, b, c, h, w), xd, xd.new_zeros(pd, b, c, h, w)])
        taps = [xd[k * dd:k * dd + sd * (d_out - 1) + 1:sd] for k in range(kd)]  # kd x [Do, B, C, H, W]
        planes = torch.stack(taps, 3).reshape(d_out * b, c * kd, h, w)
        weight = conv.weight.reshape(conv.out_channels, -1, *conv.weight.shape[3:])
        y = F.conv2d(planes, weight, conv.bias, stride, padding, dilation, conv.groups)
    else:
        # Copy the input depth-major (none if it already is) in chunks of about CHUNK_BYTES, so each tap below is a
        # view of the chunk.
        plane_bytes = b * c * h * w * x.element_size()
        n = d_out if xd.is_contiguous() else max(1, min(stack_limit_bytes, CHUNK_BYTES) // (plane_bytes * sd))
        y, out_dtype = None, None
        for o0 in range(0, d_out, n):
            o1 = min(d_out, o0 + n)
            i0 = max(0, o0 * sd - pd)
            i1 = min(d, (o1 - 1) * sd + (kd - 1) * dd - pd + 1)
            if i1 <= i0:
                continue
            xs = xd[i0:i1].contiguous()
            for k in range(kd):
                lo = max(o0, -((k * dd - pd) // sd))  # first output plane whose input plane is >= 0
                hi = min(o1, (d - 1 + pd - k * dd) // sd + 1)  # one past the last whose input plane is < d
                if hi <= lo:
                    continue
                first = lo * sd + k * dd - pd - i0
                tap = xs[first:first + sd * (hi - lo - 1) + 1:sd].reshape((hi - lo) * b, c, h, w)
                yk = F.conv2d(tap, conv.weight[:, :, k], None, stride, padding, dilation, conv.groups)
                out_dtype = yk.dtype
                # Under bf16 autocast, sum the taps in fp32 and round once, as a native 3-D convolution does: bf16
                # partial sums double the error. fp16 partial sums cost little accuracy, and fp32 ones 23% more time.
                acc_dtype = torch.float32 if out_dtype == torch.bfloat16 else out_dtype
                if y is None and hi - lo == d_out:
                    y = yk.to(acc_dtype)
                    continue
                if y is None:
                    y = yk.new_zeros(d_out * b, *yk.shape[1:], dtype=acc_dtype)
                y[lo * b:hi * b].add_(yk)
        if y is None:  # every tap reads only padding
            h_out = (h + 2 * padding[0] - dilation[0] * (conv.kernel_size[1] - 1) - 1) // stride[0] + 1
            w_out = (w + 2 * padding[1] - dilation[1] * (conv.kernel_size[2] - 1) - 1) // stride[1] + 1
            y = x.new_zeros(d_out * b, conv.out_channels, h_out, w_out)
        if conv.bias is not None:
            y.add_(conv.bias.view(1, -1, 1, 1))
        if out_dtype is not None and y.dtype != out_dtype:
            y = y.to(out_dtype)
        if y.device.type == "mps":
            # MPS frees a buffer only once the GPU work queued on it has run. This path is fast enough that, without a
            # wait, a whole decoder of these large layers gets queued ahead and the freed copies pile up.
            torch.mps.synchronize()
    y = y.reshape(d_out, b, *y.shape[1:]).permute(1, 2, 0, 3, 4).contiguous()
    return y.squeeze(0) if unbatched else y


def conv_transpose3d_by_blocks(conv: nn.ConvTranspose3d, x: torch.Tensor) -> torch.Tensor:
    """``conv(x)`` for an ``nn.ConvTranspose3d`` with kernel == stride and no padding: an ``einsum`` and a reshape."""
    unbatched = x.dim() == 4
    if unbatched:
        x = x.unsqueeze(0)
    b, _, d, h, w = x.shape
    kd, kh, kw = conv.kernel_size
    weight = conv.weight.to(x.dtype)
    y = torch.einsum("bcdhw,cokij->bodkhiwj", x, weight)
    y = y.reshape(b, weight.shape[1], d * kd, h * kh, w * kw)
    if conv.bias is not None:
        y = y + conv.bias.to(y.dtype).view(1, -1, 1, 1, 1)
    return y.squeeze(0) if unbatched else y


class MPSConv3d(nn.Conv3d):
    """``nn.Conv3d`` that runs :func:`conv3d_as_conv2d` on MPS and PyTorch's kernel elsewhere."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "mps":
            return super().forward(x)
        return conv3d_as_conv2d(self, x)


class MPSConvTranspose3d(nn.ConvTranspose3d):
    """``nn.ConvTranspose3d`` that runs :func:`conv_transpose3d_by_blocks` on MPS and PyTorch's kernel elsewhere."""

    def forward(self, x: torch.Tensor, output_size=None) -> torch.Tensor:
        if x.device.type != "mps" or output_size is not None:
            return super().forward(x, output_size)
        return conv_transpose3d_by_blocks(self, x)


def _eligible_conv(m: nn.Module) -> bool:
    return type(m) is nn.Conv3d and m.padding_mode == "zeros" and isinstance(m.padding, tuple)


def _eligible_tconv(m: nn.Module) -> bool:
    return (type(m) is nn.ConvTranspose3d and m.groups == 1 and tuple(m.kernel_size) == tuple(m.stride)
            and tuple(m.padding) == (0, 0, 0) and tuple(m.output_padding) == (0, 0, 0)
            and tuple(m.dilation) == (1, 1, 1))


def speed_up_3d_convs(model: nn.Module) -> int:
    """Swap eligible 3-D (transposed) convolutions in ``model`` to the MPS-friendly classes; returns how many."""
    n = 0
    for m in model.modules():
        if _eligible_conv(m):
            m.__class__ = MPSConv3d
            n += 1
        elif _eligible_tconv(m):
            m.__class__ = MPSConvTranspose3d
            n += 1
    return n


def speed_up_3d_convs_on_mps(model: nn.Module, device: torch.device | str) -> int:
    """:func:`speed_up_3d_convs` when ``device`` is MPS (and ``VESUVIUS_MPS_NATIVE_CONV3D`` is not 1), else 0."""
    if torch.device(device).type != "mps" or os.environ.get(ENV_KEEP_NATIVE) == "1":
        return 0
    return speed_up_3d_convs(model)
